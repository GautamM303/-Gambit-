"""Real-robot backend on the bbos daemons.  *** HARDWARE-UNVERIFIED ***

Built from the daemon sources shipped in demos/ (daemon.py, constants.py, ik.py) and the
view_*.py examples; nothing here has run on a robot yet. Every assumption that must be
checked before the first --execute run is listed in CHECKLIST below and in the README.

Channels
  <arm>.state / .ctrl / .torque   joint state; position command (motor turns); enable
  camera.head.jpeg                head camera frames
  led.ctrl                        status colour (must be re-written every < 3 s)
  speaker.audio                   PCM int16 chunks (speech via espeak-ng if installed, else chimes)
  wakeword.state                  the hands-free 'my turn is complete' signal

Safety
  * without --execute nothing moves: torque stays off, no ctrl is written, every
    waypoint is printed instead
  * every target is checked against the workspace box in the calibration before any
    IK is attempted; an IK failure aborts the step (the arm never moves part-way)
  * the stop flag is polled every control tick (20 ms); estop() drops torque on all
    joints and stops writing ctrl (the daemon also cuts torque when the ctrl writer
    disappears)
  * speeds: <= line_speed free, carry_speed with a piece, descent_speed near the board
  * the daemon's own limits (current, temperature, rate clip, gripper torque 300/1000)
    stay active underneath
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .config import GambitConfig
from .robot import EStop, Robot, RobotError, StopFlag
from .ui import tts_pcm

CHECKLIST = """
BEFORE --execute, verify on the robot (see README 'Bring-up'):
 1. `gambit.py check --robot` : IK reaches every square + tray slot from the calibration (dry run).
 2. DOWN_QUAT_XYZW really points the fingers down with the pads closing along the board's file
    axis (view_ik.py shows the eef frame; adjust GRIP_YAW if the pads close along the rank axis).
 3. The tool frame offset PAD_BELOW_TOOL / PAD_AHEAD (from the URDF eef link to the pad centre).
 4. Gripper angles: gripper_open clears neighbouring pieces; grip_empty_angle from closing on
    nothing vs on a piece (gambit.py calibrate --gripper).
 5. The home pose keeps the arm out of the head camera's view of the board.
 6. The camera can see all four ArUco markers with the arm parked.
"""

TICK = 0.02
IK_FRAME_Z = 0.0304               # arm_base sits this far above the robot origin (URDF)
DOWN_QUAT_XYZW = (1.0, 0.0, 0.0, 0.0)   # fingers down in the arm_base frame (pick_by_voice.py)
PAD_AHEAD = 0.015                 # pad centre this far ahead (+x of the hand) of the tool frame
PAD_BELOW_TOOL = 0.023            # pads reach this far below the tool frame


class BBOSRobot(Robot):
    def __init__(self, cfg: GambitConfig, execute=False, log=print):
        from bbos import Config, Reader, Type, Writer
        self.cfg, self.execute, self.log = cfg, execute, log
        self.stop = StopFlag()
        arm = cfg.arm
        self.acfg = Config(arm)
        self.acfg.ik.init()
        self.names = list(self.acfg.joint_names)
        self.chain = self._chain_indices()
        self.grip_idx = next((i for i, n in enumerate(self.names) if "gripper" in n), None)

        self.state = Reader(f"{arm}.state", Type("arm_state")).__enter__()
        self.cam = Reader("camera.head.jpeg").__enter__()
        self.wake = Reader("wakeword.state", keeptime=False).__enter__()
        self.w_led = Writer("led.ctrl", Type("led_ctrl"), keeptime=False).__enter__()
        self.w_audio = Writer("speaker.audio", Type("speaker_audio")).__enter__()
        self.spk = Config("speaker")
        self.w_ctrl = self.w_torque = None
        if execute:
            self.w_ctrl = Writer(f"{arm}.ctrl", Type("arm_ctrl")).__enter__()
            self.w_torque = Writer(f"{arm}.torque", Type("arm_torque")).__enter__()

        deadline = time.monotonic() + 5.0
        while not self.state.ready() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.state.ready():
            raise RobotError(f"no {arm}.state: is the arm daemon running?")
        self.q_motor = np.array(self.state.data["pos"], np.float64)
        self.q_urdf = np.asarray(self.acfg.q2urdf(self.q_motor.astype(np.float32)), np.float64)
        home_urdf = np.asarray(self.acfg.q2urdf(np.asarray(self.acfg.home, np.float32)), np.float64)
        self.home_chain = [float(home_urdf[i]) for i in self.chain]
        self.acfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])
        self._led_rgb, self._led_period = (0, 0, 0), 0
        self._wake_seen = False
        self._grip_ok = False
        self.arm_state = "unknown"
        self.enabled = False
        log(f"{arm}: chain {[self.names[i] for i in self.chain]}, gripper j{self.grip_idx}; "
            f"{'EXECUTE' if execute else 'DRY RUN (no motion)'}")
        log(CHECKLIST)

    # -- set-up ------------------------------------------------------------- #
    def _chain_indices(self):
        import yourdfpy
        urdf = yourdfpy.URDF.load(self.acfg.urdf_path, load_meshes=False, build_scene_graph=False)
        jd = {j.name: (j.parent, j.child) for j in urdf.robot.joints
              if j.type in ("revolute", "prismatic", "continuous") and j.name in self.names}
        roots = {p for p, _ in jd.values()} - {c for _, c in jd.values()}
        cur, chain = next(iter(roots)), []
        while True:
            nxt = next((jn for jn, (p, _) in jd.items() if p == cur and jn not in chain), None)
            if nxt is None:
                break
            chain.append(nxt)
            cur = jd[nxt][1]
        return [self.names.index(j) for j in chain[:7]]

    def _to_motor(self, q_urdf):
        return np.asarray(self.acfg.urdf2q(np.asarray(q_urdf, np.float32)), np.float64)

    # -- control tick ---------------------------------------------------------- #
    def _tick(self):
        self.stop.check()
        with self.w_led.buf() as b:
            b["rgb"] = np.array(self._led_rgb, dtype=np.uint8)
            b["brightness"] = np.int16(-1)
            b["period_ms"] = np.uint16(self._led_period)
        if self.execute and self.enabled:
            with self.w_torque.buf() as b:
                b["enable"][:] = np.ones(self.acfg.dof, dtype=np.bool_)
            if self.w_ctrl.ready():
                with self.w_ctrl.buf() as b:
                    b["pos"][:] = self._to_motor(self.q_urdf).astype(np.float32)
        time.sleep(TICK)

    def measured_urdf(self):
        return np.asarray(self.acfg.q2urdf(np.array(self.state.data["pos"], np.float32)), np.float64)

    def enable(self):
        if self.execute and not self.enabled:
            # seed the command from the real pose so enabling never jumps the arm
            self.q_urdf = self.measured_urdf()
            self.acfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])
            self.enabled = True
            for _ in range(10):
                self._tick()

    # -- io ---------------------------------------------------------------------- #
    def led(self, rgb, period_ms=0):
        self._led_rgb, self._led_period = rgb, period_ms
        self._tick()

    def chime(self, notes):
        sr, ch, cs = self.spk.sample_rate, self.spk.channels, self.spk.chunk_size
        wave = np.concatenate([0.25 * np.sin(2 * np.pi * hz * np.arange(int(s * sr)) / sr) for hz, s in notes])
        edge = int(0.005 * sr)
        wave[:edge] *= np.linspace(0, 1, edge)
        wave[-edge:] *= np.linspace(1, 0, edge)
        self._play((32767 * wave).astype(np.int16))

    def _play(self, pcm):
        ch, cs = self.spk.channels, self.spk.chunk_size
        frames = pcm[:, None].repeat(ch, axis=1)
        frames = np.concatenate([frames, np.zeros((-len(frames) % cs, ch), np.int16)])
        for chunk in frames.reshape(-1, cs, ch):
            with self.w_audio.buf() as b:
                b["audio"] = chunk

    def say(self, text):
        self.log(f"[say] {text}")
        pcm = tts_pcm(text, self.spk.sample_rate)
        if pcm is not None:
            self._play(pcm)
        else:
            self.chime([(660, 0.08), (880, 0.10)])

    def wait_signal(self, timeout):
        """Edge-triggered wake word."""
        if self.wake.ready():
            active = bool(self.wake.data["active"])
            fired = active and not self._wake_seen
            self._wake_seen = active
            return fired
        return False

    def look(self):
        deadline = time.monotonic() + 3.0
        while not self.cam.ready() and time.monotonic() < deadline:
            self._tick()
        if not self.cam.ready():
            raise RobotError("no camera.head.jpeg: is the camera daemon running?")
        d = self.cam.data
        jpeg = np.frombuffer(bytes(d["jpeg"][:int(d["jpeg_len"])]), np.uint8)
        bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RobotError("could not decode the head camera frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # -- kinematics ------------------------------------------------------------ #
    def tool_pos(self):
        pos, _ = self.acfg.ik.fk([float(self.q_urdf[i]) for i in self.chain])
        return np.asarray(pos, np.float64) + [0, 0, IK_FRAME_Z]

    def _check_workspace(self, pos):
        (x0, x1), (y0, y1), (z0, z1) = self.cfg.motion.workspace
        if not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1 and z0 <= pos[2] <= z1):
            raise RobotError(f"target {np.round(pos, 3)} is outside the allowed workspace")

    def _solve(self, pos):
        self._check_workspace(pos)
        pos_ik = np.asarray(pos, np.float64) - [0, 0, IK_FRAME_Z]
        self.acfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])
        sol = self.acfg.ik.solve([float(v) for v in pos_ik], [float(v) for v in DOWN_QUAT_XYZW])
        if sol is None or len(sol) == 0:
            raise RobotError(f"IK failed at {np.round(pos, 3)}")
        sol = np.asarray(sol, np.float64)
        reached, _ = self.acfg.ik.fk(list(sol))
        err = np.linalg.norm(np.asarray(reached) - pos_ik)
        if err > 0.008:
            raise RobotError(f"IK only reached within {err * 1000:.0f} mm of {np.round(pos, 3)}")
        return sol

    def move_line(self, target, speed):
        """Straight line at the fixed 'down' orientation. The whole line is solved before
        anything moves, so an unreachable target never leaves the arm half-way."""
        target = np.asarray(target, np.float64)
        cur = self.tool_pos()
        n = max(1, int(np.ceil(np.linalg.norm(target - cur) / (speed * TICK))))
        path = [cur + (target - cur) * (k + 1) / n for k in range(n)]
        q_saved = self.q_urdf.copy()
        sols = []
        for p in path:
            sols.append(self._solve(p))
            for k, i in enumerate(self.chain):
                self.q_urdf[i] = sols[-1][k]
        self.q_urdf = q_saved
        self.acfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])
        self.log(f"   -> {np.round(target, 3)} at {speed:.2f} m/s ({n} steps)")
        if not self.execute:
            for k, i in enumerate(self.chain):
                self.q_urdf[i] = sols[-1][k]
            self.log(f"      joints (urdf): {np.round(self.q_urdf, 3)}")
            return
        for sol in sols:
            for k, i in enumerate(self.chain):
                self.q_urdf[i] = sol[k]
            self._tick()
        for _ in range(int(0.3 / TICK)):
            self._tick()

    def gripper(self, angle, settle=0.5):
        if self.grip_idx is None:
            return
        start = self.q_urdf[self.grip_idx]
        steps = int(0.5 / TICK)
        for k in range(steps):
            self.q_urdf[self.grip_idx] = start + (angle - start) * (k + 1) / steps
            self._tick()
        for _ in range(int(settle / TICK)):
            self._tick()

    def gripper_measured(self):
        return float(self.measured_urdf()[self.grip_idx]) if self.grip_idx is not None else None

    def _joint_move(self, target_chain, rate=0.3):
        start = self.q_urdf.copy()
        target = start.copy()
        for k, i in enumerate(self.chain):
            target[i] = target_chain[k]
        steps = int(max(2.0, np.abs(target - start).max() / rate) / TICK)
        if not self.execute:
            self.q_urdf = target
            self.log(f"   joint move -> {np.round(target, 3)}")
            return
        for k in range(steps):
            self.q_urdf = start + (target - start) * (k + 1) / steps
            self._tick()
        self.acfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])

    # -- Robot interface ---------------------------------------------------------- #
    def _ready_pose(self):
        g = self.cfg.board
        cx, cy = g.board_uv_to_xy(4.0, -1.0)       # just in front of the near edge, centred
        return np.array([cx, cy, g.board_z + 0.15])

    def ready(self):
        self.enable()
        if self.arm_state != "ready":
            self.gripper(self.cfg.motion.gripper_closed, settle=0.2)
            self.move_line(self._ready_pose(), self.cfg.motion.line_speed)
            self.arm_state = "ready"

    def park(self):
        if self.arm_state == "home":
            return
        self.enable()
        if self.arm_state == "ready":
            self.gripper(self.cfg.motion.gripper_closed, settle=0.2)
        self._joint_move(self.home_chain)
        self.arm_state = "home"

    def _grasp_z(self, kind):
        g, m = self.cfg.board, self.cfg.motion
        top = g.piece_top.get(kind, g.piece_top["p"])
        return g.board_z + top - m.grasp_below_top - 0.5 * (top - g.piece_top["p"])

    def pick(self, xy, kind):
        m = self.cfg.motion
        self.ready()
        grasp = np.array([xy[0] - PAD_AHEAD, xy[1], self._grasp_z(kind)])
        up = np.array([0, 0, m.approach])
        try:
            self.move_line(grasp + up, m.line_speed)
            self.gripper(m.gripper_open, settle=0.3)
            self.move_line(grasp, m.descent_speed)
            self.gripper(m.gripper_closed, settle=0.6)
            angle = self.gripper_measured()
            if self.execute and angle is not None and abs(angle - m.gripper_closed) < m.grip_empty_angle:
                self.log(f"      gripper closed to {angle:.3f} rad: nothing between the pads")
                self._grip_ok = False
                return False
            self._grip_ok = True
            self.held_kind = kind
            self.move_line(grasp + up, m.descent_speed)
            far = grasp[0] > m.far_x
            self.move_line(self._ready_pose(), m.far_speed if far else m.carry_speed)
        except RobotError as e:
            self.log(f"      pick aborted: {e}")
            return False
        return self.in_hand()

    def place(self, xy, surface_z):
        m = self.cfg.motion
        kind = getattr(self, "held_kind", "p")
        # the piece hangs from the pads at its grasp height: release 4 mm above the surface
        z = surface_z + (self._grasp_z(kind) - self.cfg.board.board_z) + 0.004
        place = np.array([xy[0] - PAD_AHEAD, xy[1], z])
        up = np.array([0, 0, m.approach])
        try:
            self.move_line(place + up, m.carry_speed)
            if not self.in_hand():
                return False
            self.move_line(place, m.descent_speed)
            self.gripper(m.gripper_open, settle=0.4)
            self._grip_ok = False
            self.move_line(place + up, m.descent_speed)
            self.gripper(m.gripper_closed, settle=0.1)
            self.move_line(self._ready_pose(), m.line_speed)
        except RobotError as e:
            self.log(f"      place aborted: {e}")
            return False
        return True

    def in_hand(self):
        if not self.execute:
            return self._grip_ok
        a = self.gripper_measured()
        return a is not None and abs(a - self.cfg.motion.gripper_closed) >= self.cfg.motion.grip_empty_angle

    def abort_hold(self):
        m = self.cfg.motion
        if self.arm_state == "home" and not self._grip_ok:
            return                      # nothing to release; opening at home would hit the mast
        try:
            self.gripper(m.gripper_open, settle=0.3)
            self._grip_ok = False
            t = self.tool_pos()
            self.move_line([t[0], t[1], min(t[2] + m.approach, self._ready_pose()[2])], m.descent_speed)
            self.gripper(m.gripper_closed, settle=0.1)
            self.move_line(self._ready_pose(), m.line_speed)
        except RobotError as e:
            self.log(f"      abort_hold: {e}; going home on joints")
            self._joint_move(self.home_chain)
            self.arm_state = "home"

    def gesture(self, name):
        # expressive motions are LED-only on the real robot until the arm behaviour is verified
        self.led({"think": (255, 160, 0), "wave": (0, 255, 200), "check": (255, 0, 0),
                  "celebrate": (0, 255, 0), "bow": (120, 120, 255)}.get(name, (255, 255, 255)), 200)

    def estop(self):
        self.enabled = False
        if self.execute and self.w_torque is not None:
            with self.w_torque.buf() as b:
                b["enable"][:] = np.zeros(self.acfg.dof, dtype=np.bool_)
        self._led_rgb, self._led_period = (255, 0, 0), 0
        self.log("EMERGENCY STOP: torque off")

    def close(self):
        self._led_rgb = (0, 0, 0)
        try:
            if self.execute and self.w_torque is not None:
                with self.w_torque.buf() as b:
                    b["enable"][:] = np.zeros(self.acfg.dof, dtype=np.bool_)
        finally:
            for x in (self.w_ctrl, self.w_torque, self.w_led, self.w_audio, self.state, self.cam, self.wake):
                if x is not None:
                    x.__exit__(None, None, None)

    # -- calibration helpers ------------------------------------------------------ #
    def jog(self, delta):
        """Move the tool by delta (m) at descent speed; returns the new tool position."""
        self.enable()
        self.move_line(self.tool_pos() + np.asarray(delta, float), self.cfg.motion.descent_speed)
        return self.tool_pos()
