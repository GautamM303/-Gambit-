# /// script
# requires-python = ">=3.10"
# dependencies = ["bbos", "numpy<2"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Say the wake phrase and BracketBot picks a coloured block off the table.

    on the robot:   uv run pick_by_voice.py --color red [--execute]
    on a laptop:    uv run python demos/pick_by_voice.py --sim [--view] [--no-wait]

Behaviour (identical on the robot and in the simulator)
  1. LED breathes blue while waiting for the wake word.
  2. On the wake word: LED flashes white, a chime plays, one frame of the depth
     camera's coloured point cloud is grabbed and the red and green blocks are
     found in it (3-D, no camera calibration needed).
  3. The LED turns the colour of the chosen block, the right arm reaches above
     it on a straight line, opens, descends, grips, lifts, carries it to the
     drop-off point, sets it down, releases and returns home.
  4. Two-tone chime on success, a low buzz on failure; the LED goes idle.

Safety on the real robot
  * Without --execute nothing moves: torque stays off, no arm commands are
    written, and the script prints every waypoint it *would* send. Run that
    first and check the numbers against where the block really is.
  * With --execute the arm moves at <= MAX_SPEED (default 6 cm/s), only the
    right arm is enabled, and torque is dropped on exit or ctrl-c.
  * VERIFY BEFORE --execute: the assumptions in BBOSRobot's docstring.

The task itself is 60 lines (run_task). Everything else is the two Robot
backends: BBOSRobot talks to the bbos daemons through the same channels the
view_*.py demos use; SimRobot drives the MuJoCo scene from
chopped_urdf_v2/sim/pick_place.py so the behaviour can be tested off-robot.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# task parameters (robot frame: x forward, y left, z up, origin at the base)
# --------------------------------------------------------------------------- #
COLOURS = {  # name -> (led rgb, point-cloud classifier)
    "red": ((255, 0, 0), lambda r, g, b: (r > 140) & (g < 0.6 * r) & (b < 0.6 * r)),
    "green": ((0, 255, 0), lambda r, g, b: (g > 120) & (r < 0.6 * g) & (b < 0.7 * g)),
}
DROP_XY = (0.28, 0.12)      # where the block is set down (same surface the block stands on)
DROP_Z = 0.725              # that surface's height, only used by the sim's final check
APPROACH = 0.12             # m above the grasp pose for the pre-grasp / pre-place
GRASP_BELOW_TOP = 0.01      # tool frame this far below the block's top face when gripping
TOOL_AHEAD = 0.015          # finger pads sit this far ahead (+x of the hand) of the tool frame
MAX_SPEED = 0.06            # m/s, straight-line end-effector speed
APPROACH_SPEED = 0.03       # m/s, for the last 12 cm down onto the block
MIN_BLOB_POINTS = 40        # fewer coloured points than this = no block
TICK = 0.02                 # s, control period


class TaskError(RuntimeError):
    pass


class Robot:
    """What run_task needs. Positions are metres in the robot frame."""

    def wait_wake(self): ...
    def led(self, rgb, period_ms=0): ...
    def chime(self, notes): ...           # [(hz, seconds), ...]
    def point_cloud(self): ...            # -> xyz (N,3) float, rgb (N,3) uint8
    def tool_pos(self): ...               # current tool-frame position
    def move_line(self, target, speed): ...  # straight line at fixed (down) orientation
    def gripper(self, open_amount): ...   # 0 closed .. 1 open, blocks until settled
    def home(self): ...
    def held_height(self): ...            # block height above its start, or None if unknown
    def close(self): ...


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
def find_block(xyz, rgb, colour):
    """(x, y, z_top) of the largest blob of `colour` in the cloud, by its top face."""
    r, g, b = (rgb[:, i].astype(np.float32) for i in range(3))
    sel = COLOURS[colour][1](r, g, b) & np.isfinite(xyz).all(axis=1)
    pts = xyz[sel]
    if len(pts) < MIN_BLOB_POINTS:
        raise TaskError(f"no {colour} block in view ({len(pts)} matching points)")
    # Keep the densest cluster: points within 6 cm of the median, iterated once.
    centre = np.median(pts, axis=0)
    near = pts[np.linalg.norm(pts - centre, axis=1) < 0.06]
    centre = np.median(near, axis=0)
    near = pts[np.linalg.norm(pts - centre, axis=1) < 0.06]
    z_top = np.percentile(near[:, 2], 97)
    top = near[near[:, 2] > z_top - 0.006]
    x, y = top[:, :2].mean(axis=0)
    return float(x), float(y), float(z_top), len(near)


def run_task(robot, colour, wait=True):
    led_rgb = COLOURS[colour][0]
    if wait:
        robot.led((0, 40, 255), period_ms=2000)
        print("waiting for the wake word...", flush=True)
        robot.wait_wake()
    robot.led((255, 255, 255))
    robot.chime([(880, 0.12), (1320, 0.12)])

    xyz, rgb = robot.point_cloud()
    x, y, z_top, n = find_block(xyz, rgb, colour)
    print(f"{colour} block: top at ({x:.3f}, {y:.3f}, {z_top:.3f}), {n} points", flush=True)
    robot.led(led_rgb)

    grasp = np.array([x - TOOL_AHEAD, y, z_top - GRASP_BELOW_TOP])
    # The block is set down on the same surface it stood on, so the tool ends at the
    # same height it gripped at (plus a few mm so it is released, not pressed).
    place = np.array([DROP_XY[0] - TOOL_AHEAD, DROP_XY[1], grasp[2] + 0.004])
    up = np.array([0, 0, APPROACH])

    print("phase 1: reach above the block")
    robot.gripper(0.0)
    robot.move_line(grasp + up, MAX_SPEED)
    print("phase 2: open, descend, grip")
    robot.gripper(1.0)
    robot.move_line(grasp, APPROACH_SPEED)
    robot.gripper(0.0)
    print("phase 3: lift")
    robot.move_line(grasp + up, APPROACH_SPEED)
    lifted = robot.held_height()
    if lifted is not None:
        print(f"   block lifted {lifted * 1000:.0f} mm")
        if lifted < 0.06:
            raise TaskError("the block slipped out of the gripper")
    print("phase 4: carry")
    robot.move_line(place + up, MAX_SPEED)
    print("phase 5: set down, release")
    robot.move_line(place, APPROACH_SPEED)
    robot.gripper(1.0)
    robot.move_line(place + up, APPROACH_SPEED)
    print("phase 6: home")
    robot.gripper(0.0)
    robot.home()
    robot.led((0, 255, 0))
    robot.chime([(660, 0.1), (880, 0.1), (1320, 0.18)])
    print("done", flush=True)


# --------------------------------------------------------------------------- #
# real robot
# --------------------------------------------------------------------------- #
class BBOSRobot(Robot):
    """bbos backend.

    Channels used (all as in demos/view_*.py): arm_right.state/.ctrl/.torque,
    camera.points, led.ctrl, speaker.audio, wakeword.state.

    Confirmed from the arm daemon sources (demos/daemon.py, constants.py, ik.py):
      * ik.solve(pos, quat_xyzw) solves for the eef link in the ARM_BASE frame and
        returns an empty list on failure; ik.fk(q7) -> (pos, quat_xyzw); ik.reset(q7)
        warm-starts it from the real joints. arm_base is world-aligned, IK_FRAME_Z
        above the robot origin (from the URDF; ROBOT_FRAME assumes the same origin as
        the point cloud). 'Hand down' is DOWN_QUAT_XYZW in that frame.
      * arm_right.ctrl 'pos' is motor space (turns); cfg.urdf2q / q2urdf convert.
      * The gripper is the 8th joint ('..._gripper'); the URDF range 0..1 rad maps
        through urdf2q (gripper_sign included). The daemon holds it gently.
    ASSUMPTIONS TO VERIFY before --execute:
      * camera.points is expressed in the robot frame (x fwd, y left, z up).
        view_depth.py draws it z-up with the axes at the origin, which suggests
        it is. If it is camera-relative, set POINTS_TO_ROBOT below.
      * The robot frame's origin is the base origin the URDF uses (arm_base is
        IK_FRAME_Z above it); if the point cloud's origin differs, adjust
        POINTS_TO_ROBOT.
    """
    POINTS_TO_ROBOT = np.eye(4)   # set if camera.points is not already in the robot frame
    IK_FRAME_Z = 0.0304           # arm_base sits this far above the robot origin (URDF)
    DOWN_QUAT_XYZW = np.array([1.0, 0.0, 0.0, 0.0])   # fingers down, closing along y, in the arm_base frame

    def __init__(self, execute=False):
        from bbos import Config, Reader, Type, Writer
        self.execute = execute
        self.cfg = Config("arm_right")
        self.cfg.ik.init()
        self.names = list(self.cfg.joint_names)
        self.chain = self._chain_indices()
        self.grip_idx = next((i for i, n in enumerate(self.names) if "gripper" in n), None)
        self.M, self.c = self._urdf_to_motor_affine()

        self.state = Reader("arm_right.state", Type("arm_state")).__enter__()
        self.points = Reader("camera.points").__enter__()
        self.wake = Reader("wakeword.state", keeptime=False).__enter__()
        self.w_led = Writer("led.ctrl", Type("led_ctrl"), keeptime=False).__enter__()
        self.w_audio = Writer("speaker.audio", Type("speaker_audio")).__enter__()
        self.spk = Config("speaker")
        self.w_ctrl = self.w_torque = None
        if execute:
            self.w_ctrl = Writer("arm_right.ctrl", Type("arm_ctrl")).__enter__()
            self.w_torque = Writer("arm_right.torque", Type("arm_torque")).__enter__()

        print("waiting for arm_right.state...", flush=True)
        deadline = time.monotonic() + 5.0
        while not self.state.ready() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self.state.ready():
            raise TaskError("no arm_right.state: is the arm daemon running?")
        self.q_motor = np.array(self.state.data["pos"], np.float64)   # commanded pose, motor space
        self.q_urdf = np.asarray(self.cfg.q2urdf(self.q_motor.astype(np.float32)), np.float64)
        home_urdf = np.asarray(self.cfg.q2urdf(np.asarray(self.cfg.home, np.float32)), np.float64)
        self.home_chain = [float(home_urdf[i]) for i in self.chain]
        self.cfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])
        self.down_quat = self.DOWN_QUAT_XYZW
        self._led_rgb, self._led_period = (0, 0, 0), 0
        print(f"arm_right: {self.cfg.dof} dof, chain {[self.names[i] for i in self.chain]}, "
              f"gripper {'j' + str(self.grip_idx) if self.grip_idx is not None else 'NONE'}; "
              f"{'EXECUTE' if execute else 'DRY RUN (no motion)'}", flush=True)

    # -- set-up helpers ---------------------------------------------------- #
    def _chain_indices(self):
        """Indices (into joint_names) of the 7 arm joints in kinematic order, as ik.solve returns them."""
        import yourdfpy
        urdf = yourdfpy.URDF.load(self.cfg.urdf_path, load_meshes=False, build_scene_graph=False)
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

    def _urdf_to_motor_affine(self):
        """Invert cfg.q2urdf (urdf = M q + c) by probing it, for a config without urdf2q
        (the bbos arm configs have one, which is used directly)."""
        if hasattr(self.cfg, "urdf2q"):
            return None, None
        n = self.cfg.dof
        f = lambda q: np.asarray(self.cfg.q2urdf(np.asarray(q, np.float64)), np.float64)
        c = f(np.zeros(n))
        M = np.stack([f(np.eye(n)[i]) - c for i in range(n)], axis=1)
        probe = np.linspace(-1, 1, n) * 0.7
        if not np.allclose(f(probe), M @ probe + c, atol=1e-6):
            raise TaskError("cfg.q2urdf is not affine; add an urdf2q to the config before running this")
        return M, c

    def _to_motor(self, q_urdf):
        if self.M is None:
            return np.asarray(self.cfg.urdf2q(np.asarray(q_urdf, np.float32)), np.float64)
        return np.linalg.solve(self.M, q_urdf - self.c)

    # -- io ------------------------------------------------------------------ #
    def _tick(self):
        """One control period: refresh the LED (the daemon idles 3 s after the last write)
        and, when executing, send the current joint command."""
        with self.w_led.buf() as b:
            b["rgb"] = np.array(self._led_rgb, dtype=np.uint8)
            b["brightness"] = np.int16(-1)
            b["period_ms"] = np.uint16(self._led_period)
        if self.execute:
            with self.w_torque.buf() as b:
                b["enable"][:] = np.ones(self.cfg.dof, dtype=np.bool_)
            if self.w_ctrl.ready():
                with self.w_ctrl.buf() as b:
                    b["pos"][:] = self._to_motor(self.q_urdf).astype(np.float32)
        time.sleep(TICK)

    def wait_wake(self):
        while True:
            if self.wake.ready() and bool(self.wake.data["active"]):
                return
            self._tick()

    def led(self, rgb, period_ms=0):
        self._led_rgb, self._led_period = rgb, period_ms
        self._tick()

    def chime(self, notes):
        sr, ch, cs = self.spk.sample_rate, self.spk.channels, self.spk.chunk_size
        wave = np.concatenate([0.25 * np.sin(2 * np.pi * hz * np.arange(int(s * sr)) / sr) for hz, s in notes])
        edge = int(0.005 * sr)
        wave[:edge] *= np.linspace(0, 1, edge)
        wave[-edge:] *= np.linspace(1, 0, edge)
        frames = (32767 * wave).astype(np.int16)[:, None].repeat(ch, axis=1)
        frames = np.concatenate([frames, np.zeros((-len(frames) % cs, ch), np.int16)])
        for chunk in frames.reshape(-1, cs, ch):
            with self.w_audio.buf() as b:
                b["audio"] = chunk

    def point_cloud(self):
        deadline = time.monotonic() + 3.0
        while not self.points.ready() and time.monotonic() < deadline:
            self._tick()
        if not self.points.ready():
            raise TaskError("no camera.points: is the depth camera daemon running?")
        d = self.points.data
        n = int(d["num_points"])
        xyz = np.asarray(d["points"][:n], np.float32).astype(np.float64)
        rgb = np.asarray(d["colors"][:n], np.uint8) if "colors" in d.dtype.names else np.zeros((n, 3), np.uint8)
        xyz = xyz @ self.POINTS_TO_ROBOT[:3, :3].T + self.POINTS_TO_ROBOT[:3, 3]
        return xyz, rgb

    def tool_pos(self):
        pos, _ = self.cfg.ik.fk([float(self.q_urdf[i]) for i in self.chain])
        return np.asarray(pos, np.float64) + [0, 0, self.IK_FRAME_Z]   # arm_base -> robot frame

    def _solve(self, pos):
        pos_ik = np.asarray(pos, np.float64) - [0, 0, self.IK_FRAME_Z]
        self.cfg.ik.reset([float(self.q_urdf[i]) for i in self.chain])   # continue from the commanded pose
        sol = self.cfg.ik.solve([float(v) for v in pos_ik], [float(v) for v in self.down_quat])
        if sol is None or len(sol) == 0:
            raise TaskError(f"IK failed at {np.round(pos, 3)}")
        sol = np.asarray(sol, np.float64)
        reached, _ = self.cfg.ik.fk(list(sol))
        err = np.linalg.norm(np.asarray(reached) - pos_ik)
        if err > 0.01:
            raise TaskError(f"IK only reached within {err * 1000:.0f} mm of {np.round(pos, 3)}")
        return sol

    def move_line(self, target, speed):
        target = np.asarray(target, np.float64)
        print(f"   -> {np.round(target, 3)} at {speed:.2f} m/s", flush=True)
        cur = self.tool_pos()
        while True:
            delta = target - cur
            dist = np.linalg.norm(delta)
            cur = target.copy() if dist <= speed * TICK else cur + delta * (speed * TICK / dist)
            sol = self._solve(cur)
            for k, i in enumerate(self.chain):
                self.q_urdf[i] = sol[k]
            self._tick()
            if dist <= speed * TICK:
                break
        for _ in range(int(0.3 / TICK)):   # settle
            self._tick()
        if not self.execute:
            print(f"      joints (urdf): {np.round(self.q_urdf, 3)}")

    def gripper(self, open_amount):
        if self.grip_idx is None:
            return
        start = self.q_urdf[self.grip_idx]
        steps = int(0.5 / TICK)
        for k in range(steps):
            self.q_urdf[self.grip_idx] = start + (open_amount - start) * (k + 1) / steps
            self._tick()
        for _ in range(int(0.5 / TICK)):
            self._tick()

    def home(self):
        target = self.q_urdf.copy()
        for k, i in enumerate(self.chain):
            target[i] = self.home_chain[k]
        start = self.q_urdf.copy()
        steps = int(max(2.0, np.abs(target - start).max() / 0.3) / TICK)   # <= 0.3 rad/s
        for k in range(steps):
            self.q_urdf = start + (target - start) * (k + 1) / steps
            self._tick()
        self.cfg.ik.reset(self.home_chain)

    def held_height(self):
        return None   # no force sensing on the gripper; trust the grasp

    def close(self):
        self._led_rgb, self._led_period = (0, 0, 0), 0
        try:
            if self.execute:
                with self.w_torque.buf() as b:
                    b["enable"][:] = np.zeros(self.cfg.dof, dtype=np.bool_)
        finally:
            for x in (self.w_ctrl, self.w_torque, self.w_led, self.w_audio, self.state, self.points, self.wake):
                if x is not None:
                    x.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #
class SimRobot(Robot):
    """MuJoCo backend on the pick_place.py scene: same table, blocks, pads and contact
    settings, plus a synthetic coloured point cloud rendered from the head camera."""

    def __init__(self, view=False, no_wait=False, record=None):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chopped_urdf_v2" / "sim"))
        import mujoco
        import pick_place as pp
        self.mujoco, self.pp = mujoco, pp
        self.model, self.plan_model, _ = pp.build_scene()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.arm = pp.Arm(self.model, "right")
        robot_bodies = {i for i in range(self.model.nbody) if pp.part(self.model.body(i).name) in pp.ARM_PARTS}
        self.planner = pp.Planner(self.plan_model, self.arm, robot_bodies, np.random.default_rng(0))
        self.viewer = None
        if view:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.3, -0.05, 0.9]
            self.viewer.cam.distance = 2.4
            self.viewer.cam.azimuth = 150
            self.viewer.cam.elevation = -18
        self.recorder = pp.Recorder(self.model, 30, record) if record else None
        self.runner = pp.Runner(self.model, self.data, self.arm, robot_bodies, viewer=self.viewer, recorder=self.recorder)
        self.R_down = np.diag([1.0, -1.0, -1.0])
        self.no_wait = no_wait
        self.boxes = {c: self.model.body(f"box_{c}").id for c in pp.BOXES}
        self.z0 = {c: self.data.xpos[b, 2] for c, b in self.boxes.items()}
        self.held = None

    def wait_wake(self):
        if self.no_wait:
            print("[sim] (wake word skipped)")
            return
        input("[sim] press Enter to 'say' the wake word... ")

    def led(self, rgb, period_ms=0):
        print(f"[sim] LED {rgb}{' blinking' if period_ms else ''}")

    def chime(self, notes):
        print(f"[sim] chime {[hz for hz, _ in notes]} Hz")

    def point_cloud(self):
        """Render RGB + depth from the head camera and back-project to world points."""
        mujoco, pp = self.mujoco, self.pp
        w, h = 640, 480
        cam = pp.Camera(self.model, self.data, "head_cam", w, h)
        renderer = mujoco.Renderer(self.model, h, w)
        mjcam = mujoco.MjvCamera()
        mjcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mjcam.fixedcamid = cam.id
        renderer.update_scene(self.data, mjcam)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(self.data, mjcam)
        depth = renderer.render().copy()
        renderer.close()
        v, u = np.mgrid[0:h, 0:w]
        d = depth.reshape(-1)
        rays = np.stack([(u.reshape(-1) - w / 2) / cam.fy, -(v.reshape(-1) - h / 2) / cam.fy, -np.ones(w * h)], axis=1)
        pts_cam = rays * d[:, None]
        xyz = pts_cam @ cam.R.T + cam.origin
        ok = np.isfinite(d) & (d < 3.0)
        print(f"[sim] point cloud: {ok.sum()} points")
        return xyz[ok], rgb.reshape(-1, 3)[ok]

    def tool_pos(self):
        return self.runner.eef_pos()

    def move_line(self, target, speed):
        target = np.asarray(target, np.float64)
        self.planner.sync(self.data)
        ignore = self.held
        near_ok = {self.model.geom("table_top").id, self.model.geom("tray").id}
        cur = self.tool_pos()
        q = self.runner.q_cmd.copy()
        while True:
            delta = target - cur
            dist = np.linalg.norm(delta)
            cur = target.copy() if dist <= speed * TICK else cur + delta * (speed * TICK / dist)
            q_new = self.arm.ik(self.plan_model, self.planner.d, cur, self.R_down, q, iters=200)
            if q_new is None:
                raise TaskError(f"IK failed at {np.round(cur, 3)}")
            if self.planner.in_collision(q_new, ignore, near_ok):
                raise TaskError(f"predicted collision at {np.round(cur, 3)}; stopping")
            q = q_new
            self.runner.q_cmd = q
            self.data.ctrl[self.arm.act] = q
            self.runner.hold(TICK)
            if dist <= speed * TICK:
                break
        self.runner.hold(0.3)

    def gripper(self, open_amount):
        if open_amount < 0.5 and self.held is None:   # closing: remember what is between the pads
            tool = self.tool_pos()
            for c, b in self.boxes.items():
                if np.linalg.norm(self.data.xpos[b, :2] - (tool[:2] + [TOOL_AHEAD, 0])) < 0.03:
                    self.held = b
                    self.runner.held = b
        self.runner.set_gripper(open_amount, settle=0.6)
        if open_amount >= 0.5:
            self.held = None
            self.runner.held = None

    def home(self):
        self.planner.sync(self.data)
        self.runner.move(self.planner.plan(self.runner.q_cmd, np.zeros(7)))
        self.runner.hold(0.5)

    def held_height(self):
        if self.held is None:
            return 0.0
        c = next(c for c, b in self.boxes.items() if b == self.held)
        return float(self.data.xpos[self.held, 2] - self.z0[c])

    def report(self):
        ill = self.runner.illegal_steps
        print(f"[sim] illegal contacts: {ill} steps" + (f" {self.runner.illegal_pairs}" if ill else " (none)"))
        for c, b in self.boxes.items():
            p = self.data.xpos[b]
            up = self.data.xmat[b].reshape(3, 3)[2, 2] > 0.95
            on = np.linalg.norm(p[:2] - DROP_XY) < 0.06 and abs(p[2] - DROP_Z) < 0.01
            print(f"[sim] {c} block at {np.round(p, 3)} {'upright' if up else 'TIPPED'}{' on the tray' if on else ''}")

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
            print(f"[sim] wrote video ({self.recorder.frames} frames)")
        if self.viewer is not None:
            print("[sim] close the viewer window to exit")
            while self.viewer.is_running():
                self.runner.step()
            self.viewer.close()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", choices=list(COLOURS), default="red")
    ap.add_argument("--execute", action="store_true", help="really move the real arm (default: dry run)")
    ap.add_argument("--sim", action="store_true", help="run against the MuJoCo scene instead of bbos")
    ap.add_argument("--view", action="store_true", help="(sim) open the MuJoCo viewer")
    ap.add_argument("--no-wait", action="store_true", help="skip the wake word")
    ap.add_argument("--record", metavar="FILE.mp4", help="(sim) write a video of the run")
    args = ap.parse_args()

    robot = (SimRobot(view=args.view, no_wait=args.no_wait, record=args.record) if args.sim
             else BBOSRobot(execute=args.execute))
    ok = False
    try:
        run_task(robot, args.color, wait=not args.no_wait)
        ok = True
    except TaskError as e:
        print(f"FAILED: {e}", flush=True)
        robot.led((255, 0, 0), period_ms=300)
        robot.chime([(220, 0.4)])
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        if isinstance(robot, SimRobot):
            robot.report()
        robot.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
