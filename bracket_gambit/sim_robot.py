"""MuJoCo backend: the validated chess scene from chopped_urdf_v2/sim/chess_robot.py
behind the Robot interface.

What is simulated faithfully: the arm (URDF, collision-checked IK trajectories, position
actuators), the finger pads, piece physics, the head camera image the vision runs on.
What is sim-only: the base slides sideways to bring each file in front of the shoulder
(the real SDK exposes no base-drive channel), and the 'human' teleports pieces.
"""
from __future__ import annotations

import pathlib
import sys

import chess
import cv2
import numpy as np

from .config import GambitConfig
from .robot import EStop, Robot, StopFlag

SIM_DIR = pathlib.Path(__file__).resolve().parents[1] / "chopped_urdf_v2" / "sim"


class Views:
    """Offscreen renders of the scene ('wide' / 'table' camera) and the hand camera for the
    status display, refreshed from the simulation thread every `period` seconds of sim time."""

    def __init__(self, model, display, period=0.1, scene_size=(1280, 720), hand_size=(320, 240)):
        import mujoco
        self.mujoco, self.model, self.display, self.period = mujoco, model, display, period
        self.scene_r = mujoco.Renderer(model, scene_size[1], scene_size[0])
        self.hand_r = mujoco.Renderer(model, hand_size[1], hand_size[0])
        self.scene_cam, self.hand_cam = mujoco.MjvCamera(), mujoco.MjvCamera()
        for cam, name in ((self.scene_cam, "table"), (self.hand_cam, "hand_cam")):
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = model.camera(name).id
        self.hand_opt = mujoco.MjvOption()
        self.hand_opt.sitegroup[:] = 0          # the tool-frame marker would sit in the middle of the view
        self.next_t = 0.0

    def use_camera(self, name):
        self.scene_cam.fixedcamid = self.model.camera(name).id

    def tick(self, data, force=False):
        if not force and data.time < self.next_t:
            return
        self.next_t = data.time + self.period
        self.scene_r.update_scene(data, self.scene_cam)
        scene = cv2.cvtColor(self.scene_r.render(), cv2.COLOR_RGB2BGR)
        self.hand_r.update_scene(data, self.hand_cam, self.hand_opt)
        self.hand_r.scene.flags[self.mujoco.mjtRndFlag.mjRND_SHADOW] = False
        hand = cv2.cvtColor(self.hand_r.render(), cv2.COLOR_RGB2BGR)
        self.display.set_views(scene, hand)

    def close(self):
        self.scene_r.close()
        self.hand_r.close()


class SimRobot(Robot):
    def __init__(self, cfg: GambitConfig, view=False, recorder=None, log=print, display=None, view_period=0.1,
                 scene_size=(1280, 720), hand_size=(320, 240)):
        sys.path.insert(0, str(SIM_DIR))
        import mujoco                      # noqa: F401  (import check)
        import chess_robot as cr
        self.cr, self.log = cr, log
        self.stop = StopFlag()
        self.bot = cr.ChessBot(view=view, recorder=recorder)
        self.views = (Views(self.bot.model, display, period=view_period, scene_size=scene_size, hand_size=hand_size)
                      if display is not None else None)
        # the scene's geometry is the calibration (world frame == robot frame: the base
        # parks at the origin after rolling up)
        b = cfg.board
        b.square, b.a1, b.file_dir, b.rank_dir = cr.SQ, (cr.RANK1_X, cr.FILEA_Y), (0.0, -1.0), (1.0, 0.0)
        b.board_z, b.table_z, b.marker_offset = cr.BOARD_Z, cr.TABLE_TOP, 0.008 / cr.SQ
        b.tray, b.piece_top = list(cr.GRAVE), dict(cr.PIECE_SHAFT_TOP)
        cfg.vision.marker_mode = "colour"        # the scene has blue corner dots
        self.cfg = cfg
        self.bodies = dict(self.bot.piece_body)          # square -> body id (pieces on the board)
        self.kind = {sq: n[1] for sq, n in self.bot.piece_name.items()}
        self.tray_bodies = []
        self.held = None
        self.arm_state = "home"
        self.human_grave = 0
        self.helper_moves = 0
        # every simulation step checks the emergency-stop flag
        runner, step = self.bot.runner, self.bot.runner.step
        def guarded_step():
            self.stop.check()
            step()
            if self.views is not None:
                self.views.tick(self.bot.data)
        runner.step = guarded_step
        self.model, self.data = self.bot.model, self.bot.data

    # -- show ------------------------------------------------------------------ #
    def arrive(self):
        self.bot.status = "rolling up to the table"
        self.bot.drive(0.0, self.cr.BASE_Y)       # straight ahead; it never slides sideways

    def status_text(self, text):
        self.bot.status = text

    def sim_time(self):
        """Simulated seconds since the last call (what the physical move would take)."""
        now, last = self.data.time, getattr(self, "_t_mark", self.data.time)
        self._t_mark = now
        return now - last

    # -- perception ------------------------------------------------------------ #
    def look(self):
        self.stop.check()
        if self.views is not None:
            self.views.tick(self.data, force=True)
        return self.bot.look()

    def use_camera(self, name):
        if self.views is not None:
            self.views.use_camera(name)

    def reference_image(self):
        """Render the EMPTY board (pieces lifted out of view) for the reference classifier."""
        cr, m, d = self.cr, self.model, self.data
        saved = {}
        for body in list(self.bodies.values()) + self.tray_bodies:
            adr = m.jnt_qposadr[m.body_jntadr[body]]
            saved[body] = d.qpos[adr:adr + 7].copy()
            d.qpos[adr + 2] = -1.0
        self.cr.mujoco.mj_forward(m, d)
        rgb = self.bot.look()
        for body, q in saved.items():
            adr = m.jnt_qposadr[m.body_jntadr[body]]
            d.qpos[adr:adr + 7] = q
        self.cr.mujoco.mj_forward(m, d)
        return rgb

    # -- io ------------------------------------------------------------------ #
    def say(self, text):
        self.log(f"[say] {text}")

    def led(self, rgb, period_ms=0):
        pass

    # -- arm ------------------------------------------------------------------ #
    def ready(self):
        self.stop.check()
        if self.arm_state != "ready":
            self.bot.go_ready()
            self.arm_state = "ready"

    def park(self):
        """Out of the camera's view: the look pose beside the board (fast), not home."""
        self.stop.check()
        if self.arm_state == "ready":
            self.bot.go_watch()
            self.arm_state = "aside"

    def rest(self):
        """All the way home (end of the game / stop)."""
        self.stop.check()
        self.bot.go_home()
        self.arm_state = "home"

    def _carry_pose(self):
        return self.bot.carry_pose()

    def _body_near(self, xy, tol=0.015):
        best, best_d = None, tol
        for sq, body in list(self.bodies.items()) + [(None, b) for b in self.tray_bodies]:
            d = np.linalg.norm(self.data.xpos[body, :2] - xy)
            if d < best_d:
                best, best_d = (sq, body), d
        return best

    def pick(self, xy, kind):
        cr, bot = self.cr, self.bot
        self.ready()
        found = self._body_near(np.asarray(xy))
        if found is None:
            self.log(f"      nothing within 15 mm of {np.round(xy, 3)}")
            return False
        sq, body = found
        top = cr.PIECE_SHAFT_TOP.get(kind, cr.PIECE_SHAFT_TOP["p"])
        grasp = np.array([self.data.xpos[body, 0], self.data.xpos[body, 1], cr.grasp_height(top)]) + cr.PAD_SHIFT
        try:
            bot.face(self.data.xpos[body, :2])        # turn in place, never strafe
            bot.runner.set_gripper(cr.OPEN, settle=0.1)
            bot.move_line(grasp + [0, 0, cr.LIFT], cr.LINE_SPEED, ignore=body)
            bot.move_line(grasp, cr.DESCENT_SPEED, ignore=body)
            bot.runner.held = body
            self.held = body
            bot.runner.set_gripper(cr.CLOSED, settle=0.4)
            z0 = self.data.xpos[body, 2]
            bot.move_line(grasp + [0, 0, cr.LIFT], cr.DESCENT_SPEED, ignore=body)
            if self.data.xpos[body, 2] - z0 < 0.06 or not bot.in_hand(body):
                self.log("      the piece did not come up with the gripper")
                return False
            far = bot.to_base(grasp[:2])[0] > cr.FAR_X
            bot.move_line(self._carry_pose(), cr.FAR_SPEED if far else cr.CARRY_SPEED, ignore=body)
        except EStop:
            raise
        except RuntimeError as e:
            self.log(f"      pick aborted: {e}")
            return False
        if sq is not None:
            self.bodies.pop(sq, None)
            self.kind.pop(sq, None)
        else:
            self.tray_bodies.remove(body)
        self.held_kind = kind
        return bot.in_hand(body)

    def place(self, xy, surface_z):
        cr, bot = self.cr, self.bot
        body = self.held
        if body is None:
            return False
        x, y = xy
        on_board = self._square_at((x, y)) is not None
        try:
            bot.face((x, y))
            if not bot.in_hand(body):
                self.log("      piece lost while turning")
                return False
            # ChessBot.place: touch-detected release; diagonal pads only on the board
            # (the tray has no neighbours, so the tool keeps whatever twist it has)
            bot.place((x, y), body, surface_z, diagonal=on_board, auto_reset=False)
            self.held = None
            bot.move_line(self._carry_pose(), cr.LINE_SPEED)
        except EStop:
            raise
        except RuntimeError as e:
            self.log(f"      place aborted: {e}")
            if bot.runner.held is None:      # released but not verified: book-keep where it is
                self.held = None
                self._register(body, self.data.xpos[body, :2])
            return False
        err = np.linalg.norm(self.data.xpos[body, :2] - [x, y])
        upright = self.data.xmat[body].reshape(3, 3)[2, 2] > 0.95
        self._register(body, (x, y))
        return err <= 0.015 and upright

    def _register(self, body, xy):
        """Book-keep where a body ended up (a square or the tray)."""
        sq = self._square_at(xy)
        if sq is not None:
            self.bodies[sq] = body
            self.kind[sq] = self.model.body(body).name[1]
        else:
            self.tray_bodies.append(body)

    def _square_at(self, xy):
        g = self.cfg.board
        for sq in chess.SQUARES:
            x, y = g.square_xy(sq)
            if abs(x - xy[0]) < g.square / 2 and abs(y - xy[1]) < g.square / 2:
                return sq
        return None

    def in_hand(self):
        return self.held is not None and self.bot.in_hand(self.held)

    def abort_hold(self):
        cr, bot = self.cr, self.bot
        if self.arm_state == "home" and self.held is None:
            return                      # opening the gripper at home would hit the mast
        if self.arm_state == "aside" and self.held is None:
            return                      # nothing in hand and already out of the way
        body, self.held = self.held, None
        try:
            bot.escape()
        except EStop:
            raise
        except RuntimeError as e:
            self.log(f"      abort_hold could not park the arm: {e}")
        if body is not None:
            self._register(body, self.data.xpos[body, :2])

    def gesture(self, name):
        self.log(f"[gesture] {name}")

    def estop(self):
        # the runner is already frozen (guarded_step raised); hold the arm where it is
        self.bot.runner.q_cmd = self.data.qpos[self.bot.arm.qadr].copy()
        self.data.ctrl[self.bot.arm.act] = self.bot.runner.q_cmd
        self.held = self.bot.runner.held = None
        self.log("[sim] EMERGENCY STOP: arm frozen")

    # -- the simulated human and helper ---------------------------------------- #
    def teleport(self, body, xy, z=None):
        self.bot.teleport(body, xy) if z is None else None
        if z is not None:
            m, d = self.model, self.data
            adr = m.jnt_qposadr[m.body_jntadr[body]]
            d.qpos[adr:adr + 3] = [xy[0], xy[1], z]
            d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
            d.qvel[m.jnt_dofadr[m.body_jntadr[body]]:m.jnt_dofadr[m.body_jntadr[body]] + 6] = 0

    def human_plays(self, board: chess.Board, mv: chess.Move):
        """A hand moves a black piece (teleport), including captures and castling."""
        cr = self.cr
        cap = None
        if board.is_en_passant(mv):
            cap = mv.to_square + (-8 if board.turn else 8)
        elif board.is_capture(mv):
            cap = mv.to_square
        if cap is not None:
            b = self.bodies.pop(cap)
            self.kind.pop(cap, None)
            self.teleport(b, cr.HUMAN_GRAVE[self.human_grave], cr.TABLE_TOP + 0.0005)
            self.human_grave += 1
        b = self.bodies.pop(mv.from_square)
        self.teleport(b, self.cfg.board.square_xy(mv.to_square))
        self.bodies[mv.to_square] = b
        self.kind[mv.to_square] = self.kind.pop(mv.from_square)
        if board.is_castling(mv):
            rank = chess.square_rank(mv.from_square)
            rf, rt = ((chess.square(7, rank), chess.square(5, rank)) if chess.square_file(mv.to_square) == 6
                      else (chess.square(0, rank), chess.square(3, rank)))
            rb = self.bodies.pop(rf)
            self.teleport(rb, self.cfg.board.square_xy(rt))
            self.bodies[rt] = rb
            self.kind[rt] = self.kind.pop(rf)
        self.cr.mujoco.mj_forward(self.model, self.data)
        self.bot.settle(0.8)

    def human_takes_back(self, board: chess.Board, mv: chess.Move):
        """Undo a (rejected) human move on the physical board: `board` is the position
        before the move."""
        b = self.bodies.pop(mv.to_square)
        self.teleport(b, self.cfg.board.square_xy(mv.from_square))
        self.bodies[mv.from_square] = b
        self.kind[mv.from_square] = self.kind.pop(mv.to_square)
        victim = board.piece_at(mv.to_square)
        if victim is not None and self.human_grave > 0:      # bring the captured piece back
            self.human_grave -= 1
            slot = self.cr.HUMAN_GRAVE[self.human_grave]
            vb = next(bd for bd in range(self.model.nbody)
                      if np.linalg.norm(self.data.xpos[bd, :2] - slot) < 0.01 and self.model.body(bd).name[:1] in "wb")
            self.teleport(vb, self.cfg.board.square_xy(mv.to_square))
            self.bodies[mv.to_square], self.kind[mv.to_square] = vb, victim.symbol().lower()
        self.cr.mujoco.mj_forward(self.model, self.data)
        self.bot.settle(0.8)

    def helper_fix(self, expected: chess.Board):
        """A helper puts the pieces where `expected` says (what a human does when the robot
        asks for help). Goes by where every piece PHYSICALLY stands (upright, centred on a
        square), not by the book-keeping - a piece that was dropped and rolled onto another
        square is picked up and put right. Counted so the report shows it."""
        self.helper_moves += 1
        want = {sq: (p.color, p.symbol().lower()) for sq, p in expected.piece_map().items()}
        all_bodies = [b for b in range(self.model.nbody) if self.model.body(b).name[:1] in "wb"
                      and len(self.model.body(b).name) > 3 and self.model.body(b).name[2] == "_"]
        keep, loose = {}, []
        for b in all_bodies:
            name = self.model.body(b).name
            colour, kind = name[0] == "w", name[1]
            p = self.data.xpos[b]
            upright = self.data.xmat[b].reshape(3, 3)[2, 2] > 0.95
            sq = self._square_at(p[:2])
            centred = sq is not None and np.linalg.norm(np.asarray(self.cfg.board.square_xy(sq)) - p[:2]) < 0.008
            if upright and centred and want.get(sq) == (colour, kind) and sq not in keep:
                keep[sq] = (b, kind)
            else:
                loose.append(b)
        moved = []
        for sq, (c, k) in want.items():
            if sq in keep:
                continue
            match = next((b for b in loose if (self.model.body(b).name[0] == "w") == c and self.model.body(b).name[1] == k), None)
            if match is None:
                self.log(f"      helper: no spare {'white' if c else 'black'} {k} for {chess.square_name(sq)}")
                continue
            loose.remove(match)
            self.teleport(match, self.cfg.board.square_xy(sq))
            keep[sq] = (match, k)
            moved.append(f"{self.model.body(match).name}->{chess.square_name(sq)}")
        # Everything else is taken off the board: onto the human's side first (so the
        # robot's own tray is left clear), then the robot's tray, then the far table edge.
        hg = self.cr.HUMAN_GRAVE
        overflow = [(hg[i % len(hg)][0] + 0.045 * (2 + i // len(hg)), hg[i % len(hg)][1]) for i in range(24)]
        grave = list(hg) + list(self.cr.GRAVE) + overflow
        self.tray_bodies = []
        for i, b in enumerate(loose):
            slot = grave[min(i, len(grave) - 1)]
            if np.linalg.norm(self.data.xpos[b, :2] - slot) > 0.005:
                self.teleport(b, slot, self.cr.TABLE_TOP + 0.0005)
                moved.append(f"{self.model.body(b).name}->tray")
            self.tray_bodies.append(b)
        self.bodies = {sq: b for sq, (b, k) in keep.items()}
        self.kind = {sq: k for sq, (b, k) in keep.items()}
        self.log(f"      helper moved: {', '.join(moved) if moved else 'nothing (board already matched)'}")
        self.cr.mujoco.mj_forward(self.model, self.data)
        self.bot.settle(0.8)

    def report(self):
        r = self.bot.runner
        return (f"illegal contacts: {r.illegal_steps} steps" + (f" {r.illegal_pairs}" if r.illegal_pairs else " (none)")
                + f"; helper interventions: {self.helper_moves}")

    def close(self):
        if self.bot.viewer is not None:
            self.log("[sim] close the viewer window to exit")
            while self.bot.viewer.is_running():
                self.bot.runner.step()
        if self.views is not None:
            self.views.close()
