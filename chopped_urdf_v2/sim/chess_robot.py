"""BracketBot plays chess in MuJoCo: rolls up to the table, reads the board with its
head camera and moves the pieces itself.

    uv run chopped_urdf_v2/sim/chess_robot.py --moves 5 --record chess.mp4   # video + dashboard frames
    uv run chopped_urdf_v2/sim/chess_robot.py --view                          # interactive viewer

Each turn
  1. the (simulated) human moves a black piece - it is teleported, as if a hand moved it
  2. vision: head camera -> blue corner markers -> homography -> rectified board ->
     every square classified white / black / empty
  3. the human's move is inferred by matching the observed occupancy against the
     legal moves (the way real chess robots do it), then validated with python-chess
  4. a small alpha-beta engine chooses the reply
  5. the robot turns in place so the target square is in front of its shoulder, then
     picks the piece by its shaft (diagonally, so the claws clear the neighbours),
     carries it, sets it down, removes captured pieces to its graveyard, and returns
     to the ready pose - every move collision-checked as in pick_place.py

Outputs a dashboard per turn (3-D view | head camera with the detected grid | board map
from vision) into --out and, with --record, a video of the whole game.
"""
import argparse
import math
import pathlib
import random
import sys
import time

import chess
import cv2
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pick_place as pp  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# geometry (world frame: x forward from the parked base, y left, z up)
# --------------------------------------------------------------------------- #
TABLE_TOP = 0.72
SQ = 0.03                       # square size
RANK1_X, FILEA_Y = 0.175, 0.105  # centre of a1; files run to -y, ranks to +x
BOARD_Z = TABLE_TOP + 0.011     # playing surface
RIM = 0.015
MARKER_R = 0.008
MARKERS = [(RANK1_X - SQ / 2 - 0.008, FILEA_Y + SQ / 2 + 0.008), (RANK1_X - SQ / 2 - 0.008, -FILEA_Y - SQ / 2 - 0.008),
           (RANK1_X + 7.5 * SQ + 0.008, FILEA_Y + SQ / 2 + 0.008), (RANK1_X + 7.5 * SQ + 0.008, -FILEA_Y - SQ / 2 - 0.008)]
# The robot's capture tray: beside the h-file on the NEAR side, every slot within 0.37 m of
# the base (the arm's reach ends at ~0.40 m and a piece carried to a slot at 0.38 m was
# dropped; slots along the far edge were out of reach altogether). 9 slots; when they are
# full the robot asks for the tray to be cleared.
GRAVE = [(x, -FILEA_Y - SQ * 2 - 0.045 * j) for x in (0.165, 0.21, 0.255) for j in range(4)
         if (x ** 2 + (FILEA_Y + SQ * 2 + 0.045 * j + 0.03) ** 2) ** 0.5 < 0.37]
HUMAN_GRAVE = [(RANK1_X + 0.045 * i, FILEA_Y + SQ * 2 + 0.045 * j) for j in range(2) for i in range(6)]

# Shaft heights leave 4 cm of plain shaft for the pads (which are 4 cm tall and reach 2.3 cm
# below the tool frame); the ornaments on top stay within the closed pad gap (radius < 0.8 cm)
# so the pads can only ever squeeze the round shaft, never a wedge-shaped top.
PIECE_SHAFT_TOP = {"p": 0.050, "n": 0.053, "b": 0.055, "r": 0.052, "q": 0.058, "k": 0.061}
SHAFT_R, BASE_R = 0.008, 0.011
WHITE_RGBA, BLACK_RGBA = (0.93, 0.90, 0.82, 1), (0.06, 0.06, 0.07, 1)

SWEET_Y = -0.05        # base-frame y the arm likes to work at (see the reach map)
CARRY_X = 0.22         # base-frame x of the parking spot between moves
# Where the arm waits while the head camera reads the board: off to the right, out of the
# camera's view of the markers (checked: 0 misread squares from here), only one straight
# line away from the carry pose - much quicker than retracting to home every turn.
LOOK_POSE = (0.15, -0.30, 0.95)
APPROACH_SPEED = 0.09  # m/s, the empty gripper descending onto a piece
FAR_X = 0.31           # squares beyond this (base frame) need a stretched arm; retracting from
FAR_SPEED = 0.04       # them is a big wrist reconfiguration, done slowly so the piece stays put
GRASP_BELOW_TOP = 0.020        # tool frame below the shaft top: the pads' upper edge meets the shaft top
LIFT = 0.08
CARRY_Z = 0.90
OPEN, CLOSED = 0.25, 0.0        # gripper angles: 0.25 rad puts the pads 27 mm out, clear of neighbours
LINE_SPEED, DESCENT_SPEED = 0.25, 0.07
CARRY_SPEED = 0.08              # with a piece in hand (0.10 with RATE 0.6 drops tall pieces)
RATE = 0.5                      # fraction of VMAX any joint may use while CARRYING a piece
RATE_EMPTY = 1.0                # ...and with an empty gripper (nothing to whip out of the pads)
DRIVE_SPEED = 0.30
YAW_SPEED = 1.2                 # rad/s, turning in place
TICK = 0.02
# The base is a differential drive: it rolls straight up to the table and parks at a small
# lateral offset (BASE_Y), then brings each square into the arm's working fan by TURNING IN
# PLACE - never by sliding sideways. With BASE_Y = 0.03 every square is reachable with a yaw
# between -25 and +38 degrees (checked with the reach test in the README).
BASE_Y = 0.03
START_POSE = (-1.3, BASE_Y)
YAW45 = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0],
                  [math.sin(math.pi / 4), math.cos(math.pi / 4), 0], [0, 0, 1]])
R_GRASP = YAW45 @ np.diag([1.0, -1.0, -1.0])
PAD_SHIFT = -R_GRASP[:, 0] * pp.PAD_OFFSET[0]


def pad_shift(yaw=0.0):
    """Tool-frame offset that puts the pads (not the eef origin) over a point, for a grasp
    frame twisted by `yaw` about z relative to R_GRASP."""
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return -(Rz @ R_GRASP)[:, 0] * pp.PAD_OFFSET[0]


def square_xy(sq):
    """World (x, y) of a python-chess square index."""
    return RANK1_X + SQ * chess.square_rank(sq), FILEA_Y - SQ * chess.square_file(sq)


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def grasp_height(top):
    """Tool-frame height above the board for a piece whose shaft top is `top` above it:
    the pads' upper edge meets the shaft top on a pawn and sits lower on taller pieces,
    keeping the pinch closer to the centre of mass."""
    return BOARD_Z + top - GRASP_BELOW_TOP - 0.5 * (top - PIECE_SHAFT_TOP["p"])


def build_chess_scene():
    # Flat pads set parallel at the shaft width (PAD_BIAS is the pad sink assumed by
    # add_finger_pads; a weighted piece sinks ~1 mm) plus a backstop plate between the
    # pads: a shaft between slightly toed-in pads creeps along them during a long carry and
    # used to fall out of the back of the pads (see STATUS.md, "pieces creeping out").
    rs = pp.RobotSpec(grip_width=2 * SHAFT_R + pp.PAD_BIAS, mobile=True, backstop=True)
    rs.spec.option.noslip_iterations = 10   # slows the slow downward creep of a pinched piece
    wb = rs.spec.worldbody
    rs.add_table(center=(0.30, 0.0), half=(0.17, 0.32), top_z=TABLE_TOP)
    cx, cy = RANK1_X + 3.5 * SQ, 0.0
    half = 4 * SQ + RIM
    board = wb.add_body(name="board", pos=[cx, cy, TABLE_TOP])
    plate = board.add_geom(name="board_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.005],
                           pos=[0, 0, 0.005], rgba=[0.30, 0.20, 0.12, 1])
    plate.contype = plate.conaffinity = pp.STRUCTURE
    rs.env_geoms.append(plate)
    for sq in chess.SQUARES:
        x, y = square_xy(sq)
        light = (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 1
        g = board.add_geom(name=f"sq_{chess.square_name(sq)}", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[SQ / 2, SQ / 2, 0.0005], pos=[x - cx, y - cy, 0.0105],
                           rgba=[0.72, 0.56, 0.36, 1] if light else [0.36, 0.22, 0.12, 1])
        g.contype = g.conaffinity = pp.STRUCTURE
        rs.soft_geoms.append(g)
    for k, (mx, my) in enumerate(MARKERS):
        g = board.add_geom(name=f"marker{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[MARKER_R, 0.0005, 0],
                           pos=[mx - cx, my - cy, 0.0105], rgba=[0.05, 0.25, 1.0, 1])
        g.contype = g.conaffinity = 0

    pieces = {}
    start = chess.Board()
    for sq, piece in start.piece_map().items():
        x, y = square_xy(sq)
        name = f"{'w' if piece.color else 'b'}{piece.symbol().lower()}_{chess.square_name(sq)}"
        body = wb.add_body(name=name, pos=[x, y, BOARD_Z])
        body.add_freejoint()
        rgba = WHITE_RGBA if piece.color else BLACK_RGBA
        top = PIECE_SHAFT_TOP[piece.symbol().lower()]
        geoms = [body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BASE_R, 0.002, 0], pos=[0, 0, 0.002], rgba=list(rgba)),
                 body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SHAFT_R, (top - 0.004) / 2, 0],
                               pos=[0, 0, 0.004 + (top - 0.004) / 2], rgba=list(rgba))]
        s = piece.symbol().lower()
        sph, box, cyl = mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
        if s == "p":
            geoms.append(body.add_geom(type=sph, size=[0.0055, 0, 0], pos=[0, 0, top + 0.0075], rgba=list(rgba)))
        elif s == "r":
            geoms.append(body.add_geom(type=cyl, size=[0.0055, 0.004, 0], pos=[0, 0, top + 0.006], rgba=list(rgba)))
        elif s == "n":
            geoms.append(body.add_geom(type=sph, size=[0.0045, 0, 0], pos=[0, 0, top + 0.0065], rgba=list(rgba)))
            geoms.append(body.add_geom(type=sph, size=[0.003, 0, 0], pos=[0.003, 0, top + 0.012], rgba=list(rgba)))
        elif s == "b":
            geoms.append(body.add_geom(type=sph, size=[0.0045, 0, 0], pos=[0, 0, top + 0.0065], rgba=list(rgba)))
            geoms.append(body.add_geom(type=sph, size=[0.0025, 0, 0], pos=[0, 0, top + 0.0125], rgba=list(rgba)))
        elif s == "q":
            geoms.append(body.add_geom(type=sph, size=[0.0055, 0, 0], pos=[0, 0, top + 0.0075], rgba=list(rgba)))
            geoms.append(body.add_geom(type=sph, size=[0.003, 0, 0], pos=[0, 0, top + 0.0145], rgba=list(rgba)))
        else:
            geoms.append(body.add_geom(type=box, size=[0.0015, 0.005, 0.0015], pos=[0, 0, top + 0.010], rgba=list(rgba)))
            geoms.append(body.add_geom(type=box, size=[0.0015, 0.0015, 0.007], pos=[0, 0, top + 0.009], rgba=list(rgba)))
        for i, g in enumerate(geoms):
            # Weighted base, ~30 g like a real piece: MuJoCo scales contact stiffness with
            # mass, and a 10 g piece was so soft the pads sank several mm into its shaft.
            g.mass = 0.024 if i == 0 else 0.004
            g.friction = [1.5, 0.005, 0.0001]
            g.contype = g.conaffinity = pp.OBJECT
            rs.soft_geoms.append(g)
        pieces[sq] = name

    wide = (1.4, -2.6, 1.7)
    wb.add_camera(name="wide", pos=list(wide), xyaxes=pp.look_at(wide, (-0.4, -0.2, 0.8)), fovy=50)
    close = (0.95, -0.85, 1.15)
    wb.add_camera(name="table", pos=list(close), xyaxes=pp.look_at(close, (0.26, -0.03, 0.82)), fovy=40)
    model, plan_model, _ = rs.finalize()
    return model, plan_model, pieces


# --------------------------------------------------------------------------- #
# vision: board reading from the head camera
# --------------------------------------------------------------------------- #
WARP = 50   # px per square in the rectified view


def board_to_warp(xy):
    """World board-plane (x, y) -> rectified image pixel (col, row); a1 bottom-left."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    col = (FILEA_Y + SQ / 2 - xy[:, 1]) / SQ * WARP
    row = (RANK1_X + 7.5 * SQ - xy[:, 0]) / SQ * WARP
    return np.stack([col, row], axis=1)


def read_board(rgb):
    """Detect the 4 blue markers, rectify the board and classify every square.

    Returns (occupancy {square: 'w'|'b'|None}, H board->image, rectified image, marker px)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array((100, 120, 80)), np.array((130, 255, 255)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > 15:
            blobs.append((m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"]))
    blobs = sorted(blobs, key=lambda b: -b[2])[:4]
    if len(blobs) < 4:
        raise RuntimeError(f"vision: only {len(blobs)} corner markers found")
    blobs = sorted(blobs, key=lambda b: b[1])            # by row: far pair first (top of image)
    far = sorted(blobs[:2], key=lambda b: b[0])           # left (+y, file a) first
    near = sorted(blobs[2:], key=lambda b: b[0])
    img_pts = np.array([near[0][:2], near[1][:2], far[0][:2], far[1][:2]], np.float32)
    world_pts = np.array(MARKERS, np.float32)             # same order: near-a, near-h, far-a, far-h
    H_img_from_board = cv2.getPerspectiveTransform(world_pts, img_pts)
    warp_pts = board_to_warp(world_pts).astype(np.float32)
    H_warp = cv2.getPerspectiveTransform(img_pts, warp_pts)
    rect = cv2.warpPerspective(rgb, H_warp, (8 * WARP, 8 * WARP))
    hsv_r = cv2.cvtColor(rect, cv2.COLOR_RGB2HSV)
    occ = {}
    for sq in chess.SQUARES:
        c0 = chess.square_file(sq) * WARP
        r0 = (7 - chess.square_rank(sq)) * WARP
        # Sample the near (camera-side) half of the square: tall pieces smear away from the camera.
        patch = hsv_r[r0 + WARP // 2 - 4:r0 + WARP - 6, c0 + 12:c0 + WARP - 12]
        s, v = patch[..., 1].astype(int), patch[..., 2].astype(int)
        white = np.mean((v > 200) & (s < 50))
        black = np.mean(v < 50)
        occ[sq] = "w" if white > 0.15 and white >= black else ("b" if black > 0.15 else None)
    return occ, H_img_from_board, rect, img_pts


def infer_move(board, occ):
    """The legal move whose resulting occupancy best matches what the camera saw."""
    def occupancy(b):
        return {sq: (None if p is None else ("w" if p.color else "b")) for sq, p in ((s, b.piece_at(s)) for s in chess.SQUARES)}
    best = []
    for mv in board.legal_moves:
        b2 = board.copy()
        b2.push(mv)
        o2 = occupancy(b2)
        mismatch = sum(1 for sq in chess.SQUARES if o2[sq] != occ[sq])
        best.append((mismatch, mv))
    best.sort(key=lambda t: t[0])
    return best[0][1], best[0][0], (best[1][0] if len(best) > 1 else 99)


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
CENTRE = np.array([[-2, -1, 0, 1, 1, 0, -1, -2]]).T + np.array([[-2, -1, 0, 1, 1, 0, -1, -2]])


FAR_RANK_PENALTY = 400   # centipawns: the robot prefers not to pick pieces off ranks 7-8 (see README)


def evaluate(board):
    if board.is_checkmate():
        return -100000
    score = 0
    for sq, p in board.piece_map().items():
        v = VALUES[p.piece_type] + 4 * CENTRE[chess.square_rank(sq), chess.square_file(sq)]
        if p.piece_type == chess.PAWN:
            v += 6 * (chess.square_rank(sq) if p.color else 7 - chess.square_rank(sq))
        score += v if p.color == board.turn else -v
    return score


def search(board, depth, alpha=-10 ** 9, beta=10 ** 9):
    if depth == 0 or board.is_game_over():
        return evaluate(board), None
    best_mv, best = None, -10 ** 9
    moves = sorted(board.legal_moves, key=lambda m: (not board.is_capture(m), random.random()))
    robot_to_move = board.turn == chess.WHITE
    for mv in moves:
        # The physical robot handles the far ranks less reliably, so at its own root
        # moves it discounts captures there (the human's teleported moves are free).
        penalty = FAR_RANK_PENALTY if (robot_to_move and board.is_capture(mv)
                                       and chess.square_rank(mv.to_square) >= 6) else 0
        board.push(mv)
        score = -search(board, depth - 1, -beta, -alpha)[0] - penalty
        board.pop()
        if score > best:
            best, best_mv = score, mv
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best, best_mv


def human_move(board):
    """A cautious but fallible opponent: shallow search, random among the top three."""
    scored = []
    for mv in board.legal_moves:
        board.push(mv)
        scored.append((-search(board, 1)[0], mv))
        board.pop()
    scored.sort(key=lambda t: -t[0])
    return random.choice(scored[:3])[1]


# --------------------------------------------------------------------------- #
# the robot
# --------------------------------------------------------------------------- #
class ChessBot:
    def __init__(self, view=False, recorder=None):
        self.model, self.plan_model, self.piece_name = build_chess_scene()
        self.data = mujoco.MjData(self.model)
        self.data.qpos[self.model.jnt_qposadr[self.model.joint("base_x").id]] = START_POSE[0]
        self.data.qpos[self.model.jnt_qposadr[self.model.joint("base_y").id]] = START_POSE[1]
        mujoco.mj_forward(self.model, self.data)
        self.arm = pp.Arm(self.model, "right")
        robot_bodies = {i for i in range(self.model.nbody) if pp.part(self.model.body(i).name) in pp.ARM_PARTS}
        self.planner = pp.Planner(self.plan_model, self.arm, robot_bodies, np.random.default_rng(0))
        self.viewer = None
        if view:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0.1, -0.1, 0.8]
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 150
            self.viewer.cam.elevation = -20
        self.runner = pp.Runner(self.model, self.data, self.arm, robot_bodies, viewer=self.viewer, recorder=recorder)
        self.base_act = {n: self.model.actuator(n).id for n in ("base_x", "base_y", "base_yaw", "wheels")}
        self.base_q = {n: self.model.jnt_qposadr[self.model.joint(n).id] for n in ("base_x", "base_y", "base_yaw")}
        self.data.ctrl[self.base_act["base_x"]] = START_POSE[0]
        self.data.ctrl[self.base_act["base_y"]] = START_POSE[1]
        self.odometer = 0.0
        self.piece_body = {sq: self.model.body(n).id for sq, n in self.piece_name.items()}
        self.near_ok = {self.model.geom("table_top").id, self.model.geom("board_plate").id}
        self.near_ok |= {g for g in range(self.model.ngeom) if self.model.geom(g).name.startswith("sq_")}
        self.near_ok |= {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.piece_body.values()}
        self.grave_used = 0
        self.human_grave_used = 0
        self.helper_resets = 0
        self.q_ready = None
        self.status = "booting"
        # World yaw of the tool frame relative to R_GRASP. Turning the base with the joints
        # held rotates the tool with it; move_line un-twists it gradually along the line.
        self.tool_yaw = 0.0
        self.arm_out = False            # False: at home; True: out over the table (carry / look pose)
        self.settle(0.3)

    # -- base ------------------------------------------------------------- #
    def base_xy(self):
        return np.array([self.data.qpos[self.base_q["base_x"]], self.data.qpos[self.base_q["base_y"]]])

    def drive(self, x, y, speed=DRIVE_SPEED):
        start = np.array([self.data.ctrl[self.base_act["base_x"]], self.data.ctrl[self.base_act["base_y"]]])
        goal = np.array([x, y])
        dist = np.linalg.norm(goal - start)
        if dist < 1e-4:
            return
        dur = max(0.8, dist / speed)
        t0 = self.data.time
        while self.data.time < t0 + dur:
            s = 0.5 - 0.5 * math.cos(math.pi * (self.data.time - t0) / dur)
            p = start + (goal - start) * s
            self.data.ctrl[self.base_act["base_x"]], self.data.ctrl[self.base_act["base_y"]] = p
            self.data.ctrl[self.base_act["wheels"]] = -(self.odometer + dist * s) / pp.WHEEL_RADIUS
            self.runner.step()
        self.odometer += dist
        self.settle(0.4)

    def settle(self, seconds):
        self.runner.hold(seconds)
        self.planner.sync(self.data)

    # -- turning in place -------------------------------------------------- #
    def yaw(self):
        return float(self.data.qpos[self.base_q["base_yaw"]])

    def rot(self, yaw=None):
        th = self.yaw() if yaw is None else yaw
        return np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])

    def to_base(self, xy):
        """World (x, y) -> base frame (x, y) at the current yaw."""
        d = np.asarray(xy, float) - self.base_xy()
        return (self.rot().T @ [d[0], d[1], 0])[:2]

    def yaw_for(self, xy):
        """Yaw that puts world point xy at base-frame y == SWEET_Y (in front of the arm)."""
        dx, dy = np.asarray(xy, float) - self.base_xy()
        r = math.hypot(dx, dy)
        return math.atan2(dy, dx) - math.asin(max(-1.0, min(1.0, SWEET_Y / r)))

    def turn(self, yaw, speed=YAW_SPEED):
        """Rotate the base in place to `yaw` (cosine profile); the wheels counter-rotate."""
        start = float(self.data.ctrl[self.base_act["base_yaw"]])
        delta = yaw - start
        if abs(delta) < 1e-3:
            return
        dur = max(0.4, abs(delta) / speed)
        t0 = self.data.time
        odo0 = self.odometer
        while self.data.time < t0 + dur:
            s = 0.5 - 0.5 * math.cos(math.pi * (self.data.time - t0) / dur)
            self.data.ctrl[self.base_act["base_yaw"]] = start + delta * s
            self.data.ctrl[self.base_act["wheels"]] = -(odo0 + 0.2 * delta * s) / pp.WHEEL_RADIUS   # right wheel arc
            self.runner.step()
        self.odometer = odo0 + 0.2 * delta
        self.tool_yaw += delta          # the arm turned with the base
        self.settle(0.2)

    def face(self, xy):
        """Bring world point xy into the arm's working fan by turning in place."""
        self.turn(self.yaw_for(xy))

    def carry_pose(self):
        """World position of the arm's parking spot, which rotates with the base."""
        base = self.base_xy()
        p = self.rot() @ np.array([CARRY_X, SWEET_Y, 0.0])
        return np.array([base[0] + p[0], base[1] + p[1], CARRY_Z])

    def look_pose(self):
        base = self.base_xy()
        p = self.rot() @ np.array([LOOK_POSE[0], LOOK_POSE[1], 0.0])
        return np.array([base[0] + p[0], base[1] + p[1], LOOK_POSE[2]])

    # -- arm -------------------------------------------------------------- #
    def tool(self):
        return self.runner.eef_pos()

    def move_line(self, target, speed, ignore=None, rate=None, settle=0.1, check=True, yaw=0.0):
        """Straight line, eased at both ends and time-scaled so no joint exceeds `rate` x
        VMAX: near a wrist singularity a steady Cartesian speed would spin the wrist fast
        enough to whip a gripped piece out of the pads.

        Orientation: the grasp frame ends twisted by `yaw` about z relative to R_GRASP
        (0 = diagonal to the board grid, what picking/placing on a square needs). A tool
        left twisted by a base turn is un-twisted gradually along the line, never as a step.
        yaw=None keeps the current twist (used for the tray, which has no neighbours)."""
        target = np.asarray(target, float)
        self.planner.sync(self.data)
        start, q = self.tool(), self.runner.q_cmd.copy()
        length = np.linalg.norm(target - start)
        if length < 1e-4:
            return
        if rate is None:
            rate = RATE if self.runner.held is not None else RATE_EMPTY
        limit = pp.VMAX * rate * TICK
        yaw0 = self.tool_yaw
        yaw1 = yaw0 if yaw is None else yaw
        twist_len = max(length, abs(yaw1 - yaw0) * 0.15)   # >= 0.15 m of travel per radian of un-twist
        s = 0.0
        while s < 1.0:
            ease = min(1.0, max(0.12, min(s, 1.0 - s) / 0.15))
            ds = min(speed * TICK * ease / length, 1.0 - s)
            while True:
                cur = start + (target - start) * (s + ds)
                f = min(1.0, (s + ds) * length / twist_len)
                q_new = self.arm.ik(self.plan_model, self.planner.d, cur, self.rot(yaw0 + (yaw1 - yaw0) * f) @ R_GRASP, q, iters=200)
                if q_new is None:
                    raise RuntimeError(f"IK failed at {np.round(cur, 3)}")
                if np.all(np.abs(q_new - q) <= limit) or ds < 1e-5:
                    break
                ds *= 0.5   # joint-rate limited: cover less of the line this tick
            if check and self.planner.in_collision(q_new, ignore, self.near_ok):
                raise RuntimeError(f"predicted collision at {np.round(cur, 3)}: {self.collisions(ignore)}")
            s += ds
            q = q_new
            self.runner.q_cmd = q
            self.data.ctrl[self.arm.act] = q
            self.runner.hold(TICK)
        self.tool_yaw = yaw0 + (yaw1 - yaw0) * min(1.0, length / twist_len)
        if settle:
            self.runner.hold(settle)

    def collisions(self, ignore=None):
        """Penetrating pairs in the planning model after the last collision check."""
        d, m = self.planner.d, self.plan_model
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if ignore in (b1, b2) or c.dist > 0:
                continue
            if b1 not in self.planner.robot_bodies and b2 not in self.planner.robot_bodies:
                continue
            out.append(f"{m.geom(c.geom1).name or m.body(b1).name}~{m.geom(c.geom2).name or m.body(b2).name}:{c.dist * 1000:.1f}mm")
        return out

    def go_ready(self):
        """Arm parked above the near edge of the board, ready to work. The first time
        this plans a collision-free path from home; later calls replay it, so the arm
        can shuttle between home (out of the camera's view) and the board without
        re-planning."""
        if self.arm_out:                            # waiting at the look pose: one line back
            self.move_line(self.carry_pose(), LINE_SPEED)
            return
        if self.q_ready is None:
            self.turn(0.0)
            self.planner.sync(self.data)
            self.q_ready = self.planner.solve_ik(self.carry_pose(), R_GRASP, np.zeros(7))
            self.ready_path = self.planner.plan(self.runner.q_cmd, self.q_ready)
        self.runner.move(self.ready_path)          # joint space: valid at any base yaw
        self.tool_yaw = self.yaw()                  # q_ready was solved at yaw 0, so the tool turned with the base
        self.runner.hold(0.2)
        self.runner.set_gripper(OPEN, settle=0.2)   # only once clear of the mast
        self.arm_out = True

    def go_watch(self):
        """Move the arm aside so it does not block the head camera's view of the board, and
        face the board squarely so all four markers are in view."""
        self.move_line(self.carry_pose(), LINE_SPEED)
        self.turn(0.0)
        self.move_line(self.look_pose(), LINE_SPEED)

    def go_home(self):
        """Retract to the home pose (end of the game / stop)."""
        if not self.arm_out:
            return
        self.move_line(self.carry_pose(), LINE_SPEED)
        self.turn(0.0)
        self.runner.set_gripper(CLOSED, settle=0.2)
        self.runner.move(self.ready_path[::-1])
        self.runner.hold(0.2)
        self.arm_out = False

    def align(self, y_world):
        """Kept for old callers: the base no longer slides sideways (see BASE_Y)."""
        raise RuntimeError("align() is gone: use face(xy) - the base turns in place instead of strafing")

    def pick(self, sq, body):
        x, y = square_xy(sq)
        top = PIECE_SHAFT_TOP[self.piece_name[sq][1]]
        grasp = np.array([x, y, grasp_height(top)]) + PAD_SHIFT
        self.runner.set_gripper(OPEN, settle=0.1)
        self.move_line(grasp + [0, 0, LIFT], LINE_SPEED, ignore=body)
        self.move_line(grasp, APPROACH_SPEED, ignore=body)
        self.runner.held = body
        self.runner.set_gripper(CLOSED, settle=0.3)
        z0 = self.data.xpos[body, 2]
        self.move_line(grasp + [0, 0, LIFT], DESCENT_SPEED, ignore=body)
        if self.data.xpos[body, 2] - z0 < 0.06:
            raise RuntimeError(f"failed to lift the piece on {chess.square_name(sq)}")
        far = self.to_base(grasp[:2])[0] > FAR_X
        self.move_line(self.carry_pose(), FAR_SPEED if far else CARRY_SPEED, ignore=body)

    def touching(self, body):
        """Is the held piece resting on something other than the gripper?"""
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if body in (b1, b2) and not ({b1, b2} & self.runner.gripper_bodies):
                return True
        return False

    def place(self, xy, body, surface, diagonal=True, auto_reset=True):
        """Set the held piece down on `surface` (z). A piece creeps a few mm down the
        pads during a carry, so rather than trusting a nominal height the last stretch
        is descended slowly until the piece is felt to touch, then the pads open.
        diagonal=False (the tray) keeps whatever twist the tool has: nothing to clear there."""
        x, y = xy
        yaw = 0.0 if diagonal else None
        shift = pad_shift(0.0 if diagonal else self.tool_yaw)
        far = self.to_base((x, y))[0] > FAR_X
        above = np.array([x, y, surface + LIFT + 0.05]) + shift
        self.move_line(above, FAR_SPEED if far else CARRY_SPEED, ignore=body, yaw=yaw)
        if not self.in_hand(body):
            raise RuntimeError("piece lost on the way to the drop point")
        hang = self.tool()[2] - self.data.xpos[body, 2]
        near = np.array([x, y, surface + hang + 0.010]) + shift
        self.move_line(near, DESCENT_SPEED, ignore=body, yaw=yaw)
        z = near[2]
        while not self.touching(body) and z > surface + hang - 0.01:
            z -= 0.002
            self.move_line([near[0], near[1], z], 0.03, ignore=body, settle=0.0, yaw=yaw)
        self.runner.hold(0.1)
        self.runner.set_gripper(OPEN, settle=0.2)
        self.runner.held = None
        self.move_line([near[0], near[1], z + LIFT], APPROACH_SPEED, ignore=body, yaw=yaw)
        err = np.linalg.norm(self.data.xpos[body, :2] - [x, y])
        upright = self.data.xmat[body].reshape(3, 3)[2, 2] > 0.95
        print(f"      placed {self.model.body(body).name} {err * 1000:.0f} mm off centre, {'upright' if upright else 'TIPPED'}", flush=True)
        if auto_reset and (err > 0.015 or not upright):
            self.helper_reset(body, (x, y), surface)

    def in_hand(self, body):
        p = self.data.xpos[body] - self.tool()
        return abs(p[2] + 0.035) < 0.03 and np.linalg.norm(p[:2] + pad_shift(self.tool_yaw)[:2]) < 0.03

    def helper_reset(self, body, xy, surface):
        """A dropped or knocked-over piece is stood back on its square by a helper, as at a
        real chess-robot demo. Counted and reported."""
        self.helper_resets += 1
        print(f"      helper resets {self.model.body(body).name} onto its square", flush=True)
        self.runner.held = None
        self.runner.set_gripper(OPEN, settle=0.2)
        adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], surface + 0.0005]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[body]]
        self.data.qvel[vadr:vadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.5)

    def transfer(self, from_sq, to_xy, to_sq=None):
        """Physically move the piece on from_sq to to_xy (a square or a graveyard slot).
        A physical mishap never aborts the game: the piece is reset by the helper and the
        arm re-parks."""
        body = self.piece_body.pop(from_sq)
        surface = BOARD_Z if to_sq is not None else TABLE_TOP
        try:
            self.face(square_xy(from_sq))
            self.pick(from_sq, body)
            if not self.in_hand(body):
                raise RuntimeError("piece lost after the lift")
            self.face(to_xy)
            if not self.in_hand(body):
                raise RuntimeError("piece lost while turning")
            self.place(to_xy, body, surface, diagonal=to_sq is not None)
        except RuntimeError as e:
            print(f"      mishap: {e}", flush=True)
            self.helper_reset(body, to_xy, surface)
            self.escape()
        if to_sq is not None:
            self.piece_body[to_sq] = body
            self.piece_name[to_sq] = self.piece_name[from_sq]
        del self.piece_name[from_sq]
        self.move_line(self.carry_pose(), LINE_SPEED)

    def escape(self):
        """Get the (empty) gripper back to the carry pose from wherever a mishap left it:
        straight up first without collision checking (up is always away from the board),
        then a checked line, then a planned path as the last resort."""
        self.runner.held = None
        self.runner.set_gripper(OPEN, settle=0.2)
        self.planner.sync(self.data)
        tool = self.tool()
        try:
            self.move_line([tool[0], tool[1], min(tool[2] + LIFT, CARRY_Z)], DESCENT_SPEED, check=False, yaw=None)
            self.move_line(self.carry_pose(), DESCENT_SPEED)
        except RuntimeError:
            self.planner.sync(self.data)
            self.runner.move(self.planner.plan(self.runner.q_cmd, self.q_ready, ignore_body=None))
            self.tool_yaw = self.yaw()

    def play(self, board, mv):
        """Execute the engine's move on the physical board."""
        captured_sq = None
        if board.is_en_passant(mv):
            captured_sq = mv.to_square + (-8 if board.turn else 8)
        elif board.is_capture(mv):
            captured_sq = mv.to_square
        if captured_sq is not None:
            self.status = f"removing captured {self.piece_name[captured_sq]}"
            self.transfer(captured_sq, GRAVE[self.grave_used])
            self.grave_used += 1
        self.status = f"playing {mv.uci()}"
        self.transfer(mv.from_square, square_xy(mv.to_square), mv.to_square)
        if board.is_castling(mv):
            rook_from, rook_to = ((chess.H1, chess.F1) if mv.to_square == chess.G1 else (chess.A1, chess.D1))
            self.transfer(rook_from, square_xy(rook_to), rook_to)

    # -- the simulated human ------------------------------------------------ #
    def teleport(self, body, xy):
        adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
        self.data.qpos[adr:adr + 3] = [xy[0], xy[1], BOARD_Z + 0.0005]
        self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[body]]
        self.data.qvel[vadr:vadr + 6] = 0

    def human_plays(self, board, mv):
        if board.is_en_passant(mv):
            cap = mv.to_square + (-8 if board.turn else 8)
        elif board.is_capture(mv):
            cap = mv.to_square
        else:
            cap = None
        if cap is not None:
            self.teleport(self.piece_body.pop(cap), HUMAN_GRAVE[self.human_grave_used])
            self.human_grave_used += 1
            del self.piece_name[cap]
        body = self.piece_body.pop(mv.from_square)
        self.teleport(body, square_xy(mv.to_square))
        self.piece_body[mv.to_square] = body
        self.piece_name[mv.to_square] = self.piece_name.pop(mv.from_square)
        if board.is_castling(mv):
            rf, rt = ((chess.H8, chess.F8) if mv.to_square == chess.G8 else (chess.A8, chess.D8))
            rb = self.piece_body.pop(rf)
            self.teleport(rb, square_xy(rt))
            self.piece_body[rt] = rb
            self.piece_name[rt] = self.piece_name.pop(rf)
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.8)

    # -- vision --------------------------------------------------------------- #
    def look(self):
        w, h = 1280, 960
        renderer = mujoco.Renderer(self.model, h, w)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = self.model.camera("head_cam").id
        renderer.update_scene(self.data, cam)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        rgb = renderer.render().copy()
        renderer.close()
        return rgb


# --------------------------------------------------------------------------- #
# dashboard + video
# --------------------------------------------------------------------------- #
class Dashboard:
    W, H = 1920, 1080

    def __init__(self, model, out_dir, video=None, fps=30):
        self.model = model
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, 720, 1280)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = model.camera("wide").id
        self.writer = None
        if video:
            import imageio
            self.writer = imageio.get_writer(video, fps=fps, codec="libx264", quality=8)
        self.fps, self.next_t, self.frames = fps, 0.0, 0
        self.cam_panel = np.zeros((480, 640, 3), np.uint8)
        self.board_panel = np.zeros((600, 640, 3), np.uint8)
        self.status_lines = []
        self.moves = []
        self.bot = None

    def use_camera(self, name):
        self.cam.fixedcamid = self.model.camera(name).id

    def tick(self, data):
        if self.writer and data.time >= self.next_t:
            self.writer.append_data(self.compose(data))
            self.next_t += 1 / self.fps
            self.frames += 1

    def compose(self, data):
        self.renderer.update_scene(data, self.cam)
        scene = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        canvas = np.full((self.H, self.W, 3), 24, np.uint8)
        canvas[:720, :1280] = scene
        canvas[:480, 1280:] = self.cam_panel
        canvas[480:, 1280:] = self.board_panel
        y = 760
        cv2.putText(canvas, "BracketBot plays chess", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        status = self.bot.status if self.bot else ""
        cv2.putText(canvas, status, (24, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 220, 255), 2, cv2.LINE_AA)
        for i, line in enumerate(self.status_lines[-5:]):
            cv2.putText(canvas, line, (24, y + 85 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        moves = " ".join(self.moves[-12:])
        cv2.putText(canvas, moves, (24, 1060), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def update_vision(self, rgb, occ, H, marker_px, board, note):
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # grid lines and classifications projected back into the camera image
        for i in range(9):
            a = np.array([[[RANK1_X - SQ / 2 + i * SQ, FILEA_Y + SQ / 2], [RANK1_X - SQ / 2 + i * SQ, -FILEA_Y - SQ / 2]]], np.float32)
            b = np.array([[[RANK1_X - SQ / 2, FILEA_Y + SQ / 2 - i * SQ], [RANK1_X + 7.5 * SQ, FILEA_Y + SQ / 2 - i * SQ]]], np.float32)
            for seg in (a, b):
                p = cv2.perspectiveTransform(seg, H)[0].astype(int)
                cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 255, 255), 1, cv2.LINE_AA)
        for sq, o in occ.items():
            if o is None:
                continue
            x, y = square_xy(sq)
            p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H)[0, 0].astype(int)
            cv2.circle(img, tuple(p), 9, (255, 255, 255) if o == "w" else (40, 40, 40), 2)
        for p in marker_px.astype(int):
            cv2.circle(img, tuple(p), 14, (0, 200, 255), 2)
        cv2.putText(img, "head camera: markers -> homography -> squares", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        self.cam_panel = cv2.resize(img, (640, 480))
        self.board_panel = self.draw_board(occ, board, note)

    def draw_board(self, occ, board, note):
        panel = np.full((600, 640, 3), 30, np.uint8)
        s, ox, oy = 60, 80, 20
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            x0, y0 = ox + f * s, oy + (7 - r) * s
            light = (f + r) % 2 == 1
            cv2.rectangle(panel, (x0, y0), (x0 + s, y0 + s), (140, 190, 220) if light else (70, 110, 150), -1)
            o = occ.get(sq)
            if o:
                cv2.circle(panel, (x0 + s // 2, y0 + s // 2), 22, (245, 240, 235) if o == "w" else (25, 25, 25), -1)
                p = board.piece_at(sq)
                if p is not None:
                    cv2.putText(panel, p.symbol().upper(), (x0 + s // 2 - 10, y0 + s // 2 + 9), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (30, 30, 30) if o == "w" else (230, 230, 230), 2, cv2.LINE_AA)
        for i in range(8):
            cv2.putText(panel, "abcdefgh"[i], (ox + i * s + s // 2 - 8, oy + 8 * s + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(panel, str(8 - i), (ox - 28, oy + i * s + s // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        if board.move_stack:
            mv = board.peek()
            colour = (60, 200, 60) if not board.turn else (60, 140, 255)   # white just moved -> green
            for a, b_, c in [(mv.from_square, mv.to_square, colour)]:
                pa = (ox + chess.square_file(a) * s + s // 2, oy + (7 - chess.square_rank(a)) * s + s // 2)
                pb = (ox + chess.square_file(b_) * s + s // 2, oy + (7 - chess.square_rank(b_)) * s + s // 2)
                cv2.arrowedLine(panel, pa, pb, c, 4, cv2.LINE_AA, tipLength=0.25)
        cv2.putText(panel, "board map from vision", (ox, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, note, (ox, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
        return panel

    def snapshot(self, data, name):
        cv2.imwrite(str(self.out / name), cv2.cvtColor(self.compose(data), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer:
            self.writer.close()
        self.renderer.close()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moves", type=int, default=5, help="robot moves to play")
    ap.add_argument("--depth", type=int, default=3, help="engine search depth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="open the MuJoCo viewer")
    ap.add_argument("--record", metavar="FILE.mp4", help="write a video of the game")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "chess_frames", help="dashboard PNGs")
    args = ap.parse_args()
    random.seed(args.seed)

    t0 = time.time()
    dash_model = None
    bot = ChessBot(view=args.view)
    dash = Dashboard(bot.model, args.out, args.record)
    dash.bot = bot
    bot.runner.recorder = dash
    board = chess.Board()
    occ0 = {sq: (None if p is None else ("w" if p.color else "b")) for sq, p in ((s, board.piece_at(s)) for s in chess.SQUARES)}
    dash.board_panel = dash.draw_board(occ0, board, "waiting for the first look at the board")

    print("rolling up to the table...", flush=True)
    bot.status = "rolling up to the table"
    bot.drive(0.0, BASE_Y)
    dash.use_camera("table")
    bot.status = "reading the board"
    rgb = bot.look()
    occ, H, rect, px = read_board(rgb)
    n_w, n_b = sum(o == "w" for o in occ.values()), sum(o == "b" for o in occ.values())
    print(f"board read: {n_w} white, {n_b} black pieces", flush=True)
    dash.update_vision(rgb, occ, H, px, board, f"initial read: {n_w} white / {n_b} black")
    dash.status_lines.append(f"initial position read from camera: {n_w}w {n_b}b")
    dash.snapshot(bot.data, "turn00_read.png")
    cv2.imwrite(str(args.out / "turn00_rectified.png"), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
    bot.status = "moving to the ready pose"
    bot.go_ready()

    try:
        for turn in range(1, args.moves + 1):
            # --- robot (white) ---
            bot.status = "thinking..."
            score, mv = search(board, args.depth)
            san = board.san(mv)
            print(f"turn {turn}: robot plays {san} ({mv.uci()}, eval {score:+d})", flush=True)
            dash.status_lines.append(f"robot: {san}  (search depth {args.depth}, eval {score:+d})")
            bot.play(board, mv)
            board.push(mv)
            dash.moves.append(f"{turn}. {san}")
            if bot.helper_resets:
                dash.status_lines.append(f"(helper reset a knocked-over piece: {bot.helper_resets} so far)")
            dash.board_panel = dash.draw_board(occ_of(board), board, f"robot played {san}")
            dash.snapshot(bot.data, f"turn{turn:02d}_robot.png")
            if board.is_game_over():
                break
            # --- human (black) ---
            hv = human_move(board)
            hsan = board.san(hv)
            bot.status = f"human moves {hsan}"
            print(f"         human plays {hsan} ({hv.uci()})", flush=True)
            bot.human_plays(board, hv)
            truth = board.copy()
            truth.push(hv)
            # --- vision: read the board and infer what the human did ---
            bot.status = "reading the board"
            bot.go_watch()
            rgb = bot.look()
            occ, H, rect, px = read_board(rgb)
            inferred, mism, second = infer_move(board, occ)
            ok = inferred == hv
            note = f"saw {hsan if ok else board.san(inferred)}: {mism} mismatches (next best {second})"
            print(f"         vision inferred {board.san(inferred)} - {'correct' if ok else 'WRONG'} ({note})", flush=True)
            dash.status_lines.append(f"human: {hsan} -> vision inferred {board.san(inferred)} {'OK' if ok else 'MISMATCH'}")
            board.push(inferred)
            dash.moves.append(hsan)
            dash.update_vision(rgb, occ, H, px, board, note)
            dash.snapshot(bot.data, f"turn{turn:02d}_vision.png")
            cv2.imwrite(str(args.out / f"turn{turn:02d}_rectified.png"), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("vision inferred the wrong move; the physical and logical boards would diverge")
            if board.is_game_over():
                break
            bot.status = "moving to the ready pose"
            bot.go_ready()
        bot.status = "game paused - returning home"
        bot.go_home()
        print(f"illegal contacts: {bot.runner.illegal_steps} steps"
              + (f" {bot.runner.illegal_pairs}" if bot.runner.illegal_pairs else " (none)")
              + f"; helper resets: {bot.helper_resets}", flush=True)
        print(f"moves: {' '.join(dash.moves)}   ({time.time() - t0:.0f}s wall)", flush=True)
    finally:
        dash.close()
        if args.record:
            print(f"wrote {args.record} ({dash.frames} frames)")
        if bot.viewer is not None:
            while bot.viewer.is_running():
                bot.runner.step()


def occ_of(board):
    return {sq: (None if p is None else ("w" if p.color else "b")) for sq, p in ((s, board.piece_at(s)) for s in chess.SQUARES)}


if __name__ == "__main__":
    main()
