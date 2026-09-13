"""Calibration and tuning parameters.

Everything the demo needs to know about the physical set-up lives in one JSON file
(`gambit_calibration.json`) so it can be edited on the robot without touching code.
Frames: robot frame, x forward, y left, z up, origin at the base (the frame the arm
IK and the point cloud use).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass, field

import chess


@dataclass
class BoardGeometry:
    """Where the board is, in the robot frame."""
    square: float = 0.03                       # square pitch (m)
    a1: tuple = (0.175, 0.105)                 # (x, y) of the centre of a1
    file_dir: tuple = (0.0, -1.0)              # unit vector a -> h  (robot plays from the rank-1 side)
    rank_dir: tuple = (1.0, 0.0)               # unit vector rank 1 -> 8
    board_z: float = 0.731                     # playing surface height
    table_z: float = 0.72                      # height of the surface the capture tray stands on
    marker_offset: float = 0.267               # marker centres this many squares outside each board corner
    tray: list = field(default_factory=list)   # [(x, y), ...] captured-piece slots (robot's captures)
    # grasp point above the playing surface for each piece kind (pads centred a little
    # below the top of a plain shaft; see manipulation.py)
    piece_top: dict = field(default_factory=lambda: {"p": 0.050, "n": 0.053, "b": 0.055, "r": 0.052, "q": 0.058, "k": 0.061})

    def square_xy(self, sq: int) -> tuple:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x = self.a1[0] + self.square * (f * self.file_dir[0] + r * self.rank_dir[0])
        y = self.a1[1] + self.square * (f * self.file_dir[1] + r * self.rank_dir[1])
        return (float(x), float(y))

    def board_uv_to_xy(self, u: float, v: float) -> tuple:
        """Board-plane coordinates in squares (u along files from the a1 outer corner,
        v along ranks) -> robot frame."""
        du, dv = u - 0.5, v - 0.5
        x = self.a1[0] + self.square * (du * self.file_dir[0] + dv * self.rank_dir[0])
        y = self.a1[1] + self.square * (du * self.file_dir[1] + dv * self.rank_dir[1])
        return (float(x), float(y))

    def marker_xy(self) -> list:
        """Robot-frame (x, y) of the four corner markers: near-a, near-h, far-a, far-h."""
        m = self.marker_offset
        return [self.board_uv_to_xy(u, v) for u, v in ((-m, -m), (8 + m, -m), (-m, 8 + m), (8 + m, 8 + m))]

    def default_tray(self, n: int = 12) -> list:
        """Two rows of capture slots beside the h-file, 1.5 pitches from the board."""
        pitch = 1.5 * self.square
        out = []
        for row in range(2):
            for i in range(n // 2):
                out.append(self.board_uv_to_xy(8 + 2.0 + row * 1.5, 0.5 + i * 1.5))
        return out


@dataclass
class VisionConfig:
    marker_mode: str = "aruco"                 # "aruco" (printed markers, recommended) | "colour" (blue dots; the sim)
    aruco_dict: str = "DICT_4X4_50"
    aruco_ids: tuple = (0, 1, 2, 3)            # marker ids at the near-a, near-h, far-a, far-h corners
    colour_hsv_lo: tuple = (100, 120, 80)
    colour_hsv_hi: tuple = (130, 255, 255)
    method: str = "reference"                  # "reference" (diff vs empty board) | "colour" (HSV thresholds)
    occupied_diff: float = 18.0                # mean |gray - reference| over the sample patch => occupied
    colour_split: float | None = None          # gray level separating black from white pieces (learned at start)
    white_v: int = 200                         # colour mode: pixel is 'white piece' if V > white_v and S < white_s
    white_s: int = 50
    black_v: int = 50                          # colour mode: pixel is 'black piece' if V < black_v
    min_fraction: float = 0.15                 # colour mode: fraction of such pixels to call a square occupied
    # sample patch inside a square, as fractions of the square (rows from the far edge).
    # A band straddling the square centre, where the piece's BASE sits: tall pieces smear
    # away from the camera, so the near strip of a square receives the top of the piece in
    # front of it and the far strip its own shaft. Swept on rendered positions
    # (bracket_gambit/STATUS.md): 0 errors with (0.45, 0.65) vs 4-11 with the near half.
    patch: tuple = (0.45, 0.65, 0.24, 0.76)    # row0, row1, col0, col1
    min_marker_area: float = 15.0
    camera_width: int = 1280                   # requested capture size (sim render / real jpeg is whatever it is)
    camera_height: int = 960


@dataclass
class MotionConfig:
    approach: float = 0.10                     # m above the grasp pose for the pre-grasp / pre-place
    grasp_below_top: float = 0.020             # tool frame this far below the piece's grasp-point height
    line_speed: float = 0.10                   # m/s free moves (empty hand)
    carry_speed: float = 0.06                  # m/s with a piece
    descent_speed: float = 0.04                # m/s last stretch down onto / off a piece
    far_speed: float = 0.025                   # m/s retracting from the far ranks (stretched arm)
    far_x: float = 0.31                        # base-frame x beyond which a square counts as 'far'
    gripper_open: float = 0.25                 # gripper joint angle (rad, URDF) with the pads clear of neighbours
    gripper_closed: float = 0.0
    grip_empty_angle: float = 0.02             # real robot: measured gripper angle below this after closing => nothing gripped
    max_attempts: int = 2                      # pick attempts before asking for help
    workspace: tuple = ((0.10, 0.45), (-0.40, 0.20), (0.70, 1.05))   # allowed tool positions (x, y, z ranges)


@dataclass
class GambitConfig:
    board: BoardGeometry = field(default_factory=BoardGeometry)
    vision: VisionConfig = field(default_factory=VisionConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    stockfish: str | None = None               # path to the binary; None = search PATH / $STOCKFISH
    skill: int = 5                             # Stockfish Skill Level 0..20
    think_time: float = 1.0                    # s per move
    arm: str = "arm_right"
    robot_color: str = "white"                 # the robot's pieces stand on the ranks nearest to it (see README)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "GambitConfig":
        raw = json.loads(text)
        cfg = cls()
        for section in ("board", "vision", "motion"):
            if section in raw:
                obj = getattr(cfg, section)
                for k, v in raw[section].items():
                    if hasattr(obj, k):
                        setattr(obj, k, tuple(v) if isinstance(getattr(obj, k), tuple) else v)
        for k in ("stockfish", "skill", "think_time", "arm", "robot_color"):
            if k in raw:
                setattr(cfg, k, raw[k])
        if isinstance(cfg.board.tray, list):
            cfg.board.tray = [tuple(t) for t in cfg.board.tray]
        return cfg

    @classmethod
    def load(cls, path: str | pathlib.Path | None) -> "GambitConfig":
        if path is None or not pathlib.Path(path).exists():
            cfg = cls()
        else:
            cfg = cls.from_json(pathlib.Path(path).read_text())
        if not cfg.board.tray:
            cfg.board.tray = cfg.board.default_tray()
        return cfg

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(self.to_json())


def occupancy_of(board: chess.Board) -> dict:
    """{square: 'w' | 'b' | None} for a python-chess position."""
    return {sq: (None if p is None else ("w" if p.color else "b"))
            for sq, p in ((s, board.piece_at(s)) for s in chess.SQUARES)}
