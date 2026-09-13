"""Board perception.

    image --(4 corner markers)--> homography --> rectified top-down board (WARP px / square)
          --> per-square occupancy: 'w' | 'b' | None

Two marker detectors (ArUco ids for the real board, blue dots for the simulator) and two
square classifiers:
  * reference: compare each square with a rectified image of the EMPTY board taken at
    calibration; a square whose sample patch differs is occupied, and the brightness of
    the differing pixels says which colour. Works for any board / piece colours; the
    black-vs-white split is learned from the 32 pieces of the starting position.
  * colour: fixed HSV thresholds (what the simulator was validated with).

Everything here is pure numpy/OpenCV and works on any image, so it is unit-tested with
synthetic renders (tests/test_vision.py) and can be tuned from saved photos of the real
board without the robot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import cv2
import numpy as np

from .config import VisionConfig, occupancy_of

WARP = 50   # px per square in the rectified view


class VisionError(RuntimeError):
    pass


@dataclass
class Observation:
    occupancy: dict                               # {square: 'w' | 'b' | None}
    rectified: np.ndarray                         # 8*WARP x 8*WARP RGB
    H_img_from_board: np.ndarray                  # 3x3, board (u, v in squares) -> image px
    marker_px: np.ndarray                         # 4x2 near-a, near-h, far-a, far-h
    image: np.ndarray
    scores: dict = field(default_factory=dict)    # {square: (occupied_score, brightness)}

    def counts(self):
        return (sum(o == "w" for o in self.occupancy.values()), sum(o == "b" for o in self.occupancy.values()))


# --------------------------------------------------------------------------- #
# board <-> rectified image coordinates
# --------------------------------------------------------------------------- #
def board_to_warp(uv):
    """Board-plane coords in squares (u along files, v along ranks, origin at the a1
    outer corner) -> rectified pixel (col, row); rank 8 at the top."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    return np.stack([uv[:, 0] * WARP, (8.0 - uv[:, 1]) * WARP], axis=1)


def square_uv(sq):
    return (chess.square_file(sq) + 0.5, chess.square_rank(sq) + 0.5)


def marker_uv(offset):
    return np.array([(-offset, -offset), (8 + offset, -offset), (-offset, 8 + offset), (8 + offset, 8 + offset)], np.float64)


def square_patch(sq, cfg: VisionConfig):
    """Row/col slice of a square's sample patch in the rectified image."""
    c0 = chess.square_file(sq) * WARP
    r0 = (7 - chess.square_rank(sq)) * WARP
    p = cfg.patch
    return (slice(r0 + int(p[0] * WARP), r0 + int(p[1] * WARP)), slice(c0 + int(p[2] * WARP), c0 + int(p[3] * WARP)))


# --------------------------------------------------------------------------- #
# marker detection
# --------------------------------------------------------------------------- #
def detect_markers_colour(rgb, cfg: VisionConfig):
    """Four colour blobs; ordered near-a, near-h, far-a, far-h assuming the camera looks
    along the ranks from the rank-1 side with file a on the left."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(cfg.colour_hsv_lo), np.array(cfg.colour_hsv_hi))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        m = cv2.moments(c)
        if m["m00"] > cfg.min_marker_area:
            blobs.append((m["m10"] / m["m00"], m["m01"] / m["m00"], m["m00"]))
    blobs = sorted(blobs, key=lambda b: -b[2])[:4]
    if len(blobs) < 4:
        raise VisionError(f"only {len(blobs)} corner markers found")
    blobs = sorted(blobs, key=lambda b: b[1])          # by row: far pair first (top of image)
    far = sorted(blobs[:2], key=lambda b: b[0])
    near = sorted(blobs[2:], key=lambda b: b[0])
    return np.array([near[0][:2], near[1][:2], far[0][:2], far[1][:2]], np.float32)


def detect_markers_aruco(rgb, cfg: VisionConfig):
    """Four ArUco markers whose ids give the corner order (no orientation assumption)."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, cfg.aruco_dict))
    params = aruco.DetectorParameters()
    corners, ids, _ = aruco.ArucoDetector(dictionary, params).detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    found = {} if ids is None else {int(i): c.reshape(4, 2).mean(axis=0) for i, c in zip(ids.ravel(), corners)}
    missing = [i for i in cfg.aruco_ids if i not in found]
    if missing:
        raise VisionError(f"ArUco markers {missing} not found (saw {sorted(found)})")
    return np.array([found[i] for i in cfg.aruco_ids], np.float32)


def detect_markers(rgb, cfg: VisionConfig):
    return detect_markers_aruco(rgb, cfg) if cfg.marker_mode == "aruco" else detect_markers_colour(rgb, cfg)


# --------------------------------------------------------------------------- #
# rectification + classification
# --------------------------------------------------------------------------- #
def rectify(rgb, marker_px, marker_offset):
    """(rectified 8*WARP square image, H image<-board)."""
    uv = marker_uv(marker_offset).astype(np.float32)
    H_img_from_board = cv2.getPerspectiveTransform(uv, marker_px)
    H_warp = cv2.getPerspectiveTransform(marker_px, board_to_warp(uv).astype(np.float32))
    return cv2.warpPerspective(rgb, H_warp, (8 * WARP, 8 * WARP)), H_img_from_board


def classify_colour(rect, cfg: VisionConfig):
    hsv = cv2.cvtColor(rect, cv2.COLOR_RGB2HSV)
    occ, scores = {}, {}
    for sq in chess.SQUARES:
        rs, cs = square_patch(sq, cfg)
        patch = hsv[rs, cs]
        s, v = patch[..., 1].astype(int), patch[..., 2].astype(int)
        white = float(np.mean((v > cfg.white_v) & (s < cfg.white_s)))
        black = float(np.mean(v < cfg.black_v))
        occ[sq] = "w" if white > cfg.min_fraction and white >= black else ("b" if black > cfg.min_fraction else None)
        scores[sq] = (max(white, black), white - black)
    return occ, scores


def classify_reference(rect, reference, cfg: VisionConfig):
    """Occupied where the square differs from the empty-board reference; the colour is
    the brightness of the differing pixels against cfg.colour_split (None => every
    occupied square is reported as 'w'; call learn_colour_split first)."""
    g = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    occ, scores = {}, {}
    for sq in chess.SQUARES:
        rs, cs = square_patch(sq, cfg)
        d = np.abs(g[rs, cs] - r[rs, cs])
        score = float(d.mean())
        changed = d > cfg.occupied_diff * 0.5
        bright = float(g[rs, cs][changed].mean()) if changed.any() else float(g[rs, cs].mean())
        if score > cfg.occupied_diff:
            occ[sq] = "w" if (cfg.colour_split is None or bright > cfg.colour_split) else "b"
        else:
            occ[sq] = None
        scores[sq] = (score, bright)
    return occ, scores


def learn_colour_split(scores, board=None):
    """Gray threshold between black and white pieces from the brightness of the occupied
    squares (Otsu's criterion). With `board` given, only its occupied squares are used."""
    vals = np.array(sorted(b for sq, (s, b) in scores.items() if board is None or board.piece_at(sq) is not None))
    if len(vals) < 2:
        raise VisionError("not enough occupied squares to learn the piece colours")
    best, split = -1.0, float(vals.mean())
    for k in range(1, len(vals)):
        lo, hi = vals[:k], vals[k:]
        between = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
        if between > best:
            best, split = between, float((lo[-1] + hi[0]) / 2)
    return split


class BoardReader:
    """Stateful reader: holds the config and the empty-board reference."""

    def __init__(self, cfg: VisionConfig, reference=None):
        self.cfg = cfg
        self.reference = reference      # rectified RGB of the empty board (reference method)

    def set_reference_image(self, rgb):
        """Calibration: `rgb` shows the EMPTY board with its markers."""
        px = detect_markers(rgb, self.cfg)
        self.reference, _ = rectify(rgb, px, marker_offset_for(self.cfg))
        return self.reference

    def read(self, rgb) -> Observation:
        px = detect_markers(rgb, self.cfg)
        rect, H = rectify(rgb, px, marker_offset_for(self.cfg))
        if self.cfg.method == "reference":
            if self.reference is None:
                raise VisionError("no empty-board reference image: run the calibration step")
            occ, scores = classify_reference(rect, self.reference, self.cfg)
        else:
            occ, scores = classify_colour(rect, self.cfg)
        return Observation(occ, rect, H, px, rgb, scores)

    def learn_colours(self, obs: Observation, board: chess.Board):
        """Set colour_split from the starting position and re-classify obs in place."""
        if self.cfg.method != "reference":
            return
        self.cfg.colour_split = learn_colour_split(obs.scores, board)
        obs.occupancy, obs.scores = classify_reference(obs.rectified, self.reference, self.cfg)


def marker_offset_for(cfg: VisionConfig):
    # the board geometry owns the physical offset; the reader only needs it in squares,
    # so it is mirrored onto the vision config by the coordinator (see game.py)
    return getattr(cfg, "_marker_offset", 0.267)


# --------------------------------------------------------------------------- #
# move inference and verification
# --------------------------------------------------------------------------- #
@dataclass
class Inference:
    move: chess.Move | None
    mismatches: int          # squares that disagree with the best legal move
    runner_up: int           # mismatches of the second-best move
    candidates: list         # [(mismatches, move), ...] sorted

    @property
    def confident(self):
        return self.move is not None and self.mismatches == 0 and self.runner_up >= 2

    @property
    def plausible(self):
        return self.move is not None and self.mismatches <= 1 and self.runner_up > self.mismatches


def infer_move(board: chess.Board, occ: dict) -> Inference:
    """The legal move whose resulting occupancy best matches what the camera saw."""
    scored = []
    for mv in board.legal_moves:
        b2 = board.copy(stack=False)
        b2.push(mv)
        o2 = occupancy_of(b2)
        scored.append((sum(1 for sq in chess.SQUARES if o2[sq] != occ[sq]), mv))
    scored.sort(key=lambda t: t[0])
    if not scored:
        return Inference(None, 99, 99, [])
    return Inference(scored[0][1], scored[0][0], scored[1][0] if len(scored) > 1 else 99, scored)


def mismatched_squares(board: chess.Board, occ: dict) -> list:
    """Squares where the camera disagrees with the expected position, as (name, expected, seen)."""
    exp = occupancy_of(board)
    return [(chess.square_name(sq), exp[sq], occ[sq]) for sq in chess.SQUARES if exp[sq] != occ[sq]]


def piece_offset(rect, reference, sq, cfg: VisionConfig):
    """Where the piece on `sq` actually stands, as (du, dv) in squares from the centre,
    from the centroid of the pixels that differ from the empty board. Only the file
    component is trustworthy (tall pieces smear along the view direction)."""
    if reference is None:
        return (0.0, 0.0)
    g = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c0, r0 = chess.square_file(sq) * WARP, (7 - chess.square_rank(sq)) * WARP
    d = np.abs(g[r0:r0 + WARP, c0:c0 + WARP] - r[r0:r0 + WARP, c0:c0 + WARP]) > cfg.occupied_diff * 0.5
    if d.sum() < 10:
        return (0.0, 0.0)
    rows, cols = np.nonzero(d)
    du = (cols.mean() + 0.5) / WARP - 0.5
    dv = 0.5 - (rows.mean() + 0.5) / WARP
    return (float(np.clip(du, -0.3, 0.3)), float(np.clip(dv, -0.3, 0.3)))


# --------------------------------------------------------------------------- #
# overlays for the status display
# --------------------------------------------------------------------------- #
def draw_overlay(obs: Observation, marker_offset):
    """Camera image with the detected grid, markers and classifications (BGR)."""
    img = cv2.cvtColor(obs.image, cv2.COLOR_RGB2BGR)
    H = obs.H_img_from_board
    for i in range(9):
        for seg in ([[i, 0], [i, 8]], [[0, i], [8, i]]):
            p = cv2.perspectiveTransform(np.array([seg], np.float32), H)[0].astype(int)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 255, 255), 1, cv2.LINE_AA)
    for sq, o in obs.occupancy.items():
        if o is None:
            continue
        p = cv2.perspectiveTransform(np.array([[square_uv(sq)]], np.float32), H)[0, 0].astype(int)
        cv2.circle(img, tuple(p), 9, (255, 255, 255) if o == "w" else (40, 40, 40), 2)
    for p in obs.marker_px.astype(int):
        cv2.circle(img, tuple(p), 14, (0, 200, 255), 2)
    return img
