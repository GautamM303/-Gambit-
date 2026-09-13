"""Synthetic camera images of a chessboard for testing the vision pipeline without a robot.

render_board() draws a top-down board (squares, corner markers, pieces as discs) and
warps it with a perspective transform that mimics a head camera looking down at the
board from the rank-1 side, file a on the left.
"""
import chess
import cv2
import numpy as np

from bracket_gambit.config import VisionConfig

PX = 60          # px per square in the top-down drawing
MARGIN = 2       # squares of border around the board (markers live in it)


def top_down(occupancy, cfg: VisionConfig, marker_offset=0.5, jitter=None, light=(190, 160, 110), dark=(110, 70, 40)):
    """Top-down RGB image; a1 bottom-left. occupancy: {square: 'w'|'b'|None}.
    jitter: {square: (du, dv)} piece offsets in squares."""
    size = (8 + 2 * MARGIN) * PX
    img = np.full((size, size, 3), (60, 60, 60), np.uint8)
    o = MARGIN * PX
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x0, y0 = o + f * PX, o + (7 - r) * PX
        cv2.rectangle(img, (x0, y0), (x0 + PX, y0 + PX), light if (f + r) % 2 else dark, -1)
    m = marker_offset * PX
    corners = [(o - m, o + 8 * PX + m), (o + 8 * PX + m, o + 8 * PX + m), (o - m, o - m), (o + 8 * PX + m, o - m)]  # near-a, near-h, far-a, far-h
    if cfg.marker_mode == "aruco":
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))
        for mid, (cx, cy) in zip(cfg.aruco_ids, corners):
            mk = cv2.aruco.generateImageMarker(d, mid, 40)
            mk = cv2.cvtColor(cv2.copyMakeBorder(mk, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2RGB)
            h, w = mk.shape[:2]
            x0, y0 = int(cx - w / 2), int(cy - h / 2)
            img[y0:y0 + h, x0:x0 + w] = mk
    else:
        for cx, cy in corners:
            cv2.circle(img, (int(cx), int(cy)), 10, (30, 60, 255), -1)
    for sq, occ in occupancy.items():
        if occ is None:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        du, dv = (jitter or {}).get(sq, (0.0, 0.0))
        cx, cy = o + (f + 0.5 + du) * PX, o + (7.5 - r - dv) * PX
        col = (238, 232, 220) if occ == "w" else (18, 18, 20)
        cv2.circle(img, (int(cx), int(cy)), int(0.32 * PX), (40, 40, 40), -1)       # shadow / base
        cv2.circle(img, (int(cx), int(cy)), int(0.27 * PX), col, -1)
    return img


def camera_view(top, width=1280, height=960, tilt=0.25, seed=0):
    """Perspective-warp the top-down drawing as seen from the rank-1 side: the near edge
    (rank 1, image bottom) appears wider than the far edge."""
    h, w = top.shape[:2]
    rng = np.random.default_rng(seed)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)            # far-left, far-right, near-right, near-left
    W, H = width, height
    far_w, near_w = 0.55 * W, 0.90 * W
    dst = np.array([[(W - far_w) / 2, 0.12 * H], [(W + far_w) / 2, 0.12 * H],
                    [(W + near_w) / 2, 0.95 * H], [(W - near_w) / 2, 0.95 * H]], np.float32)
    dst += rng.uniform(-8, 8, dst.shape).astype(np.float32) * tilt
    Hm = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(top, Hm, (W, H), borderValue=(70, 70, 70))
    noise = rng.normal(0, 3, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def render(occupancy, cfg: VisionConfig, marker_offset=0.5, jitter=None, seed=0):
    return camera_view(top_down(occupancy, cfg, marker_offset, jitter), cfg.camera_width, cfg.camera_height, seed=seed)
