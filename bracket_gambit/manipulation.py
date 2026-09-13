"""Chess move -> physical pick-and-place, with verification and retries.

A move becomes a list of transfers executed in a safe order:
  capture      the captured piece goes to the tray first, then the mover moves
  en passant   the captured pawn (not on the destination square) goes to the tray
  castling     king first, then the rook
  promotion    NOT supported (the piece would have to be swapped) -> the engine is
               told not to play it; a human promotion is handled by asking for help

Each transfer: pick (backend verifies something is in the gripper) -> place. A failed
pick is retried once at the position the camera saw the piece at (if a refinement
callback is available); a failure after that stops the move and asks the human for
help. The chess position is NOT updated here - the game does that only after the
board has been re-observed and matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .config import BoardGeometry, MotionConfig
from .robot import EStop, Robot


@dataclass
class Transfer:
    from_xy: tuple
    to_xy: tuple
    kind: str                 # piece symbol lower-case
    to_surface: float
    label: str
    from_sq: int | None = None


@dataclass
class ExecResult:
    ok: bool
    message: str = ""
    completed: list = field(default_factory=list)   # labels of transfers that succeeded
    failed: Transfer | None = None
    attempts: int = 0
    tray_full: bool = False                          # nothing was moved: the tray must be cleared first


def plan_transfers(board: chess.Board, move: chess.Move, geom: BoardGeometry, tray_next: int) -> list:
    """Ordered physical steps for `move` in `board`. tray_next = index of the next free tray slot."""
    steps = []
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError(f"no piece on {chess.square_name(move.from_square)}")
    if move.promotion:
        raise ValueError("promotion is not supported by the manipulator")
    captured_sq = None
    if board.is_en_passant(move):
        captured_sq = move.to_square + (-8 if board.turn == chess.WHITE else 8)
    elif board.is_capture(move):
        captured_sq = move.to_square
    if captured_sq is not None:
        victim = board.piece_at(captured_sq)
        if tray_next >= len(geom.tray):
            raise ValueError("the capture tray is full")
        steps.append(Transfer(geom.square_xy(captured_sq), tuple(geom.tray[tray_next]), victim.symbol().lower(),
                              geom.table_z, f"captured {victim.symbol()} {chess.square_name(captured_sq)} -> tray", captured_sq))
    steps.append(Transfer(geom.square_xy(move.from_square), geom.square_xy(move.to_square), piece.symbol().lower(),
                          geom.board_z, f"{piece.symbol()} {chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}",
                          move.from_square))
    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) == 6
        rank = chess.square_rank(move.from_square)
        rf, rt = (chess.square(7, rank), chess.square(5, rank)) if kingside else (chess.square(0, rank), chess.square(3, rank))
        steps.append(Transfer(geom.square_xy(rf), geom.square_xy(rt), "r", geom.board_z,
                              f"rook {chess.square_name(rf)} -> {chess.square_name(rt)}", rf))
    return steps


class MoveExecutor:
    def __init__(self, robot: Robot, geom: BoardGeometry, motion: MotionConfig, refine=None, log=print):
        """refine(square) -> (dx, dy) metres: where the camera last saw the piece relative to
        the square centre (optional; used for the retry)."""
        self.robot, self.geom, self.motion, self.refine, self.log = robot, geom, motion, refine, log
        self.tray_used = 0

    def execute(self, board: chess.Board, move: chess.Move, status=lambda s: None) -> ExecResult:
        try:
            steps = plan_transfers(board, move, self.geom, self.tray_used)
        except ValueError as e:
            return ExecResult(False, str(e), tray_full="tray" in str(e))
        result = ExecResult(True)
        try:
            for step in steps:
                status(step.label)
                ok, attempts = self.transfer(step)
                result.attempts += attempts
                if not ok:
                    result.ok, result.failed = False, step
                    result.message = f"could not complete '{step.label}' after {attempts} attempt(s)"
                    return result
                result.completed.append(step.label)
                if step.from_sq is not None and step.to_xy in [tuple(t) for t in self.geom.tray]:
                    self.tray_used += 1
        except EStop as e:
            result.ok, result.message = False, f"emergency stop: {e}"
        return result

    def transfer(self, step: Transfer):
        """Pick at the nominal centre; on a miss, back off and retry at the camera-refined
        position. Returns (ok, attempts)."""
        xy = step.from_xy
        for attempt in range(1, self.motion.max_attempts + 1):
            picked = self.robot.pick(xy, step.kind)
            if picked and self.robot.in_hand():
                break
            self.log(f"      grasp attempt {attempt} on {step.label} failed")
            self.robot.abort_hold()
            if attempt == self.motion.max_attempts:
                return False, attempt
            if self.refine is not None and step.from_sq is not None:
                dx, dy = self.refine(step.from_sq)
                xy = (step.from_xy[0] + dx, step.from_xy[1] + dy)
                self.log(f"      retrying at the observed position ({dx * 1000:+.0f}, {dy * 1000:+.0f}) mm")
        else:
            return False, self.motion.max_attempts
        placed = self.robot.place(step.to_xy, step.to_surface)
        if not placed:
            self.log(f"      place failed on {step.label}")
            self.robot.abort_hold()
            return False, attempt
        return True, attempt
