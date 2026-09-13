"""Regression fixtures: head-camera renders from the MuJoCo scene (7 positions: the start,
captures, a knight move, castling) with the rectified empty-board reference. Pins the
sample-patch placement that survived the smear of tall far-rank pieces (see STATUS.md)."""
import pathlib

import chess
import cv2
import pytest

from bracket_gambit.config import VisionConfig
from bracket_gambit.vision import BoardReader, mismatched_squares, infer_move

DATA = pathlib.Path(__file__).parent / "data"
SIM_MARKER_OFFSET = 0.008 / 0.03


def load():
    fens = (DATA / "sim_positions.txt").read_text().splitlines()
    imgs = [cv2.cvtColor(cv2.imread(str(DATA / f"sim_pos{i}.jpg")), cv2.COLOR_BGR2RGB) for i in range(len(fens))]
    ref = cv2.cvtColor(cv2.imread(str(DATA / "sim_reference.png")), cv2.COLOR_BGR2RGB)
    return fens, imgs, ref


def make_reader(**overrides):
    cfg = VisionConfig(marker_mode="colour", method="reference", **overrides)
    cfg._marker_offset = SIM_MARKER_OFFSET
    fens, imgs, ref = load()
    reader = BoardReader(cfg, reference=ref)
    start = next(i for i, f in enumerate(fens) if f == "START")
    reader.learn_colours(reader.read(imgs[start]), chess.Board())
    return reader, fens, imgs


def test_default_patch_reads_every_position():
    reader, fens, imgs = make_reader()
    for fen, img in zip(fens, imgs):
        board = chess.Board() if fen == "START" else chess.Board(fen)
        obs = reader.read(img)
        assert mismatched_squares(board, obs.occupancy) == [], fen


def test_near_half_patch_is_fooled_by_far_rank_smear():
    """Documents why the patch is where it is: the near half of a square sees the top of
    the piece standing in front of it."""
    reader, fens, imgs = make_reader(patch=(0.42, 0.88, 0.24, 0.76), occupied_diff=22.0)
    errors = 0
    for fen, img in zip(fens, imgs):
        board = chess.Board() if fen == "START" else chess.Board(fen)
        errors += len(mismatched_squares(board, reader.read(img).occupancy))
    assert errors >= 3


def test_moves_between_fixture_positions_are_inferred():
    reader, fens, imgs = make_reader()
    boards = [chess.Board(f) for f in fens if f != "START"]
    for prev, nxt, img in zip(boards[:-1], boards[1:], imgs[1:]):
        # the fixture positions are two plies apart: find the intermediate position, then
        # check that the second ply (the 'human' move) is inferred from the image
        found = None
        for mv in prev.legal_moves:
            mid = prev.copy(stack=False)
            mid.push(mv)
            for m2 in mid.legal_moves:
                after = mid.copy(stack=False)
                after.push(m2)
                if after.board_fen() == nxt.board_fen():
                    found = (mid, m2)
                    break
            if found:
                break
        assert found is not None
        mid, m2 = found
        inf = infer_move(mid, reader.read(img).occupancy)
        assert inf.move == m2 and inf.mismatches == 0, (mid.fen(), m2)
