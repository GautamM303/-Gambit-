import chess
import numpy as np
import pytest

from bracket_gambit.config import VisionConfig, occupancy_of
from bracket_gambit.vision import (BoardReader, VisionError, infer_move, mismatched_squares, piece_offset,
                                   learn_colour_split)
from synthetic import render

OFFSET = 0.5


def make_reader(mode):
    cfg = VisionConfig(marker_mode=mode, method="reference")
    cfg._marker_offset = OFFSET
    reader = BoardReader(cfg)
    reader.set_reference_image(render({sq: None for sq in chess.SQUARES}, cfg, OFFSET))
    return reader


@pytest.mark.parametrize("mode", ["aruco", "colour"])
def test_reads_starting_position(mode):
    reader = make_reader(mode)
    board = chess.Board()
    obs = reader.read(render(occupancy_of(board), reader.cfg, OFFSET))
    reader.learn_colours(obs, board)
    assert obs.counts() == (16, 16)
    assert mismatched_squares(board, obs.occupancy) == []


def test_reads_midgame_position_with_jitter():
    reader = make_reader("aruco")
    board = chess.Board()
    reader.learn_colours(reader.read(render(occupancy_of(board), reader.cfg, OFFSET)), board)
    for san in ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]:
        board.push_san(san)
    rng = np.random.default_rng(1)
    jitter = {sq: tuple(rng.uniform(-0.15, 0.15, 2)) for sq in chess.SQUARES}
    obs = reader.read(render(occupancy_of(board), reader.cfg, OFFSET, jitter, seed=3))
    assert mismatched_squares(board, obs.occupancy) == []


def test_missing_markers_raise():
    reader = make_reader("aruco")
    with pytest.raises(VisionError):
        reader.read(np.zeros((480, 640, 3), np.uint8))


def test_infer_move_prefers_exact_match():
    board = chess.Board()
    board.push_san("e4")
    truth = board.copy()
    truth.push_san("e5")
    inf = infer_move(board, occupancy_of(truth))
    assert inf.move == chess.Move.from_uci("e7e5")
    assert inf.mismatches == 0 and inf.runner_up >= 2 and inf.confident


def test_infer_move_flags_illegal_change():
    board = chess.Board()
    board.push_san("e4")
    occ = occupancy_of(board)
    occ[chess.E7], occ[chess.E4] = None, "b"        # black pawn 'captures' e4 from e7: illegal
    inf = infer_move(board, occ)
    assert not inf.plausible


def test_infer_move_no_change_is_not_plausible():
    board = chess.Board()
    inf = infer_move(board, occupancy_of(board))
    assert inf.mismatches >= 2 and not inf.plausible


def test_capture_and_castling_inferred():
    board = chess.Board("r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R b KQkq - 0 1")
    truth = board.copy()
    truth.push_san("O-O")
    inf = infer_move(board, occupancy_of(truth))
    assert inf.move == chess.Move.from_uci("e8g8") and inf.confident
    truth = board.copy()
    truth.push_san("dxe4")
    assert infer_move(board, occupancy_of(truth)).move == chess.Move.from_uci("d5e4")


def test_piece_offset_tracks_file_shift():
    reader = make_reader("aruco")
    board = chess.Board()
    obs = reader.read(render(occupancy_of(board), reader.cfg, OFFSET, jitter={chess.E2: (0.2, 0.0)}))
    du, dv = piece_offset(obs.rectified, reader.reference, chess.E2, reader.cfg)
    assert 0.1 < du < 0.3
    du0, _ = piece_offset(obs.rectified, reader.reference, chess.D2, reader.cfg)
    assert abs(du0) < 0.08


def test_learn_colour_split_separates_two_clusters():
    scores = {sq: (30.0, 230.0 if sq < 16 else 25.0) for sq in range(32)}
    split = learn_colour_split(scores)
    assert 25 < split < 230
