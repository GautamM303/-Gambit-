import chess
import pytest

from bracket_gambit import engine as eng


def test_fallback_engine_plays_legal_moves():
    e = eng.FallbackEngine(depth=2)
    board = chess.Board()
    for _ in range(6):
        c = e.choose(board)
        assert c.move in board.legal_moves and c.source == "fallback"
        board.push(c.move)


def test_fallback_takes_free_queen():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert eng.FallbackEngine(depth=2).choose(board).move == chess.Move.from_uci("d1d5")


def test_explanations_are_grounded():
    board = chess.Board("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    c = eng.Choice(chess.Move.from_uci("d1d5"), 800, None, [chess.Move.from_uci("d1d5")], 2, "fallback")
    text = eng.explain(board, c)
    assert "wins a queen" in text and "fallback engine" in text and "Stockfish" not in text.split("fallback")[0]
    assert eng.describe_move(board, c.move) == "rook from d1 takes the queen on d5"
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert eng.describe_move(board, chess.Move.from_uci("e1g1")) == "castles kingside"


def test_stockfish_if_available():
    exe = eng.find_stockfish()
    if exe is None:
        pytest.skip("stockfish not installed")
    e = eng.StockfishEngine(exe, skill=5, think_time=0.2)
    try:
        c = e.choose(chess.Board())
        assert c.move in chess.Board().legal_moves and c.source.startswith("Stockfish")
        assert c.score_cp is not None and c.pv
        assert e.analyse(chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")).score(mate_score=10000) > 500
    finally:
        e.close()


def test_open_engine_falls_back(monkeypatch):
    monkeypatch.setattr(eng, "find_stockfish", lambda path=None: None)
    e = eng.open_engine(None, log=lambda *_: None)
    assert isinstance(e, eng.FallbackEngine)
    with pytest.raises(FileNotFoundError):
        eng.open_engine(None, allow_fallback=False, log=lambda *_: None)


def test_project_local_binary_is_found_if_present():
    exe = eng.find_stockfish()
    if exe is None:
        pytest.skip("no stockfish anywhere")
    assert eng.os.path.isfile(exe)
