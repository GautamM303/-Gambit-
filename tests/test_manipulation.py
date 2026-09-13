import chess

from bracket_gambit.config import BoardGeometry, MotionConfig
from bracket_gambit.manipulation import MoveExecutor, plan_transfers
from bracket_gambit.robot import MockRobot, StopFlag, EStop


def geom():
    g = BoardGeometry()
    g.tray = g.default_tray()
    return g


def test_square_geometry():
    g = geom()
    assert g.square_xy(chess.A1) == g.a1
    x, y = g.square_xy(chess.H8)
    assert abs(x - (g.a1[0] + 7 * g.square)) < 1e-9 and abs(y - (g.a1[1] - 7 * g.square)) < 1e-9
    m = g.marker_xy()
    assert m[0][0] < g.a1[0] and m[0][1] > g.a1[1]          # near-a marker is behind and left of a1


def test_plan_quiet_capture_castle_enpassant():
    g = geom()
    b = chess.Board()
    assert [t.label for t in plan_transfers(b, chess.Move.from_uci("e2e4"), g, 0)] == ["P e2 -> e4"]
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
    steps = plan_transfers(b, chess.Move.from_uci("d4e5"), g, 0)
    assert steps[0].label.startswith("captured p e5 -> tray") and steps[0].to_xy == tuple(g.tray[0])
    assert steps[1].label == "P d4 -> e5"
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    steps = plan_transfers(b, chess.Move.from_uci("e1g1"), g, 0)
    assert [t.label for t in steps] == ["K e1 -> g1", "rook h1 -> f1"]
    b = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    steps = plan_transfers(b, chess.Move.from_uci("e5f6"), g, 0)
    assert steps[0].from_xy == g.square_xy(chess.F5)


def test_executor_retries_then_succeeds():
    g = geom()
    robot = MockRobot(pick_results=[False, True])
    ex = MoveExecutor(robot, g, MotionConfig(), refine=lambda sq: (0.004, -0.002), log=lambda *_: None)
    res = ex.execute(chess.Board(), chess.Move.from_uci("e2e4"))
    assert res.ok and res.attempts == 2
    picks = [c for c in robot.log if c[0] == "pick"]
    assert len(picks) == 2 and picks[1][1] != picks[0][1]        # second try at the refined position
    assert ("abort_hold",) in robot.log
    assert robot.log[-1][0] == "place"


def test_executor_gives_up_and_reports():
    g = geom()
    robot = MockRobot(pick_results=[False, False])
    ex = MoveExecutor(robot, g, MotionConfig(), log=lambda *_: None)
    res = ex.execute(chess.Board(), chess.Move.from_uci("g1f3"))
    assert not res.ok and res.failed is not None and "attempt" in res.message
    assert not any(c[0] == "place" for c in robot.log)


def test_capture_uses_tray_slots_in_order():
    g = geom()
    robot = MockRobot()
    ex = MoveExecutor(robot, g, MotionConfig(), log=lambda *_: None)
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
    assert ex.execute(b, chess.Move.from_uci("d4e5")).ok and ex.tray_used == 1
    places = [c for c in robot.log if c[0] == "place"]
    assert places[0][1] == (round(g.tray[0][0], 3), round(g.tray[0][1], 3))


def test_estop_aborts_execution():
    g = geom()
    robot = MockRobot()
    robot.stop.set("test")
    ex = MoveExecutor(robot, g, MotionConfig(), log=lambda *_: None)
    res = ex.execute(chess.Board(), chess.Move.from_uci("e2e4"))
    assert not res.ok and "emergency stop" in res.message
