"""End-to-end loop with a mock robot whose 'physical' board is rendered synthetically:
the human moves pieces by editing that board; the robot's pick/place edits it too."""
import chess

from bracket_gambit.config import GambitConfig, occupancy_of
from bracket_gambit.engine import FallbackEngine
from bracket_gambit.game import Game, Phase
from bracket_gambit.robot import MockRobot
from bracket_gambit.vision import BoardReader
from synthetic import render


class PhysicalMock(MockRobot):
    """Pieces live on a python-chess board; xy <-> square through the geometry."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.world = chess.Board()
        self.tray = []
        self.fail_next_pick = 0

    def _sq(self, xy):
        g = self.cfg.board
        for sq in chess.SQUARES:
            x, y = g.square_xy(sq)
            if abs(x - xy[0]) < g.square / 2 and abs(y - xy[1]) < g.square / 2:
                return sq
        return None

    def look(self):
        return render(occupancy_of(self.world), self.cfg.vision, self.cfg.board.marker_offset)

    def pick(self, xy, kind):
        self.log.append(("pick", self._key(xy), kind))
        if self.fail_next_pick:
            self.fail_next_pick -= 1
            return False
        sq = self._sq(xy)
        if sq is None or self.world.piece_at(sq) is None:
            return False
        self.holding = self.world.remove_piece_at(sq)
        return True

    def place(self, xy, surface_z):
        self.log.append(("place", self._key(xy), round(surface_z, 3)))
        sq = self._sq(xy)
        if sq is None:
            self.tray.append(self.holding)
        else:
            self.world.set_piece_at(sq, self.holding)
        self.holding = None
        return True

    def human_move(self, san):
        mv = self.world.parse_san(san)                # pieces are edited directly, no move stack
        piece = self.world.remove_piece_at(mv.from_square)
        self.world.set_piece_at(mv.to_square, piece)
        self.world.turn = not self.world.turn


def make_game(seed=0):
    cfg = GambitConfig()
    cfg.board.marker_offset = 0.5
    cfg.board.tray = cfg.board.default_tray()
    robot = PhysicalMock(cfg)
    reader = BoardReader(cfg.vision)
    reader.cfg._marker_offset = cfg.board.marker_offset
    reader.set_reference_image(render({sq: None for sq in chess.SQUARES}, cfg.vision, cfg.board.marker_offset))
    game = Game(robot, reader, FallbackEngine(depth=2, seed=seed), cfg, log=lambda *_: None)
    return game, robot


def test_full_turns_with_mock_robot():
    game, robot = make_game()
    game.handle("start")
    assert game.status.phase == Phase.WAITING_FOR_HUMAN and game.board.turn == chess.BLACK
    assert len(game.status.moves) == 1                    # the robot (white) opened
    for san in ["e6", "d5"]:
        # the robot has moved; the human replies on the physical board and signals
        legal = [game.board.san(m) for m in game.board.legal_moves]
        san = san if san in legal else legal[0]
        robot.world.turn = chess.BLACK
        robot.human_move(san)
        game.handle("done")
        assert game.status.phase == Phase.WAITING_FOR_HUMAN, game.status.detail
        assert game.status.last_human == san
        # internal position == physical board after both moves
        assert occupancy_of(game.board) == occupancy_of(robot.world)
    assert len(game.status.moves) == 5


def test_illegal_human_move_is_rejected_and_recovered():
    game, robot = make_game()
    game.handle("start")
    # human 'moves' a pawn two squares sideways: not legal
    p = robot.world.remove_piece_at(chess.E7)
    robot.world.set_piece_at(chess.C6, p)
    game.handle("done")
    assert game.status.phase == Phase.WAITING_FOR_HUMAN and game.board.turn == chess.BLACK
    assert any("does not correspond to a legal move" in e for e in game.status.events)
    # fix it: put the pawn back and make a legal move
    robot.world.remove_piece_at(chess.C6)
    robot.world.set_piece_at(chess.E7, p)
    robot.world.turn = chess.BLACK
    robot.human_move("e5")
    game.handle("done")
    assert game.status.last_human == "e5" and game.status.phase == Phase.WAITING_FOR_HUMAN


def test_no_change_is_reported():
    game, robot = make_game()
    game.handle("start")
    game.handle("done")
    assert game.status.phase == Phase.WAITING_FOR_HUMAN
    assert any("do not see any change" in e for e in game.status.events)


def test_failed_grasp_asks_for_help_then_verifies():
    game, robot = make_game()
    game.handle("start")
    robot.fail_next_pick = 5
    robot.world.turn = chess.BLACK
    robot.human_move("e5")
    game.handle("done")
    assert game.status.phase == Phase.NEEDS_HELP
    assert game.board.turn == chess.WHITE            # internal position NOT advanced
    # the helper performs the robot's move by hand, then signals
    mv = game._pending_robot_move
    robot.fail_next_pick = 0
    piece = robot.world.remove_piece_at(mv.from_square)
    robot.world.set_piece_at(mv.to_square, piece)
    game.handle("done")
    assert game.status.phase == Phase.WAITING_FOR_HUMAN and game.board.turn == chess.BLACK
    assert game.status.helper_interventions == 1


def test_pause_resume_stop_estop():
    game, robot = make_game()
    game.handle("start")
    game.handle("pause")
    assert game.status.phase == Phase.PAUSED
    game.handle("done")                                   # ignored while paused
    assert game.status.phase == Phase.PAUSED
    game.handle("resume")
    assert game.status.phase == Phase.WAITING_FOR_HUMAN
    try:
        game.handle("estop")
    except Exception as e:
        game._on_estop(str(e))
    assert game.status.phase == Phase.STOPPED and ("estop",) in robot.log
    game.handle("resume")
    assert game.status.phase == Phase.IDLE and not robot.stop.is_set()
    game.handle("stop")
    assert not game.running


def test_bad_setup_is_reported():
    game, robot = make_game()
    robot.world.remove_piece_at(chess.B1)
    game.handle("start")
    assert game.status.phase == Phase.IDLE
    assert any("b1" in e for e in game.status.events) or game.status.mismatches
