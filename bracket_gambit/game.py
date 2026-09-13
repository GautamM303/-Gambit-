"""The game loop as an explicit state machine.

    IDLE -> (start) -> OBSERVING (setup check) -> WAITING_FOR_HUMAN
    WAITING_FOR_HUMAN -> (turn complete) -> OBSERVING -> [human move inferred + legal]
        -> THINKING -> ANNOUNCING -> MOVING -> VERIFYING -> WAITING_FOR_HUMAN
    any observation that does not match a legal move / the expected position -> RECOVERING
        (the robot says what it saw and what it expects, then waits for 'turn complete' again)
    a failed grasp/place after retries -> NEEDS_HELP (the robot asks a human to finish the
        step, then re-verifies before continuing)
    pause / resume / reset / stop / e-stop from any state

Invariant: `self.board` (the internal position) is only pushed after the board was
re-observed and matches the expected position - for the human's move and for the
robot's own.
"""
from __future__ import annotations

import enum
import queue
import time
from dataclasses import dataclass, field

import chess

from . import engine as eng
from .config import GambitConfig, occupancy_of
from .manipulation import MoveExecutor
from .robot import EStop, Robot
from .vision import BoardReader, Observation, VisionError, infer_move, mismatched_squares, piece_offset


class Phase(enum.Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    WAITING_FOR_HUMAN = "waiting for the human"
    THINKING = "thinking"
    ANNOUNCING = "announcing"
    MOVING = "moving"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    NEEDS_HELP = "needs help"
    PAUSED = "paused"
    STOPPED = "stopped"
    GAME_OVER = "game over"


LED = {Phase.IDLE: ((0, 0, 40), 0), Phase.OBSERVING: ((255, 255, 255), 0), Phase.WAITING_FOR_HUMAN: ((0, 40, 255), 2000),
       Phase.THINKING: ((255, 160, 0), 600), Phase.ANNOUNCING: ((0, 255, 120), 0), Phase.MOVING: ((255, 0, 200), 0),
       Phase.VERIFYING: ((255, 255, 255), 300), Phase.RECOVERING: ((255, 60, 0), 400), Phase.NEEDS_HELP: ((255, 0, 0), 300),
       Phase.PAUSED: ((60, 60, 60), 0), Phase.STOPPED: ((255, 0, 0), 0), Phase.GAME_OVER: ((0, 255, 0), 1000)}


@dataclass
class Status:
    phase: Phase = Phase.IDLE
    detail: str = ""
    last_human: str = ""
    last_robot: str = ""
    explanation: str = ""
    engine: str = ""
    eval: str = ""
    observation: Observation | None = None
    mismatches: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    events: list = field(default_factory=list)
    helper_interventions: int = 0
    result: str = ""


class Game:
    def __init__(self, robot: Robot, reader: BoardReader, engine: eng.Engine, cfg: GambitConfig,
                 log=print, display=None, sim_human=None):
        self.robot, self.reader, self.engine, self.cfg = robot, reader, engine, cfg
        self.log, self.display = log, display
        self.sim_human = sim_human            # callable(board) -> Move, only in simulation
        self.board = chess.Board()
        self.status = Status(engine=engine.name)
        self.executor = MoveExecutor(robot, cfg.board, cfg.motion, refine=self._refine, log=log)
        self.commands = queue.Queue()
        self.running = True
        self._resume_phase = Phase.IDLE
        self._pending_robot_move = None
        self._last_obs = None
        self._last_choice = None
        self.robot_color = chess.WHITE if cfg.robot_color == "white" else chess.BLACK
        self.move_limit = 0                   # stop the demo after this many robot moves (0 = play on)
        self.auto = False                     # simulation: the scripted opponent signals 'turn complete' itself
        self.auto_help = False                # simulation: a simulated helper answers 'needs help'
        self._manual = False
        self.robot_moves = 0
        self.on_phase = None                  # optional callback(phase) for snapshots
        self.help_rounds = 0                  # consecutive 'needs help' rounds on the same move
        self.max_help_rounds = 3
        reader.cfg._marker_offset = cfg.board.marker_offset

    # ------------------------------------------------------------------ status
    def set_phase(self, phase: Phase, detail=""):
        self.status.phase, self.status.detail = phase, detail
        rgb, period = LED[phase]
        self.robot.led(rgb, period)
        self.log(f"[{phase.value}] {detail}" if detail else f"[{phase.value}]")
        self._refresh()
        if self.on_phase is not None:
            self.on_phase(phase)

    def event(self, text):
        self.status.events.append(f"{time.strftime('%H:%M:%S')} {text}")
        self.log(f"   {text}")
        self._refresh()

    def _refresh(self):
        if self.display is not None:
            self.display.update(self)

    def say(self, text):
        self.event(f'says: "{text}"')
        self.robot.say(text)

    # ------------------------------------------------------------ perception
    def observe(self) -> Observation:
        self.robot.park()
        rgb = self.robot.look()
        obs = self.reader.read(rgb)
        self._last_obs = obs
        self.status.observation = obs
        self._refresh()
        return obs

    def _refine(self, sq):
        """Metres offset of the piece on sq from its square centre, from the last observation."""
        if self._last_obs is None or self.reader.reference is None:
            return (0.0, 0.0)
        du, dv = piece_offset(self._last_obs.rectified, self.reader.reference, sq, self.reader.cfg)
        g = self.cfg.board
        # only the file component (dv is smeared by the perspective of tall pieces)
        return (du * g.square * g.file_dir[0], du * g.square * g.file_dir[1])

    # ---------------------------------------------------------------- commands
    def submit(self, cmd: str):
        """Thread-safe: 'start' 'done' 'pause' 'resume' 'reset' 'stop' 'estop' 'what' 'why' 'help_done'."""
        self.commands.put(cmd)

    def run(self):
        """Main loop: consume commands until stopped."""
        self.set_phase(Phase.IDLE, "say 'start' to begin")
        while self.running:
            try:
                cmd = self.commands.get(timeout=0.2)
            except queue.Empty:
                ph = self.status.phase
                if ph == Phase.WAITING_FOR_HUMAN and (self.auto or self.robot.wait_signal(0.0)):
                    cmd = "done"
                elif ph == Phase.NEEDS_HELP and self.auto_help and hasattr(self.robot, "helper_fix"):
                    expected = self.board.copy(stack=False)
                    if self._pending_robot_move is not None and not getattr(self, "_tray_clear_pending", False):
                        expected.push(self._pending_robot_move)
                    self.event("(simulated helper " + ("clears the tray)" if getattr(self, "_tray_clear_pending", False) else "fixes the board)"))
                    self.robot.helper_fix(expected)
                    cmd = "done"
                else:
                    continue
            try:
                self.handle(cmd)
            except EStop as e:
                self._on_estop(str(e))
            except VisionError as e:
                self.set_phase(Phase.RECOVERING, f"vision: {e}")
                self.say("I cannot see the board markers. Please clear the view and say turn complete again.")
                self.set_phase(Phase.WAITING_FOR_HUMAN if self._pending_robot_move is None else Phase.NEEDS_HELP,
                               "camera view blocked")
            except RuntimeError as e:          # backend trouble outside a move (IK, camera, daemon)
                self.event(f"error: {e}")
                self.set_phase(Phase.NEEDS_HELP, str(e))
                self.say("I ran into a problem. Please check the arm and the board, then say turn complete.")

    def handle(self, cmd: str):
        ph = self.status.phase
        if cmd == "estop":
            self.robot.stop.set("operator e-stop")
            raise EStop("operator e-stop")
        if cmd == "stop":
            self.stop()
        elif cmd == "pause":
            if ph not in (Phase.PAUSED, Phase.STOPPED):
                self._resume_phase = ph
                self.set_phase(Phase.PAUSED, "paused; 'resume' to continue")
                self.say("Paused.")
        elif cmd == "resume":
            if ph == Phase.PAUSED:
                self.set_phase(self._resume_phase, "resumed")
                self.say("Resuming.")
            elif ph == Phase.STOPPED:
                self.robot.stop.clear()
                self.set_phase(Phase.IDLE, "cleared; 'start' for a new game")
        elif cmd == "start":
            self.start()
        elif cmd == "reset":
            self.reset()
        elif cmd == "done":
            if ph in (Phase.WAITING_FOR_HUMAN, Phase.RECOVERING):
                self.human_turn_complete()
            elif ph == Phase.NEEDS_HELP:
                self.help_done()
            else:
                self.event(f"ignored 'turn complete' while {ph.value}")
        elif cmd == "help_done":
            self.help_done()
        elif cmd == "what":
            self.what_did_you_see()
        elif cmd == "why":
            self.why()
        elif cmd.startswith("move:"):
            self.user_move(cmd[5:])
        else:
            self.event(f"unknown command {cmd!r}")

    def user_move(self, uci):
        """A move entered on the board display. In the simulator this IS the human's hand
        (the piece is moved on the table); on the real robot it is only a claim - the human
        still moves the real piece and the camera decides what happened."""
        if self.status.phase not in (Phase.WAITING_FOR_HUMAN, Phase.RECOVERING) or self.board.turn == self.robot_color:
            self.event(f"ignored move {uci}: not the human's turn ({self.status.phase.value})")
            return
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            self.event(f"bad move text {uci!r}")
            return
        if mv not in self.board.legal_moves:
            promo = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
            if promo in self.board.legal_moves:
                mv = promo
            else:
                self.say(f"{uci} is not a legal move.")
                return
        if hasattr(self.robot, "human_plays"):
            self.event(f"you move {self.board.san(mv)} on the simulated board")
            self.robot.human_plays(self.board, mv)
            self._manual = True
        else:
            self.event(f"you say you played {self.board.san(mv)}; checking the board")
        self.human_turn_complete()

    # ------------------------------------------------------------------ flow
    def start(self):
        if self.robot.stop.is_set():
            self.event("stop flag is set; 'resume' to clear it first")
            return
        self.board = chess.Board()
        self.status = Status(engine=self.engine.name)
        self.executor.tray_used = 0
        self.set_phase(Phase.OBSERVING, "checking the starting position")
        self.say("Let's play. Checking the board.")
        obs = self.observe()
        if self.reader.cfg.method == "reference":
            try:
                self.reader.learn_colours(obs, self.board)
            except VisionError as e:
                self.event(str(e))
        w, b = obs.counts()
        bad = mismatched_squares(self.board, obs.occupancy)
        self.event(f"initial read: {w} white, {b} black; {len(bad)} squares differ from the starting position")
        if bad:
            self.set_phase(Phase.RECOVERING, f"setup differs on {', '.join(n for n, _, _ in bad[:6])}")
            self.say("The board does not look like the starting position. " + self._describe_mismatch(bad)
                     + " Please fix it and say start again.")
            self.status.mismatches = bad
            self.status.phase = Phase.IDLE
            self._refresh()
            return
        self.robot.ready()
        self.robot.gesture("wave")
        if self.robot_color == chess.WHITE:
            self.say("You are black; I will open.")
            self.robot_turn()
        else:
            self.say("You are white. Make your move, then tell me your turn is complete.")
            self.set_phase(Phase.WAITING_FOR_HUMAN, "make your move, then say 'turn complete'")

    def human_turn_complete(self):
        if self.board.turn == self.robot_color:
            self.event("it is not the human's turn")
            return
        self.set_phase(Phase.OBSERVING, "reading the board")
        if self.sim_human is not None and not getattr(self, "_manual", False):
            self.sim_human(self.board)          # the simulated hand moves a piece now
        self._manual = False
        obs = self.observe()
        inf = infer_move(self.board, obs.occupancy)
        self.status.mismatches = []
        if inf.move is None:
            self._recover("I do not have any legal moves.", [])
            return
        if not inf.plausible:
            bad = mismatched_squares(self.board, obs.occupancy)
            expected_after = self.board.copy(stack=False)
            expected_after.push(inf.move)
            bad_after = mismatched_squares(expected_after, obs.occupancy)
            detail = (f"best match {self.board.san(inf.move)} with {inf.mismatches} mismatches "
                      f"(runner-up {inf.runner_up})")
            self.event(f"observation does not correspond to a legal move: {detail}")
            self.status.mismatches = bad_after
            if not bad:
                self._recover("I do not see any change on the board. Make your move and tell me again.", [])
            else:
                self._recover(f"That does not look like a legal move. The closest legal move is "
                              f"{self.board.san(inf.move)}, but " + self._describe_mismatch(bad_after)
                              + " Please correct the board and say turn complete again.", bad_after)
            return
        san = self.board.san(inf.move)
        self.event(f"human move inferred: {san} ({inf.mismatches} mismatches, runner-up {inf.runner_up})")
        self.board.push(inf.move)
        self.status.last_human, self.status.mismatches = san, []
        self.status.moves.append(san)
        self.say(f"You played {san}." + (" Check!" if self.board.is_check() else ""))
        if self._check_game_over():
            return
        self.robot_turn()

    def robot_turn(self):
        self.set_phase(Phase.THINKING, f"{self.engine.name} is thinking")
        self.robot.gesture("think")
        choice = self.engine.choose(self.board)
        # promotions cannot be executed physically: pick the best non-promotion instead
        if choice.move.promotion and any(not m.promotion for m in self.board.legal_moves):
            self.event("engine chose a promotion (not executable); choosing the best other move")
            choice = self._choose_excluding(choice)
        self._last_choice = choice
        san = self.board.san(choice.move)
        self.status.last_robot = san
        self.status.eval = (f"mate in {abs(choice.mate_in)}" if choice.mate_in is not None
                            else f"{choice.score_cp / 100:+.2f}" if choice.score_cp is not None else "?")
        self.status.explanation = eng.explain(self.board, choice)
        self.event(f"robot chooses {san} [{choice.source}] eval {self.status.eval} depth {choice.depth}")
        self.set_phase(Phase.ANNOUNCING, san)
        self.say(f"I play {san}: {eng.describe_move(self.board, choice.move)}.")
        self._pending_robot_move = choice.move
        self._execute_pending()

    def _execute_pending(self):
        """Physically execute the pending robot move, then verify it."""
        mv = self._pending_robot_move
        san = self.board.san(mv)
        self.set_phase(Phase.MOVING, san)
        t0 = time.time()
        if hasattr(self.robot, "sim_time"):
            self.robot.sim_time()             # reset the mark
        self.robot.ready()
        res = self.executor.execute(self.board, mv, status=lambda s: self.set_phase(Phase.MOVING, s))
        sim_t = self.robot.sim_time() if hasattr(self.robot, "sim_time") else None
        self.event(f"move executed in {time.time() - t0:.0f}s wall" + (f", {sim_t:.0f}s robot time" if sim_t else ""))
        if not res.ok:
            self.event(res.message)
            if self.robot.stop.is_set():
                raise EStop(res.message)
            if res.tray_full:
                self._tray_clear_pending = True
                self.set_phase(Phase.NEEDS_HELP, "capture tray full")
                self.robot.park()
                self.say("My capture tray is full. Please take the captured pieces off it, then say turn complete.")
                return
            self.help_rounds += 1
            self.set_phase(Phase.NEEDS_HELP, res.message)
            self.robot.abort_hold()
            self.robot.park()
            self.say(f"I need a hand. Please finish my move {san} by hand"
                     + (f": {res.failed.label}" if res.failed else "") + ", then say turn complete.")
            return
        self.verify_robot_move()

    def verify_robot_move(self):
        mv = self._pending_robot_move
        self.set_phase(Phase.VERIFYING, "checking the board after my move")
        expected = self.board.copy(stack=False)
        expected.push(mv)
        obs = self.observe()
        bad = mismatched_squares(expected, obs.occupancy)
        self.status.mismatches = bad
        if bad:
            self.help_rounds += 1
            self.event(f"verification failed on {[n for n, _, _ in bad]} (round {self.help_rounds}/{self.max_help_rounds})")
            if self.help_rounds >= self.max_help_rounds:
                # The board cannot be made to match: stop cleanly instead of asking forever.
                # (Either a piece is physically where the camera says, or the camera is wrong;
                # both need a person to look - see the snapshot in the frames folder.)
                self.set_phase(Phase.STOPPED, f"board still wrong after {self.help_rounds} attempts: {self._describe_mismatch(bad)}")
                self.say("I cannot get the board to match what I expect. " + self._describe_mismatch(bad)
                         + " Stopping the game. Please check the camera view and reset.")
                self._rest()
                self.running = False
                return
            self.set_phase(Phase.NEEDS_HELP, "board does not match after my move")
            self.say("Something moved. " + self._describe_mismatch(bad) + " Please fix it and say turn complete.")
            return
        self.help_rounds = 0
        san = self.board.san(mv)
        self.board.push(mv)
        self._pending_robot_move = None
        self.status.moves.append(san)
        self.event(f"robot move {san} verified on the board")
        if self.board.is_check() and not self.board.is_checkmate():
            self.say("Check.")
            self.robot.gesture("check")
        if self._check_game_over():
            return
        self.robot_moves += 1
        if self.move_limit and self.robot_moves >= self.move_limit:
            self.set_phase(Phase.GAME_OVER, f"demo complete after {self.robot_moves} robot moves")
            self.status.result = "demo complete"
            self.say("That is the end of the demo. Thank you for playing!")
            self.robot.gesture("wave")
            self._rest()
            self.running = False
            return
        self.robot.ready()
        self.set_phase(Phase.WAITING_FOR_HUMAN, "your move")

    def help_done(self):
        """The human finished a step by hand: re-verify the pending robot move (or, after
        clearing the tray, execute it)."""
        if getattr(self, "_tray_clear_pending", False):
            self._tray_clear_pending = False
            self.executor.tray_used = 0
            self.status.helper_interventions += 1
            self.say("Thank you.")
            self._execute_pending()
        elif self._pending_robot_move is not None:
            self.status.helper_interventions += 1
            self.verify_robot_move()
        else:
            self.set_phase(Phase.WAITING_FOR_HUMAN, "your move")

    def _recover(self, text, bad):
        self.set_phase(Phase.RECOVERING, "waiting for the board to be corrected")
        self.say(text)
        self.set_phase(Phase.WAITING_FOR_HUMAN, "correct the board, then say 'turn complete'")

    def _describe_mismatch(self, bad):
        parts = []
        for name, exp, seen in bad[:3]:
            what = {None: "empty", "w": "a white piece", "b": "a black piece"}
            parts.append(f"{name} should be {what[exp]} but I see {what[seen]}")
        return "; ".join(parts) + "."

    def _choose_excluding(self, bad_choice):
        alt = [m for m in self.board.legal_moves if not m.promotion]
        best = None
        for m in alt:
            b2 = self.board.copy(stack=False)
            b2.push(m)
            s = -self.engine.analyse(b2).score(mate_score=100000) if hasattr(self.engine, "analyse") else 0
            if best is None or s > best[0]:
                best = (s, m)
        return eng.Choice(best[1], best[0], None, [best[1]], None, bad_choice.source)

    def _check_game_over(self):
        if not self.board.is_game_over():
            return False
        out = self.board.outcome()
        if out.winner is None:
            text, self.status.result = "It's a draw. Good game!", "draw"
        elif out.winner == self.robot_color:
            text, self.status.result = "Checkmate! I win. Thank you for the game.", "robot wins"
        else:
            text, self.status.result = "Checkmate. You win, well played!", "human wins"
        self.set_phase(Phase.GAME_OVER, self.status.result)
        self.say(text)
        self.robot.gesture("celebrate" if out.winner == self.robot_color else "bow")
        self._rest()
        return True

    def _rest(self):
        """Arm all the way home (end of the game / stop); park() is the quick out-of-view pose."""
        getattr(self.robot, "rest", self.robot.park)()

    def what_did_you_see(self):
        if self._last_obs is None:
            self.say("I have not looked at the board yet.")
            return
        w, b = self._last_obs.counts()
        text = f"I saw {w} white and {b} black pieces."
        if self.status.last_human:
            text += f" Your last move was {self.status.last_human}."
        if self.status.mismatches:
            text += " " + self._describe_mismatch(self.status.mismatches)
        self.say(text)

    def why(self):
        if self._last_choice is None:
            self.say("I have not moved yet.")
        else:
            self.say(self.status.explanation)

    def reset(self):
        self.robot.stop.clear()
        self._rest()
        self.board = chess.Board()
        self.executor.tray_used = 0
        self._pending_robot_move = None
        self.status = Status(engine=self.engine.name)
        self.set_phase(Phase.IDLE, "set the pieces up, then 'start'")
        self.say("Please set the board up in the starting position, then say start.")

    def stop(self):
        self.set_phase(Phase.STOPPED, "stopped by the operator")
        try:
            if self.robot.in_hand():
                self.robot.abort_hold()
            self._rest()
        except (EStop, RuntimeError) as e:
            self.event(f"while parking: {e}")
        self.say("Stopping.")
        self.running = False

    def _on_estop(self, reason):
        self.robot.estop()
        self.set_phase(Phase.STOPPED, f"EMERGENCY STOP: {reason}")
        self.say("Emergency stop.")
        # stay alive so the operator can 'resume' (clears the flag) or 'stop'
