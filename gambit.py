# /// script
# requires-python = ">=3.10"
# dependencies = ["bbos", "numpy<2", "opencv-python", "chess", "yourdfpy"]
# [tool.uv.sources]
# bbos = { path = "/home/bracketbot/bbos", editable = true }
# ///
"""Bracket Gambit - BracketBot plays chess.

    laptop, simulation:   uv run python gambit.py play --sim [--view] [--record chess.mp4] [--moves 4]
    robot, dry run:       uv run gambit.py play --robot                (no motion, prints waypoints)
    robot, for real:      uv run gambit.py play --robot --execute
    calibration:          uv run gambit.py calibrate --robot [--execute]
    reach check:          uv run gambit.py check --robot

Controls: the terminal (Enter = my turn is complete, e = E-STOP, ...), the web page on
port 8010 (phone-friendly buttons + spectator display), and the wake word (= turn complete).
"""
import argparse
import json
import os
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chess  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from bracket_gambit import engine as eng  # noqa: E402
from bracket_gambit.config import GambitConfig, occupancy_of  # noqa: E402
from bracket_gambit.game import Game, Phase  # noqa: E402
from bracket_gambit.robot import EStop  # noqa: E402
from bracket_gambit.ui import Console, Display, HttpUI, Window, speak_local  # noqa: E402
from bracket_gambit.vision import BoardReader, detect_markers, rectify  # noqa: E402

DEFAULT_CALIB = HERE / "gambit_calibration.json"
OPENINGS = {   # the simulated human's replies (SAN); anything illegal falls back to the engine
    "sicilian": ["c5", "d6", "Nf6", "a6", "e5", "Be7"],
    "french": ["e6", "d5", "Nf6", "Be7", "O-O", "c5"],
    "kings-pawn": ["e5", "Nc6", "Nf6", "Bc5", "d6", "O-O"],
    "gambit": ["d5", "e5", "Nc6", "Nf6", "Bd6", "O-O"],     # offers pawns early: shows captures to the tray
    "random": [],
}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- #
def build_sim(args, cfg):
    from bracket_gambit.sim_robot import SimRobot
    display = Display()
    recorder = None
    if args.record:
        from bracket_gambit.sim_recorder import SimRecorder
        recorder = SimRecorder(display, args.record)
    # The dashboard needs scene/hand-camera renders, which run inside the physics loop:
    # full size at 30 Hz only when recording; small and 3-4 Hz when live (with --view the
    # loop is paced to real time and every render would otherwise stall the 3-D window).
    live = not args.record
    robot = SimRobot(cfg, view=args.view, recorder=recorder, log=log, display=display,
                     view_period=(0.33 if args.view else 0.25) if live else 1 / 30,
                     scene_size=(960, 540) if live else (1280, 720),
                     hand_size=(240, 180) if live else (320, 240))
    if recorder is not None:
        recorder.attach(robot)
    if args.voice:
        robot.say = lambda text: (log(f"[say] {text}"), speak_local(text))
    return robot, display, recorder


class SimHuman:
    """Moves the black pieces in the simulator: scripted opening, then a shallow engine."""

    def __init__(self, robot, opening, seed=0):
        self.robot, self.script = robot, list(OPENINGS.get(opening, []))
        self.fallback = eng.FallbackEngine(depth=1, seed=seed)
        self.rng = random.Random(seed)
        self.last = None            # (fen before the move, move, retries) of a move not yet accepted

    def __call__(self, board):
        # Like a real player: if the previous move was not accepted, first insist ("look
        # again") twice, then take it back and play something else.
        if self.last is not None and self.last[0] == board.fen():
            fen, prev, retries = self.last
            if retries < 2:
                self.last = (fen, prev, retries + 1)
                log("         (simulated human: 'look again')")
                return
            log(f"         (simulated human takes back {board.san(prev)})")
            self.robot.human_takes_back(board, prev)
        mv = None
        while self.script and mv is None:
            san = self.script.pop(0)
            try:
                mv = board.parse_san(san)
            except ValueError:
                mv = None
        if mv is None:
            scored = []
            for m in board.legal_moves:
                board.push(m)
                scored.append((-self.fallback.search(board, 1)[0], m))
                board.pop()
            scored.sort(key=lambda t: -t[0])
            mv = self.rng.choice(scored[:3])[1]
        log(f"         (simulated human plays {board.san(mv)})")
        self.last = (board.fen(), mv, 0)
        self.robot.human_plays(board, mv)


def cmd_play(args):
    cfg = GambitConfig.load(args.calib)
    if args.stockfish:
        cfg.stockfish = args.stockfish
    if args.skill is not None:
        cfg.skill = args.skill
    engine = eng.open_engine(cfg.stockfish, skill=cfg.skill, think_time=cfg.think_time, log=log)
    recorder = None
    if args.sim:
        robot, display, recorder = build_sim(args, cfg)
    else:
        from bracket_gambit.bbos_robot import BBOSRobot
        robot = BBOSRobot(cfg, execute=args.execute, log=log)
        display = Display()
    reader = BoardReader(cfg.vision)
    reader.cfg._marker_offset = cfg.board.marker_offset
    game = Game(robot, reader, engine, cfg, log=log, display=display)
    game.move_limit = args.moves
    http = None
    if args.http:
        http = HttpUI(game, display, port=args.http)
        http.start()
    if args.window:
        Window(display, game).start()
    Console(game).start()
    display.update(game)
    try:
        if args.sim:
            robot.use_camera("wide")           # the roll-up from afar...
            robot.arrive()
            robot.use_camera("table")          # ...then the close view for the game
            if cfg.vision.method == "reference":
                reader.set_reference_image(robot.reference_image())
            game.auto_help = True
            if args.opponent == "auto":
                game.sim_human = SimHuman(robot, args.opening, args.seed)
                game.auto = True
            else:
                if not args.window and not args.http:
                    log("note: --opponent me needs --window (or --http) to enter moves; opening the window")
                    Window(display, game).start()
                log("you play black: click a piece and then its destination on the board; f = full screen")
            if recorder is not None:
                frames_dir = pathlib.Path(args.record).with_suffix("") .parent / "gambit_frames"
                frames_dir.mkdir(exist_ok=True)
                counter = [0]
                timeline = []

                def snap(phase):
                    timeline.append({"frame": recorder.frames, "t": recorder.frames / recorder.fps,
                                     "phase": phase.name, "detail": game.status.detail})
                    pathlib.Path(args.record).with_suffix(".timeline.json").write_text(json.dumps(timeline, indent=1))
                    if phase in (Phase.WAITING_FOR_HUMAN, Phase.OBSERVING, Phase.NEEDS_HELP, Phase.GAME_OVER, Phase.RECOVERING):
                        counter[0] += 1
                        recorder.snapshot(robot.data, frames_dir / f"{counter[0]:02d}_{phase.name.lower()}.png")
                game.on_phase = snap
            game.submit("start")
        else:
            ref = pathlib.Path(args.calib).with_suffix(".reference.png")
            if cfg.vision.method == "reference":
                if not ref.exists():
                    sys.exit(f"no empty-board reference image {ref}: run `gambit.py calibrate --robot` first")
                reader.reference = cv2.cvtColor(cv2.imread(str(ref)), cv2.COLOR_BGR2RGB)
            log("press n (or the web 'Start game' button) when the pieces are set up")
        game.run()
    except KeyboardInterrupt:
        robot.stop.set("ctrl-c")
        try:
            robot.estop()
        except Exception:
            pass
        log("interrupted: torque off")
    finally:
        if hasattr(robot, "report"):
            log(robot.report())
        log("moves: " + " ".join(game.status.moves) + (f"   result: {game.status.result}" if game.status.result else ""))
        if recorder is not None:
            recorder.close()
        if http is not None:
            http.stop()
        engine.close()
        robot.close()


# --------------------------------------------------------------------------- #
def cmd_check(args):
    """Dry-run reachability: solve IK for every square and tray slot at the approach and grasp heights."""
    cfg = GambitConfig.load(args.calib)
    from bracket_gambit.bbos_robot import BBOSRobot, PAD_AHEAD
    robot = BBOSRobot(cfg, execute=False, log=lambda *a: None)
    bad = []
    targets = [(chess.square_name(sq), cfg.board.square_xy(sq)) for sq in chess.SQUARES] + \
              [(f"tray{i}", xy) for i, xy in enumerate(cfg.board.tray)]
    for name, (x, y) in targets:
        for z in (cfg.board.board_z + 0.03, cfg.board.board_z + 0.03 + cfg.motion.approach):
            try:
                robot._solve([x - PAD_AHEAD, y, z])
            except Exception as e:
                bad.append(f"{name} z={z:.3f}: {e}")
    print(f"{len(targets) * 2 - len(bad)}/{len(targets) * 2} targets reachable")
    for b in bad:
        print("  UNREACHABLE", b)
    robot.close()


def cmd_calibrate(args):
    """(1) empty-board reference image, (2) teach a1 / h1 / a8 by jogging the tool, (3) gripper angles."""
    cfg = GambitConfig.load(args.calib)
    from bracket_gambit.bbos_robot import BBOSRobot, PAD_AHEAD, PAD_BELOW_TOOL
    robot = BBOSRobot(cfg, execute=args.execute, log=log)
    reader = BoardReader(cfg.vision)
    reader.cfg._marker_offset = cfg.board.marker_offset
    try:
        if args.step in ("all", "camera"):
            input("Step 1: EMPTY board with its four markers in view, arm parked. Enter to capture... ")
            robot.park()
            rgb = robot.look()
            px = detect_markers(rgb, cfg.vision)
            rect, _ = rectify(rgb, px, cfg.board.marker_offset)
            ref = pathlib.Path(args.calib).with_suffix(".reference.png")
            cv2.imwrite(str(ref), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(ref.with_suffix(".camera.png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            log(f"markers at {np.round(px).astype(int).tolist()}; reference saved to {ref}")
        if args.step in ("all", "board"):
            log("Step 2: jog the OPEN gripper so the pads straddle the square centre with their lower edge on the board.")
            log("keys: w/s = +x/-x  a/d = +y/-y  q/e = +z/-z  (5 mm; capital = 20 mm)  Enter = accept  0 = abort")
            robot.ready()
            robot.gripper(cfg.motion.gripper_open, settle=0.2)
            taught = {}
            for name in ("a1", "h1", "a8"):
                log(f"--- move to {name} ---")
                while True:
                    k = input(f"{name} tool at {np.round(robot.tool_pos(), 3)} > ").strip()
                    if k == "":
                        taught[name] = robot.tool_pos() + [PAD_AHEAD, 0, 0]
                        break
                    if k == "0":
                        return
                    step = 0.02 if k.isupper() else 0.005
                    d = {"w": (step, 0, 0), "s": (-step, 0, 0), "a": (0, step, 0), "d": (0, -step, 0),
                         "q": (0, 0, step), "e": (0, 0, -step)}.get(k.lower())
                    if d:
                        robot.jog(d)
            a1, h1, a8 = (taught[k] for k in ("a1", "h1", "a8"))
            fd, rd = (h1 - a1) / 7, (a8 - a1) / 7
            sq = float((np.linalg.norm(fd[:2]) + np.linalg.norm(rd[:2])) / 2)
            cfg.board.square = sq
            cfg.board.a1 = (float(a1[0]), float(a1[1]))
            cfg.board.file_dir = tuple((fd[:2] / np.linalg.norm(fd[:2])).round(4).tolist())
            cfg.board.rank_dir = tuple((rd[:2] / np.linalg.norm(rd[:2])).round(4).tolist())
            cfg.board.board_z = float(np.mean([a1[2], h1[2], a8[2]]) - PAD_BELOW_TOOL)
            cfg.board.table_z = cfg.board.board_z - 0.011
            cfg.board.tray = cfg.board.default_tray()
            log(f"square {sq * 1000:.1f} mm, a1 {cfg.board.a1}, file_dir {cfg.board.file_dir}, rank_dir {cfg.board.rank_dir}, board_z {cfg.board.board_z:.3f}")
            robot.park()
        if args.step in ("all", "gripper"):
            input("Step 3: nothing between the pads. Enter to close... ")
            robot.enable()
            robot.gripper(cfg.motion.gripper_closed, settle=0.6)
            empty = robot.gripper_measured()
            input("Now hold a piece between the pads. Enter to close... ")
            robot.gripper(cfg.motion.gripper_open, settle=0.3)
            input("Enter when the piece is in place... ")
            robot.gripper(cfg.motion.gripper_closed, settle=0.6)
            held = robot.gripper_measured()
            robot.gripper(cfg.motion.gripper_open, settle=0.3)
            if empty is not None and held is not None:
                cfg.motion.grip_empty_angle = float(abs(held - empty) / 2 + abs(empty - cfg.motion.gripper_closed))
                log(f"gripper: empty {empty:.3f} rad, on a piece {held:.3f} rad -> grip_empty_angle {cfg.motion.grip_empty_angle:.3f}")
        cfg.save(args.calib)
        log(f"saved {args.calib}")
    finally:
        robot.close()


def cmd_init(args):
    cfg = GambitConfig.load(None)
    cfg.save(args.calib)
    log(f"wrote default calibration to {args.calib}; edit it or run `gambit.py calibrate --robot`")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--sim", action="store_true", help="MuJoCo simulation (laptop)")
        g.add_argument("--robot", action="store_true", help="the real robot (bbos)")
        p.add_argument("--execute", action="store_true", help="(robot) really move; default is a dry run")
        p.add_argument("--calib", default=str(DEFAULT_CALIB), help="calibration JSON")

    p = sub.add_parser("play", help="play a game")
    common(p)
    p.add_argument("--stockfish", help="path to the Stockfish binary (default: $STOCKFISH or PATH)")
    p.add_argument("--skill", type=int, help="Stockfish Skill Level 0..20")
    p.add_argument("--moves", type=int, default=0, help="stop after this many robot moves (0 = play on)")
    p.add_argument("--opponent", default="auto", choices=["auto", "me"],
                   help="(sim) auto = scripted/engine opponent; me = YOU move the black pieces by clicking the board")
    p.add_argument("--opening", default="sicilian", choices=list(OPENINGS), help="(sim, --opponent auto) the human's scripted replies")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", action="store_true", help="(sim) MuJoCo viewer")
    p.add_argument("--record", metavar="FILE.mp4", help="(sim) record the status display + 3-D view")
    p.add_argument("--window", action="store_true", help="the dashboard window: play on its board, f = full screen")
    p.add_argument("--http", type=int, default=0, help="optional web UI port for phones/spectators (default off)")
    p.add_argument("--voice", action="store_true", help="(sim) speak with the laptop's TTS")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("check", help="(robot) dry-run IK reachability of every square")
    common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("calibrate", help="(robot) reference image, board corners, gripper angles")
    common(p)
    p.add_argument("--step", default="all", choices=["all", "camera", "board", "gripper"])
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("init", help="write a default calibration file")
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.set_defaults(fn=cmd_init)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
