# Bracket Gambit — BracketBot plays chess

A physically embodied chess opponent: BracketBot watches the board with its head camera,
works out the human's move from the change in occupancy, asks Stockfish for a reply,
announces it, and moves the piece with its arm — then checks its own work before it
trusts the new position.

```
laptop (simulation):   uv sync
                       uv run python gambit.py play --sim --moves 4            # + --view / --record demo.mp4 / --voice
robot (dry run):       uv run gambit.py check --robot                          # IK reaches every square?
                       uv run gambit.py calibrate --robot --execute            # camera reference, board corners, gripper
                       uv run gambit.py play --robot                           # prints every waypoint, moves nothing
robot (for real):      uv run gambit.py play --robot --execute
tests:                 uv run pytest tests
```

> **Status in one line:** the whole perception → Stockfish → manipulation → verification
> loop is implemented and runs end-to-end in MuJoCo with the robot's own URDF; the real-robot
> backend is written against the bbos daemon sources but has **not been run on hardware**.
> See [STATUS.md](STATUS.md) for the feature-by-feature breakdown and [DEMO_SCRIPT.md](DEMO_SCRIPT.md)
> for the runbook.

---

## 1. The interaction

| step | what happens | where |
|---|---|---|
| 1 | `start`: the arm parks out of view, the camera reads the board, the 32 pieces are checked against the starting position (and the black/white brightness split is learned from them) | `game.start` |
| 2 | the robot (white, its pieces on the ranks nearest to it) opens: Stockfish picks, the robot says the move ("I play e4: pawn from e2 to e4"), picks the piece by its shaft, places it, parks, re-reads the board and only then pushes the move internally | `game.robot_turn`, `manipulation`, `game.verify_robot_move` |
| 3 | the human moves and signals **turn complete** (Enter, the web button, or the wake word) | `ui`, `bbos_robot.wait_signal` |
| 4 | the camera reads the board; the observed occupancy is matched against every legal move; a unique zero-mismatch match is accepted, anything else is rejected with a spoken description of what is wrong ("c6 should be empty but I see a black piece") and the robot waits for a corrected board | `vision.infer_move`, `game.human_turn_complete` |
| 5 | repeat; `why` speaks a one-line explanation grounded in the move's concrete features and Stockfish's evaluation; `what` repeats what the camera saw | `engine.explain` |
| — | a grasp that comes up empty is retried once at the position the camera saw the piece; a second failure parks the arm and asks the human to finish the step by hand (`h`), after which the board is re-verified | `manipulation.MoveExecutor` |
| — | `pause` / `resume` / `reset` / `stop` at any time; **`e` (or the red web button, or Ctrl-C) is the emergency stop**: torque off within one 20 ms tick, the game stays alive so `resume` can clear it | `game.handle`, `robot.StopFlag` |

Captures go to a tray beside the h-file before the capturing piece moves; castling moves
the king then the rook. Promotion is not executable (a piece swap would be needed): the
engine's promotion choice is replaced by its best non-promoting move, and a human
promotion is handled as a "needs help" board correction.

## 2. Architecture

```
gambit.py                     CLI: play / check / calibrate / init
bracket_gambit/
  config.py      BoardGeometry (a1, square pitch, axes, heights, tray), VisionConfig, MotionConfig -> gambit_calibration.json
  vision.py      markers (ArUco or blue dots) -> homography -> rectified 400x400 board -> per-square w/b/empty
                 infer_move (occupancy vs legal moves), mismatched_squares, piece_offset (grasp refinement)
  engine.py      StockfishEngine (python-chess UCI, Skill Level, think time) | FallbackEngine (labelled) ; explain()
  game.py        the state machine: IDLE OBSERVING WAITING THINKING ANNOUNCING MOVING VERIFYING RECOVERING NEEDS_HELP PAUSED STOPPED GAME_OVER
  manipulation.py chess move -> ordered transfers (capture-to-tray, mover, castling rook); retries; never touches the position
  robot.py       the backend interface + StopFlag + MockRobot (tests)
  sim_robot.py   MuJoCo backend on chopped_urdf_v2/sim/chess_robot.py (validated arm/gripper/piece physics)
  bbos_robot.py  real backend on the bbos daemons (arm ctrl via the daemon's IK, head jpeg, LED, speaker, wake word)
  ui.py          Console (stdin keys), HttpUI (port 8010: buttons + live status frame), Display (spectator frame), speech
tests/           synthetic-camera vision tests, engine, manipulation, full loop with a mock robot
```

**Data flow per turn:** `Robot.look()` → `BoardReader.read()` → `Observation` (occupancy, rectified
image, homography) → `infer_move()` → `chess.Board` → `Engine.choose()` → `MoveExecutor.execute()`
(`Robot.pick/place`) → `Robot.look()` → `mismatched_squares()` → `board.push()`.

**Invariant:** the internal position advances only after a camera observation matches the
expected occupancy — for the human's move and for the robot's own.

## 3. Setup

### Laptop (simulation, tests, tuning vision on photos)
```
uv sync                        # mujoco, opencv, python-chess, imageio, pytest
uv run pytest tests            # 30 tests, ~7 s
uv run python gambit.py play --sim --moves 4 --record demo.mp4
```
Stockfish: the Windows binary is already in `tools/stockfish/stockfish.exe` and is found
automatically. Other platforms: drop the matching release binary in `tools/stockfish/`
(see the README there), or put it on `PATH`, or `set STOCKFISH=<path>`, or `--stockfish <path>`.
Without it the built-in fallback searcher is used and every status line says so.

### Robot
0. Connect: the robot is a Linux box on the LAN, user `bracketbot`, hostname like
   `bracketbot-092` (it prints `http://<hostname>.local:...` in every demo). From Windows
   (OpenSSH is built in), macOS or Linux:
   ```
   ssh bracketbot@bracketbot-092.local        # or ssh bracketbot@<ip>
   ```
   If `.local` does not resolve (some venue Wi-Fi blocks mDNS), use the IP from the
   robot's screen / your router, and `ssh-copy-id bracketbot@<host>` once to skip the
   password. Same LAN is required for the web UI (port 8010) too.
1. Copy `gambit.py` and `bracket_gambit/` to the robot (e.g. `~/bbapps/gambit/`) — on Windows
   `.\deploy_to_robot.ps1 -RobotHost bracketbot-092.local [-InstallDeps]` does the SSH check,
   the copy, optional `apt install stockfish espeak-ng`, and the first `uv` resolve; elsewhere
   `scp -r gambit.py bracket_gambit bracketbot@<host>:~/bbapps/gambit/`. `gambit.py` carries
   the same PEP 723 header as the `demos/`, so `uv run gambit.py ...` resolves `bbos` from
   `/home/bracketbot/bbos`.
2. Stockfish: `sudo apt install stockfish` (arm64 package), or download
   `stockfish-linux-arm64-universal.tar.gz` from the Stockfish releases and put the binary in
   `tools/stockfish/` next to `gambit.py` (copy that folder to the robot too), or
   `export STOCKFISH=/path/to/stockfish`.
3. Speech: `sudo apt install espeak-ng` — `say()` renders text to PCM and plays it on `speaker.audio`.
   Without it the robot chimes and the text goes to the console/web page.
4. Daemons needed: the arm (`arm_right`), head camera, LED, speaker, wakeword.

## 4. Physical set-up and calibration

Assumptions (all encoded in `gambit_calibration.json`, `gambit.py init` writes the defaults):

* **Robot parked by hand** square to the table; there is no base-drive channel in the SDK
  we were given, so the base does not move during the game. (In the simulator the base
  turns in place — never sideways — to bring each square in front of the arm; a fixed base
  reaches only ~42 of 64 squares of a 3 cm board, so on hardware either expose the base yaw
  or use a smaller board. See STATUS.md "Base motion".)
* **Board**: compact — 30–35 mm squares (the arm's reach fan is only ~24 cm deep at table
  height: x ≈ 0.16–0.40 m ahead of the base), the a1–h1 edge nearest the robot, centred at
  y ≈ −0.10 m (the right arm's sweet spot), on a table ≈ 0.72 m high. The robot plays white
  so its pieces are on the near ranks.
* **Markers**: four printed ArUco tags (DICT_4X4_50, ids 0 near-a1 corner, 1 near-h1, 2 far-a8,
  3 far-h8), centred `marker_offset` squares outside each board corner (default 0.267 ≈ 8 mm
  on a 30 mm board). The simulator uses blue dots (`marker_mode: colour`) instead.
* **Pieces**: light, with a plain cylindrical shaft ≥ 4 cm tall and ≤ 16 mm diameter that the
  pads can pinch (`piece_top` = height of the shaft top per kind); weighted bases help.
  Matt white and black; nothing else on the board in those colours.
* **Camera**: head camera sees all four markers with the arm parked; lighting fixed for the
  demo (the reference image is taken under the same light).
* **Tray**: two rows of six slots beside the h-file (`board.tray`), on the table surface.

Calibration (`gambit.py calibrate --robot --execute`, or `--step camera|board|gripper`):

1. **camera** — empty board in view: saves the rectified reference image
   (`gambit_calibration.reference.png`). The black/white split is learned at every `start`.
2. **board** — jog the open gripper (w/s/a/d/q/e keys, 5 mm; capitals 20 mm) so the pads
   straddle the centre of a1, then h1, then a8, with the pad lower edge on the board. This
   yields `a1`, `square`, `file_dir`, `rank_dir`, `board_z` and a default tray.
3. **gripper** — close on nothing, then on a piece: sets `grip_empty_angle`, the threshold
   that turns the gripper's measured angle into "did I get it?".

Then `gambit.py check --robot` solves IK for every square and tray slot at grasp and
approach height and lists anything unreachable — move the board and repeat until all
152 targets pass.

## 5. Operating the demo

Three equivalent control surfaces (use whichever the room allows):

| action | terminal | web page `http://<robot>:8010/` | voice |
|---|---|---|---|
| start a new game | `n` | Start game | — |
| my turn is complete | **Enter** | My turn is complete | wake word |
| what move did you see? | `w` | What did you see? | — |
| why did you play that? | `y` | Why? | — |
| pause / resume | `p` / `r` | Pause / Resume | — |
| reset (pieces back to start) | `x` | Reset | — |
| I fixed the board by hand | `h` (or Enter) | Fixed by hand | wake word |
| stop (park, exit) | `s` | Stop | — |
| **EMERGENCY STOP** | **`e`** or Ctrl-C | **EMERGENCY STOP** | — |

The web page doubles as the spectator display: 3-D view (sim) or camera, the camera image
with the detected grid and classifications, the internal position with the last move,
squares that disagree with the camera outlined in red, the engine line and evaluation, the
spoken explanation and the event log. `--window` shows the same frame in an OpenCV window.

The LED mirrors the phase: breathing blue = waiting for you, white = looking, amber =
thinking, magenta = moving, blinking red = needs help / stopped.

### Emergency stop
* `e` in the terminal, the red web button, or Ctrl-C set the stop flag **from the input
  thread**, so the motion loop sees it on its next 20 ms tick regardless of what the game is
  doing; `BBOSRobot.estop()` writes `enable = False` on `arm.torque` for every joint.
* The arm daemon independently drops torque if the ctrl writer disappears (kill the process).
* `resume` after an e-stop clears the flag and returns to IDLE; the game position is kept
  only if you `start` again from the starting position (a mid-game restart is not supported).
* Physical: the arm's daemon current/temperature limits stay active; keep the mains
  e-stop of the robot within reach as the last resort.

## 6. Tests and simulation

* `tests/test_vision.py` — synthetic camera renders (ArUco and colour markers, perspective,
  noise, jittered pieces): starting position 16/16, a mid-game position, missing markers,
  legal-move inference (capture, castling, illegal changes, no change), grasp refinement,
  colour-split learning.
* `tests/test_vision_fixtures.py` — real head-camera renders from the MuJoCo scene (7 positions
  with captures, a knight move and castling, 640×480) + the rectified empty-board reference:
  every square read correctly, every move inferred; also pins *why* the sample patch sits at
  the square centre (the near-half patch is shown to be fooled by far-rank smear).
* `tests/test_engine.py` — fallback legality, free-queen capture, grounded explanations,
  Stockfish integration (skipped if no binary), fallback selection.
* `tests/test_manipulation.py` — geometry, step planning (quiet, capture, castling, en passant),
  retry at the refined position, giving up, tray order, e-stop abort.
* `tests/test_game.py` — the full loop against a mock robot whose "physical" board is
  rendered synthetically: several alternating turns, illegal move rejected and recovered,
  no-change detection, failed grasp → needs help → verified after the human fixes it,
  pause/resume/e-stop/stop, bad setup reported.
* `gambit.py play --sim` — MuJoCo: the robot's URDF, finger pads, piece physics, the head
  camera image the vision runs on, collision-checked trajectories. Recorded run:
  `chopped_urdf_v2/sim/gambit.mp4` and `chopped_urdf_v2/sim/gambit_frames/`.

What only the robot can validate is listed in [STATUS.md](STATUS.md) § hardware-unverified.

## 7. Playing against it yourself (simulation)

```
uv run python gambit.py play --sim --opponent me --moves 0 --view --window
```
You are black. On the board in the web page (http://localhost:8010/) or in the `--window`
dashboard, click a piece and then its destination: in the simulator that moves the piece on
the table (it is your hand) and signals "turn complete"; the robot then reads the board with
its camera exactly as it would for a real move. Legal destinations are marked when a piece
is selected; the last move is tinted, disputed squares are outlined red. On the real robot
clicking a move is only a hint — you still move the physical piece and the camera decides.

The dashboard shows the 3-D scene with two insets (the rectified board the vision works on
and the **hand camera** looking down between the fingers), the head camera with the detected
grid, and the internal position with real chess glyphs.

## 8. Full games and speed

Nothing in the chess is scripted: Stockfish picks every robot move for the live position and
the human's moves come from the camera. `--moves 0` plays until checkmate / stalemate / draw
(`--moves N` only caps the demo). In the simulator the opponent is `SimHuman`: `--opening
random` makes it choose from its own search, the named openings script its first replies.
Robot moves take ≈ 15–25 s of robot time (a capture ≈ 2×); the speeds are in
`chopped_urdf_v2/sim/chess_robot.py` (`LINE_SPEED`, `CARRY_SPEED`, `FAR_SPEED`, `RATE`) and were
raised as far as the piece stays in the gripper on the far ranks.

## 9. Presentation material (`media/`)

* `Bracket_Gambit_demo.mp4` — ~3-minute narrated demo video cut from a recorded simulation
  (`media/make_video.py`, uses the phase timeline that `play --sim --record` writes).
* `Bracket_Gambit.pptx` — 12-slide judges' deck (`media/make_slides.py`): the interaction,
  the Bracket Bot capabilities used, perception, move inference, Stockfish, manipulation,
  safety, results, what is real / simulated / unverified, next steps.
* Regenerate after a new recording: `uv run python bracket_gambit/media/make_video.py --video
  chopped_urdf_v2/sim/gambit.mp4 --timeline chopped_urdf_v2/sim/gambit.timeline.json` and
  `uv run python bracket_gambit/media/make_slides.py`.
