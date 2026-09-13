# Bracket Gambit — status report

Written 2026-09-12, before any run on the physical robot.

## Legend
* **done / sim-verified** — implemented; exercised end-to-end in the MuJoCo simulation with the robot's URDF and/or by the unit tests
* **done / tested** — implemented; covered by tests with synthetic inputs or a mock robot
* **hardware-unverified** — implemented against the bbos daemon sources in `demos/`; has never run on a robot
* **mocked** — stands in for the real thing
* **partial** / **omitted** — as stated

## Core loop

| requirement | status | notes |
|---|---|---|
| maintain a valid internal position | done / tested | python-chess; only advanced after camera verification (`game.py`) |
| determine the human's move from the board | done / sim-verified | occupancy diff vs legal moves (`vision.infer_move`); 6/6 correct in the recorded sim game, synthetic tests incl. captures/castling |
| reject / recover from a non-legal observation | done / tested | spoken description of the offending squares, waits for correction; no-change detection |
| Stockfish selects a legal response | done / tested | UCI via python-chess, Skill Level + think time; Stockfish 19 used in the sim run |
| convert a move to pick-and-place | done / sim-verified | capture-to-tray first, castling king then rook, en passant; promotion excluded |
| move ordinary pieces reliably | done / sim-verified (**hardware-unverified**) | sim: shaft pinch with tilted pads, collision-checked IK lines, touch-detected release; real arm: same sequence via the daemon IK |
| announce move and status | done (**speech hardware-unverified**) | text always; espeak-ng → `speaker.audio` on the robot, SAPI/pyttsx3 on a laptop, chimes otherwise |
| keep physical and internal state synchronised | done / tested | verification after every robot move; mismatches shown in red on the display |
| start / pause / reset / stop | done / tested | terminal keys, web buttons |
| fail safely | done (**hardware-unverified**) | stop flag polled every tick; torque-off e-stop; workspace box; full line solved before motion; speed caps; retries then "needs help" |
| turn-complete signal | done | Enter / web / wake word (wake word **hardware-unverified**) |
| status visibility | done | phases on console, LED colours, web page/OpenCV window/video with board map + camera overlay + engine + log |

## Physical behaviour

| requirement | status | notes |
|---|---|---|
| approach without disturbing neighbours | sim-verified | vertical approach, diagonal pad orientation, gripper opening chosen to clear neighbours (sim); **on the real robot depends on the pad geometry check in the bring-up list** |
| controlled, conservative motions | sim-verified / implemented for real | eased, joint-rate-limited lines (sim); 0.04–0.10 m/s lines, daemon low-pass + rate clip (real) |
| verify grasp and placement | done | sim: lift height + contact; real: gripper angle after closing (`grip_empty_angle`); placement: camera verification of the whole board |
| retry safely | done / tested | one retry at the camera-refined position, then park + ask for help |
| stop and request help | done / tested | `NEEDS_HELP` phase; `h` re-verifies |
| position updated only after success | done / tested | see invariant in README |
| immediate e-stop | done (**hardware-unverified**) | `e`, web, Ctrl-C |
| expressive behaviours | partial | LED + speech reactions (check, mate, wave, thinking); arm/head gestures **omitted** on the real robot until the arm behaviour is verified (they are logged in sim) |

## Captures and special moves

| feature | status |
|---|---|
| non-capturing moves | done / sim-verified |
| captures with a tray | done / sim-verified (two captures in the recorded game) |
| castling (both pieces, verified) | done / tested (plan + inference); not exercised in the sim run |
| en passant | done / tested in planning; not exercised physically |
| promotion | omitted — engine avoids it, human promotion → needs help |
| arbitrary start positions | omitted — start requires the standard position (setup verification is implemented) |
| full tournament rules | omitted (python-chess enforces legality; clocks, draw claims, resignation not handled) |

## Interaction

| feature | status |
|---|---|
| "Start", "turn complete", "what", "why", "pause", "reset", "stop" | done via keyboard and web |
| voice commands beyond the wake word | omitted — no speech-to-text in the SDK; the wake word is mapped to "turn complete" / "fixed by hand" |
| grounded explanations | done / tested (`engine.explain`) |
| adjustable strength | done (`--skill`, `think_time` in the calibration) |
| personalities, assisted mode, teleop learning | omitted (not started) |
| spectator display | done (web page / window / video) |

## Base motion (changed 2026-09-13)

The simulated base used to slide sideways to line up each file — impossible for a two-wheeled
robot. It now does what a differential drive can do: it rolls straight up to the table, parks
3 cm to the left of the board centre line (`BASE_Y`), and **turns in place** so the target
square sits in front of the arm (`ChessBot.face()`, yaw between −25° and +38°). Measured with the
reach test (`tests`/README): turn-in-place alignment reaches **64/64 squares**; a base that does
not move at all reaches only ~42 (ranks 1–5, files b–g), so on the real robot either the base
yaw channel is needed or the board must be smaller than 8×8 at 3 cm. The camera looks at the
board only at yaw 0, so the markers stay in view.

## Mocked or simulated things — never presented as hardware

* `SimRobot` (MuJoCo): the base turns in place between moves — the real SDK we were given has no base-drive channel, so the real backend keeps the base still; see "Base motion" above for what that costs in reach.
* The simulated human teleports pieces; the simulated helper teleports pieces when the robot asks for help (counted and printed).
* `MockRobot` / `PhysicalMock` in the tests.
* Sim vision uses blue-dot markers; the real board uses ArUco (both detectors are tested on synthetic images).

## Hardware-unverified (must be validated on the robot, in this order)

1. `arm_right` frame conventions: `DOWN_QUAT_XYZW` points the pads down and closes them across the file axis; `IK_FRAME_Z`, `PAD_AHEAD`, `PAD_BELOW_TOOL` offsets (bbos_robot.py constants).
2. The straight-line home→ready→board motions do not sweep through the mast/table (the daemon IK has collision cylinders; the sim's planner is not used on the real robot).
3. Gripper open angle clears neighbouring pieces; `grip_empty_angle` from the calibration step separates empty from holding.
4. The home pose leaves the camera's view of the board clear; the head camera sees all four markers; `occupied_diff` threshold and the learned colour split under venue lighting.
5. Wake-word latency and false triggers; espeak-ng audio level on `speaker.audio`.
6. E-stop: `arm.torque enable=False` behaviour under load (the arm may sag: keep hands clear).
7. Capture-tray slots reachable (`gambit.py check`).

## Vision finding from the first recorded run

The first recorded game rejected a correct human move (Nf6) because the vacated g8 still
read as occupied: on the far ranks a piece's top smears ~0.6–0.8 square *away* from the
camera, into the near strip of the square behind it, and the sample patch (near half of
the square, inherited from the earlier sim) sat exactly there. The system behaved as
designed — it refused the observation, and later caught the resulting physical/logical
divergence at verification — but the classifier was wrong. Sweeping the patch on rendered
positions (`tests/test_vision_fixtures.py`, 7 positions incl. castling and captures):
near-half patch 4–11 errors, the old HSV colour method 10, a band straddling the square
centre (rows 0.45–0.65, where the base sits) 0. That band and `occupied_diff = 18` are now
the defaults; re-check them under venue lighting at calibration (`--step camera`, then
`start` reports how many squares differ from the starting position).

## Feasibility on the physical Bracket Bot (assessment, 2026-09-12)

Grounded in the repo (CAD-accurate URDF, daemon/servo limits, SDK channels) plus the public
fact that Bracket Bot is a two-wheeled self-balancing base (hoverboard motors).

| factor | verdict | evidence / mitigation |
|---|---|---|
| reach | OK for a ≤ 3 cm-square board only | arm fan ~24 cm deep at a 0.72 m table (URDF); no base-drive channel in the SDK → `gambit.py check --robot` |
| payload / precision | OK | 30 g pieces vs ~3.9 Nm shoulder limit; 4096-count encoders; ±30 % of a square tolerated by vision |
| gripper | OK with pads added | 1-DOF mimic claws, gentle torque limit 300/1000 + current relief; glue foam/rubber strips on the claws; grasp check from the measured angle |
| **balancing base** | **main risk** | extending the arm shifts the CoM → the balancer leans/creeps by mm at the board. Prop/lock the base for the demo, keep far ranks out of play; test first on hardware |
| camera | OK, lighting-dependent | mast camera ~1.5 m above a 24 cm board ≈ 4–6 px/mm: fine for ArUco + occupancy |
| compute | OK | Stockfish arm64 at 1 s/move + homography/patch statistics are light for a Pi 5 / Jetson |
| speech / wake word | OK (unverified) | `speaker.audio` PCM + espeak-ng; `wakeword.state` |
| rolling up / turning to face squares | not with this SDK | needs a base-drive channel; sim-only. Without it a fixed base reaches ~42/64 squares → use a smaller board or add the channel |

## Known weaknesses

* Far ranks (7–8) are at the edge of the reach fan; in the sim, retraction from there is slowed to 2.5 cm/s and the arm is fully stretched. On the real robot this is the most likely place for an IK failure → the move is refused before any motion and the robot asks for help.
* Vision assumes the human centres pieces roughly (±30 % of a square is tolerated by the sample patch); pieces knocked askew show up as mismatches, which is intended.
* Only one arm is used. The second arm is idle (see next improvements).

## Next improvements, in priority order

1. **Bring-up on the robot** through the checklist above; record `occupied_diff`, colour split and gripper angles in `gambit_calibration.json`.
2. **Grasp-quality signal from the gripper current** (`arm_state.current[7]`) in addition to the angle, to detect a slipping piece during the carry.
3. **Piece-position refinement before every pick** (not only on retry) once the camera offsets have been validated on real pieces.
4. **Left arm for files a–b** if `gambit.py check` shows them unreachable from the parked pose (the interface already routes each transfer through `Robot.pick/place`; a two-arm backend chooses the arm by y).
5. **Head/arm gestures** (look at the player, a small "thinking" sway) behind a `--gestures` flag, verified separately from manipulation.
6. **Board-corner re-detection each turn** is already done; add a *camera-blocked* retry loop with a 2 s wait so a hand in view does not need a manual re-signal.
7. **Castling and en passant on the physical board** (exercise in sim with a scripted opening that castles: `--opening french`).
8. **Resume a game after an e-stop** by re-reading the board and matching it against the last verified position.
9. **Promotion** via a piece swap from the tray.
10. **Assisted / coaching mode** using `engine.explain` on Stockfish's top line for the human.
