# Bracket Gambit — demo runbook

Written for the team member running the demo. Total time ≈ 8–10 minutes for six moves.

## T-30 min: bring-up (once per venue)

1. Park the robot square to the table, board nearest edge ≈ 16 cm ahead of the base,
   board centred ≈ 10 cm to the robot's right (y ≈ −0.10). Tape the four ArUco markers.
   Nothing else white/black on the table. Fix the lighting.
2. Daemons up (arm_right, camera, led, speaker, wakeword). `uv run view_arms.py arm_right`
   should show the arm at home.
3. Empty board → `uv run gambit.py calibrate --robot --execute --step camera`.
4. Teach corners → `uv run gambit.py calibrate --robot --execute --step board`
   (a1, h1, a8; pads on the board surface). Then `--step gripper`.
5. `uv run gambit.py check --robot` → "152/152 targets reachable". If not, shift the board
   toward the robot's right and repeat 3–5.
6. Dry run: `uv run gambit.py play --robot` (no motion), press `n`, watch the printed
   waypoints for the opening move: all x within 0.16–0.40, z between board_z and ~0.95.
7. Set the pieces up. Open `http://<robot>:8010/` on a phone (this is also the spectator
   screen — mirror it to a monitor if there is one).

## T-0: the show

```
uv run gambit.py play --robot --execute --moves 6
```

| you | the robot | if it goes wrong |
|---|---|---|
| press **n** (or *Start game*) | "Let's play. Checking the board." Camera reads 16/16. "You are black; I will open." Arm goes to the ready pose, LED amber, "I play e4: pawn from e2 to e4." Picks, places, parks, re-reads, LED breathing blue. | "The board does not look like the starting position: b1 should be a white piece but I see empty" → fix the piece, press **n** again. |
| invite the audience member to reply (any legal move, piece in the middle of its square), then press **Enter** / say the wake word | "You played c5." Thinks (amber), announces, moves. | "That does not look like a legal move… c6 should be empty but I see a black piece" → have them correct it and press **Enter** again. "I do not see any change" → they forgot to move / hand in view. |
| press **y** after a robot move | one-sentence explanation ("I played this because it develops a piece. Stockfish rates the position +0.4 pawns for me, expecting …") | — |
| press **w** any time | "I saw 16 white and 15 black pieces. Your last move was c5." | — |
| a capture happens | the robot first carries the captured piece to the tray, then moves its own piece | the tray is beside the h-file; keep hands away while the arm moves |
| the robot says "I need a hand…" | LED blinking red, arm parked | do the requested step by hand, press **h** (or Enter). The robot re-reads the board and continues. |
| after 6 robot moves | "That is the end of the demo. Thank you for playing!" arm parks | — |

## Safety brief (say it out loud before starting)

* **`e` in the terminal, the red web button, or Ctrl-C = emergency stop** (torque off in
  < 50 ms). The mains e-stop is the last resort.
* Nobody reaches over the board while the LED is magenta (moving) or white (looking).
* The human moves pieces only after the robot says it is waiting (breathing blue LED).
* `p` pauses (arm stays where it is, torque on); `r` resumes.

## Talking points while it thinks

* Vision is a difference against the empty-board reference, rectified through four ArUco
  markers — no piece recognition needed, because the move is inferred by matching the
  occupancy change against the legal moves of the internal position.
* The internal position is only advanced after the camera confirms the physical board —
  for both players. If the board and the engine ever disagree, the robot says so instead
  of guessing.
* Stockfish 19, Skill Level 5 by default (`--skill 0..20`); one second per move.
* The same code runs in the MuJoCo simulation with the robot's URDF (`gambit.py play --sim`),
  which is where the manipulation was tuned.

## After the demo

`s` to stop (parks the arm, torque stays on until the process exits). The move list and
helper-intervention count are printed at the end.
