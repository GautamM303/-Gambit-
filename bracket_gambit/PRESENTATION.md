# Gambit — presentation script and judges' Q&A

For the person presenting. Timings assume a 5-minute slot: 3 minutes of pitch + live demo,
2 minutes of questions. Cut the bracketed lines if you have less time.

Before you start: `uv run python gambit.py play --sim --opponent me --moves 0 --window`,
press `f` for full screen, and have `Bracket_Gambit_demo.mp4` open in another window as a
backup. Stockfish must be in `tools/stockfish/` — the first console line says which engine
is running.

---

## The pitch (≈ 1 min, while the sim starts and reads the board)

> This is **Gambit**: Bracket Bot playing chess against you. Not a chess engine bolted to an
> arm — a robot that *watches* the board, works out what you did, decides with Stockfish,
> tells you its move, and makes it with its own hand. Then it looks again to make sure the
> board is what it thinks it is.

> Everything you see runs on the robot's real CAD model in MuJoCo, on the Bracket Bot SDK's
> channels — head camera, arm, gripper, speaker, wake word, LED. We'll be straight about
> what's simulated and what's been on hardware.

[If asked why sim: "We had one robot slot and wanted a reliable loop before touching hardware;
the real backend is written, dry-run-able, and the calibration procedure is in the repo."]

## Live demo (≈ 2 min)

1. **Start.** Point at the console/dashboard.
   > It reads the board first: four corner markers, a homography, and each square compared to
   > an empty-board photo. Sixteen white, sixteen black — it verified the setup before playing.

2. **Robot opens.** While the arm moves:
   > Stockfish 19 chose that. It announced the move, then picked the piece by its shaft —
   > the pads are angled so the claws clear the neighbours — carried it, and set it down
   > until it *felt* the board. Watch the hand camera inset.
   > Now it parks the arm beside the board and re-reads it. The internal position only advances
   > when the camera agrees.

3. **Your move.** Click a piece, then a square (pick something simple: e5 or Nf6).
   > That's me moving on the table. The robot doesn't recognise pieces — it sees which
   > squares changed and matches that against every legal move. One match, zero mismatches:
   > it accepts.

4. **Press `y`** after the robot replies:
   > "Why?" gets an explanation grounded in the position and Stockfish's evaluation — not a
   > made-up story.

5. **Show a rejection** (optional, strong moment): click an *illegal* move, e.g. a pawn
   sideways.
   > It refuses and names the squares that are wrong, then waits for me to fix the board.
   > Same thing happens if a piece gets knocked over on the real board.

6. **Press `e`** at any time:
   > Emergency stop — the flag is polled every 20 ms in the motion loop; on the real robot it
   > cuts torque on all joints.

7. **Close** (or switch to the video's capture segment if time allows):
   > Captures go to a tray first; castling moves king then rook. A full unscripted game ran to
   > checkmate in the sim with no drops after we fixed the grip — that fix, an end-stop on the
   > finger pads, is a design note we're taking to the physical gripper.

---

## Questions the judges will ask — and honest answers

**Has it run on the real robot?**
Not yet. The real backend (`bbos_robot.py`) is written against the daemon sources in the SDK
and has a dry-run mode that prints every waypoint without moving. The bring-up order is
documented: reachability check → calibration (empty-board photo, teach three corners, gripper
angles) → dry run → execute. Seven things need verifying on hardware; they're listed in
STATUS.md. Everything else — vision, move inference, Stockfish, the state machine, retries,
the UI — is tested on synthetic images, on rendered boards and in the full simulation.

**What does the simulation actually prove?**
That the *decision loop* is correct end to end and that the manipulation strategy works with
this arm's kinematics: reach, collision-free approaches between neighbouring pieces, grasp
geometry, placement accuracy (≤ 1 mm in sim). It does not prove contact behaviour on real
rubber pads or camera performance under venue lighting — those are exactly the calibration
steps.

**How does it know what move I made? Do you type it in?**
No. It compares the board before and after: which squares became empty, occupied, or changed
colour. That pattern is matched against every legal move of the current position; captures,
castling and en passant all produce distinct patterns. If nothing matches uniquely it rejects
the observation and tells you which squares disagree.

**What if it misreads the board?**
It never trusts a single ambiguous read. A move is accepted only if exactly one legal move
explains the board with zero mismatches. After its own move it re-reads and compares to the
expected position; if they differ it asks for a hand, and after three failed rounds it stops
rather than guessing. All of that is exercised in the tests.

**Why Stockfish and not your own engine?**
Stockfish is the strongest open engine; strength is a slider (skill 0–20) so it's beatable for
a demo. There is a small built-in fallback if the binary is missing, and every screen labels it
as *not Stockfish* so nobody is misled.

**How long does a move take?**
About 16–21 s of robot time in simulation: pick, carry, place, park, camera check. Speed is
limited by the piece staying in the gripper, not by the arm — we measured carry speeds until
tall pieces dropped and backed off.

**Why does the base turn instead of driving sideways?**
It's a differential drive: it can't strafe. A fixed base reaches only ~42 of 64 squares of a
3 cm board; turning in place to face each square reaches all 64. On the real robot the SDK we
had exposes no base-drive channel, so on hardware you either park it and use a smaller board or
add that channel — that's documented.

**What about the self-balancing base?**
It's the main hardware risk: extending the arm shifts the centre of mass and the balancer
leans, which is millimetres at the board. For the demo we'd prop the base. It's the first thing
to check on hardware.

**Can it play a whole game?**
Yes — nothing is scripted. Stockfish picks every move for the live position, the camera reads
every human move, python-chess enforces the rules, and it recognises check, mate, stalemate.
Promotion is the one thing the gripper can't do (piece swap), so the engine avoids it and a
human promotion is handled as a board correction.

**What's mocked or simplified?**
The simulated human teleports pieces (that's your hand); the simulated helper puts pieces back
when the robot asks for help; the base motion is simulation-only. The pieces are simplified
shapes with a plain cylindrical shaft — that's also the recommendation for the physical set:
lightweight pieces with a 16 mm shaft.

**What was the hardest problem?**
Grip reliability. Grip force was fine (11 N on a 30 g piece); the pieces still crept out of the
flat pads because a round shaft between pads that aren't perfectly parallel gets squeezed
toward the wider end. The fix was geometric — parallel pads plus a lip at each end of the pad —
and the same applies to real fingers.

**What would you do with another day?**
Bring it up on the robot through the checklist; use the gripper current as a second grasp
signal; refine the pick position from the camera on every move; use the left arm for the
far files; add the expressive gestures behind a flag once the manipulation is verified.

**Is voice required?**
No. The wake word means "my turn is complete"; everything else is keys or the window (a web
page for phones is optional). Voice is never a dependency for the chess loop.

**Safety?**
E-stop from keyboard, window, web button or Ctrl-C cuts torque within a control tick; every
line is solved by IK and checked against a workspace box before the arm moves; speeds are
capped; the daemon's own current and temperature limits stay active underneath; nothing moves
on the robot without `--execute`.

---

## If something goes wrong live

| symptom | do |
|---|---|
| robot rejects a correct move | say "it's telling us the board doesn't match — that's the safety net"; re-click the move |
| "I need a hand" | press `h` (the sim helper fixes it); on the robot, fix the piece by hand then `h` |
| the sim is slow | close other windows; it's physics-paced, not the robot |
| console says fallback engine | Stockfish binary missing from `tools/stockfish/` — the demo still works, just say so |
| nothing responds | `e` to stop, then restart the command; the video is the backup |
