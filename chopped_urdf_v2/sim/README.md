# MuJoCo simulation of chopped_urdf_v2

Runs natively on Windows/macOS/Linux — no ROS or Gazebo needed. Dependencies are
declared in the repo's `pyproject.toml`; from the repo root:

```
uv sync                                                # provisions .venv (mujoco, opencv, imageio, ...)
uv run chopped_urdf_v2/sim/build_mjcf.py               # URDF -> chopped_urdf_v2.xml (sim-ready MJCF)
uv run chopped_urdf_v2/sim/demo.py                     # interactive viewer: arms lift, rotate, grip, slide the mast
uv run chopped_urdf_v2/sim/demo.py --record            # same routine written to demo.mp4
uv run chopped_urdf_v2/sim/pick_place.py --color red   # vision-guided pick-and-place (see below)
```

## Files

| File | Purpose |
|---|---|
| `build_mjcf.py` | Converts `../urdf/chopped_urdf_v2.urdf` into MuJoCo MJCF. Recomputes masses from mesh hulls (the Onshape inertials are unscaled, ~6 g total), adds position actuators to all 16 controllable joints, keeps the gripper mimic joints as equality constraints, welds the base to the world and adds a floor/lights/camera. |
| `chopped_urdf_v2.xml` | Generated model. Mesh paths are relative to `../meshes`. |
| `demo.py` | Keyframe controller. Edit `KEYFRAMES` to change the routine. |
| `pick_place.py` | Vision-guided pick-and-place with collision-checked motion planning (builds its scene on top of the generated XML in memory). |
| `vision_<colour>.png`, `pick_place.mp4` | Outputs of `pick_place.py`: the annotated head-camera frame and, with `--record`, the video. |

## Joint map

| Actuator | Motion | Range |
|---|---|---|
| `rj0` / `lj0` | slide arm down the mast (prismatic) | −1.03 … 0 m |
| `rj1` / `lj1` | shoulder swing forward/back | ±120° |
| `rj2` / `lj2` | shoulder raise outward (mirror: `lj2 = −rj2`) | ±120° |
| `rj3` / `lj3` | elbow | ±120° |
| `rj4`–`rj6` / `lj4`–`lj6` | forearm roll, wrist pitch, wrist roll | ±120° |
| `right_left_gripper` / `left_left_gripper` | gripper open (second finger mimics) | 0 … 1 rad |

Control by writing joint targets to `data.ctrl[i]` in actuator order (`model.actuator(i).name`).

## Pick-and-place demo (`pick_place.py`)

```
uv run chopped_urdf_v2/sim/pick_place.py                 # red box, interactive viewer
uv run chopped_urdf_v2/sim/pick_place.py --color green   # green box
uv run chopped_urdf_v2/sim/pick_place.py --record        # write pick_place.mp4 instead of a window
uv run chopped_urdf_v2/sim/pick_place.py --headless      # console only; --seed N varies the planner
```

The robot faces a table with a red and a green 4 cm cube and a drop tray. The run:

1. **Vision** — renders the camera mounted in the head, segments red and green in
   HSV with OpenCV, and fits each cube's table position by matching the predicted
   silhouette centroid to the observed one (≈0.5 mm error). The chosen target is
   annotated in `vision_<colour>.png`.
2. **Reach** — RRT-Connect in the right arm's 7-D joint space (mast slide + 6
   revolute joints). Every node and edge is checked against self-collision,
   the other arm, the mast/base, the table and both cubes, using the convex
   hulls of every link with a 6 mm self-collision / 10 mm environment margin.
   Overlaps that exist by design (finger vs finger, the shoulder carriage sliding
   along the mast, ...) are found automatically and excluded.
3. **Grasp** — the gripper opens only once it is over the table (at the home pose
   an open inner finger would hit the mast), descends on a straight line, servos
   out the residual tracking error and closes.
4. **Carry and place** — straight Cartesian lines at fixed orientation while the
   cube is in hand, a gentle release onto the tray, retreat and return home.
5. **Verification** — the script reports the vision error, that the cube was
   lifted, that it was still in hand after the carry, whether it stands upright
   on the tray, and how many physics steps contained an illegal contact (any
   arm link touching itself, the other arm, the mast, the table or a cube it is
   not holding). A clean run prints `0 steps (none)`.

Grasping notes: the finger meshes are hollow fork-shaped claws whose convex hulls
cannot hold a box, so `pick_place.py` adds rubber pads to the claw tips and lets the
boxes touch only the pads and the environment (contact bitmasks), not the claw
hulls. It also enables elliptic friction cones with a high `impratio`, drives the
mimic finger with its own actuator, adds gravity compensation to the arm links and
retunes the servo gains — all in memory, so `chopped_urdf_v2.xml` and `demo.py`
are unchanged.

## Chess (`chess_robot.py`)

```
uv run chopped_urdf_v2/sim/chess_robot.py --moves 6 --record chess.mp4   # video + dashboard PNGs in chess_frames/
uv run chopped_urdf_v2/sim/chess_robot.py --view                          # interactive viewer
```

The robot starts 1.3 m from a table with a chess set on it, rolls up (the base is a
kinematic planar x/y/yaw platform with spinning wheels, added by `RobotSpec(mobile=True)`),
reads the board with its head camera and plays white against a simulated human whose
moves are teleported onto the board as if a hand had moved them. Each turn:

1. **Vision** — head camera → four blue corner markers → homography → rectified 8×8 view →
   every square classified white / black / empty (`read_board`).
2. **Move inference** — the human's move is the legal move whose resulting occupancy best
   matches what the camera saw (`infer_move`), then applied to the `python-chess` board.
3. **Engine** — a small alpha-beta search (`--depth`, default 3) picks the reply.
4. **Execution** — the arm's gripper-down reach is only ~25 cm deep, so the base slides
   sideways to put the target file ~5 cm right of the shoulder; the piece is gripped by
   its shaft on the square's diagonal (so the claws clear the neighbours), lifted, carried
   at a parked height, set down, and captured pieces go to the robot's graveyard beside the
   board. All arm motion is straight-line IK with the same collision checks as
   `pick_place.py`; the arm retracts to home before every look so it does not block the camera.
5. **Dashboard** — `chess_frames/turnNN_*.png` and the video show the 3-D scene, the camera
   image with the detected markers/grid/occupancy, the rectified board, and the board map
   with the last move.

The console reports each move, whether vision inferred the human's move correctly, how far
off-centre each piece was placed, and the illegal-contact count for the whole game.

Known weakness: picking pieces off ranks 7-8 needs a fully stretched arm, and the retraction
from there occasionally shakes a piece out of the pads (the pads pinch a 16 mm shaft; the
contact model is the limiting factor). The robot therefore retracts from those squares at a
crawl, its engine discounts captures on those ranks (`FAR_RANK_PENALTY`), and if a piece is
still dropped or knocked over, a "helper" stands it back on its square — counted and shown on
the dashboard, as would happen at a real chess-robot demo.

## Running it on the real robot (`demos/pick_by_voice.py`)

`demos/pick_by_voice.py` is the same task written against the robot's `bbos` SDK
(the channels used by `demos/view_*.py`): wake word → LED/chime → depth-camera point
cloud → find the red/green block in 3-D → straight-line moves through the arm's own
IK → grip, carry, set down → home. It has two backends behind one `Robot` interface:

```
uv run python demos/pick_by_voice.py --sim --no-wait [--view]   # laptop: MuJoCo backend (tested)
uv run demos/pick_by_voice.py --color red                        # robot: dry run, prints waypoints, no motion
uv run demos/pick_by_voice.py --color red --execute              # robot: moves the right arm (<= 6 cm/s)
```

The bbos backend has not been run on hardware; its docstring lists the assumptions
to verify (point-cloud frame, IK quaternion convention, motor↔URDF joint mapping,
gripper joint) before using `--execute`.

## Known limitations

* **Base is welded.** The wheels are fixed joints in the URDF; the balancing base isn't simulated.
* **Masses are estimates** (convex hull × assumed density, see `DENSITY` in `build_mjcf.py`); actuator gains/force limits are placeholders, retuned in `pick_place.py`.
* `demo.py` still runs the visual-only model (links pass through each other); collision hulls, pads and contact settings live in `pick_place.py`.
* The pick-and-place uses the right arm only and assumes the cubes are axis-aligned.
* Same fixes are prerequisites for Gazebo, if you move there later (plus `<transmission>`/ros2_control tags).
