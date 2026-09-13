"""Drive the BracketBot arms through a lift / rotate / grip routine in MuJoCo.

    python demo.py            # interactive viewer (drag to orbit, space to pause)
    python demo.py --record   # write demo.mp4 next to this script instead

Requires chopped_urdf_v2.xml (run build_mjcf.py first).
"""
import argparse
import math
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE / "chopped_urdf_v2.xml"

# Joint targets (rad, or m for j0) reached at each time (s); the controller
# eases between consecutive keyframes. Joints not listed hold their last value.
# Right arm: rj0 slide down the mast, rj1 swing forward, rj2 raise outward,
# rj3 elbow, rj4-rj6 wrist. Left arm mirrors with lj*; only lj2 needs the
# opposite sign to move symmetrically.
KEYFRAMES = [
    (0.0,  {}),
    (1.0,  {}),                                                   # settle
    (3.0,  {"rj1": 1.2, "lj1": 1.2}),                             # lift both forward
    (5.0,  {"rj2": 0.9, "lj2": -0.9}),                            # raise outward
    (7.0,  {"rj3": 1.3, "lj3": 1.3}),                             # bend elbows
    (9.0,  {"right_left_gripper": 0.9, "left_left_gripper": 0.9}),  # open
    (10.5, {"right_left_gripper": 0.0, "left_left_gripper": 0.0}),  # close
    (13.0, {"rj1": -0.6, "lj1": -0.6}),                           # rotate shoulders back
    (15.0, {"rj1": 1.2, "lj1": 1.2}),                             # ...and forward again
    (17.0, {"rj0": -0.4, "lj0": -0.4}),                           # slide down the mast
    (19.0, {"rj0": 0.0, "lj0": 0.0}),                             # and back up
    (21.0, {"rj1": 0, "rj2": 0, "rj3": 0, "lj1": 0, "lj2": 0, "lj3": 0}),  # home
    (22.0, {}),
]


def targets_at(t: float, act_names: list[str]) -> np.ndarray:
    """Cosine-eased interpolation through KEYFRAMES for every actuator."""
    # expand sparse keyframes into a full pose per keyframe
    poses, pose = [], dict.fromkeys(act_names, 0.0)
    for kt, kv in KEYFRAMES:
        pose = {**pose, **kv}
        poses.append((kt, pose))
    if t >= poses[-1][0]:
        return np.array([poses[-1][1][n] for n in act_names])
    for (t0, p0), (t1, p1) in zip(poses, poses[1:]):
        if t0 <= t < t1:
            s = 0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0))
            return np.array([p0[n] + s * (p1[n] - p0[n]) for n in act_names])
    return np.array([poses[0][1][n] for n in act_names])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write demo.mp4 instead of opening a window")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    act_names = [model.actuator(i).name for i in range(model.nu)]
    hand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand__hand")
    duration = KEYFRAMES[-1][0]

    def step_to(t_target: float) -> None:
        while data.time < t_target:
            data.ctrl[:] = targets_at(data.time, act_names)
            mujoco.mj_step(model, data)

    if args.record:
        import imageio
        renderer = mujoco.Renderer(model, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        out = HERE / "demo.mp4"
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8) as w:
            n = int(duration * args.fps)
            for i in range(n):
                step_to(i / args.fps)
                renderer.update_scene(data, cam)
                w.append_data(renderer.render())
                if i % args.fps == 0:
                    print(f"t={data.time:5.1f}s  right hand xyz={data.xpos[hand].round(3)}")
        print(f"wrote {out}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.9]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -12
        wall0 = time.time()
        while viewer.is_running():
            # loop the routine forever, real-time
            t = (time.time() - wall0) % duration
            if t < data.time:  # wrapped around
                mujoco.mj_resetData(model, data)
            step_to(t)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
