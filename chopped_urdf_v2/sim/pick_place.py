"""Vision-guided pick-and-place for the BracketBot right arm in MuJoCo.

    uv run chopped_urdf_v2/sim/pick_place.py                  # pick the red box, interactive viewer
    uv run chopped_urdf_v2/sim/pick_place.py --color green
    uv run chopped_urdf_v2/sim/pick_place.py --record         # write pick_place.mp4 instead of a window
    uv run chopped_urdf_v2/sim/pick_place.py --headless       # console only (fast check)

Pipeline
  1. scene    chopped_urdf_v2.xml + collision hulls on every link and the mast,
              rubber pads on the claw tips, a table, a red and a green box, a
              drop tray and a head camera (built in memory; demo.py is untouched)
  2. vision   render the head camera, segment red/green in HSV (OpenCV), then fit
              the box position so its predicted silhouette centroid matches the
              observed one (sub-millimetre; writes vision_<colour>.png)
  3. IK       limit-aware damped least squares on the 7 right-arm joints
              (mast slide + 6 revolute)
  4. planning free-space moves: RRT-Connect in joint space, every node and edge
              checked for self-collision, the other arm, the mast/base, the
              table and the boxes (6 mm self / 10 mm environment margins);
              moves with the box in hand: straight Cartesian lines at fixed
              orientation, checked the same way
  5. execute  position actuators track the time-parameterised path while a
              monitor counts any illegal contact (expected: 0) and the script
              verifies the box was lifted, carried and set down upright

Requires chopped_urdf_v2.xml (run build_mjcf.py first).
"""
import argparse
import math
import pathlib
import re
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MODEL_XML = HERE / "chopped_urdf_v2.xml"

# Layout sits inside the right arm's gripper-down workspace (x 0.20-0.35, y -0.35..+0.15).
TABLE_TOP = 0.72
TABLE_CENTER = (0.30, 0.0)
TABLE_HALF = (0.16, 0.32)
BOX_HALF = (0.02, 0.02, 0.02)
BOX_HEIGHT = 2 * BOX_HALF[2]
BOXES = {  # colour -> (x, y, rgba)
    "red": (0.28, -0.26, (0.85, 0.12, 0.10, 1.0)),
    "green": (0.28, -0.08, (0.12, 0.70, 0.18, 1.0)),
}
TRAY = (0.28, 0.12)
TRAY_TOP = TABLE_TOP + 0.005
# The claw hulls only meet 3-7 cm above the eef frame and pinch a box by an edge, which
# is unreliable, so rubber pads are added to the claw tips (which hang 2.3 cm below the
# frame), covering the cube's upper 3 cm face-on. The pads are made parallel at the
# opening the fingers actually settle at when squeezing (the soft contacts let each
# pad sink ~1.5 mm into the cube), so both contact rows carry load; a pinch that
# converges either way collapses onto one edge and the cube swings or pitches out.
PAD_HALF = (0.01, 0.002, 0.02)
PAD_OFFSET = (0.015, 0.003, 0.0)  # pad centre in the eef frame (x along the claw, y to its side, z down)
PAD_BIAS = 0.003                  # pads parallel at a gap this much narrower than the cube
HINGE_TO_PAD = 0.096
GRASP_HEIGHT = 0.03   # eef frame above the box's bottom: pads span the cube's upper 3 cm
APPROACH = 0.12       # pre-grasp / pre-place height above the grasp pose
CAM_POS = (0.10, 0.0, 1.57)
CAM_LOOK = (0.30, 0.0, TABLE_TOP)
PLAN_MARGIN = 0.006      # planning clearance between the arm's own links
ENV_MARGIN = 0.01        # planning clearance to the mast, table, tray and boxes (covers tracking lag)

ARM_PARTS = {"shoulder", "shoulder_cover", "shoulder_knuckle", "bicep", "bicep_cover",
             "forearm", "forearm_cover", "forearm_rotation", "wrist_knuckle", "hand",
             "hand_motor_mount", "wrist_cam", "left_finger", "right_finger", "right_eef", "left_eef"}
GRIPPER_PARTS = {"hand", "hand_motor_mount", "wrist_cam", "left_finger", "right_finger", "right_eef", "left_eef"}
# (kp, kv, |force|) per joint family; the URDF values are placeholders too weak to track.
# kv is near critical damping: heavier damping lags the hand by kv/kp seconds of travel.
GAINS = {"j1": (400, 15, 80), "j2": (400, 15, 80), "j3": (250, 10, 60),
         "j4": (120, 5, 25), "j5": (120, 5, 25), "j6": (80, 4, 20),
         "gripper": (10, 1.0, 3.0)}
VMAX = np.array([0.25, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5])  # m/s for j0, rad/s otherwise


def part(body_name: str) -> str:
    return body_name.split("__")[-1]


def joint_family(joint_name: str) -> str:
    if "gripper" in joint_name:
        return "gripper"
    m = re.fullmatch(r"[lr](j\d)", joint_name)
    return m.group(1) if m else ""


def look_at(pos, target, up=(0, 0, 1)):
    """xyaxes for a MuJoCo camera at `pos` looking at `target`."""
    z = np.subtract(pos, target); z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return [*x, *y]


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def add_finger_pads(spec, grip_width=2 * BOX_HALF[1], vee=None):
    """Rubber pads on the claw tips, tilted so their faces are parallel when closed on an
    object `grip_width` across. With `vee` (radians) each pad is a V-groove of two faces
    opening toward the object instead of a flat face: a cylinder then rests on four
    lines and cannot pitch or roll in the grip, the way chess-robot fingers are made."""
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # home pose, fingers closed
    theta = math.asin((grip_width / 2 - PAD_BIAS) / HINGE_TO_PAD)  # claw angle at which the pads are parallel
    for side, prefix in (("right", ""), ("left", "l_")):
        eef = m.body(f"{side}_eef").id
        p_e, R_e = d.xpos[eef], d.xmat[eef].reshape(3, 3)
        for fname in (f"{prefix}left_finger__left_finger", f"{prefix}right_finger__right_finger"):
            fid = m.body(fname).id
            p_f, R_f = d.xpos[fid], d.xmat[fid].reshape(3, 3)
            s = 1.0 if (d.geom_xpos[m.body_geomadr[fid]] - p_e) @ R_e[:, 1] > 0 else -1.0
            centre = p_e + R_e @ (np.array(PAD_OFFSET) * [1, s, 1])
            phi = s * theta  # rotate about the claw hinge axis so the pad bottom swings inward
            rot = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
            R_pad = R_e @ rot
            body = next(b for b in spec.bodies if b.name == fname)
            if vee is None:
                faces = [(np.zeros(3), np.eye(3), PAD_HALF)]
            else:
                faces = []
                dy = -s   # local direction toward the object
                half_len = 0.005   # short faces: longer ones sweep into neighbouring objects
                for k in (-1.0, 1.0):
                    ang = math.atan2(dy * math.sin(vee), k * math.cos(vee))  # face direction, about local z
                    mid = np.array([k * half_len * math.cos(vee), dy * (0.002 + half_len * math.sin(vee)), 0])
                    normal = np.array([-k * math.sin(vee), dy * math.cos(vee), 0])  # toward the object
                    Rz = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
                    faces.append((mid - 0.002 * normal, Rz, (half_len, 0.002, PAD_HALF[2])))
            for i, (offset, Rz, size) in enumerate(faces):
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, (R_f.T @ R_pad @ Rz).ravel())
                pos = R_f.T @ (centre + R_pad @ offset - p_f)
                g = body.add_geom(name=f"{fname}_pad" + ("" if vee is None else f"{i}"), type=mujoco.mjtGeom.mjGEOM_BOX,
                                  size=list(size), pos=list(pos), quat=list(quat), rgba=[0.12, 0.12, 0.12, 1])
                g.mass = 0.002
                g.friction = [2.0, 0.02, 0.001]
                g.condim = 4
                g.contype = g.conaffinity = 3


def add_backstop(spec, side="right", half_w=0.004):
    """Two thin plates on the eef, one behind and one in front of the gripped object (a
    cage around the pads' span). A round shaft pinched between two flat pads that are not
    perfectly parallel is squeezed toward the wider end of the wedge and creeps along the
    pad faces until it leaves them - forwards or backwards depending on the toe angle; the
    stops limit that creep to ~3 mm and turn it into a stable rest against a plate."""
    body = next(b for b in spec.bodies if b.name == f"{side}_eef")
    geoms = []
    for name, x in ((f"{side}_backstop", PAD_OFFSET[0] - PAD_HALF[0] - 0.002), (f"{side}_frontstop", PAD_OFFSET[0] + PAD_HALF[0] + 0.002)):
        g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.001, half_w, PAD_HALF[2]], pos=[x, 0.0, PAD_OFFSET[2] + 0.003],
                          rgba=[0.12, 0.12, 0.12, 1])
        g.mass = 0.002
        g.friction = [1.0, 0.02, 0.001]
        g.contype = g.conaffinity = 3
        geoms.append(g)
    return geoms


# Contact bitmasks: bit 1 = structure (links, mast, table, ...), bit 2 = manipulation
# (objects, pads), bit 4 = claw hulls, bit 8 = floor. Objects touch pads, structure and the
# floor but not the claw hulls: the convex hull of a hollow fork-shaped claw is phantom
# material whose edge would otherwise lever a gripped object out of the pads. The robot
# does not touch the floor (it is welded or kinematically driven).
STRUCTURE, OBJECT, CLAW_HULL, FLOOR = 3, 10, 5, 8

# Position gains for the (kinematic) mobile base: (kp, kv).
BASE_GAINS = {"base_x": (20000, 600), "base_y": (20000, 600), "base_yaw": (3000, 100), "wheels": (50, 5)}
WHEEL_RADIUS = 0.0875


class RobotSpec:
    """The robot's MjSpec with pads, contact masks, actuators and cameras applied, plus the
    bookkeeping a scene needs to finalise it (see build_scene for the pattern)."""

    def __init__(self, grip_width=2 * BOX_HALF[1], mobile=False, vee=None, backstop=False):
        spec = mujoco.MjSpec.from_file(str(MODEL_XML))
        # Pyramidal soft contacts let a held object creep out of the fingers; elliptic
        # cones with stiff friction impedance are the standard MuJoCo grasping setup.
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.option.noslip_iterations = 3
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960
        add_finger_pads(spec, grip_width, vee)
        if backstop:
            add_backstop(spec, "right")
        self.spec = spec
        self.robot_bodies = {b.name for b in spec.bodies if b.name != "world"}
        self.arm_geoms, self.env_geoms, self.soft_geoms = [], [], []
        for body in spec.bodies:
            for g in body.geoms:
                if body.name == "world":
                    g.contype = g.conaffinity = FLOOR
                elif part(body.name) in ("left_finger", "right_finger") and "_pad" not in g.name:
                    g.contype = g.conaffinity = CLAW_HULL
                else:
                    g.contype = g.conaffinity = STRUCTURE
                if part(body.name) in ARM_PARTS:
                    self.arm_geoms.append(g)
                elif body.name != "world":
                    self.env_geoms.append(g)
            if part(body.name) in ARM_PARTS:
                body.gravcomp = 1.0  # the URDF's placeholder servos otherwise sag under the arm's weight

        for bname, sname in (("right_eef", "right_eef_site"), ("left_eef", "left_eef_site")):
            body = next(b for b in spec.bodies if b.name == bname)
            body.add_site(name=sname, pos=[0, 0, 0], size=[0.004, 0.004, 0.004], rgba=[1, 0.3, 0, 0.6])

        # The mimic finger is only coupled through a soft equality, which leaves it
        # limp when squeezing: drive both fingers and stiffen the coupling.
        for joint in spec.joints:
            if joint.name.endswith("_right_gripper"):
                act = spec.add_actuator(name=joint.name, target=joint.name,
                                        trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                                        biastype=mujoco.mjtBias.mjBIAS_AFFINE)
                act.ctrlrange = joint.range
                act.ctrllimited = True
        for eq in spec.equalities:
            eq.solref = [0.005, 1.0]
        for act in spec.actuators:
            fam = joint_family(act.target)
            if fam in GAINS:
                kp, kv, _ = GAINS[fam]
                act.gainprm[0] = kp
                act.biasprm[1] = -kp
                act.biasprm[2] = -kv
        for joint in spec.joints:
            fam = joint_family(joint.name)
            if fam in GAINS:
                joint.actfrcrange = [-GAINS[fam][2], GAINS[fam][2]]

        if mobile:
            self._add_mobile_base()
        self._add_head_camera()
        self._add_hand_camera()

    def _add_hand_camera(self):
        """A camera on the right hand (where the URDF's wrist_cam sits), looking down along
        the fingers: the eef frame has z toward the fingertips, so the camera's -z (its view
        axis) is the eef +z, i.e. a 180-degree turn about x."""
        body = next(b for b in self.spec.bodies if b.name == "right_eef")
        body.add_camera(name="hand_cam", pos=[-0.035, 0.0, -0.055], quat=[0, 1, 0, 0], fovy=80)

    def _add_mobile_base(self):
        """Planar x / y / yaw joints on the root body (position-driven, so the base is
        kinematic) and a cosmetic hinge that spins the wheels with the distance driven."""
        root = next(b for b in self.spec.bodies if b.name == "root")
        for name, jtype, axis in (("base_x", mujoco.mjtJoint.mjJNT_SLIDE, [1, 0, 0]),
                                  ("base_y", mujoco.mjtJoint.mjJNT_SLIDE, [0, 1, 0]),
                                  ("base_yaw", mujoco.mjtJoint.mjJNT_HINGE, [0, 0, 1])):
            root.add_joint(name=name, type=jtype, axis=axis)
        m = self.spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        tire = m.body("right_wheel_tire__right_wheel_tire").id
        p, R = d.xpos[tire], d.xmat[tire].reshape(3, 3)
        centre = d.geom_xpos[m.body_geomadr[tire]]
        tire_body = next(b for b in self.spec.bodies if b.name == "right_wheel_tire__right_wheel_tire")
        # The left tire is a fixed child of the right one and lies on the same axle, so one
        # hinge through the right tire's centre spins both.
        tire_body.add_joint(name="wheels", type=mujoco.mjtJoint.mjJNT_HINGE,
                            axis=list(R.T @ np.array([0, 1.0, 0])), pos=list(R.T @ (centre - p)))
        for name, (kp, kv) in BASE_GAINS.items():
            act = self.spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                         gaintype=mujoco.mjtGain.mjGAIN_FIXED, biastype=mujoco.mjtBias.mjBIAS_AFFINE)
            act.gainprm[0] = kp
            act.biasprm[1] = -kp
            act.biasprm[2] = -kv

    def _add_head_camera(self):
        """Mount head_cam on the head body so it travels with the robot."""
        m = self.spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        head = m.body("head__head__head__head").id
        p_h, R_h = d.xpos[head], d.xmat[head].reshape(3, 3)
        xy = look_at(CAM_POS, CAM_LOOK)
        R_cam = np.stack([xy[:3], xy[3:], np.cross(xy[:3], xy[3:])], axis=1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_h.T @ R_cam).ravel())
        body = next(b for b in self.spec.bodies if b.name == "head__head__head__head")
        body.add_camera(name="head_cam", pos=list(R_h.T @ (np.array(CAM_POS) - p_h)), quat=list(quat), fovy=60)

    def add_table(self, center=TABLE_CENTER, half=TABLE_HALF, top_z=TABLE_TOP):
        wb = self.spec.worldbody
        table = wb.add_body(name="table", pos=[center[0], center[1], top_z - 0.02])
        top = table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[half[0], half[1], 0.02], rgba=[0.55, 0.40, 0.28, 1])
        top.contype = top.conaffinity = STRUCTURE
        self.env_geoms.append(top)
        leg_h = (top_z - 0.04) / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                leg = table.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, leg_h, 0],
                                     pos=[sx * (half[0] - 0.03), sy * (half[1] - 0.03), leg_h - (top_z - 0.02)],
                                     rgba=[0.35, 0.25, 0.18, 1])
                leg.contype = leg.conaffinity = 0
        return table

    def finalize(self):
        """Compile: (physics model, planning model with margins, excluded body pairs).

        By-design overlaps are excluded so they never count as collisions: pairs that
        touch at the rest pose (finger vs finger, ...) and, because the shoulder carriage
        clamps around the mast, carriage-vs-mast pairs found anywhere along the slide."""
        spec = self.spec
        for g in self.arm_geoms:
            g.margin = PLAN_MARGIN
        for g in self.env_geoms:
            g.margin = ENV_MARGIN
        for g in self.soft_geoms:
            g.margin = 0.002
        m = spec.compile()
        d = mujoco.MjData(m)
        carriage = {"shoulder", "shoulder_cover", "shoulder_knuckle"}
        robot_bodies = self.robot_bodies

        def is_base(b):
            return b in robot_bodies and part(b) not in ARM_PARTS

        excluded = set()
        slides = [m.joint(n).id for n in ("rj0", "lj0")]
        for s in np.linspace(0, 1, 30):
            mujoco.mj_resetData(m, d)
            for j in slides:
                d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j, 0] * s
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                arm_pair = part(b1) in ARM_PARTS or part(b2) in ARM_PARTS
                slide_pair = (part(b1) in carriage and is_base(b2)) or (part(b2) in carriage and is_base(b1))
                if (s == 0 and arm_pair) or slide_pair:
                    excluded.add(tuple(sorted((b1, b2))))
        # ...and finger-vs-hand pairs anywhere in the gripper's own travel.
        grips = [m.joint(n).id for n in ("right_left_gripper", "right_right_gripper",
                                         "left_left_gripper", "left_right_gripper")]
        for s in np.linspace(0, 1, 11):
            mujoco.mj_resetData(m, d)
            for j in grips:
                d.qpos[m.jnt_qposadr[j]] = s
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                same_hand = b1.startswith("l_") == b2.startswith("l_")
                if part(b1) in GRIPPER_PARTS and part(b2) in GRIPPER_PARTS and same_hand:
                    excluded.add(tuple(sorted((b1, b2))))
        # ...and parent/child arm links joined by a hinge: their joint limits keep them apart
        # physically, but the planning margins make them look like a collision when the
        # elbow folds (shoulder vs bicep came within 12 mm on the way to the look pose).
        names = [m.body(i).name for i in range(m.nbody)]
        covers = {n: [c for c in names if c.startswith(n.split("__")[0] + "_cover")] for n in names}
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            child = m.body(m.jnt_bodyid[j])
            up = child
            for _ in range(2):                       # parent and grandparent (the shoulder
                up = m.body(up.parentid[0])          # carriage vs the bicep two joints down)
                if part(child.name) in ARM_PARTS and part(up.name) in ARM_PARTS:
                    for a in [up.name] + covers.get(up.name, []):
                        for b in [child.name] + covers.get(child.name, []):
                            excluded.add(tuple(sorted((a, b))))
        for k, (b1, b2) in enumerate(sorted(excluded)):
            spec.add_exclude(name=f"bydesign{k}", bodyname1=b1, bodyname2=b2)
        plan_model = spec.compile()
        for g in self.arm_geoms + self.env_geoms + self.soft_geoms:
            g.margin = 0.0
        model = spec.compile()
        return model, plan_model, sorted(excluded)


def build_scene():
    """The pick-and-place scene: (physics model, planning model with margins, excluded pairs)."""
    rs = RobotSpec()
    wb = rs.spec.worldbody
    rs.add_table()
    tray = wb.add_geom(name="tray", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.07, 0.07, 0.0025],
                       pos=[TRAY[0], TRAY[1], TABLE_TOP + 0.0025], rgba=[0.82, 0.82, 0.88, 1])
    tray.contype = tray.conaffinity = STRUCTURE
    rs.env_geoms.append(tray)

    for name, (x, y, rgba) in BOXES.items():
        b = wb.add_body(name=f"box_{name}", pos=[x, y, TABLE_TOP])  # origin at the bottom face
        b.add_freejoint()
        g = b.add_geom(name=f"box_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=list(BOX_HALF), pos=[0, 0, BOX_HALF[2]], rgba=list(rgba))
        g.mass = 0.03
        g.friction = [1.5, 0.005, 0.0001]
        g.contype = g.conaffinity = OBJECT
        rs.env_geoms.append(g)

    demo_pos = (1.5, -1.3, 1.35)
    wb.add_camera(name="demo", pos=list(demo_pos), xyaxes=look_at(demo_pos, (0.3, -0.05, 0.9)), fovy=45)
    return rs.finalize()


# --------------------------------------------------------------------------- #
# vision
# --------------------------------------------------------------------------- #
HSV_RANGES = {
    "red": [((0, 110, 50), (8, 255, 255)), ((170, 110, 50), (180, 255, 255))],
    "green": [((40, 110, 50), (85, 255, 255))],
}


def detect_boxes(rgb):
    """Return {colour: (u, v, area, contour)} for every colour blob found."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = {}
    for colour, ranges in HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 30]
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        mom = cv2.moments(c)
        found[colour] = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"], cv2.contourArea(c), c)
    return found


class Camera:
    """Pinhole model of a MuJoCo fixed camera (looks down -z, x right, y up)."""

    def __init__(self, model, data, name, width, height):
        self.id = model.camera(name).id
        self.width, self.height = width, height
        self.fy = (height / 2) / math.tan(math.radians(model.cam_fovy[self.id]) / 2)
        self.R = data.cam_xmat[self.id].reshape(3, 3).copy()
        self.origin = data.cam_xpos[self.id].copy()

    def project(self, pts):
        pc = (np.asarray(pts) - self.origin) @ self.R
        return np.stack([self.width / 2 + self.fy * pc[:, 0] / -pc[:, 2],
                         self.height / 2 - self.fy * pc[:, 1] / -pc[:, 2]], axis=1)

    def ray_to_plane(self, u, v, z_plane):
        ray = self.R @ np.array([(u - self.width / 2) / self.fy, -(v - self.height / 2) / self.fy, -1.0])
        return self.origin + ray * (z_plane - self.origin[2]) / ray[2]


CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def predicted_centroid(cam, xy, scale=4):
    """Silhouette centroid (px) of a box standing at xy, rasterised at `scale`x resolution."""
    canvas = np.zeros((cam.height * scale, cam.width * scale), np.uint8)
    corners = np.array([xy[0], xy[1], TABLE_TOP + BOX_HALF[2]]) + CORNERS * BOX_HALF
    uv = cam.project(corners) * scale
    cv2.fillConvexPoly(canvas, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    m = cv2.moments(canvas, binaryImage=True)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) / scale


def locate(cam, uv_obs, iters=6):
    """Box (x, y) whose predicted silhouette centroid matches the observed one (Gauss-Newton)."""
    xy = cam.ray_to_plane(*uv_obs, TABLE_TOP + BOX_HEIGHT / 2)[:2]
    for _ in range(iters):
        r = uv_obs - predicted_centroid(cam, xy)
        if np.linalg.norm(r) < 0.05:
            break
        J = np.zeros((2, 2))
        for k, d in enumerate(np.eye(2) * 0.002):
            J[:, k] = (predicted_centroid(cam, xy + d) - predicted_centroid(cam, xy - d)) / 0.004
        xy = xy + np.linalg.solve(J, r)
    return xy


def run_vision(model, data, colour, out_path):
    """Render the head camera, find the requested box and return its world (x, y)."""
    width, height = 1280, 960
    cam = Camera(model, data, "head_cam", width, height)
    renderer = mujoco.Renderer(model, height, width)
    mjcam = mujoco.MjvCamera()
    mjcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    mjcam.fixedcamid = cam.id
    renderer.update_scene(data, mjcam)
    rgb = renderer.render().copy()
    renderer.close()

    found = detect_boxes(rgb)
    if colour not in found:
        raise SystemExit(f"vision: no {colour} box in view (found: {sorted(found)})")
    world = {c: locate(cam, np.array(f[:2])) for c, f in found.items()}

    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for c, (u, v, area, contour) in found.items():
        chosen = c == colour
        bgr = (0, 255, 255) if chosen else (200, 200, 200)
        cv2.drawContours(canvas, [contour], -1, bgr, 2)
        cv2.circle(canvas, (int(u), int(v)), 5, bgr, -1)
        label = f"{c} {'<- TARGET' if chosen else ''} ({world[c][0]:.3f}, {world[c][1]:.3f})"
        cv2.putText(canvas, label, (int(u) + 12, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"head_cam  target: {colour}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)

    truth = data.xpos[model.body(f"box_{colour}").id]
    err = np.linalg.norm(world[colour] - truth[:2])
    print(f"vision: found {sorted(found)}; {colour} at px=({found[colour][0]:.0f},{found[colour][1]:.0f}) "
          f"-> world ({world[colour][0]:.4f}, {world[colour][1]:.4f}), error {err * 1000:.1f} mm; wrote {out_path.name}")
    return world[colour]


# --------------------------------------------------------------------------- #
# kinematics / planning
# --------------------------------------------------------------------------- #
class Arm:
    def __init__(self, model, side="right"):
        p = "r" if side == "right" else "l"
        self.joints = [f"{p}j{i}" for i in range(7)]
        jid = [model.joint(n).id for n in self.joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.lo = model.jnt_range[jid, 0].copy()
        self.hi = model.jnt_range[jid, 1].copy()
        self.act = np.array([model.actuator(n).id for n in self.joints])
        self.grip_act = [model.actuator(f"{side}_{f}_gripper").id for f in ("left", "right")]
        self.site = model.site(f"{side}_eef_site").id

    def fk(self, model, data, q):
        data.qpos[self.qadr] = q
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, model, data, pos, rot, q0, z_only=False, iters=300):
        """Damped least squares to `pos` (3) and `rot` (3x3); z_only constrains the approach axis only.

        Joints that would be pushed past a limit are locked for that step and the
        step is re-solved with the rest, so clipping cannot stall the iteration."""
        q = q0.copy()
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(iters):
            p, R = self.fk(model, data, q)
            ep = pos - p
            if z_only:
                er = np.cross(R[:, 2], rot[:, 2])
            else:
                er = 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(ep) < 1e-3 and np.linalg.norm(er) < 5e-3:
                return q
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacSite(model, data, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dofadr], jacr[:, self.dofadr]])
            e = np.concatenate([ep, er])
            active = np.ones(7, bool)
            for _ in range(7):
                Ja = J[:, active]
                dq = np.zeros(7)
                dq[active] = Ja.T @ np.linalg.solve(Ja @ Ja.T + 0.05 ** 2 * np.eye(6), e)
                n = np.linalg.norm(dq)
                if n > 0.3:
                    dq *= 0.3 / n
                blocked = active & (((q + dq < self.lo) & (dq < 0)) | ((q + dq > self.hi) & (dq > 0)))
                if not blocked.any():
                    break
                active &= ~blocked
            q = np.clip(q + dq, self.lo, self.hi)
        return None


class Planner:
    """Collision checking + RRT-Connect on the planning model (arm hulls inflated by PLAN_MARGIN)."""

    def __init__(self, plan_model, arm, robot_bodies, rng):
        self.m = plan_model
        self.d = mujoco.MjData(plan_model)
        self.arm = arm
        self.robot_bodies = robot_bodies
        self.rng = rng
        self.checks = 0

    def sync(self, data):
        self.d.qpos[:] = data.qpos
        self.d.qvel[:] = 0

    def in_collision(self, q, ignore_body=None, near_ok=()):
        """True if the arm at q touches anything (within the planning margins).

        Contacts with a body in `ignore_body` are skipped; for geoms in `near_ok`
        only real penetration counts, so a slow vertical approach may come within
        the margin of the surface it is heading for."""
        self.checks += 1
        self.d.qpos[self.arm.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 in self.robot_bodies or b2 in self.robot_bodies:
                if ignore_body is not None and ignore_body in (b1, b2):
                    continue
                if c.dist > 0 and (c.geom1 in near_ok or c.geom2 in near_ok):
                    continue
                return True
        return False

    def edge_ok(self, qa, qb, ignore_body=None, res=0.04, near_ok=()):
        n = max(2, int(math.ceil(np.linalg.norm(qb - qa) / res)))
        return all(not self.in_collision(qa + (qb - qa) * k / n, ignore_body, near_ok) for k in range(1, n + 1))

    def cartesian(self, p_from, p_to, rot, q_seed, ignore_body=None, step=0.02, near_ok=()):
        """Joint path that tracks the straight segment p_from -> p_to at fixed orientation."""
        n = max(1, int(math.ceil(np.linalg.norm(p_to - p_from) / step)))
        path = [np.asarray(q_seed, float)]
        for k in range(1, n + 1):
            p = p_from + (p_to - p_from) * (k / n)
            q = self.arm.ik(self.m, self.d, p, rot, path[-1], iters=600)
            if q is None:  # restart from random seeds, but only accept a nearby branch
                q = self.solve_ik(p, rot, path[-1], ignore_body)
                if np.max(np.abs(q - path[-1])) > 0.3:
                    raise RuntimeError(f"cartesian move: no continuous IK at {np.round(p, 3)}")
            if self.in_collision(q, ignore_body, near_ok) or not self.edge_ok(path[-1], q, ignore_body, near_ok=near_ok):
                raise RuntimeError(f"cartesian move: collision near {np.round(p, 3)}")
            path.append(q)
        return path

    def solve_ik(self, pos, rot, seed, ignore_body=None, tries=40):
        """Collision-free IK with random restarts; relaxes to approach-axis-only in the last third."""
        for k in range(tries):
            q0 = seed if k == 0 else self.rng.uniform(self.arm.lo, self.arm.hi)
            if k:
                q0[0] = self.rng.uniform(-0.8, 0.0)
            q = self.arm.ik(self.m, self.d, pos, rot, q0, z_only=k >= 2 * tries // 3)
            if q is not None and not self.in_collision(q, ignore_body):
                return q
        raise RuntimeError(f"no collision-free IK solution for {np.round(pos, 3)}")

    def plan(self, q_start, q_goal, ignore_body=None, max_iters=4000, step=0.15):
        t0 = time.time()
        self.checks = 0
        if self.in_collision(q_start, ignore_body):
            raise RuntimeError("planner: start configuration is in collision")
        if self.in_collision(q_goal, ignore_body):
            raise RuntimeError("planner: goal configuration is in collision")
        if self.edge_ok(q_start, q_goal, ignore_body):
            print(f"plan: straight line ok ({self.checks} checks, {time.time() - t0:.1f}s)")
            return [q_start, q_goal]

        lo, hi = self.arm.lo.copy(), self.arm.hi.copy()
        lo[0] = -0.9
        trees = [[(q_start, -1)], [(q_goal, -1)]]  # trees[0] grows from start

        def extend(tree, target):
            i = int(np.argmin([np.linalg.norm(n[0] - target) for n in tree]))
            qn = tree[i][0]
            dist = np.linalg.norm(target - qn)
            qnew = target if dist <= step else qn + (target - qn) * (step / dist)
            if self.edge_ok(qn, qnew, ignore_body):
                tree.append((qnew, i))
                return ("reached" if dist <= step else "advanced"), len(tree) - 1
            return "trapped", None

        def trace(tree, i):
            out = []
            while i != -1:
                out.append(tree[i][0])
                i = tree[i][1]
            return out

        a = 0
        for it in range(max_iters):
            b = 1 - a
            status, ia = extend(trees[a], self.rng.uniform(lo, hi))
            if status != "trapped":
                while True:
                    status, ib = extend(trees[b], trees[a][ia][0])
                    if status != "advanced":
                        break
                if status == "reached":
                    pa, pb = trace(trees[a], ia), trace(trees[b], ib)
                    path = pa[::-1] + pb if a == 0 else pb[::-1] + pa
                    path = self.shortcut(path, ignore_body)
                    print(f"plan: RRT-Connect {it + 1} iters, {len(path)} waypoints, "
                          f"{self.checks} checks, {time.time() - t0:.1f}s")
                    return path
            a = b
        raise RuntimeError("planner: no path found")

    def shortcut(self, path, ignore_body, iters=150):
        path = list(path)
        for _ in range(iters):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.edge_ok(path[i], path[j], ignore_body):
                path = path[:i + 1] + path[j:]
        return path


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
class Runner:
    """Steps the physics, drives the actuators, records video and monitors contacts."""

    def __init__(self, model, data, arm, robot_bodies, viewer=None, recorder=None):
        self.m, self.d, self.arm = model, data, arm
        self.robot_bodies = robot_bodies
        self.gripper_bodies = {i for i in range(model.nbody)
                               if part(model.body(i).name) in GRIPPER_PARTS and not model.body(i).name.startswith("l_")}
        self.viewer, self.recorder = viewer, recorder
        self.held = None
        self.illegal_steps = 0
        self.illegal_pairs = {}
        self.q_cmd = np.zeros(7)
        self.wall0 = time.time() - data.time
        self.frame_every = 8
        self.k = 0

    def _monitor(self):
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if b1 not in self.robot_bodies and b2 not in self.robot_bodies:
                continue
            if self.held is not None and self.held in (b1, b2) and (b1 in self.gripper_bodies or b2 in self.gripper_bodies):
                continue
            self.illegal_steps += 1
            key = tuple(sorted((self.m.body(b1).name, self.m.body(b2).name)))
            self.illegal_pairs[key] = self.illegal_pairs.get(key, 0) + 1

    def step(self):
        mujoco.mj_step(self.m, self.d)
        self._monitor()
        if self.recorder:
            self.recorder.tick(self.d)
        self.k += 1
        if self.viewer and self.k % self.frame_every == 0:
            self.viewer.sync()
            lag = self.wall0 + self.d.time - time.time()
            if lag > 0:
                time.sleep(lag)

    def hold(self, seconds):
        for _ in range(int(seconds / self.m.opt.timestep)):
            self.step()

    def set_gripper(self, open_amount, settle=0.8, ramp=0.4):
        """Ramp the finger targets so the pads neither slam shut nor flick the box on release."""
        start = float(self.d.ctrl[self.arm.grip_act[0]])
        steps = int(ramp / self.m.opt.timestep)
        for k in range(steps):
            for a in self.arm.grip_act:
                self.d.ctrl[a] = start + (open_amount - start) * (k + 1) / steps
            self.step()
        self.hold(settle)

    def move(self, path, speed=1.0):
        """Track `path` (list of joint vectors) with cosine ease-in/out at constant weighted speed."""
        path = [np.asarray(p, float) for p in path]
        durs = [max(0.05, np.max(np.abs(b - a) / (VMAX * speed))) for a, b in zip(path, path[1:])]
        total = sum(durs)
        cum = np.concatenate([[0], np.cumsum(durs)])
        t_end = self.d.time + total
        while self.d.time < t_end:
            u = 1 - (t_end - self.d.time) / total
            s = (0.5 - 0.5 * math.cos(math.pi * u)) * total
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(durs) - 1)
            f = (s - cum[i]) / durs[i]
            self.q_cmd = path[i] + (path[i + 1] - path[i]) * f
            self.d.ctrl[self.arm.act] = self.q_cmd
            self.step()
        self.q_cmd = path[-1]
        self.d.ctrl[self.arm.act] = self.q_cmd

    def eef_pos(self):
        return self.d.site_xpos[self.arm.site].copy()

    def servo(self, planner, pos, rot, ignore_body=None, near_ok=(), iters=3):
        """Cancel steady-state tracking error by offsetting the IK target."""
        offset = np.zeros(3)
        for _ in range(iters):
            self.hold(0.4)
            err = pos - self.eef_pos()
            if np.linalg.norm(err) < 0.003:
                break
            offset += err
            planner.sync(self.d)
            q = self.arm.ik(planner.m, planner.d, pos + offset, rot, self.q_cmd)
            if q is None or planner.in_collision(q, ignore_body, near_ok):
                break
            self.move([self.q_cmd, q], speed=0.4)
        self.hold(0.3)
        return np.linalg.norm(pos - self.eef_pos())


class Recorder:
    """Offscreen renderer for the 'demo' camera: optional mp4 stream plus PNG snapshots."""

    def __init__(self, model, fps, video_path=None, cam_name="demo"):
        self.renderer = mujoco.Renderer(model, 720, 1280)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = model.camera(cam_name).id
        self.writer = None
        if video_path:
            import imageio
            self.writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8)
        self.fps, self.next_t, self.frames = fps, 0.0, 0

    def tick(self, data):
        if self.writer and data.time >= self.next_t:
            self.renderer.update_scene(data, self.cam)
            self.writer.append_data(self.renderer.render())
            self.next_t += 1 / self.fps
            self.frames += 1

    def snapshot(self, data, path):
        self.renderer.update_scene(data, self.cam)
        cv2.imwrite(str(path), cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer:
            self.writer.close()
        self.renderer.close()


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
def pick_and_place(model, data, arm, planner, runner, colour, target_xy, snap=None):
    box = model.body(f"box_{colour}").id
    surfaces = {model.geom("table_top").id, model.geom("tray").id}  # approached vertically at slow speed
    q_home = np.zeros(7)
    planner.sync(data)
    # Fingers straight down, closing along y, pads square to the axis-aligned boxes. (The
    # URDF's eef frame is yawed ~3 deg at home; using it would leave the pads touching
    # the box along a single edge, about which it can pitch out.)
    R_grasp = np.diag([1.0, -1.0, -1.0])
    pad_shift = -R_grasp[:, 0] * PAD_OFFSET[0]  # put the box's centre under the pads, not the eef frame
    p_grasp = np.array([target_xy[0], target_xy[1], TABLE_TOP + GRASP_HEIGHT]) + pad_shift
    p_pre = p_grasp + [0, 0, APPROACH]
    p_place = np.array([TRAY[0], TRAY[1], TRAY_TOP + GRASP_HEIGHT + 0.006]) + pad_shift
    p_preplace = p_place + [0, 0, APPROACH]
    z0 = data.xpos[box, 2]

    # At home the hands hang beside the mast and an open inner finger would hit
    # it, so the gripper stays closed until the hand is over the table.
    print(f"phase 1: reach above the {colour} box at {np.round(p_pre, 3)}")
    q_pre = planner.solve_ik(p_pre, R_grasp, q_home)
    runner.move(planner.plan(q_home, q_pre))
    snap and snap("1_pregrasp")

    # With something in hand every move is a straight Cartesian line at fixed
    # orientation: the block hangs from two edge contacts and swings if the
    # wrist wanders, as a joint-space interpolation would let it.
    print("phase 2: open gripper, descend and grasp")
    runner.set_gripper(1.0)
    planner.sync(data)
    if planner.in_collision(q_pre):
        raise RuntimeError("open gripper collides at the pre-grasp pose")
    descent = planner.cartesian(p_pre, p_grasp, R_grasp, q_pre, ignore_body=box, near_ok=surfaces)
    runner.move(descent, speed=0.4)
    err = runner.servo(planner, p_grasp, R_grasp, ignore_body=box, near_ok=surfaces)
    print(f"   gripper at {np.round(runner.eef_pos(), 3)}, {err * 1000:.1f} mm from target")
    runner.held = box
    runner.set_gripper(0.0, settle=1.0)
    snap and snap("2_grasp")

    print("phase 3: lift")
    runner.move([runner.q_cmd] + descent[::-1], speed=0.4)  # retrace the descent
    runner.hold(0.5)
    lifted = data.xpos[box, 2] - z0
    print(f"   box lifted {lifted * 1000:.0f} mm -> {'grasp OK' if lifted > 0.06 else 'GRASP FAILED'}")
    if lifted <= 0.06:
        raise RuntimeError("the box slipped out of the gripper")

    print(f"phase 4: carry to the tray at {np.round(p_preplace, 3)}")
    planner.sync(data)
    try:
        carry = planner.cartesian(runner.eef_pos(), p_preplace, R_grasp, runner.q_cmd, ignore_body=box)
        print(f"   straight-line carry, {len(carry) - 1} waypoints")
    except RuntimeError as e:
        print(f"   {e}; falling back to RRT")
        q_prep = planner.solve_ik(p_preplace, R_grasp, runner.q_cmd, ignore_body=box)
        carry = planner.plan(runner.q_cmd, q_prep, ignore_body=box)
    runner.move(carry, speed=0.4 if len(carry) > 2 else 0.25)
    runner.hold(0.5)
    snap and snap("3_carry")
    in_hand = data.xpos[box, 2] > z0 + 0.06 and np.linalg.norm(data.xpos[box, :2] - runner.eef_pos()[:2]) < 0.03
    if not in_hand:
        raise RuntimeError(f"the box was dropped during the carry (now at {np.round(data.xpos[box], 3)})")

    print("phase 5: lower, release, retreat")
    planner.sync(data)
    lowering = planner.cartesian(runner.eef_pos(), p_place, R_grasp, runner.q_cmd, ignore_body=box, near_ok=surfaces)
    runner.move(lowering, speed=0.3)
    runner.servo(planner, p_place, R_grasp, ignore_body=box, near_ok=surfaces)
    runner.hold(0.5)
    runner.set_gripper(1.0)
    runner.held = None
    runner.move([runner.q_cmd] + lowering[::-1], speed=0.4)
    snap and snap("4_released")

    print("phase 6: close gripper, return home")
    runner.set_gripper(0.0)
    planner.sync(data)
    runner.move(planner.plan(runner.q_cmd, q_home))
    runner.hold(0.5)
    snap and snap("5_home")

    final = data.xpos[box]
    upright = data.xmat[box].reshape(3, 3)[2, 2] > 0.95
    on_tray = upright and np.linalg.norm(final[:2] - TRAY) < 0.06 and abs(final[2] - TRAY_TOP) < 0.01
    print(f"done: {colour} box at {np.round(final, 3)} {'upright' if upright else 'TIPPED'} -> "
          f"{'on the tray' if on_tray else 'NOT on the tray'}")
    print(f"illegal contacts during execution: {runner.illegal_steps} steps"
          + (f" {runner.illegal_pairs}" if runner.illegal_pairs else " (none)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--color", choices=list(BOXES), default="red", help="which box to pick (default red)")
    ap.add_argument("--record", action="store_true", help="write pick_place.mp4 instead of opening a window")
    ap.add_argument("--headless", action="store_true", help="no window, no video (fast check)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap-dir", type=pathlib.Path, help="save a PNG at each phase (record/headless only)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model, plan_model, excluded = build_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    kinds = sorted({f"{part(a)}/{part(b)}" for a, b in excluded})
    print(f"scene: {model.nbody} bodies, {model.ngeom} colliding hulls; {len(excluded)} by-design "
          f"contact pairs excluded ({len(kinds)} kinds, e.g. {', '.join(kinds[:4])}, ...)")

    target_xy = run_vision(model, data, args.color, HERE / f"vision_{args.color}.png")

    arm = Arm(model, "right")
    robot_bodies = {i for i in range(model.nbody) if part(model.body(i).name) in ARM_PARTS}
    planner = Planner(plan_model, arm, robot_bodies, rng)

    if args.record or args.headless:
        video = HERE / "pick_place.mp4" if args.record else None
        recorder = Recorder(model, args.fps, video) if (video or args.snap_dir) else None
        snap = None
        if args.snap_dir:
            args.snap_dir.mkdir(parents=True, exist_ok=True)
            snap = lambda tag: recorder.snapshot(data, args.snap_dir / f"{tag}.png")  # noqa: E731
        runner = Runner(model, data, arm, robot_bodies, recorder=recorder)
        try:
            pick_and_place(model, data, arm, planner, runner, args.color, target_xy, snap)
        finally:
            if recorder:
                recorder.close()
            if video:
                print(f"wrote {video} ({recorder.frames} frames)")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, -0.05, 0.9]
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -18
        runner = Runner(model, data, arm, robot_bodies, viewer=viewer)
        pick_and_place(model, data, arm, planner, runner, args.color, target_xy)
        print("finished - close the window to exit")
        while viewer.is_running():
            runner.step()


if __name__ == "__main__":
    main()
