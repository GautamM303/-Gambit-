"""Build a simulation-ready MuJoCo model (chopped_urdf_v2.xml) from the URDF.

The Onshape export is visual-only: no actuators, no collision geometry, and
inertials that were never scaled by material density (total mass ~6 g).
This script:
  * loads the URDF through MuJoCo's MjSpec URDF importer
  * recomputes link mass/inertia from the mesh convex hulls with per-part
    densities (the base is welded to the world, so only the arms matter)
  * adds position-controlled actuators on every arm joint (the gripper's
    mimicking finger is driven by the equality constraint the importer
    creates from the URDF <mimic> tag)
  * adds a floor, lights and a default camera
  * writes sim/chopped_urdf_v2.xml (mesh paths relative to ../meshes)

Run:  python build_mjcf.py
"""
import pathlib
import re
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
URDF = PKG / "urdf" / "chopped_urdf_v2.urdf"
MESHDIR = PKG / "meshes"
OUT = HERE / "chopped_urdf_v2.xml"

# kg/m^3 applied to each link's convex hull. Printed parts are hollow shells,
# so their effective density is well below solid PLA (~1240).
DENSITY = {
    "shoulder": 500, "shoulder_cover": 150, "shoulder_knuckle": 400,
    "bicep": 400, "bicep_cover": 100, "forearm": 350, "forearm_cover": 100,
    "forearm_rotation": 400, "wrist_knuckle": 400, "hand": 350,
    "hand_motor_mount": 400, "wrist_cam": 300,
    "left_finger": 300, "right_finger": 300,
}

# (kp, kv) for the position actuators, by joint family.
GAINS = {
    "j0": (3000.0, 300.0),   # prismatic mast slide carries the whole arm
    "j1": (150.0, 15.0), "j2": (150.0, 15.0), "j3": (80.0, 8.0),
    "j4": (30.0, 3.0), "j5": (30.0, 3.0), "j6": (20.0, 2.0),
    "gripper": (5.0, 0.5),
}


def link_part(link_name: str) -> str:
    """'l_forearm_cover__forearm_cover' -> 'forearm_cover'."""
    return link_name.split("__")[-1]


def joint_family(joint_name: str) -> str:
    if "gripper" in joint_name:
        return "gripper"
    m = re.fullmatch(r"[lr](j\d)", joint_name)
    return m.group(1) if m else "j6"


def main() -> None:
    text = URDF.read_text()
    text = text.replace("package://chopped_urdf_v2/meshes/", "")
    text = text.replace(
        '<robot name="chopped_urdf_v2">',
        '<robot name="chopped_urdf_v2">\n'
        f'<mujoco><compiler meshdir="{MESHDIR}" balanceinertia="true" '
        'discardvisual="false" strippath="false" fusestatic="false"/></mujoco>',
    )
    spec = mujoco.MjSpec.from_string(text)
    spec.modelname = "chopped_urdf_v2"
    # Keep fixed-joint links as (static) child bodies: fusing them into their
    # parents drops the fixed-joint offsets when the spec is written back out.
    spec.compiler.fusestatic = False
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    spec.compiler.balanceinertia = True
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # --- masses: give every geom a density; the compiler then derives
    # body mass and inertia from the hull. Base links are static so their
    # values are irrelevant.
    for body in spec.bodies:
        part = link_part(body.name)
        for geom in body.geoms:
            geom.density = DENSITY.get(part, 300.0)
            # URDF visuals are imported as non-colliding; keep it that way so
            # the tightly packed arm links don't fight each other at rest.
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = 1 if part in DENSITY else 2

    # --- actuators: one position servo per non-mimic joint. The importer
    # already turns each URDF <mimic> into a joint equality constraint, so the
    # mimicking finger just needs no actuator of its own.
    mimics = {}  # child joint -> parent joint, from the URDF <mimic> tags
    for m in re.finditer(
        r'<joint name="([^"]+)"[^>]*>(.*?)</joint>', URDF.read_text(), re.S):
        mm = re.search(r'<mimic joint="([^"]+)"', m.group(2))
        if mm:
            mimics[m.group(1)] = mm.group(1)

    for joint in spec.joints:
        if joint.name in mimics:
            continue
        fam = joint_family(joint.name)
        kp, kv = GAINS[fam]
        if fam == "j0":
            # URDF effort="10" is a placeholder; 10 N can't hold the arm up.
            joint.actfrcrange = [-300, 300]
        act = spec.add_actuator(name=joint.name)
        act.target = joint.name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[1] = -kp
        act.biasprm[2] = -kv
        act.ctrlrange = joint.range
        act.ctrllimited = True
        joint.damping[0] = 0.5

    # --- scene dressing.
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           width=512, height=512,
                           rgb1=[0.2, 0.25, 0.3], rgb2=[0.3, 0.35, 0.4])
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [8, 8]
    mat.reflectance = 0.1
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[3, 3, 0.05], material="grid")
    floor.contype = 1
    floor.conaffinity = 1
    spec.worldbody.add_light(pos=[1, -1, 3], dir=[-0.4, 0.4, -1], castshadow=True)
    spec.worldbody.add_light(pos=[-2, 2, 2.5], dir=[0.6, -0.6, -1], castshadow=False)
    spec.worldbody.add_camera(name="front", pos=[2.4, -1.6, 1.4],
                              xyaxes=[0.55, 0.83, 0, -0.25, 0.17, 0.95])
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    model = spec.compile()
    xml = spec.to_xml()
    # make the mesh path portable
    xml = xml.replace(str(MESHDIR), "../meshes").replace(str(MESHDIR).replace("\\", "/"), "../meshes")
    OUT.write_text(xml)

    arm_mass = model.body_mass[model.body_weldid != 0].sum()  # not welded to world
    print(f"wrote {OUT}")
    print(f"bodies={model.nbody} joints={model.njnt} actuators={model.nu} "
          f"equalities={model.neq} moving mass={arm_mass:.2f} kg")


if __name__ == "__main__":
    main()
