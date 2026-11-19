"""
Rebuild the SG4's rigid fingers as the calibrated, compliant fingers of the real gripper.

Per finger, in the generated MJCF:

- gripper_finger_<side>_joint keeps its name, class and actuator, so commands and status work
  unchanged. Its range widens to Q_RANGE: 0 is closed (the cups meet), 0.5 fully open and below 0
  squeezes. It maps to the servo as pos_pct = 600 q.
- The rigid finger link it swung is removed. The fingertip body (meshes, ArUco tag, rubber contacts,
  compliance hinges) hangs on x, y and yaw joints in the gripper plane, tied to sg4_<side>_follow by
  joint equalities with the fingertip calibration's cubics, so unloaded the tips follow the robot's
  measured path.
- The motor pulls sg4_<side>_follow through a spring (fixed tendon, FLEX_K). When the cups are
  blocked the fingertips stop and the motor still reaches its command, as the steel fingers bow out
  on the robot. The spring's stretch is the grip force.
- Each fingertip gains a third compliance hinge about z at the cup's rim, so the cups turn flat onto
  each other or onto an object.
- The two steel strips per finger are hinge chains whose shape follows sg4_<side>_band. Each strip's
  end is pinned to the fingertip. They collide with the world but not the gripper body or fingertips.
- The cups' collision meshes stay excluded from each other; a flat pad on each rim and one contact
  pair between the pads stop the cups on each other.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from stretch4_mujoco.models.stretch_4.sg4_tool import calibration

# Finger joint q to the calibration's q = pos_pct / 100.
Q_SCALE = 6.0
# Finger joint range: -0.16 squeezed (pos_pct -95), 0 the cups meet, 0.5 fully open (pos_pct 300).
Q_RANGE = (-0.16, 0.5)
# Stiff ties keeping the fingertips and strips on the calibrated path of sg4_<side>_follow.
TIP_SOLREF = "0.004 1"
TIP_SOLIMP = "0.995 0.999 0.001"
# The finger's give, motor to fingertip, in N m per rad of finger joint. ESTIMATE: only its ratio to
# the fingertip_compliance hinge stiffness is set, by the real gripper's video (cups flush about
# 0.2 s after they touch). The grip force it gives needs a force measurement on a robot.
FLEX_K = 30.0
FLEX_DAMP = 1.5
# The strips' collision boxes are thicker than the 0.8 mm steel, so thin objects cannot pass between
# contacts in one step.
STRIP_COLLISION_HALF_THICKNESS = 0.0015
# The strips collide with any geom whose conaffinity has bit 0 set. Robot geoms the strips run into
# (the gripper body they slide into, the fingertips they are pinned to) are moved to conaffinity bit
# 1, which keeps their contacts with the world and the rest of the robot.
STRIP_CONTYPE, STRIP_CONAFFINITY = 1, 0
# Strip contact. MuJoCo softens a contact by the mass of the bodies in it, so the segments carry
# about the steel's own weight (1 g per 14 mm) and a stiff solimp, or objects sink into them.
STRIP_SEGMENT_MASS = 0.001
STRIP_SOLREF = "0.004 1"
STRIP_SOLIMP = "0.99 0.999 0.001"
ROBOT_NEAR_STRIP_CONAFFINITY = 2
# Suction cup rim radius (CAD: 35.8 mm across).
CUP_R = 0.0179
# Cup-on-cup pad contact. In the video the pressed cups squash about 0.5 mm each.
CUP_SOLREF = "0.004 1"
CUP_SOLIMP = "0.99 0.999 0.001"

def _fmt(values: float | np.ndarray) -> str:
    """
    Format numbers for an MJCF attribute.

    Args:
        values: A number or a 1D array.

    Returns:
        Space-separated values at 9 significant digits.
    """
    return " ".join(f"{x:.9g}" for x in np.atleast_1d(values))


def _band(side: str, follow: str, motor: str) -> tuple[ET.Element, ET.Element, ET.Element]:
    """
    Build the joint the strips' shape follows: band = follow + BAND_GIVE * (motor - follow).

    It is held by a tendon equality, so it adds no spring between follow and motor.

    Args:
        side: "left" or "right".
        follow: Name of the follower joint.
        motor: Name of the finger joint the actuator drives.

    Returns:
        The band body (for the gripper frame body), its fixed tendon and its tendon equality.
    """
    name = f"sg4_{side}_band"
    body = ET.Element("body", name=name, pos="0 0 0")
    ET.SubElement(body, "joint", name=name, type="hinge", axis="0 0 1", armature="0.002")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-8 1e-8 1e-8")

    tendon = ET.Element("fixed", name=f"{name}mix")
    ET.SubElement(tendon, "joint", joint=name, coef="1")
    ET.SubElement(tendon, "joint", joint=follow, coef=f"{calibration.BAND_GIVE - 1:.6g}")
    ET.SubElement(tendon, "joint", joint=motor, coef=f"{-calibration.BAND_GIVE:.6g}")

    equality = ET.Element(
        "tendon",
        name=f"{name}mix",
        tendon1=f"{name}mix",
        polycoef="0 0 0 0 0",
        solref=TIP_SOLREF,
        solimp=TIP_SOLIMP,
    )
    return body, tendon, equality


def _springs(side: str, band: str, design: dict, shapes: dict) -> tuple[list[ET.Element], list[ET.Element]]:
    """
    Build one finger's two strips as hinge chains with collision boxes.

    Each hinge is tied to the band joint by a quartic. A body's static transform is its joint's value
    at band = 0 and the joint's ref is that same value, so qpos stays absolute and qpos0 is the coupled
    pose. Each strip's draw and root hinge are free, and its end is pinned to where it enters the
    fingertip, so the strips stay on a fingertip that tilts off the calibrated pose.

    Args:
        side: "left" or "right".
        band: Name of the band joint from _band().
        design: This side's entry of calibration.spring_design().
        shapes: The parsed spring shapes file, for the strip and heat-shrink dimensions.

    Returns:
        The strips' root bodies (for the gripper frame body) and their equalities.
    """
    scale = Q_SCALE ** np.arange(calibration.SPRING_DEGREE + 1)
    equalities = []

    def tie(joint: str, coefs: np.ndarray, free: bool = False) -> float:
        """
        Tie a strip joint to the band joint.

        Args:
            joint: Joint name; also used as the equality's name.
            coefs: Polynomial coefficients in the calibration's q.
            free: True to leave the joint untied and only return its offset.

        Returns:
            The joint's value at band = 0, for its body transform and ref.
        """
        c = np.r_[np.asarray(coefs, float), np.zeros(calibration.SPRING_DEGREE + 1 - len(coefs))] * scale
        offset = float(c[0])
        # MuJoCo couples offsets from qpos0, which the joint's ref already holds.
        c[0] = 0.0
        if not free:
            equalities.append(
                ET.Element(
                    "joint",
                    name=joint,
                    joint1=joint,
                    joint2=band,
                    polycoef=_fmt(c),
                    solref=TIP_SOLREF,
                    solimp=TIP_SOLIMP,
                )
            )
        return offset

    t_steel, t_shrink = shapes["strip_t"] / 1000, shapes["shrink_t"] / 1000
    h_steel = (shapes["band"][1] - shapes["band"][0]) / 1000
    h_shrink = h_steel + 2 * shapes["shrink_dz"] / 1000

    def box(body: ET.Element, x0: float, x1: float, steel: bool) -> None:
        """
        Add a visual strip segment along the body's x axis.

        Args:
            body: The segment's body.
            x0: Segment start in meters.
            x1: Segment end in meters, greater than x0.
            steel: True for bare steel, False for heat shrink.

        Returns:
            None.
        """
        thickness, height, rgba = (
            (t_steel, h_steel, calibration.STEEL_RGBA) if steel else (t_shrink, h_shrink, calibration.SHRINK_RGBA)
        )
        ET.SubElement(
            body,
            "geom",
            type="box",
            pos=f"{(x0 + x1) / 2:.6g} 0 0",
            size=f"{(x1 - x0) / 2 + 0.0004:.6g} {thickness / 2:.6g} {height / 2:.6g}",
            contype="0",
            conaffinity="0",
            group="2",
            rgba=rgba,
        )

    roots = []
    for name in ("outer", "inner"):
        strip = design[name]
        prefix = f"sg4_{side}_{name}"
        x0, y0, a0 = strip["root"]
        root = ET.Element("body", name=f"{prefix}_root", pos=f"{x0:.6g} {y0:.6g} 0", euler=f"0 0 {a0:.9g}")
        roots.append(root)

        # The strip is drawn back into the body along its exit direction.
        d0 = tie(f"{prefix}_draw", strip["draw"], free=True)
        parent = ET.SubElement(root, "body", name=f"{prefix}_slide", pos=f"{d0:.9g} 0 0")
        ET.SubElement(
            parent,
            "joint",
            name=f"{prefix}_draw",
            type="slide",
            axis="1 0 0",
            ref=f"{d0:.9g}",
            armature="0.0005",
            damping="0.05",
        )
        ET.SubElement(parent, "inertial", pos="0 0 0", mass=f"{STRIP_SEGMENT_MASS}", diaginertia="1e-8 1e-8 1e-8")

        ds = strip["ds"]
        for k, coefs in enumerate(strip["hinge"]):
            r0 = tie(f"{prefix}_{k}", coefs, free=(k == 0))
            parent = ET.SubElement(
                parent, "body", name=f"{prefix}_{k}", pos=f"{0 if k == 0 else ds:.6g} 0 0", euler=f"0 0 {r0:.9g}"
            )
            ET.SubElement(
                parent,
                "joint",
                name=f"{prefix}_{k}",
                type="hinge",
                axis="0 0 1",
                ref=f"{r0:.9g}",
                armature="1e-5",
                damping="0.0005",
            )
            ET.SubElement(parent, "inertial", pos="0 0 0", mass=f"{STRIP_SEGMENT_MASS}", diaginertia="1e-8 1e-8 1e-8")
            steel = (k + 0.5) * ds < strip["steel"]
            box(parent, 0.0, ds, steel)
            ET.SubElement(
                parent,
                "geom",
                name=f"{prefix}_{k}_collision",
                type="box",
                pos=f"{ds / 2:.6g} 0 0",
                size=f"{ds / 2:.6g} {STRIP_COLLISION_HALF_THICKNESS:.6g} {h_steel / 2:.6g}",
                contype=str(STRIP_CONTYPE),
                conaffinity=str(STRIP_CONAFFINITY),
                solref=STRIP_SOLREF,
                solimp=STRIP_SOLIMP,
                group="3",
            )
            if k == 0 and name == "outer":
                # The outer strip carries on inside the body, so sliding out under load opens no gap.
                box(parent, -0.016, 0.0, steel)

        equalities.append(
            ET.Element(
                "connect",
                name=f"{prefix}_pin",
                body1=f"{prefix}_{len(strip['hinge']) - 1}",
                body2=f"gripper_fingertip_{side}_link",
                anchor=f"{ds:.6g} 0 0",
                solref=TIP_SOLREF,
                solimp=TIP_SOLIMP,
            )
        )
    return roots, equalities


def is_sg4(worldbody: ET.Element) -> bool:
    """
    Whether the MJCF carries the SG4 tool's rigid fingers.

    Args:
        worldbody: The MJCF <worldbody> element.

    Returns:
        True if gripper_slider_link holds both SG4 finger links.
    """
    slider = worldbody.find(".//body[@name='gripper_slider_link']")
    return slider is not None and all(
        slider.find(f"body[@name='gripper_finger_{side}_link']") is not None for side in ("left", "right")
    )


def make_fingers_compliant(
    asset: ET.Element,
    worldbody: ET.Element,
    model_dir: Path,
    fingertip_calibration: dict = calibration.DEFAULT_CALIBRATION,
) -> ET.Element:
    """
    Replace the SG4's rigid fingers with the calibrated, compliant fingers, in place.

    Expects the generator to have already wrapped the fingers in gripper_slider_link and added the
    fingertips' compliance hinges. The finger joints' new range is picked up by the actuator
    ctrlrange the generator writes afterwards.

    Args:
        asset: The MJCF <asset> element; the rigid finger meshes are removed from it.
        worldbody: The MJCF <worldbody> element holding gripper_slider_link.
        model_dir: Directory the spring fit is cached in.
        fingertip_calibration: The fingertip calibration to follow, as in
            calibration_4026.CALIBRATION.

    Returns:
        A <mujoco> element whose <equality>, <tendon> and <contact> children belong in the model's
        contact file.

    Raises:
        ValueError: If the SG4 finger structure is not as expected.
    """
    polys = calibration.tip_polys(fingertip_calibration)
    design = calibration.spring_design(
        fingertip_calibration, cache_path=Path(model_dir) / "autogenerated_sg4_spring_design.json"
    )
    with open(calibration.SPRING_SHAPES) as f:
        shapes = json.load(f)

    slider = worldbody.find(".//body[@name='gripper_slider_link']")
    if slider is None:
        raise ValueError("No gripper_slider_link: the SG4 fingers have not been wrapped yet.")

    equalities, tendons = [], []
    removed_meshes = set()
    for side in ("left", "right"):
        finger = slider.find(f"body[@name='gripper_finger_{side}_link']")
        tip = finger.find(f"body[@name='gripper_fingertip_{side}_link']") if finger is not None else None
        joint = finger.find(f"joint[@name='gripper_finger_{side}_joint']") if finger is not None else None
        if finger is None or tip is None or joint is None:
            raise ValueError(f"Unexpected SG4 structure on the {side} finger.")
        removed_meshes |= {g.get("mesh") for g in finger.findall("geom") if g.get("mesh")}
        slider.remove(finger)

        # The drive: a weightless body carrying the actuated finger joint.
        drive = ET.SubElement(slider, "body", name=f"gripper_finger_{side}_link", pos="0 0 0")
        joint.set("range", f"{Q_RANGE[0]} {Q_RANGE[1]}")
        joint.set("limited", "true")
        drive.append(joint)
        ET.SubElement(drive, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-7 1e-7 1e-7")

        # The follower the fingertip and strips are tied to; the motor pulls it through FLEX_K.
        follow = ET.SubElement(slider, "body", name=f"sg4_{side}_follow", pos="0 0 0")
        ET.SubElement(
            follow,
            "joint",
            name=f"sg4_{side}_follow",
            type="hinge",
            axis="0 0 1",
            range=f"{Q_RANGE[0]} {Q_RANGE[1]}",
            limited="true",
            armature="0.01",
            damping="0.2",
        )
        ET.SubElement(follow, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-7 1e-7 1e-7")

        # The fingertip on x, y and yaw in the gripper frame. qpos0 must be the coupled pose at
        # finger q = 0, or every load starts the fingertips buried in the wrist where the coupling
        # cannot pull them out. Each body is offset by that pose and its joint's ref set to match,
        # so qpos stays absolute.
        P = polys[side]
        o = {k: float(P[k][0]) for k in ("x", "y", "yaw")}
        frames = {
            "x": {"pos": f"{o['x']:.9g} 0 0"},
            "y": {"pos": f"0 {o['y']:.9g} 0"},
            "yaw": {"pos": "0 0 0", "quat": f"{np.cos(o['yaw'] / 2):.9g} 0 0 {np.sin(o['yaw'] / 2):.9g}"},
        }
        parent = slider
        for k, joint_type, axis, armature in (
            ("x", "slide", "1 0 0", "0.002"),
            ("y", "slide", "0 1 0", "0.002"),
            ("yaw", "hinge", "0 0 1", "0.0002"),
        ):
            parent = ET.SubElement(parent, "body", name=f"sg4_{side}_tip_{k}", **frames[k])
            ET.SubElement(
                parent,
                "joint",
                name=f"sg4_{side}_tip_{k}",
                type=joint_type,
                axis=axis,
                ref=f"{o[k]:.9g}",
                armature=armature,
                damping="0.5" if joint_type == "slide" else "0.005",
            )
            ET.SubElement(parent, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-8 1e-8 1e-8")
        for attr in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
            tip.attrib.pop(attr, None)
        tip.set("pos", "0 0 0")
        parent.append(tip)

        # The cup tilts in the gripper plane about its rim centre.
        compliant_x = tip.find(f"joint[@name='gripper_fingertip_{side}_link_compliant_x']")
        if compliant_x is None:
            raise ValueError(f"No gripper_fingertip_{side}_link_compliant_x joint on the SG4 fingertip.")
        tip.insert(
            list(tip).index(compliant_x),
            ET.Element(
                "joint",
                {
                    "name": f"gripper_fingertip_{side}_link_compliant_z",
                    "axis": "0 0 1",
                    "pos": f"{calibration.CUP_RIM} 0 0",
                    "class": "fingertip_compliance",
                },
            ),
        )
        # A flat pad over the cup, for the cup-on-cup pair only.
        ET.SubElement(
            tip,
            "geom",
            name=f"sg4_{side}_cupface",
            type="box",
            pos=f"{calibration.CUP_RIM / 2} 0 0",
            size=f"{calibration.CUP_RIM / 2} {CUP_R} 0.012",
            contype="0",
            conaffinity="0",
            mass="0",
            group="3",
            rgba="0 0.8 0.2 0.4",
        )

        band_body, band_tendon, band_equality = _band(side, f"sg4_{side}_follow", f"gripper_finger_{side}_joint")
        slider.append(band_body)
        tendons.append(band_tendon)
        equalities.append(band_equality)
        spring_roots, spring_equalities = _springs(side, f"sg4_{side}_band", design[side], shapes)
        slider.extend(spring_roots)
        equalities.extend(spring_equalities)

        for k in ("x", "y", "yaw"):
            c = np.r_[P[k] * Q_SCALE ** np.arange(4), 0.0]
            # MuJoCo couples offsets from qpos0, which the joint's ref already holds.
            c[0] = 0.0
            equalities.append(
                ET.Element(
                    "joint",
                    name=f"sg4_{side}_{k}",
                    joint1=f"sg4_{side}_tip_{k}",
                    joint2=f"sg4_{side}_follow",
                    polycoef=_fmt(c),
                    solref=TIP_SOLREF,
                    solimp=TIP_SOLIMP,
                )
            )

    # Keep the strips off the gripper body they slide into and the fingertips they are pinned to.
    parents = {child: parent for parent in worldbody.iter() for child in parent}
    near_strips = list(parents[slider].findall("geom"))
    for side in ("left", "right"):
        near_strips += list(slider.find(f".//body[@name='gripper_fingertip_{side}_link']").iter("geom"))
    for geom in near_strips:
        if geom.get("class") in ("collision", "rubber") and geom.get("contype", "1") != "0":
            geom.set("contype", "1")
            geom.set("conaffinity", str(ROBOT_NEAR_STRIP_CONAFFINITY))

    # The rest of the gripper is gravity compensated; so are the bodies added here.
    for body in slider.iter("body"):
        if body.get("name", "").startswith(("sg4_", "gripper_finger_")):
            body.set("gravcomp", "1")

    still_used = {g.get("mesh") for g in worldbody.iter("geom")}
    for mesh in asset.findall("mesh"):
        if mesh.get("name") in removed_meshes - still_used:
            asset.remove(mesh)

    constraints = ET.Element("mujoco")
    ET.SubElement(constraints, "equality").extend(equalities)
    tendon = ET.SubElement(constraints, "tendon")
    for side in ("left", "right"):
        flex = ET.SubElement(
            tendon,
            "fixed",
            name=f"sg4_{side}_flex",
            stiffness=f"{FLEX_K}",
            damping=f"{FLEX_DAMP}",
            springlength="0",
        )
        ET.SubElement(flex, "joint", joint=f"gripper_finger_{side}_joint", coef="1")
        ET.SubElement(flex, "joint", joint=f"sg4_{side}_follow", coef="-1")
    tendon.extend(tendons)
    contact = ET.SubElement(constraints, "contact")
    # The pads stop the cups on each other; pressed face to face, the cup meshes slip through.
    ET.SubElement(contact, "exclude", body1="gripper_fingertip_left_link", body2="gripper_fingertip_right_link")
    ET.SubElement(
        contact,
        "pair",
        name="sg4_cups",
        geom1="sg4_left_cupface",
        geom2="sg4_right_cupface",
        condim="4",
        friction="1.8 1.8 0.01 0.001 0.001",
        solref=CUP_SOLREF,
        solimp=CUP_SOLIMP,
    )
    return constraints
