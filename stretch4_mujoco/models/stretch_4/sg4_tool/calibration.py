"""
The SG4 fingertip calibration and the spring shapes, expressed in the gripper frame.

The gripper frame is quick_connect_interface_link: x forward, the fingers moving in its XY plane.

The calibration (stretch4_compliant_gripper's latest_model_planar.yaml, embedded as a dict in
calibration_4026.py) gives each fingertip frame in
the right gripper colour camera's optical frame: position = centroid + u X_plane + v Y_plane and
rotation = R(theta about the plane normal) @ base_frame, with u, v and theta single cubic B-splines of
the servo's pos_pct. The camera frame is carried into the gripper frame by the URDF's camera mount.

The spring shapes (sg4_spring_shapes.json) are exported from the commercial render rig, sampled over
the stroke. They set the steel strips' shape; the grip force comes from compliant_fingers.FLEX_K.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from stretch4_mujoco.models.stretch_4.sg4_tool.calibration_4026 import CALIBRATION as DEFAULT_CALIBRATION

SPRING_SHAPES = Path(__file__).resolve().parent / "sg4_spring_shapes.json"

# The calibrated stroke, in pos_pct: squeezed shut to fully open.
POS_PCT_RANGE = (-95.0, 300.0)

# The calibration's fingertip point is the bottom of the suction cup's inside, 7.4 mm above the cup's
# base (stretch4_compliant_gripper's suctioncup_height['cup_bottom']). The URDF's fingertip frame has
# its origin at the cup's base, so the link sits this far behind the calibration point along the cup
# axis. With it the two rims first touch at pos_pct 0, as on the robot.
CUP_BASE = 0.0074
# The cup's rim is 15 mm ahead of its base along the fingertip link's x (gripper_fingertip_*_link.STL).
CUP_RIM = 0.015

# Hinged segments per free part of each strip.
SPRING_SEGMENTS = {"outer": 12, "inner": 14}
# Degree of the polynomials the strip hinges follow; MuJoCo's polycoef allows up to quartic.
SPRING_DEGREE = 4
STEEL_RGBA = "0.6 0.61 0.63 1"
SHRINK_RGBA = "0.93 0.93 0.91 1"

# The rig is placed on the sim by the parts themselves: the body's nose (rig x 216 mm is the quick
# connect's front, x 114.6 mm in the URDF mesh) and the suction cup (the rig's cup base and opening
# direction in its rest pose, measured from its KeyShot cup).
RIG_DX = 0.2160 - 0.1146
RIG_CUP_BASE = np.array([0.3192, 0.0951])
RIG_CUP_DIR = math.radians(-46.8)
# Half-width of the SG4 body where the outer strip leaves its side (URDF mesh).
BODY_HALF_WIDTH = 0.0315

# How far the springs' shape runs on from stalled fingertips toward the motor when the fingers are
# loaded: 0 stays with the fingertips, 1 follows the motor. ESTIMATE from the real gripper's video,
# where the outer bands' span grows from 1.80 to 1.91 body widths between the cups going flush and
# full squeeze.
BAND_GIVE = 0.5

# Bump when spring_design()'s output changes, to invalidate its on-disk cache.
_SPRING_DESIGN_VERSION = 1


def _transform(xyz: list[float], rpy: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """
    Build a homogeneous transform from a translation and fixed-axis roll, pitch, yaw.

    Args:
        xyz: Translation in meters.
        rpy: Roll, pitch and yaw in radians, URDF convention.

    Returns:
        The 4x4 transform.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


# The URDF's camera mount (stretch4_urdf SE4 stretch_main.xacro): gripper_camera_joint and
# gripper_right_camera_color_optical_joint on wrist_roll_link, with quick_connect_interface_link
# 18.6 mm ahead of wrist_roll_link. The calibration camera is the right colour optical frame.
CAMERA_IN_GRIPPER = (
    np.linalg.inv(_transform([0.0186, 0, 0]))
    @ _transform([0.0192, 0, 0.057], (0, 0.0523598775598298, 0))
    @ _transform([0, -0.01, 0], (-1.51843644923507, 0, -1.5707963267949))
)


# Parsed calibrations, keyed by id(); each entry keeps its calibration dict alive so the id stays unique.
_PARSED: dict[int, tuple[dict, tuple]] = {}


def _parse_calibration(calibration: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build the splines of a planar fingertip calibration, once per calibration dict.

    Args:
        calibration: A latest_model_planar.yaml's contents, at least its "plane_geometry" and
            "fingertips", as in calibration_4026.CALIBRATION.

    Returns:
        The plane's centroid, X axis, Y axis and normal in the camera frame, and per side a dict of
        the u, v and theta splines of pos_pct and the fingertip's base rotation.
    """
    cached = _PARSED.get(id(calibration))
    if cached is not None and cached[0] is calibration:
        return cached[1]
    plane = calibration["plane_geometry"]
    splines = {}
    for side in ("left", "right"):
        fingertip = calibration["fingertips"][side]
        splines[side] = {
            key: BSpline(
                np.array(fingertip[key]["knots"]),
                np.array(fingertip[key]["coeffs"]),
                fingertip[key]["degree"],
                extrapolate=True,
            )
            for key in ("spline_u", "spline_v", "spline_theta")
        }
        splines[side]["base"] = np.array(fingertip["base_frame"])
    parsed = (
        np.array(plane["centroid"]),
        np.array(plane["X_plane"]),
        np.array(plane["Y_plane"]),
        np.array(plane["normal"]),
        splines,
    )
    _PARSED[id(calibration)] = (calibration, parsed)
    return parsed


def camera_pose(side: str, pos_pct: float, calibration: dict = DEFAULT_CALIBRATION) -> tuple[np.ndarray, np.ndarray]:
    """
    Calibrated fingertip frame in the right gripper colour camera's optical frame.

    Args:
        side: "left" or "right".
        pos_pct: Gripper servo position in percent.
        calibration: The fingertip calibration, as in calibration_4026.CALIBRATION.

    Returns:
        The fingertip position in meters and its 3x3 rotation.
    """
    centroid, X, Y, normal, splines = _parse_calibration(calibration)
    s = splines[side]
    position = centroid + float(s["spline_u"](pos_pct)) * X + float(s["spline_v"](pos_pct)) * Y
    rotation = Rotation.from_rotvec(float(s["spline_theta"](pos_pct)) * normal).as_matrix() @ s["base"]
    return position, rotation


def tip_pose(side: str, pos_pct: float, calibration: dict = DEFAULT_CALIBRATION) -> tuple[float, float, float]:
    """
    Planar pose of the SG4 fingertip link in the gripper frame.

    The origin is the calibration's fingertip point moved CUP_BASE back along the pad normal. The yaw
    is the direction the cup opens, the calibration frame's z axis, which is the link's +x.

    Args:
        side: "left" or "right".
        pos_pct: Gripper servo position in percent.
        calibration: The fingertip calibration, as in calibration_4026.CALIBRATION.

    Returns:
        (x, y, yaw) in meters, meters and radians.
    """
    R_gripper_camera, p_gripper_camera = CAMERA_IN_GRIPPER[:3, :3], CAMERA_IN_GRIPPER[:3, 3]
    position, rotation = camera_pose(side, pos_pct, calibration)
    z = R_gripper_camera @ rotation[:, 2]
    origin = R_gripper_camera @ position + p_gripper_camera - CUP_BASE * z
    return float(origin[0]), float(origin[1]), math.atan2(z[1], z[0])


def tip_polys(calibration: dict = DEFAULT_CALIBRATION) -> dict[str, dict[str, Any]]:
    """
    Cubic fits of each fingertip's planar pose over the calibrated stroke.

    The calibration's u, v and theta are single cubics, but the cup-base offset and the projection
    into the gripper plane are not polynomial, so the fits are within about 0.1 mm and 0.01 mrad.

    Args:
        calibration: The fingertip calibration, as in calibration_4026.CALIBRATION.

    Returns:
        Per side, the coefficients c0..c3 in q = pos_pct / 100 for "x", "y" and "yaw", and "resid",
        the largest fit residual.
    """
    out = {}
    q = np.linspace(*POS_PCT_RANGE, 300) / 100
    for side in ("left", "right"):
        P = np.array([tip_pose(side, v * 100, calibration) for v in q])
        P[:, 2] = np.unwrap(P[:, 2])
        out[side] = {
            k: np.polynomial.polynomial.polyfit(q, P[:, i], 3) for i, k in enumerate(("x", "y", "yaw"))
        }
        out[side]["resid"] = max(
            float(np.abs(np.polynomial.polynomial.polyval(q, out[side][k]) - P[:, i]).max())
            for i, k in enumerate(("x", "y", "yaw"))
        )
    return out


def grasp_reach(radius: float, calibration: dict = DEFAULT_CALIBRATION) -> tuple[float, float] | None:
    """
    Where to put a ball's centre for a centred grasp.

    The tips creep forward and then pull back toward the wrist as they close, so smaller objects are
    met later in the stroke and further back than the open tips.

    Args:
        radius: Ball radius in meters.
        calibration: The fingertip calibration, as in calibration_4026.CALIBRATION.

    Returns:
        (distance along the gripper from the quick connect in meters, pos_pct) at which the closing
        cup rims first reach the ball's sides, or None if they never close that far.
    """
    for p in np.linspace(POS_PCT_RANGE[1], POS_PCT_RANGE[0], 4000):
        rims = []
        for side in ("left", "right"):
            x, y, yaw = tip_pose(side, p, calibration)
            rims.append((x + CUP_RIM * math.cos(yaw), y + CUP_RIM * math.sin(yaw)))
        if (rims[0][1] - rims[1][1]) / 2 <= radius:
            return (rims[0][0] + rims[1][0]) / 2, float(p)
    return None


def _rot2(angle: float) -> np.ndarray:
    """
    Build a 2D rotation matrix.

    Args:
        angle: Rotation in radians, counter-clockwise.

    Returns:
        The 2x2 rotation.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _resample(curve: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a polyline at evenly spaced arc lengths.

    Args:
        curve: (m, 2) polyline points.
        n: Number of points to return, including both ends.

    Returns:
        (n, 2) points.
    """
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    t = np.linspace(0, s[-1], n)
    return np.c_[np.interp(t, s, curve[:, 0]), np.interp(t, s, curve[:, 1])]


def _design_springs(calibration: dict) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Fit each strip's hinge chain to the rig's shapes over the stroke.

    At every opening each strip takes the rig's free-strip curve, turned and scaled to run from its
    body exit to where it enters the tip block on the calibrated fingertip. Length the curve does not
    use is drawn back into the body along the exit direction. The rig's frame is in mm, x along the
    gripper and y from the centre line, for the finger on +y (the left); the right is its mirror.

    Args:
        calibration: The fingertip calibration, as in calibration_4026.CALIBRATION.

    Returns:
        Per side and strip ("outer", "inner"): "root" (x, y, slide direction) in the gripper frame,
        "ds" the segment length, "hinge" the quartic coefficients in q = pos_pct / 100 of each hinge
        angle (the first relative to the slide direction, the rest to the previous segment), "draw"
        those of how far the strip is drawn into the body, "steel" the bare-steel length and "miss"
        the fitted chain's worst miss at the tip block, all in meters and radians.
    """
    with open(SPRING_SHAPES) as f:
        shapes = json.load(f)
    steps = shapes["steps"]
    pos_pct = 300 - 395 * np.array([step["u"] for step in steps])
    q = pos_pct / 100
    tip0 = np.array(shapes["TIP0"]) / 1000
    design = {}
    for side, sign in (("left", 1), ("right", -1)):
        out = {}
        for name in ("outer", "inner"):
            n = SPRING_SEGMENTS[name]
            s0 = np.array(steps[0][name]) / 1000
            roll0 = math.radians(-steps[0]["roll"])
            root = np.array([s0[0, 0] - RIG_DX, sign * s0[0, 1]])
            slide_angle = sign * math.atan2(*(s0[1] - s0[0])[::-1]) if name == "outer" else 0.0
            if name == "outer":
                # The rig's body is wider here: start the strip on the sim body's side.
                root = root - (abs(root[1]) - BODY_HALF_WIDTH) / abs(math.sin(slide_angle)) * np.array(
                    [math.cos(slide_angle), math.sin(slide_angle)]
                )
            # Where the strip enters the tip block, in the fingertip link frame.
            entry_rest = tip0 + _rot2(-roll0) @ (s0[-1] - np.array(steps[0]["tip"]) / 1000)
            entry = _rot2(-RIG_CUP_DIR) @ (entry_rest - RIG_CUP_BASE)
            entry = np.array([entry[0], sign * entry[1]])

            curves = []
            for i, step in enumerate(steps):
                C = np.array(step[name]) / 1000
                C = np.c_[C[:, 0], sign * C[:, 1]]
                x, y, yaw = tip_pose(side, pos_pct[i], calibration)
                end = np.array([x, y]) + _rot2(yaw) @ entry
                # Similarity taking the rig curve's ends onto root and the tip block entry.
                a, b = C[-1] - C[0], end - root
                k = np.linalg.norm(b) / np.linalg.norm(a)
                rot = math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])
                curves.append(root + k * (C - C[0]) @ _rot2(rot).T)

            lengths = [float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) for c in curves]
            total = max(lengths)
            ds = total / n
            slide_dir = np.array([math.cos(slide_angle), math.sin(slide_angle)])
            rows, slides, theta, miss = [], [], None, 0.0
            for i, c in enumerate(curves):
                back = total - lengths[i]
                start = root - back * slide_dir
                target = _resample(np.vstack([start, c]) if back > 1e-6 else c, n + 1)
                end = c[-1]

                def points(t: np.ndarray, start: np.ndarray = start) -> np.ndarray:
                    """
                    Chain joint positions for absolute segment angles.

                    Args:
                        t: Absolute angle of each segment in radians.
                        start: Position of the chain's first joint.

                    Returns:
                        (n + 1, 2) joint positions.
                    """
                    return start + np.r_[[np.zeros(2)], np.cumsum(ds * np.c_[np.cos(t), np.sin(t)], axis=0)]

                if theta is None:
                    theta = np.arctan2(*np.diff(target, axis=0).T[::-1])
                theta = minimize(
                    lambda t: float(np.sum((points(t) - target) ** 2)),
                    theta,
                    constraints=[{"type": "eq", "fun": lambda t, end=end: points(t)[-1] - end}],
                    method="SLSQP",
                    options={"maxiter": 800, "ftol": 1e-15},
                ).x
                rows.append(np.r_[theta[0] - slide_angle, np.diff(theta)])
                slides.append(-back)

            rows = np.unwrap(np.array(rows), axis=0)
            hinge = [np.polynomial.polynomial.polyfit(q, rows[:, k], SPRING_DEGREE) for k in range(n)]
            draw = np.polynomial.polynomial.polyfit(q, slides, SPRING_DEGREE)
            for i, c in enumerate(curves):
                t = slide_angle + np.cumsum([np.polynomial.polynomial.polyval(q[i], h) for h in hinge])
                end = (
                    root
                    + np.polynomial.polynomial.polyval(q[i], draw) * slide_dir
                    + ds * np.array([np.cos(t).sum(), np.sin(t).sum()])
                )
                miss = max(miss, float(np.hypot(*(end - c[-1]))))
            out[name] = {
                "root": (float(root[0]), float(root[1]), slide_angle),
                "ds": ds,
                "hinge": hinge,
                "draw": draw,
                "steel": (total - min(lengths)) if name == "inner" else 0.0,
                "miss": miss,
            }
        design[side] = out
    return design


def spring_design(
    calibration: dict = DEFAULT_CALIBRATION, cache_path: Path | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    The strips' hinge chains, fitted to the rig's shapes over the calibrated stroke.

    The fit takes several seconds, so it is cached in cache_path, keyed by the calibration and the
    shapes file's contents.

    Args:
        calibration: The fingertip calibration, as in calibration_4026.CALIBRATION.
        cache_path: JSON file to read the fit from and write it to. None always refits.

    Returns:
        See _design_springs().
    """
    digest = hashlib.sha256(
        str(_SPRING_DESIGN_VERSION).encode()
        + json.dumps(calibration, sort_keys=True).encode()
        + SPRING_SHAPES.read_bytes()
    ).hexdigest()
    if cache_path is not None and Path(cache_path).exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached["key"] == digest:
                design = cached["design"]
                for strips in design.values():
                    for strip in strips.values():
                        strip["root"] = tuple(strip["root"])
                        strip["hinge"] = [np.array(h) for h in strip["hinge"]]
                        strip["draw"] = np.array(strip["draw"])
                return design
        except (OSError, ValueError, KeyError):
            pass

    design = _design_springs(calibration)
    if cache_path is not None:
        serial = {
            side: {
                name: {
                    **strip,
                    "hinge": [h.tolist() for h in strip["hinge"]],
                    "draw": strip["draw"].tolist(),
                }
                for name, strip in strips.items()
            }
            for side, strips in design.items()
        }
        with open(cache_path, "w") as f:
            json.dump({"key": digest, "design": serial}, f)
    return design
