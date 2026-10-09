"""
Measure the real Stretch 4 gripper's `--grasp-offset-mm` from its gripper camera: the ArUco
markers on its fingertips give where the fingertips really are relative to `grasp_center_link`,
and that is compared to where the Franka's Robotiq 2F-85 closes relative to its `grasp_site`.

    gripper camera image --ArUco--> fingertip markers in the camera
        --URDF camera mount--> fingertip markers in grasp_center_link
        --URDF finger geometry--> fingertip contact points, their midpoint (the pinch point)
        and their distance apart (the jaw width)

    Robotiq at the same jaw width --> its pinch point in grasp_site (Stretch's TCP axes)
    grasp offset = Robotiq pinch point - Stretch pinch point

That is the offset that puts Stretch's pinch point on the Robotiq's (see
`FrankaStretchRetargeter.stretch_tool_target_world`): x along the approach, y between the
fingers, z across the hand. Both grippers' fingertips move along the approach as they close,
so it depends on the jaw width; measure at the width the robot grasps at.

It sees only the gripper: the camera is fixed to the wrist roll like `grasp_center_link`, so
anything before the wrist (lift, arm, wrist calibration) does not show up here. Each
measurement also says how far the markers are from where the URDF puts them at the best-fitting
finger angle: a few mm and degrees is the URDF being a little off; tens of mm or ~90/180
degrees is the camera mount or a sticker not being where the URDF has it, and the numbers are
not to be trusted.

On the robot, run the image + joint-state publisher from stretch4_compliant_gripper first, as
for run_stretch4_real.py (any wrist camera side; a higher wrist resolution helps):

    python send_gripper_and_head_images_with_joint_states.py -r --wrist_camera left --wrist_resolution 800

then open the gripper to the width to measure at, with the fingertip markers in view, and press
Space here to measure (Q or Esc quits). Each measurement is printed and appended to
calculate_gripper_offset_real.md beside this script, with the frame it ended on, annotated and
raw, embedded in it.

The tool (Stretch gripper or parallel jaw gripper) is read from the gripper's joint state; each
has its own fingertip geometry in FINGERTIPS.

Usage:
    python -m examples.vla.molmobot_droid.franka_retarget.calculate_gripper_offset_real --robot_ip 10.0.0.12
"""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass
from pathlib import Path

import click
import cv2
import mujoco
import numpy as np

from examples.vla.molmobot_droid.checkpoint import ROBOTIQ_DRIVER_CLOSED, ROBOTIQ_DRIVER_OPEN
from examples.vla.molmobot_droid.droid import FRANKA_PREFIX, FrankaKinematics, site_transform
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    DEFAULT_GRASP_OFFSET_MM,
    PARALLEL_GRIPPER_TOOL,
    STRETCH_GRIPPER_TOOL,
    STRETCH_TCP,
    TCP_ALIGN,
    _rotation_angle,
    _stretch_urdf,
)
from examples.vla.molmobot_droid.run_stretch4_real import (
    GRIPPER_PCT_OPEN,
    PARALLEL_GRIPPER_OPEN_MM,
    ImageAndJointReceiver,
)

SIDES = ("left", "right")

MARKER_DICTIONARY = cv2.aruco.DICT_6X6_250
MARKER_IDS = {"left": 200, "right": 201}
"""stretch4_compliant_gripper's `aruco_marker_info`: `finger_left` and `finger_right`."""

MARKER_SIZE_MM = 14.0
"""The black square, without its white border (the URDF's sticker meshes are 18 mm with it)."""

SUCTION_CUP_RIM_M = 0.017
"""The Stretch gripper's suction cup rims, where it grips, along `gripper_fingertip_*_link`'s +x
(the cup's axis) from its origin: stretch4_compliant_gripper's `aruco_to_fingertips`, 2 mm from
the link to the cup's base plus the cup's 15 mm height."""

FINGERTIPS = {
    STRETCH_GRIPPER_TOOL: ("gripper_finger_{side}_joint", "gripper_fingertip_{side}_link", (SUCTION_CUP_RIM_M, 0.0, 0.0)),
    # The fingers' origins meet at grasp_center_link when the jaws are closed: their pad centres.
    PARALLEL_GRIPPER_TOOL: ("finger_{side}_joint", "finger_{side}_link", (0.0, 0.0, 0.0)),
}
"""Per tool: each finger's joint, and the point it grips with, as a link and a point in it."""

FIT_SAMPLES = 401


def _point(transform: np.ndarray, point) -> np.ndarray:
    return transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3]


def _pose(rvec, tvec) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


# ---------------------------------------------------------------------------
# The Robotiq 2F-85
# ---------------------------------------------------------------------------


class RobotiqPads:
    """
    Where the Robotiq 2F-85's pads close, against how far apart they are, in its `grasp_site`
    frame turned into Stretch's TCP axes (TCP_ALIGN), from molmospaces' model of it.

    Its fingers are four-bar linkages closed by `connect` equalities, which forward kinematics
    leaves open: for each driver angle the passive joints are solved for them.
    """

    def __init__(self, samples: int = 200):
        model = FrankaKinematics().model
        data = mujoco.MjData(model)
        prefix = f"{FRANKA_PREFIX}gripper/"
        hinges = [
            j for j in range(model.njnt)
            if model.joint(j).name.startswith(prefix) and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        drivers = [model.jnt_qposadr[j] for j in hinges if model.joint(j).name.endswith("driver_joint")]
        passive = np.array([model.jnt_qposadr[j] for j in hinges if not model.joint(j).name.endswith("driver_joint")])
        connects = [e for e in range(model.neq) if model.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT]
        pads = {side: [model.geom(f"{prefix}{side}_pad{i}").id for i in (1, 2)] for side in SIDES}
        site = f"{prefix}grasp_site"

        def violation(x: np.ndarray) -> np.ndarray:
            data.qpos[passive] = x
            mujoco.mj_kinematics(model, data)
            gaps = []
            for e in connects:
                body1, body2 = model.eq_obj1id[e], model.eq_obj2id[e]
                anchor1 = data.xpos[body1] + data.xmat[body1].reshape(3, 3) @ model.eq_data[e][:3]
                anchor2 = data.xpos[body2] + data.xmat[body2].reshape(3, 3) @ model.eq_data[e][3:6]
                gaps.append(anchor1 - anchor2)
            return np.concatenate(gaps)

        x = np.zeros(len(passive))
        self.widths, self.pinches = [], []
        for driver in np.linspace(ROBOTIQ_DRIVER_OPEN, ROBOTIQ_DRIVER_CLOSED, samples):
            data.qpos[drivers] = driver
            for _ in range(50):  # Gauss-Newton from the last solution
                gap = violation(x)
                if np.abs(gap).max() < 1e-9:
                    break
                jacobian = np.empty((len(gap), len(x)))
                for i in range(len(x)):
                    step = np.zeros_like(x)
                    step[i] = 1e-6
                    jacobian[:, i] = (violation(x + step) - gap) / 1e-6
                x = x - np.linalg.lstsq(jacobian, gap, rcond=None)[0]
            else:
                raise RuntimeError(f"The Robotiq linkage does not close at driver angle {driver:.3f}")
            violation(x)
            tcp_from_world = np.linalg.inv(site_transform(data, site) @ TCP_ALIGN)
            centres = {side: _point(tcp_from_world, data.geom_xpos[pads[side]].mean(axis=0)) for side in SIDES}
            # The pads are boxes facing each other across their thin (x) side.
            thickness = sum(model.geom_size[pads[side][0]][0] for side in SIDES)
            self.widths.append(np.linalg.norm(centres["left"] - centres["right"]) - thickness)
            self.pinches.append((centres["left"] + centres["right"]) / 2)
        # Closing narrows the jaws; np.interp wants them widening.
        self.widths = np.array(self.widths[::-1])
        self.pinches = np.array(self.pinches[::-1])

    @property
    def max_width(self) -> float:
        return float(self.widths[-1])

    def pinch_at_width(self, width: float) -> np.ndarray:
        """The pinch point (m) with the pads `width` apart, or fully open if wider."""
        return np.array([np.interp(width, self.widths, self.pinches[:, i]) for i in range(3)])


# ---------------------------------------------------------------------------
# Stretch's fingers
# ---------------------------------------------------------------------------


@dataclass
class FingerSighting:
    """One fingertip marker in one image."""

    side: str
    camera_from_marker: np.ndarray
    tool_from_marker: np.ndarray
    """In `grasp_center_link`."""
    tip: np.ndarray
    """The fingertip's contact point in `grasp_center_link`, m."""
    finger_joint: float
    """The finger angle (or travel) that puts the URDF's marker nearest this one."""
    urdf_position_error: float
    """m, from the URDF's marker at `finger_joint`."""
    urdf_rotation_error: float
    """rad."""
    corners: np.ndarray


class StretchFingers:
    """
    The URDF's gripper camera mount and finger geometry for one tool: where each fingertip
    marker can be (tabulated over its finger's travel), and where the fingertip is relative to it.
    """

    def __init__(self, tool_name: str):
        joint_name, tip_link, tip_in_link = FINGERTIPS[tool_name]
        urdf = _stretch_urdf(tool_name)
        self.tool_from_optical = {
            side: urdf.get_transform(f"gripper_{side}_camera_color_optical_frame", STRETCH_TCP) for side in SIDES
        }
        self.joints, self.tool_from_marker, self.tool_tip, self.tip_in_marker = {}, {}, {}, {}
        for side in SIDES:
            joint = joint_name.format(side=side)
            limit = urdf.joint_map[joint].limit
            self.joints[side] = np.linspace(limit.lower, limit.upper, FIT_SAMPLES)
            markers, tips = [], []
            for q in self.joints[side]:
                urdf.update_cfg({joint: float(q)})
                markers.append(urdf.get_transform(f"aruco_fingertip_{side}_link", STRETCH_TCP))
                tips.append(_point(urdf.get_transform(tip_link.format(side=side), STRETCH_TCP), tip_in_link))
            urdf.update_cfg({joint: 0.0})
            self.tool_from_marker[side] = np.array(markers)
            self.tool_tip[side] = np.array(tips)
            # Marker and fingertip are on the same rigid finger.
            self.tip_in_marker[side] = _point(np.linalg.inv(markers[0]), tips[0])

    def fit(self, side: str, tool_from_marker: np.ndarray) -> tuple[int, float, float]:
        """The finger position whose URDF marker is nearest (index, m off, rad off)."""
        distances = np.linalg.norm(self.tool_from_marker[side][:, :3, 3] - tool_from_marker[:3, 3], axis=1)
        index = int(np.argmin(distances))
        rotation = self.tool_from_marker[side][index][:3, :3].T @ tool_from_marker[:3, :3]
        return index, float(distances[index]), _rotation_angle(rotation)


class FingertipMarkers:
    """Finds the fingertip markers in gripper camera images and places the fingertips."""

    def __init__(self, fingers: StretchFingers, marker_size_m: float):
        self.fingers = fingers
        # AprilTag's corners, fitted to the whole border, place these small, obliquely seen
        # markers several mm closer than SUBPIX's (synthetic images: <1 mm against ~5 mm off);
        # SUBPIX finds them where AprilTag's do not, in poor light, as in stretch4_compliant_gripper.
        self.detectors = []
        for refinement in (cv2.aruco.CORNER_REFINE_APRILTAG, cv2.aruco.CORNER_REFINE_SUBPIX):
            parameters = cv2.aruco.DetectorParameters()
            parameters.cornerRefinementMethod = refinement
            dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
            self.detectors.append(cv2.aruco.ArucoDetector(dictionary, parameters))
        half = marker_size_m / 2
        # SOLVEPNP_IPPE_SQUARE's corner order, which is ArUco's: x right, y up, z out of the marker.
        self.marker_corners = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])

    def find_corners(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Each fingertip marker's corners in the image, by the first detector that finds it once."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        found = {}
        for detector in self.detectors:
            corners, ids, _ = detector.detectMarkers(gray)
            ids = np.array([]) if ids is None else ids.reshape(-1)
            for side in SIDES:
                matches = np.flatnonzero(ids == MARKER_IDS[side])
                if side not in found and len(matches) == 1:
                    found[side] = corners[matches[0]].reshape(4, 2).astype(np.float64)
            if len(found) == len(SIDES):
                break
        return found

    def detect(
        self, image_rgb: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray, camera_side: str
    ) -> dict[str, FingerSighting]:
        tool_from_camera = self.fingers.tool_from_optical[camera_side]
        sightings = {}
        for side, image_corners in self.find_corners(image_rgb).items():
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                self.marker_corners, image_corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            # A marker this small seen this close is ambiguous: two poses, mirrored about the line
            # of sight, fit the corners about as well. Take the one the URDF's finger can be in.
            candidates = []
            for rvec, tvec in zip(rvecs, tvecs):
                _, _, rotation_error = self.fingers.fit(side, tool_from_camera @ _pose(rvec, tvec))
                candidates.append((rotation_error, rvec, tvec))
            _, rvec, tvec = min(candidates, key=lambda c: c[0])
            rvec, tvec = cv2.solvePnPRefineLM(self.marker_corners, image_corners, camera_matrix, distortion, rvec, tvec)
            camera_from_marker = _pose(rvec, tvec)
            tool_from_marker = tool_from_camera @ camera_from_marker
            index, position_error, rotation_error = self.fingers.fit(side, tool_from_marker)
            sightings[side] = FingerSighting(
                side=side,
                camera_from_marker=camera_from_marker,
                tool_from_marker=tool_from_marker,
                tip=_point(tool_from_marker, self.fingers.tip_in_marker[side]),
                finger_joint=float(self.fingers.joints[side][index]),
                urdf_position_error=position_error,
                urdf_rotation_error=rotation_error,
                corners=image_corners,
            )
        return sightings


# ---------------------------------------------------------------------------
# The offset
# ---------------------------------------------------------------------------


@dataclass
class GraspOffset:
    """Stretch's fingertips against the Robotiq's at the same jaw width, in Stretch's TCP frame (m)."""

    tips: dict[str, np.ndarray]
    width: float
    pinch: np.ndarray
    robotiq_pinch: np.ndarray
    offset: np.ndarray
    """Robotiq pinch point - Stretch's: `--grasp-offset-mm` / 1000."""
    wider_than_robotiq: bool

    @staticmethod
    def compare(tips: dict[str, np.ndarray], robotiq: RobotiqPads) -> "GraspOffset":
        width = float(np.linalg.norm(tips["left"] - tips["right"]))
        pinch = (tips["left"] + tips["right"]) / 2
        robotiq_pinch = robotiq.pinch_at_width(width)
        return GraspOffset(tips, width, pinch, robotiq_pinch, robotiq_pinch - pinch, width > robotiq.max_width)

    def jaw_tilt_deg(self) -> tuple[float, float]:
        """
        (roll, yaw): how far the line from the right fingertip to the left is turned about the
        approach (x) and about z from the TCP's y axis, along which the Robotiq closes.
        """
        u = (self.tips["left"] - self.tips["right"]) / self.width
        roll = math.atan2(-u[2], u[1])
        yaw = math.atan2(u[0], math.hypot(u[1], u[2]))
        return math.degrees(roll), math.degrees(yaw)


@dataclass
class Measurement:
    """Several frames' sightings of both fingertips, combined."""

    frames: int
    measured: GraspOffset
    urdf: GraspOffset
    """The URDF's fingertips at the finger positions fitted to the markers."""
    offset_spread: np.ndarray
    """Per-axis standard deviation of the single-frame offsets, m."""
    urdf_errors: dict[str, tuple[float, float]]
    """Per side, the median (m, rad) the markers were off the URDF's."""


def combine(frames: list[dict[str, FingerSighting]], fingers: StretchFingers, robotiq: RobotiqPads) -> Measurement:
    tips = {side: np.median([f[side].tip for f in frames], axis=0) for side in SIDES}
    urdf_tips = {}
    for side in SIDES:
        joint = np.median([f[side].finger_joint for f in frames])
        index = int(np.argmin(np.abs(fingers.joints[side] - joint)))
        urdf_tips[side] = fingers.tool_tip[side][index]
    per_frame = [GraspOffset.compare({s: f[s].tip for s in SIDES}, robotiq).offset for f in frames]
    return Measurement(
        frames=len(frames),
        measured=GraspOffset.compare(tips, robotiq),
        urdf=GraspOffset.compare(urdf_tips, robotiq),
        offset_spread=np.std(per_frame, axis=0),
        urdf_errors={
            side: (
                float(np.median([f[side].urdf_position_error for f in frames])),
                float(np.median([f[side].urdf_rotation_error for f in frames])),
            )
            for side in SIDES
        },
    )


# ---------------------------------------------------------------------------
# The robot's stream
# ---------------------------------------------------------------------------


def stream_tool(message: dict) -> str:
    """The tool the robot has, from the gripper's joint state: the parallel gripper reports mm."""
    state = message.get("closest_joint_state")
    if not state or "gripper" not in state:
        raise click.ClickException("The robot's stream has no gripper joint state; is the publisher up to date?")
    return PARALLEL_GRIPPER_TOOL if "pos_mm" in state["gripper"] else STRETCH_GRIPPER_TOOL


def gripper_opening(message: dict) -> str:
    gripper = (message.get("closest_joint_state") or {}).get("gripper", {})
    if "pos_mm" in gripper:
        return f"{gripper['pos_mm']:.1f} of {PARALLEL_GRIPPER_OPEN_MM:g} mm"
    if "pos_pct" in gripper:
        return f"{gripper['pos_pct']:.0f} of {GRIPPER_PCT_OPEN:g} %"
    return "unknown"


def camera_calibration(message: dict) -> tuple[np.ndarray, np.ndarray]:
    matrix = message.get("wrist_camera_matrix", message.get("camera_matrix"))
    distortion = message.get("wrist_distortion_coefficients", message.get("distortion_coefficients"))
    if matrix is None or distortion is None:
        raise click.ClickException(
            "The robot's stream has no gripper camera calibration; the publisher could not read the camera's"
        )
    return np.asarray(matrix, dtype=np.float64), np.asarray(distortion, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _mm(vector) -> str:
    return " ".join(f"{v * 1000:+7.1f}" for v in vector)


def _flag(vector) -> str:
    return ",".join(f"{v * 1000:.1f}" for v in vector)


def table_rows(measurement: Measurement) -> list[tuple[str, np.ndarray | float, np.ndarray | float | None]]:
    """(label, measured, URDF) rows, mm-converted by the caller; vectors are m, widths m."""
    measured, urdf = measurement.measured, measurement.urdf
    return [
        ("left fingertip", measured.tips["left"], urdf.tips["left"]),
        ("right fingertip", measured.tips["right"], urdf.tips["right"]),
        ("Stretch pinch point", measured.pinch, urdf.pinch),
        ("Robotiq pinch point", measured.robotiq_pinch, urdf.robotiq_pinch),
        ("jaw width", measured.width, urdf.width),
        ("grasp offset", measured.offset, urdf.offset),
        ("frame-to-frame (1σ)", measurement.offset_spread, None),
    ]


def notes(measurement: Measurement, robotiq: RobotiqPads) -> list[tuple[str, str | None]]:
    """(line, color) under the table."""
    lines = []
    if measurement.measured.wider_than_robotiq:
        lines.append((
            f"The jaws are wider than the Robotiq opens ({measurement.measured.width * 1000:.0f} > "
            f"{robotiq.max_width * 1000:.0f} mm); compared with it fully open.",
            "yellow",
        ))
    lines.append((
        "Markers off the URDF's: " + ", ".join(
            f"{side} {e[0] * 1000:.1f} mm {math.degrees(e[1]):.1f}°" for side, e in measurement.urdf_errors.items()
        ),
        None,
    ))
    if any(e[0] > 0.01 or e[1] > math.radians(15) for e in measurement.urdf_errors.values()):
        lines.append(("That is more than the URDF being slightly off: check the camera side and the stickers.", "red"))
    roll, yaw = measurement.measured.jaw_tilt_deg()
    lines.append((f"Jaw line tilted roll {roll:+.1f}°, yaw {yaw:+.1f}° from the TCP's y (not in the offset).", None))
    return lines


def report(measurement: Measurement, robotiq: RobotiqPads, tool: str, opening: str) -> None:
    click.secho(f"\n== {measurement.frames} frames · gripper at {opening} ==", bold=True)
    click.echo("   mm in Stretch's TCP frame: x along the approach, y between the fingers, z across\n")
    click.echo(f"   {'':24s}{'measured':>24s}    {'URDF, same finger angles':>24s}")
    for label, measured, urdf in table_rows(measurement):
        if np.ndim(measured) == 0:
            line = f"   {label:24s}{measured * 1000:24.1f}    {urdf * 1000:24.1f}"
        else:
            line = f"   {label:24s}{_mm(measured)}    " + (_mm(urdf) if urdf is not None else "")
        click.secho(line, bold=label == "grasp offset")
    click.echo()
    for line, color in notes(measurement, robotiq):
        click.secho(f"   {line}", fg=color)
    click.secho(f"\n   --grasp-offset-mm {_flag(measurement.measured.offset)}", fg="green", bold=True)
    default = DEFAULT_GRASP_OFFSET_MM[tool]
    click.echo(f"   (default for {tool[-3:].upper()}: {','.join(f'{v:g}' for v in default)})\n")


RESULTS_PATH = Path(__file__).with_suffix(".md")
"""Every measurement is appended here, its images embedded."""

IMAGE_QUALITY = 95
"""JPEG quality of the embedded images; the stream's own are JPEG already, and PNG would be ~5x larger."""


def save_results(
    path: Path,
    measurement: Measurement,
    robotiq: RobotiqPads,
    *,
    tool: str,
    opening: str,
    camera_side: str,
    robot_id: str | None,
    annotated_bgr: np.ndarray,
    raw_rgb: np.ndarray,
) -> None:
    """Append a measurement to `path`, with the last frame used, annotated and as received."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def embedded(image_bgr: np.ndarray) -> str:
        _, jpeg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")

    def cell(value) -> str:
        if value is None:
            return ""
        if np.ndim(value) == 0:
            return f"{value * 1000:.1f}"
        return ", ".join(f"{v * 1000:+.1f}" for v in value)

    lines = []
    if not path.exists():
        lines += [
            "# Real Stretch 4 grasp offsets",
            "",
            "Measured with `python -m examples.vla.molmobot_droid.franka_retarget.calculate_gripper_offset_real`"
            " from the gripper camera's view of the fingertip ArUco markers; see that script for how.",
            "All values are mm in Stretch's TCP frame (`grasp_center_link`): x along the approach,"
            " y between the fingers, z across. The grasp offset is the Robotiq's pinch point minus"
            " Stretch's at the same jaw width, i.e. `--grasp-offset-mm`.",
            "",
        ]
    lines += [
        f"## {stamp} · {tool} · {robot_id or 'unknown robot'}",
        "",
        f"Gripper at {opening}, {camera_side} gripper camera, {measurement.frames} frames.",
        "",
        f"**`--grasp-offset-mm {_flag(measurement.measured.offset)}`** "
        f"(default: `{','.join(f'{v:g}' for v in DEFAULT_GRASP_OFFSET_MM[tool])}`)",
        "",
        "| | measured | URDF, same finger angles |",
        "|---|---|---|",
    ]
    lines += [f"| {label} | {cell(measured)} | {cell(urdf)} |" for label, measured, urdf in table_rows(measurement)]
    lines += [""] + [f"- {line}" for line, _ in notes(measurement, robotiq)]
    lines += [
        "",
        f"![annotated]({embedded(annotated_bgr)})",
        f"![raw]({embedded(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))})",
        "",
        "",
    ]
    with path.open("a") as file:
        file.write("\n".join(lines))
    click.secho(f"   Saved to {path}", dim=True)


def draw(image_rgb, sightings, fingers, camera_side, camera_matrix, distortion, status: list[str]) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    camera_from_tool = np.linalg.inv(fingers.tool_from_optical[camera_side])
    for sighting in sightings.values():
        cv2.polylines(image, [sighting.corners.astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        rvec = cv2.Rodrigues(sighting.camera_from_marker[:3, :3])[0]
        cv2.drawFrameAxes(image, camera_matrix, distortion, rvec, sighting.camera_from_marker[:3, 3], 0.01)
    points = [s.tip for s in sightings.values()]
    if len(points) == 2:
        points.append((points[0] + points[1]) / 2)
    if points:
        in_camera = np.array([_point(camera_from_tool, p) for p in points])
        pixels, _ = cv2.projectPoints(in_camera, np.zeros(3), np.zeros(3), camera_matrix, distortion)
        for i, pixel in enumerate(pixels.reshape(-1, 2)):
            color = (0, 0, 255) if i == 2 else (255, 255, 255)
            cv2.circle(image, tuple(int(round(v)) for v in pixel), 4, color, -1, cv2.LINE_AA)
    for i, line in enumerate(status):
        position = (10, 22 + 20 * i)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


WINDOW = "Stretch 4 fingertips"


@click.command()
@click.option("--robot_ip", "--robot-ip", "robot_ip", required=True, help="Stretch 4's IP address.")
@click.option("--frames", type=int, default=30, show_default=True,
              help="Frames with both fingertip markers in view to combine into one measurement.")
@click.option("--marker-size-mm", type=float, default=MARKER_SIZE_MM, show_default=True,
              help="The markers' black square. Measure the stickers: 1 mm off here is several mm of depth.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=RESULTS_PATH, show_default=True,
              help="Markdown file each measurement is appended to, its images embedded.")
def main(robot_ip, frames, marker_size_mm, output):
    receiver = ImageAndJointReceiver(robot_ip)
    try:
        click.secho(f"Waiting for the gripper camera from {robot_ip}...", dim=True)
        message = receiver.receive()
        tool = stream_tool(message)
        camera_side = message.get("wrist_camera_side")
        if camera_side not in SIDES:
            raise click.ClickException(f"The robot publishes wrist camera side {camera_side!r}, not left or right")
        click.secho("Loading the gripper models...", dim=True)
        fingers = StretchFingers(tool)
        robotiq = RobotiqPads()
        markers = FingertipMarkers(fingers, marker_size_mm / 1000)
        click.echo(
            click.style("Stretch 4 ", bold=True) + f"{robot_ip} · {tool} · {camera_side} gripper camera\n"
            "Open the gripper to the width to measure at, then press Space here to measure; Q or Esc quits."
        )

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        collected: list[dict[str, FingerSighting]] | None = None
        while True:
            message = receiver.receive()
            camera_matrix, distortion = camera_calibration(message)
            image = message["wrist_rgb"]
            sightings = markers.detect(image, camera_matrix, distortion, camera_side)
            opening = gripper_opening(message)
            status = [f"gripper {opening} - markers: " + (", ".join(sightings) or "none")]
            if len(sightings) == 2:
                single = GraspOffset.compare({s: sightings[s].tip for s in SIDES}, robotiq)
                status.append(f"width {single.width * 1000:.1f} mm, offset {_mm(single.offset)} mm")
            measurement = None
            if collected is not None:
                if len(sightings) == 2:
                    collected.append(sightings)
                status.append(f"measuring {len(collected)}/{frames}")
                if len(collected) >= frames:
                    measurement = combine(collected, fingers, robotiq)
                    collected = None
            else:
                status.append("Space: measure   Q: quit")
            annotated = draw(image, sightings, fingers, camera_side, camera_matrix, distortion, status)
            cv2.imshow(WINDOW, annotated)
            if measurement is not None:
                report(measurement, robotiq, tool, opening)
                save_results(
                    output, measurement, robotiq, tool=tool, opening=opening, camera_side=camera_side,
                    robot_id=message.get("robot_id"), annotated_bgr=annotated, raw_rgb=image,
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and collected is None:
                collected = []
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        receiver.close()


if __name__ == "__main__":
    main()
