"""
Retarget MolmoBot-DROID's Franka actions onto Stretch 4, and Stretch 4's state back onto the
Franka, by matching the tool centre point (TCP).

    Franka joints --FK--> grasp_site pose --(mount, TCP alignment, grasp offset)-->
    Stretch grasp_center_link pose --stretch4_kinematics IK--> base rotation, lift, arm,
    wrist yaw/pitch/roll

The virtual Franka stands where `custom_scene` puts it: `fr3_link0` above Stretch's starting
footprint, facing the same way, at the height molmospaces trains at. Both robots reach along
their +x, so no extra mount rotation is needed. FK/IK of the Franka is `droid.FrankaKinematics`;
Stretch's is `stretch4_kinematics.StretchKinematics` (6 DOF for a 6-D pose: base rotation,
lift, arm, and the three wrist joints; base translation is not used).

The grasp offset (`--grasp-offset-mm`, `--grasp-offset-deg`) moves where Stretch's real tool
goes relative to the Franka's TCP, in the TCP frame. The policy never sees it: Stretch's state
has it removed again before it is mapped back to Franka joints.

Frames: Franka's `grasp_site` approaches along +z with the fingers on y; Stretch's
`grasp_center_link` approaches along +x with the fingers on y. `TCP_ALIGN` maps one onto the
other and puts both wrist cameras on the same side.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass

import click
import cv2
import numpy as np

from examples.vla.molmobot_droid.checkpoint import (
    ACTION_HORIZON,
    DROID_IMAGE_SIZE,
    FRANKA_HOME_QPOS,
    GRIPPER_CLOSED,
    POLICY_DT,
    ROBOTIQ_DRIVER_CLOSED,
    ROBOTIQ_DRIVER_OPEN,
)
from examples.vla.molmobot_droid.droid import (
    EXO_CAMERAS,
    HEAD_CAMERAS,
    FrankaKinematics,
    FrankaSpawn,
    Observation,
    RobotPose,
    make_transform,
    pose_to_transform,
    rotz,
)

STRETCH_TCP = "grasp_center_link"

TCP_ALIGN = np.array(
    [
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
"""Pose of Stretch's `grasp_center_link` axes in Franka's `grasp_site` frame (same origin).

Camera-aligned: both wrist cameras end up on the same side of the grasp, looking the same way
up, so Stretch's gripper image is what the Franka's wrist camera would see. The grasp is never
turned the other way round: where Stretch's wrist cannot roll as far as the Franka's grasp
needs, it goes to the closest pose it can reach instead."""

GRIPPER_CAMERAS = ("left", "right")
HEAD_CROPS = ("droid", "none")

SLOW_FACTOR = 0.1
"""`--slow` runs every joint at 20% of its default speed (80% slower)."""

ARM_RANGE = (0.0, 0.52)
"""Stretch 4 telescoping arm travel, m (the `arm` actuator's range)."""

LIFT_RANGE = (0.0, 1.2)

END_STOP_MARGIN = 0.005
"""m. Lift and arm targets stay this far inside their travel: driven into an end stop, a joint
pushes on it and trips its guarded contact."""

MIN_BASE_ROTATION = math.radians(0.3)
"""Base rotations smaller than this are skipped rather than commanded."""

STEP_ARRIVAL_TIMEOUT = 3.0
"""s. A step moves each joint 0.2 rad at most, so this only runs out when something holds a
joint back (a guarded contact), and then waiting longer does not help."""

CUSTOM_START_JOINT6 = math.radians(125)
"""The Franka's joint 6 in --custom_franka_start_pose, instead of home's 90 degrees. Every degree
over 90 tilts the grasp a degree up from straight down: 125 points it 55 degrees below level."""

CUSTOM_START_QPOS = (*FRANKA_HOME_QPOS[:5], CUSTOM_START_JOINT6, FRANKA_HOME_QPOS[6])
"""--custom_franka_start_pose: the Franka's home pose with joint 6 at CUSTOM_START_JOINT6."""

MAX_CLAMPED_POSITION_ERROR = 0.05
MAX_CLAMPED_ROTATION_ERROR = math.radians(20)
"""How far from an unreachable target Stretch may go instead (m, rad); further is a failure."""


# ---------------------------------------------------------------------------
# Parameters and CLI flags
# ---------------------------------------------------------------------------


STRETCH_GRIPPER_TOOL = "eoa_wrist_dw4_tool_sg4"
PARALLEL_GRIPPER_TOOL = "eoa_wrist_dw4_tool_pg4"

DEFAULT_GRASP_OFFSET_MM = {
    STRETCH_GRIPPER_TOOL: (-9.0, 0.0, 0.0),
    PARALLEL_GRIPPER_TOOL: (4.0 , 0, 0.0),
    # PARALLEL_GRIPPER_TOOL: (4 + 33, -21.0, 0.0),
}
"""Per tool, along the approach axis: where its fingers close relative to its grasp_center_link,
compared to the Robotiq's relative to its grasp_site, so the fingers line up with the Franka's."""

OVERLAY_GRASP_OFFSET_MM = {
    STRETCH_GRIPPER_TOOL: (-9, 21.0, 17.0),
    # PARALLEL_GRIPPER_TOOL: (4 + 33, -21.0, 0.0),
    PARALLEL_GRIPPER_TOOL: (4 + 15, -21.0, 0.0),
}
"""Per tool, the grasp offset to default to with --overlay_franka_gripper (run_stretch4_sim): the
one that puts the overlaid Robotiq's fingers where the tool's are in its gripper camera's view.
A tool not listed keeps DEFAULT_GRASP_OFFSET_MM."""


@dataclass
class RetargetParams:
    slow: bool = False
    wait_for_arrival: bool = True
    head_crop: str = "droid"
    execute_horizon: int = 8
    execute_first_n: int = 2
    grasp_offset_mm: tuple[float, float, float] | None = None
    """x (approach), y (between the fingers), z. None: DEFAULT_GRASP_OFFSET_MM for the tool."""
    grasp_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """(roll, yaw, pitch) about the TCP's x (approach), z and y axes."""
    exo_camera: str = "left"
    gripper_camera: str = "left"
    use_parallel_gripper: bool = False

    def __post_init__(self):
        if not 1 <= self.execute_first_n <= self.execute_horizon <= ACTION_HORIZON:
            raise ValueError(
                "Need 1 <= --execute-horizon-do-only-first-n-steps <= --execute-horizon <= "
                f"{ACTION_HORIZON}, got {self.execute_first_n} and {self.execute_horizon}"
            )
        if self.exo_camera not in EXO_CAMERAS:
            raise ValueError(f"--exo_camera must be one of {EXO_CAMERAS}")
        if self.gripper_camera not in GRIPPER_CAMERAS:
            raise ValueError(f"--gripper_camera must be one of {GRIPPER_CAMERAS}")
        if self.head_crop not in HEAD_CROPS:
            raise ValueError(f"--head-crop must be one of {HEAD_CROPS}")

    @property
    def tool_name(self) -> str:
        """The stretch4_urdf / stretch4_mujoco tool: the Stretch gripper (SG4) or parallel (PG4)."""
        return PARALLEL_GRIPPER_TOOL if self.use_parallel_gripper else STRETCH_GRIPPER_TOOL

    @property
    def effective_grasp_offset_mm(self) -> tuple[float, float, float]:
        if self.grasp_offset_mm is not None:
            return tuple(self.grasp_offset_mm)
        return DEFAULT_GRASP_OFFSET_MM[self.tool_name]

    @property
    def tcp_offset(self) -> np.ndarray:
        """Stretch's tool relative to the Franka's TCP, in the (Stretch-axes) TCP frame."""
        roll, yaw, pitch = (math.radians(a) for a in self.grasp_offset_deg)
        rotation = rotz(yaw) @ _roty(pitch) @ _rotx(roll)
        return make_transform(rotation, np.asarray(self.effective_grasp_offset_mm, dtype=float) / 1000.0)

    def flags(self) -> dict[str, object]:
        """The CLI flags these came from (spelled as on the command line), for reports and run names."""
        return {
            "exo_camera": self.exo_camera,
            "gripper_camera": self.gripper_camera,
            **execution_flags(self.execute_horizon, self.execute_first_n),
            "head-crop": self.head_crop,
            "slow": self.slow,
            "wait-for-arrival": self.wait_for_arrival,
            "use_parallel_gripper": self.use_parallel_gripper,
            "grasp-offset-mm": ",".join(f"{v:g}" for v in self.effective_grasp_offset_mm),
            "grasp-offset-deg": ",".join(f"{v:g}" for v in self.grasp_offset_deg),
        }


def execution_flags(execute_horizon: int, execute_first_n: int) -> dict[str, int]:
    return {
        "execute-horizon": execute_horizon,
        "execute-horizon-do-only-first-n-steps": execute_first_n,
    }


def _rotx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _roty(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _parse_triplet(ctx, param, value):
    if value is None:
        return None
    try:
        values = tuple(float(v) for v in value.split(","))
    except ValueError:
        values = ()
    if len(values) != 3:
        raise click.BadParameter("expected three comma-separated numbers, e.g. 0,0,10")
    return values


def execution_options(function):
    """--execute-horizon and --execute-horizon-do-only-first-n-steps."""
    function = click.option(
        "--execute-horizon-do-only-first-n-steps",
        "execute_first_n",
        type=int,
        default=2,
        show_default=True,
        help="Of the actions kept, execute this many, then discard the rest and query again.",
    )(function)
    function = click.option(
        "--execute-horizon",
        type=int,
        default=8,
        show_default=True,
        help=f"How many of the {ACTION_HORIZON} actions in each predicted chunk to keep.",
    )(function)
    return function


def exo_camera_option(default: str):
    return click.option(
        "--exo_camera",
        "--exo-camera",
        "exo_camera",
        type=click.Choice(EXO_CAMERAS),
        default=default,
        show_default=True,
        help="droid: the DROID shoulder camera on the Franka. left/right/center: Stretch 4's "
        "head camera (on the Franka: transplanted to the same place relative to the floor).",
    )


def head_crop_option(function):
    return click.option(
        "--head-crop",
        "--head_crop",
        "head_crop",
        type=click.Choice(HEAD_CROPS),
        default="droid",
        show_default=True,
        help=f"droid: center-crop head images to the DROID exo camera's "
        f"{DROID_IMAGE_SIZE[0]}x{DROID_IMAGE_SIZE[1]} aspect, then resize to it.",
    )(function)


def custom_start_option(function):
    """--custom_franka_start_pose, for the Stretch 4 scripts (see `custom_start_link0_height()`)."""
    return click.option(
        "--custom_franka_start_pose",
        "--custom-franka-start-pose",
        "custom_franka_start_pose",
        is_flag=True,
        help=f"Start the Franka at home with joint 6 at {math.degrees(CUSTOM_START_JOINT6):g} degrees "
        "instead of 90 (the grasp tilted up from straight down), on a pedestal of the height that "
        "puts Stretch's tool as high as Stretch can put it at that tilt (instead of the height "
        "molmospaces trains at for the object).",
    )(function)


def overlay_gripper_option(function):
    """--overlay_franka_gripper, for the Stretch 4 scripts (see `apply_overlay_grasp_offset()`)."""
    return click.option(
        "--overlay_franka_gripper",
        "--overlay-franka-gripper",
        "overlay_franka_gripper",
        is_flag=True,
        help="Show the policy the Franka instead of Stretch, where it is told its arm and hand are "
        "(grasp offset included): the Robotiq's fingers for Stretch's gripper in the wrist view, the "
        "whole Franka for Stretch's arm in the exo view. Defaults the grasp offset to the overlay's: "
        + ", ".join(f"{','.join(f'{v:g}' for v in o)} for {t[-3:].upper()}" for t, o in OVERLAY_GRASP_OFFSET_MM.items())
        + ".",
    )(function)


def apply_overlay_grasp_offset(params: RetargetParams) -> None:
    """With --overlay_franka_gripper: OVERLAY_GRASP_OFFSET_MM's grasp offset, unless one was given.
    Call it once the tool is known, before anything that depends on the offset."""
    if params.grasp_offset_mm is None:
        params.grasp_offset_mm = OVERLAY_GRASP_OFFSET_MM.get(params.tool_name)


def retarget_options(function=None, *, gripper_option: bool = True):
    """
    Every `RetargetParams` flag, for the Stretch 4 scripts. Without `gripper_option` there is
    no --use_parallel_gripper (the real robot reports its gripper).
    """
    if function is None:
        return lambda f: retarget_options(f, gripper_option=gripper_option)
    options = [
        click.option("--slow", is_flag=True, help="Run every joint at 20% of its default speed."),
        click.option(
            "--wait-for-arrival/--no-wait-for-arrival",
            default=True,
            show_default=True,
            help="Wait for Stretch to reach each action before executing the next.",
        ),
        head_crop_option,
        execution_options,
        click.option(
            "--grasp-offset-mm",
            default=None,
            callback=_parse_triplet,
            help="x,y,z offset of Stretch's tool from the Franka TCP, in the TCP frame "
            "(x along the approach, y between the fingers). Hidden from the policy. "
            "Default: the tool's, to line its fingers up with the Franka's: "
            + ", ".join(f"{','.join(f'{v:g}' for v in o)} for {t[-3:].upper()}" for t, o in DEFAULT_GRASP_OFFSET_MM.items())
            + ".",
        ),
        click.option(
            "--grasp-offset-deg",
            default="0,0,0",
            show_default=True,
            callback=_parse_triplet,
            help="roll,yaw,pitch offset of Stretch's tool from the Franka TCP, about the TCP's "
            "x, z and y axes. Hidden from the policy.",
        ),
        exo_camera_option("left"),
        click.option(
            "--gripper_camera",
            "--gripper-camera",
            "gripper_camera",
            type=click.Choice(GRIPPER_CAMERAS),
            default="left",
            show_default=True,
            help="Which gripper camera stands in for the Franka's wrist camera.",
        ),
    ]
    if gripper_option:
        options.append(
            click.option(
                "--use_parallel_gripper",
                "--use-parallel-gripper",
                "use_parallel_gripper",
                is_flag=True,
                help="Stretch 4 with the parallel jaw gripper (PG4) instead of the Stretch gripper (SG4).",
            )
        )
    for option in reversed(options):
        function = option(function)
    return function


def params_from_kwargs(kwargs: dict) -> RetargetParams:
    """Pop the `retarget_options()` values out of a click command's kwargs."""
    return RetargetParams(
        slow=kwargs.pop("slow"),
        wait_for_arrival=kwargs.pop("wait_for_arrival"),
        head_crop=kwargs.pop("head_crop"),
        execute_horizon=kwargs.pop("execute_horizon"),
        execute_first_n=kwargs.pop("execute_first_n"),
        grasp_offset_mm=kwargs.pop("grasp_offset_mm"),
        grasp_offset_deg=kwargs.pop("grasp_offset_deg"),
        exo_camera=kwargs.pop("exo_camera"),
        gripper_camera=kwargs.pop("gripper_camera"),
        use_parallel_gripper=kwargs.pop("use_parallel_gripper", False),
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def crop_to_aspect(image: np.ndarray, aspect: float) -> np.ndarray:
    """Largest centered crop with width/height == aspect."""
    height, width = image.shape[:2]
    if width / height > aspect:
        new_width = round(height * aspect)
        x0 = (width - new_width) // 2
        return image[:, x0 : x0 + new_width]
    new_height = round(width / aspect)
    y0 = (height - new_height) // 2
    return image[y0 : y0 + new_height]


def to_droid_frame(image: np.ndarray) -> np.ndarray:
    """Center-crop to the DROID camera aspect and resize to DROID_IMAGE_SIZE."""
    width, height = DROID_IMAGE_SIZE
    cropped = crop_to_aspect(image, width / height)
    return cv2.resize(cropped, DROID_IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def wrist_view(gripper_image: np.ndarray) -> np.ndarray:
    """The wrist image the policy sees: Stretch's gripper camera, DROID-framed. TCP_ALIGN keeps
    it the right way up, as the Franka's wrist camera would see it."""
    return to_droid_frame(gripper_image)


def prepare_exo(image: np.ndarray, params: RetargetParams) -> np.ndarray:
    """The exo image the policy sees: head images are cropped per `--head-crop`."""
    if params.exo_camera in HEAD_CAMERAS and params.head_crop == "droid":
        return to_droid_frame(image)
    return image


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


@dataclass
class StretchJoints:
    """Stretch 4's arm joints, as its simulator and RobotClient report them."""

    lift: float
    arm: float
    wrist_yaw: float
    wrist_pitch: float
    wrist_roll: float
    gripper_open_fraction: float
    """0 closed .. 1 fully open."""


@dataclass
class StretchTargets:
    base_rotate_by: float
    lift: float
    arm: float
    wrist_yaw: float
    wrist_pitch: float
    wrist_roll: float
    gripper_closed: bool
    clamped: bool = False
    """The target was out of reach and this is the closest pose found."""
    tool_error_m: float = 0.0


@functools.cache
def measure_arm_offset() -> float:
    """
    How much further out stretch4_kinematics puts the tool than the URDF does, for the same
    arm extension (m).

    stretch4_urdf's IK-URDF generator merges the four telescoping arm joints into one, and
    sums their offsets from `joint.origin[3, :3]` (the homogeneous row, always zero) instead of
    `joint.origin[:3, 3]`, dropping the inner links' offsets. Measuring it here, against the URDF
    the simulator is built from, keeps this right whether or not that is fixed upstream.
    """
    from stretch4_kinematics import StretchJointPositions

    urdf = _stretch_urdf(STRETCH_GRIPPER_TOOL)
    kinematics = stretch_kinematics()

    def urdf_tcp(arm):
        cfg = {"lift_joint": 0.6, "wrist_yaw_joint": 0.0, "wrist_pitch_joint": 0.0, "wrist_roll_joint": 0.0}
        cfg.update({f"arm_l{i}_joint": arm / 4 for i in range(1, 5)})
        urdf.update_cfg(cfg)
        return urdf.get_transform(STRETCH_TCP, "base_footprint")[:3, 3]

    axis = urdf_tcp(0.3) - urdf_tcp(0.2)
    axis /= np.linalg.norm(axis)
    kin_tcp = kinematics.forward(StretchJointPositions(0, 0, 0, 0.6, 0.2, 0, 0, 0), STRETCH_TCP).translation
    error = kin_tcp - urdf_tcp(0.2)
    offset = float(error @ axis)
    residual = np.linalg.norm(error - offset * axis)
    if residual > 1e-3:
        raise RuntimeError(f"stretch4_kinematics disagrees with the URDF by {residual * 1000:.1f} mm off the arm axis")
    return offset


@functools.cache
def _stretch_urdf(tool_name: str):
    import yourdfpy

    from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

    return yourdfpy.URDF.load(Stretch4MujocoSimulator.get_urdf_path(tool_name))


@functools.cache
def tool_tcp_correction(tool_name: str) -> np.ndarray:
    """
    The tool's `grasp_center_link` in the Stretch gripper's, both hung off `wrist_roll_link`,
    from the stretch4_mujoco URDFs. Everything up to the wrist roll is the same for every tool,
    so IK solved for the Stretch gripper's TCP at `goal @ inv(this)` puts this tool's at `goal`.

    (stretch4_kinematics models the parallel gripper only on an unreleased branch; this needs
    nothing but the Stretch gripper model it ships.)
    """
    if tool_name == STRETCH_GRIPPER_TOOL:
        return np.eye(4)
    roll_from_sg4 = _stretch_urdf(STRETCH_GRIPPER_TOOL).get_transform(STRETCH_TCP, "wrist_roll_link")
    roll_from_tool = _stretch_urdf(tool_name).get_transform(STRETCH_TCP, "wrist_roll_link")
    return np.linalg.inv(roll_from_sg4) @ roll_from_tool


WRIST_ROLL_RANGE = (-1.135, 4.276)
"""The wrist roll's travel on the robot (stretch4_body `SE4_wrist_roll_DW4` range_deg, -65..245
degrees) and in the simulator (mjcf_generator's FLIP_WRIST_ROLL_RANGE), in the URDF's rounded
radians so a roll at the limit is one the simulator accepts. The URDF, and so older
stretch4_kinematics, has it mirrored as [-4.276, 1.135]."""


@functools.cache
def stretch_kinematics():
    """stretch4_kinematics' model of Stretch 4 with the Stretch gripper, roll limits corrected."""
    from stretch4_kinematics import StretchKinematics

    kinematics = StretchKinematics()
    for model in (kinematics.model, kinematics.model_ik):
        index = model.joints[model.getJointId("wrist_roll_joint")].idx_q
        model.lowerPositionLimit[index], model.upperPositionLimit[index] = WRIST_ROLL_RANGE
    return kinematics


def custom_start_link0_height(params: RetargetParams, floor_z: float) -> float:
    """
    Where `fr3_link0` goes for --custom_franka_start_pose: the virtual Franka's pedestal lowered
    (or raised) until, at CUSTOM_START_QPOS, Stretch's tool is as high as Stretch can put it, at
    the wrist pitch that pose's tilt needs.
    """
    from stretch4_kinematics import StretchJointPositions

    # In fr3_link0, which stands level over Stretch's footprint: only the heights differ.
    tool = FrankaKinematics().fk(CUSTOM_START_QPOS) @ TCP_ALIGN @ params.tcp_offset
    # Stretch's tool approaches along its x, and its wrist pitch is how far below level that
    # points (pi/2 straight down). The yaw and roll turn it about the vertical and the approach,
    # so leave its height alone.
    pitch = math.asin(float(np.clip(-tool[2, 0], -1.0, 1.0)))
    top = StretchJointPositions(0, 0, 0, LIFT_RANGE[1] - END_STOP_MARGIN, 0, 0, pitch, 0)
    highest = stretch_kinematics().forward(top, STRETCH_TCP).homogeneous @ tool_tcp_correction(params.tool_name)
    return floor_z + highest[2, 3] - tool[2, 3]


def franka_yaw_for_start(params: RetargetParams, link0_height: float, start_q7, floor_z: float = 0.0) -> float:
    """
    The virtual Franka's yaw relative to Stretch's footprint (rad) that lets Stretch reach the
    Franka's `start_q7` without turning its base.

    Stretch's arm and tool sit off to the side of its footprint's centre, so with the Franka
    facing the way Stretch does, the start pose has Stretch turn its base (~12-16 degrees). The
    Franka is turned by minus that instead, about the footprint, where its `fr3_link0` stands:
    the start pose relative to the Franka is the same, and so is Stretch's head relative to it
    (the base had turned it by just as much), but nothing moves to get there.

    Re-solved a few times, as the turned target can pick a slightly different wrist pose.
    Returns 0 when Stretch cannot reach the start pose, which reaching it reports.
    """
    seed = StretchJoints(
        lift=LIFT_RANGE[0] + SPAWN_LIFT_FRACTION * (LIFT_RANGE[1] - LIFT_RANGE[0]),
        arm=SPAWN_ARM_EXTENSION,
        wrist_yaw=0.0,
        wrist_pitch=0.0,
        wrist_roll=0.0,
        gripper_open_fraction=1.0,
    )
    yaw = 0.0
    for _ in range(5):
        franka = FrankaSpawn("", RobotPose(0.0, 0.0, yaw), floor_z, floor_z, link0_height - floor_z, None, None, "")
        targets = FrankaStretchRetargeter(franka, params).franka_to_stretch(
            np.append(start_q7, 0.0), planar_transform(0.0, 0.0, 0.0, floor_z), seed
        )
        if targets is None:
            return 0.0
        if abs(targets.base_rotate_by) < MIN_BASE_ROTATION:
            break
        yaw -= targets.base_rotate_by
    return _wrap(yaw)


class FrankaStretchRetargeter:
    """
    Maps Franka joint actions to Stretch 4 joint targets and Stretch 4 joints back to the
    Franka state, for one virtual Franka (`franka`, fixed in the world).
    """

    def __init__(self, franka: FrankaSpawn, params: RetargetParams):
        self.franka = franka
        self.params = params
        self.franka_kinematics = FrankaKinematics()
        self.stretch_kinematics = stretch_kinematics()
        self.arm_offset = measure_arm_offset()
        # IK runs on the Stretch gripper model; this carries its TCP to the tool's.
        self.tool_correction = tool_tcp_correction(params.tool_name)
        self.tool_correction_inv = np.linalg.inv(self.tool_correction)
        # The IK model's arm runs `arm_offset` ahead of the real one; give it the real travel.
        for model in (self.stretch_kinematics.model, self.stretch_kinematics.model_ik):
            index = model.joints[model.getJointId("arm_l4_joint")].idx_q
            model.lowerPositionLimit[index] = ARM_RANGE[0] + END_STOP_MARGIN - self.arm_offset
            model.upperPositionLimit[index] = ARM_RANGE[1] - END_STOP_MARGIN - self.arm_offset
        self.tcp_offset = params.tcp_offset
        self.tcp_offset_inv = np.linalg.inv(self.tcp_offset)
        self.franka_seed = np.array(FRANKA_HOME_QPOS, dtype=float)
        self.last_state_q7 = np.array(FRANKA_HOME_QPOS, dtype=float)
        self.ik_failures = 0
        self.ik_clamped = 0
        self.at_start = True
        """The next solve picks a start pose: most room to the wrist limits, not least motion."""
        self.last_targets: StretchTargets | None = None
        self.last_target_tool_world: np.ndarray | None = None

    def set_grasp_offset(self, offset_mm, offset_deg=None) -> None:
        """
        Change the grasp offset (--grasp-offset-mm, and --grasp-offset-deg if given) from the next
        step on, in `params` too. What was worked out from it at the start (the custom start
        pose's pedestal, the Franka's yaw) stays as it was.
        """
        self.params.grasp_offset_mm = tuple(float(v) for v in offset_mm)
        if offset_deg is not None:
            self.params.grasp_offset_deg = tuple(float(v) for v in offset_deg)
        self.tcp_offset = self.params.tcp_offset
        self.tcp_offset_inv = np.linalg.inv(self.tcp_offset)

    # -- Franka -> Stretch --------------------------------------------------

    def stretch_tool_target_world(self, franka_q7) -> np.ndarray:
        """
        Where Stretch's `grasp_center_link` should be, in the world, for this Franka pose: the
        Franka's TCP in Stretch's axes, then the grasp offset in Stretch's own tool frame.
        """
        return self.franka.world_from_link0 @ self.franka_kinematics.fk(franka_q7) @ TCP_ALIGN @ self.tcp_offset

    def franka_to_stretch(
        self, action8: np.ndarray, world_from_footprint: np.ndarray, current: StretchJoints
    ) -> StretchTargets | None:
        """
        Stretch targets for a Franka action, from Stretch's current footprint pose and joints.

        Out of reach (Stretch's lift tops out ~0.12 m below a Franka TCP pointing down from
        1.2 m, for one), the closest pose within MAX_CLAMPED_* is used and counted in
        `ik_clamped`. Returns None, counted in `ik_failures`, if there is none.
        """
        from stretch4_kinematics import StretchJointPositions

        self.franka_seed = np.asarray(action8[:7], dtype=float)
        footprint_from_world = np.linalg.inv(world_from_footprint)
        seed = StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=0.0,
            lift=current.lift,
            arm=current.arm - self.arm_offset,
            wrist_yaw=current.wrist_yaw,
            wrist_pitch=current.wrist_pitch,
            wrist_roll=current.wrist_roll,
        )
        target_world = self.stretch_tool_target_world(action8[:7])
        self.last_target_tool_world = target_world
        goal = footprint_from_world @ target_world @ self.tool_correction_inv
        solution, clamped = solve_stretch_ik(self.stretch_kinematics, goal, seed, prefer_margin=self.at_start)
        reached = self.stretch_kinematics.forward(solution, STRETCH_TCP).homogeneous
        position_error = float(np.linalg.norm(reached[:3, 3] - goal[:3, 3]))
        rotation_error = _rotation_angle(reached[:3, :3].T @ goal[:3, :3])
        arm = solution.arm + self.arm_offset
        if (
            position_error > MAX_CLAMPED_POSITION_ERROR
            or rotation_error > MAX_CLAMPED_ROTATION_ERROR
            or not ARM_RANGE[0] - 1e-3 <= arm <= ARM_RANGE[1] + 1e-3
        ):
            self.ik_failures += 1
            return None
        lift = float(np.clip(solution.lift, LIFT_RANGE[0] + END_STOP_MARGIN, LIFT_RANGE[1] - END_STOP_MARGIN))
        arm_target = float(np.clip(arm, ARM_RANGE[0] + END_STOP_MARGIN, ARM_RANGE[1] - END_STOP_MARGIN))
        # Held off an end stop by more than a hair is not reaching the target either.
        clamped = clamped or abs(lift - solution.lift) > 1e-3 or abs(arm_target - arm) > 1e-3
        if clamped:
            self.ik_clamped += 1
        self.at_start = False  # once a pose is reached; until then the next solve may still pick

        targets = StretchTargets(
            base_rotate_by=_wrap(solution.base_theta),
            lift=lift,
            arm=arm_target,
            wrist_yaw=solution.wrist_yaw,
            wrist_pitch=solution.wrist_pitch,
            wrist_roll=solution.wrist_roll,
            gripper_closed=bool(action8[7] >= GRIPPER_CLOSED / 2),
            clamped=clamped,
            tool_error_m=position_error,
        )
        self.last_targets = targets
        return targets

    # -- Stretch -> Franka --------------------------------------------------

    def stretch_tool_world(self, world_from_footprint: np.ndarray, joints: StretchJoints) -> np.ndarray:
        from stretch4_kinematics import StretchJointPositions

        q = StretchJointPositions(
            0.0, 0.0, 0.0, joints.lift, joints.arm - self.arm_offset,
            joints.wrist_yaw, joints.wrist_pitch, joints.wrist_roll,
        )
        return world_from_footprint @ self.stretch_kinematics.forward(q, STRETCH_TCP).homogeneous @ self.tool_correction

    def stretch_to_franka(
        self, world_from_footprint: np.ndarray, joints: StretchJoints
    ) -> tuple[np.ndarray, bool]:
        """
        The Franka state (7 joints + Robotiq driver angle) whose TCP is where Stretch's tool is,
        with the grasp offset taken back out. Returns (state8, ik_converged).

        The Franka is redundant, so which of its arm poses this is depends on the seed: the
        last action commanded (see `franka_to_stretch()`), so that when Stretch tracks, the
        policy sees the arm where it put it.
        """
        tool_world = self.stretch_tool_world(world_from_footprint, joints) @ self.tcp_offset_inv
        franka_tcp = np.linalg.inv(self.franka.world_from_link0) @ tool_world @ TCP_ALIGN.T
        # When Stretch could not follow the last command, the arm pose it is in is nearer the
        # previous state than the command, so fall back on that, then on home.
        for seed in (self.franka_seed, self.last_state_q7, FRANKA_HOME_QPOS):
            q7, converged = self.franka_kinematics.ik(franka_tcp, seed)
            if converged:
                break
        self.franka_seed = q7
        self.last_state_q7 = q7
        driver = ROBOTIQ_DRIVER_OPEN + (1.0 - np.clip(joints.gripper_open_fraction, 0, 1)) * (
            ROBOTIQ_DRIVER_CLOSED - ROBOTIQ_DRIVER_OPEN
        )
        return np.concatenate([q7, [driver]]), converged

    def reset(self) -> None:
        self.franka_seed = np.array(FRANKA_HOME_QPOS, dtype=float)
        self.last_state_q7 = np.array(FRANKA_HOME_QPOS, dtype=float)
        self.ik_failures = 0
        self.ik_clamped = 0
        self.at_start = True
        self.last_targets = None
        self.last_target_tool_world = None


IK_EXACT_POSITION = 1e-3
IK_EXACT_ROTATION = math.radians(0.5)
"""m, rad. Closer than this to the target counts as reaching it rather than being clamped."""


MAX_STEP_JUMP = math.radians(45)
"""Most the base rotation or any wrist joint may move for one policy step. A policy step moves
each Franka joint 0.2 rad at most, so following it never needs more; a bigger change is the IK
swapping to another way of reaching the same pose (pointing down, yaw + 180 with roll - 180)."""

WRIST_LIMIT_MARGIN = 0.15
"""rad. Exact IK solutions with a wrist joint closer than this to a limit are passed over for
ones that are not, so the next step has room to move."""


def solve_stretch_ik(kinematics, target: np.ndarray, seed, prefer_margin: bool = False):
    """
    Stretch 4 IK for a `grasp_center_link` pose in the current base_footprint frame:
    (solution, clamped).

    The 6 DOF (base rotation, lift, arm, wrist yaw/pitch/roll) usually reach a pose several
    ways, and from a seed against a joint limit the solver stalls, so it starts from several
    seeds: the current joints, and canonical wrist poses with the base turned toward the target.
    Of the exact solutions it prefers, in order:
      1. every wrist joint WRIST_LIMIT_MARGIN inside its limits;
      2. the least change from the current joints, so the wrist follows the Franka's smoothly.
    Unless `prefer_margin`, only solutions within MAX_STEP_JUMP of the current base rotation and
    wrist joints count: when the pose needs the wrist swung round to another configuration, the
    closest pose reachable without that is used instead (clamped; solved with those joints
    bounded to MAX_STEP_JUMP around where they are), so the wrist never flips.

    With `prefer_margin` (for a start pose, where there is no motion to keep smooth) it prefers
    instead, of those WRIST_LIMIT_MARGIN inside the wrist limits, the one that turns the base the
    least (none, for a Franka mounted by `franka_yaw_for_start()`), and then the one with the most
    room to every wrist limit, so the moves that follow have room before they run into one.

    This is the solver behind `StretchKinematics.inverse_6dof_local()`, called directly because
    that keeps only exact solutions; for an unreachable target the closest one is wanted.
    """
    import pinocchio as pin
    from stretch4_kinematics import Stretch4IKModes, StretchJointPositions

    target_pose = pin.SE3(target[:3, :3], target[:3, 3])
    toward = math.atan2(target[1, 3], target[0, 3])
    seeds = [seed] + [
        StretchJointPositions(0, 0, base, seed.lift, seed.arm, yaw, pitch, roll)
        for base, yaw, pitch, roll in [
            # Camera-aligned, a Franka grasp pointing down needs Stretch's roll near 180 degrees:
            # mid-range, with ~65 degrees of room either way.
            (toward, 0, math.pi / 2, math.pi),
            (0, 0, math.pi / 2, math.pi),
            (toward, 0, math.pi / 2, 0),
            (toward, math.pi / 2, math.pi / 2, 0),
            (toward, 0, 0, 0),
            (0, 0, math.pi / 2, 0),
            (0, math.pi, math.pi / 2, 0),
            (0, math.pi / 2, 0, 0),
        ]
    ]
    model = kinematics.model_ik
    current = seed.to_numpy()
    exact: list[tuple[tuple, object]] = []
    best, best_error = None, np.inf
    for start in seeds:
        q = kinematics._closed_loop_inverse_kinematics(
            model,
            kinematics.data_ik,
            STRETCH_TCP,
            target_pose,
            q_guess=start.to_pinocchio_q(Stretch4IKModes.BASE_ROTATE),
            max_iter=300,
        )
        solution = StretchJointPositions.from_pinocchio_q(q)
        # base rotation and the wrist: indices 2, 5, 6, 7 of [x, y, theta, lift, arm, yaw, pitch, roll]
        jump = np.max(np.abs(solution.to_numpy()[[2, 5, 6, 7]] - current[[2, 5, 6, 7]]))
        if not prefer_margin and jump > MAX_STEP_JUMP:
            continue
        reached = kinematics.forward(solution, STRETCH_TCP)
        position_error = np.linalg.norm(reached.translation - target[:3, 3])
        rotation_error = _rotation_angle(reached.rotation.T @ target[:3, :3])
        if position_error < IK_EXACT_POSITION and rotation_error < IK_EXACT_ROTATION:
            # q is [base, lift, arm, yaw, pitch, roll]; the wrist is the last three.
            margin = min(np.min(q[3:] - model.lowerPositionLimit[3:]), np.min(model.upperPositionLimit[3:] - q[3:]))
            if prefer_margin:
                # A turn the base would skip anyway (MIN_BASE_ROTATION) counts as none.
                turn = max(abs(_wrap(solution.base_theta)) - MIN_BASE_ROTATION, 0.0)
                score = (margin < WRIST_LIMIT_MARGIN, turn, -margin)
            else:
                score = (margin < WRIST_LIMIT_MARGIN, float(np.linalg.norm(solution.to_numpy()[2:] - current[2:])))
            exact.append((score, solution))
            continue
        # Closest in position, among those not far off in orientation.
        error = position_error + (np.inf if rotation_error > MAX_CLAMPED_ROTATION_ERROR else 0)
        if best is None or error < best_error:
            best, best_error = solution, error
    if exact:
        return min(exact, key=lambda pair: pair[0])[1], False
    if prefer_margin:
        return best, True

    # The closest the base and wrist get to the pose moving at most MAX_STEP_JUMP from where
    # they are: the solver clips to its joint limits every iteration, so narrow those.
    lower, upper = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()
    start_q = seed.to_pinocchio_q(Stretch4IKModes.BASE_ROTATE)
    try:
        for index in (0, 3, 4, 5):  # base rotation, yaw, pitch, roll in [base, lift, arm, yaw, pitch, roll]
            model.lowerPositionLimit[index] = max(lower[index], start_q[index] - MAX_STEP_JUMP)
            model.upperPositionLimit[index] = min(upper[index], start_q[index] + MAX_STEP_JUMP)
        q = kinematics._closed_loop_inverse_kinematics(
            model, kinematics.data_ik, STRETCH_TCP, target_pose, q_guess=np.clip(start_q, model.lowerPositionLimit, model.upperPositionLimit), max_iter=300
        )
    finally:
        model.lowerPositionLimit[:], model.upperPositionLimit[:] = lower, upper
    bounded = StretchJointPositions.from_pinocchio_q(q)
    reached = kinematics.forward(bounded, STRETCH_TCP)
    error = np.linalg.norm(reached.translation - target[:3, 3]) + (
        np.inf if _rotation_angle(reached.rotation.T @ target[:3, :3]) > MAX_CLAMPED_ROTATION_ERROR else 0
    )
    return (best if best is not None and best_error <= error else bounded), True


def _rotation_angle(rotation: np.ndarray) -> float:
    return float(math.acos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def planar_transform(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    return make_transform(rotz(yaw), [x, y, z])


def planar_from_pose(pos, quat_wxyz, floor_z: float = 0.0) -> np.ndarray:
    """The footprint-on-floor transform of a body pose: its x, y and heading, at floor_z."""
    rotation = pose_to_transform(pos, quat_wxyz)[:3, :3]
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return planar_transform(pos[0], pos[1], yaw, floor_z)


# ---------------------------------------------------------------------------
# Stretch 4 in simulation
# ---------------------------------------------------------------------------


def stretch_cameras_to_use(params: RetargetParams):
    """Only the cameras the policy needs, so the simulator renders nothing else."""
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    gripper = {
        "left": StretchCameras.cam_gripper_se4_left_rgb,
        "right": StretchCameras.cam_gripper_se4_right_rgb,
    }[params.gripper_camera]
    cameras = [gripper]
    if params.exo_camera in HEAD_CAMERAS:
        # The simulator's center camera is the low-resolution one unless asked otherwise, and
        # it is reported under `cam_nav_rgb_se4_center`.
        cameras.append(
            StretchCameras.cam_nav_rgb_se4_center
            if params.exo_camera == "center"
            else HEAD_CAMERAS[params.exo_camera]
        )
    return cameras


SPAWN_LIFT_FRACTION = 0.9
"""Stretch spawns with its lift this far up its travel, clear of table tops."""

SPAWN_ARM_EXTENSION = 0.2
"""m. Stretch's arm goes this far out as it starts, nearer where the Franka's start poses need it."""


def spawn_stretch4(stretch_scene, params: RetargetParams):
    """
    A `Stretch4MujocoSimulator` for a `custom_scene.Stretch4Scene` (not started), rendering only
    the cameras `params` needs and publishing the robot's and the target object's poses.

    Stretch spawns with its lift SPAWN_LIFT_FRACTION up, and homes its arm SPAWN_ARM_EXTENSION out
    (see `spawn_with_arm_out()`): at the model's lift of 0 the gripper is often under a table,
    which `start()`'s homing would then drive it up into.
    """
    from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

    spawn_with_arm_out(stretch_scene.model, SPAWN_LIFT_FRACTION, SPAWN_ARM_EXTENSION)
    sim = Stretch4MujocoSimulator(
        model=stretch_scene.model,
        cameras_to_use=stretch_cameras_to_use(params),
        tool_name=params.tool_name,
    )
    sim.watch_bodies(stretch_scene.watched_bodies)
    return sim


def spawn_with_arm_out(model, lift_fraction: float, arm_extension: float) -> None:
    """
    Start Stretch's lift `lift_fraction` of the way up, and make the `home` keyframe that
    `Stretch4MujocoSimulator.start()` homes to hold it there and put the arm `arm_extension` (m)
    out. The other joints keep the model's home pose.

    The arm starts retracted and homing extends it: spawned already out at a raised lift, the
    robot settles onto the floor with its weight that far off centre, and its omni wheels let
    the base twist (~8 degrees, and 2 cm aside), which the start pose then turns back.
    """
    import mujoco

    low, high = model.joint("lift_joint").range
    lift = low + lift_fraction * (high - low)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    _start_slide_joint_at(model, "lift_joint", lift, home)
    if home != -1:
        model.key_ctrl[home][model.actuator("lift").id] = lift
        model.key_ctrl[home][model.actuator("arm").id] = arm_extension


def _start_slide_joint_at(model, name: str, value: float, home: int) -> None:
    """
    Start a slide joint at `value`, in the `home` keyframe too (if `home` is not -1).

    MuJoCo places a slide joint's body at `body_pos + axis * (qpos - qpos0)`, so raising `qpos0`
    alone would leave the link where it was while the joint reads higher. The link's origin is
    moved along the axis by the same amount, so the link is where its joint says.
    """
    import mujoco

    joint = model.joint(name)
    adr = joint.qposadr[0]
    body = joint.bodyid[0]
    axis_in_parent = np.zeros(3)
    mujoco.mju_rotVecQuat(axis_in_parent, model.jnt_axis[joint.id], model.body_quat[body])
    model.body_pos[body] += axis_in_parent * (value - model.qpos0[adr])
    model.qpos0[adr] = value
    if home != -1:
        model.key_qpos[home][adr] = value


def sim_gripper_open_aperture() -> float:
    """The simulator's gripper command (aperture angle, rad) for fully open."""
    from stretch4_mujoco.config import robot_settings_se4

    conversion = robot_settings_se4["gripper_conversion"]
    return 2 * math.asin(conversion["aperture_open_m"] / (2 * conversion["finger_length_m"]))


SIM_GRIPPER_CLOSED = 0.0
"""rad. Fingertips touching: the fingers' ctrlrange stops there, and with an object between them
the position actuators squeeze it in proportion to how far they are held open."""


def joint_speed(actuator: str, params: RetargetParams) -> float | None:
    """The velocity to command an actuator at: its default, or 20% of it with --slow."""
    from stretch4_mujoco.config import get_actuator_motion_limits

    if not params.slow:
        return None
    limits = get_actuator_motion_limits(actuator, "default")
    return None if limits is None else SLOW_FACTOR * limits[0]


class Stretch4SimEnv:
    """
    Stretch 4 in stretch4_mujoco, driven by Franka actions: what the rollout loop and the
    benchmark step, the same way they step `droid.FrankaDroidEnv`.

    `observe()` returns the Franka-equivalent state and Stretch's own camera images; `step()`
    retargets an action and moves Stretch, waiting for it to arrive with --wait-for-arrival
    and otherwise for one policy period of simulated time.
    """

    robot_root = "stretch4"

    def __init__(self, sim, stretch_scene, params: RetargetParams, mirror):
        self.sim = sim
        self.stretch_scene = stretch_scene
        self.params = params
        self.mirror = mirror
        self.retargeter = FrankaStretchRetargeter(stretch_scene.franka, params)
        self.gripper_open = sim_gripper_open_aperture()
        self._gripper_closed: bool | None = None
        self.ik_converged = True
        self.reverse_ik_failures = 0
        self.start_q7 = np.array(FRANKA_HOME_QPOS, dtype=float)
        """The Franka pose `reset()` goes back to."""
        self.overlay_franka_gripper = False
        """Show the policy the Franka in place of Stretch: the Robotiq's fingers for Stretch's
        gripper in the wrist view, the whole Franka for Stretch's arm in the exo view
        (`SceneMirror.render_with_franka()`; needs the ghost Franka)."""

    # -- state ------------------------------------------------------------

    def world_from_footprint(self) -> np.ndarray:
        pos, quat = self.sim.pull_body_poses()["stretch4"]
        return planar_from_pose(pos, quat, self.stretch_scene.scene.floor_z)

    def joints(self, status=None) -> StretchJoints:
        status = status or self.sim.pull_status()
        return StretchJoints(
            lift=status.lift.pos,
            arm=status.arm.pos,
            wrist_yaw=status.wrist_yaw.pos,
            wrist_pitch=status.wrist_pitch.pos,
            wrist_roll=status.wrist_roll.pos,
            gripper_open_fraction=float(np.clip(status.gripper.pos / self.gripper_open, 0, 1)),
        )

    def camera_images(self) -> tuple[np.ndarray | None, np.ndarray]:
        """(head image or None, gripper image), RGB and upright, as the simulator renders them."""
        cameras = self.sim.pull_camera_data()
        gripper_camera, *head = stretch_cameras_to_use(self.params)
        gripper = cameras.get_camera_data(gripper_camera, auto_correct_rgb=False)
        head_image = cameras.get_camera_data(head[0], auto_correct_rgb=False) if head else None
        return head_image, gripper

    def observe(self) -> Observation:
        status = self.sim.pull_status()
        poses = self.sim.pull_body_poses()
        footprint = planar_from_pose(*poses["stretch4"], self.stretch_scene.scene.floor_z)
        state8, self.ik_converged = self.retargeter.stretch_to_franka(footprint, self.joints(status))
        self.reverse_ik_failures += not self.ik_converged
        self.mirror.update(status, poses)
        self.mirror.set_ghost(state8[:7])

        head, gripper = self.camera_images()
        if self.params.exo_camera == "droid":
            from examples.vla.molmobot_droid.molmospaces.custom_scene import DROID_EXO_IN_STRETCH_SCENE

            exo = (
                self.mirror.render_with_franka(DROID_EXO_IN_STRETCH_SCENE, state8[7], "franka")
                if self.overlay_franka_gripper
                else self.mirror.render(DROID_EXO_IN_STRETCH_SCENE)
            )
        else:
            if self.overlay_franka_gripper:
                # The head camera with the Franka in place of Stretch's arm.
                head_camera = stretch_cameras_to_use(self.params)[1]
                exo = prepare_exo(self.mirror.render_with_franka(head_camera, state8[7], "franka"), self.params)
            else:
                exo = prepare_exo(head, self.params)
        extra = {"gripper_raw": gripper}
        if head is not None:
            extra["head_raw"] = head
        if self.overlay_franka_gripper:
            # The policy's wrist view with the Franka's gripper where it is told its hand is.
            gripper = self.mirror.render_with_franka(stretch_cameras_to_use(self.params)[0], state8[7], "fingers")
        return Observation(exo_rgb=exo, wrist_rgb=wrist_view(gripper), state8=state8, extra_cameras=extra)

    def tool_error_m(self) -> float | None:
        """How far Stretch's tool is from where the policy wants it."""
        target = self.retargeter.last_target_tool_world
        if target is None:
            return None
        actual = self.retargeter.stretch_tool_world(self.world_from_footprint(), self.joints())
        return float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))

    def render_scene(self) -> np.ndarray:
        from examples.vla.molmobot_droid.molmospaces.custom_scene import SCENE_CAMERA

        return self.mirror.render(SCENE_CAMERA, show_ghost=True)

    # -- acting -------------------------------------------------------------

    def step(self, action8: np.ndarray) -> StretchTargets | None:
        targets = self.retargeter.franka_to_stretch(action8, self.world_from_footprint(), self.joints())
        start = self.sim.pull_status().time
        if targets is not None:
            self.send(targets)
        else:
            self.send_gripper(bool(action8[7] >= GRIPPER_CLOSED / 2))
        self.wait(start)
        return targets

    def send(self, targets: StretchTargets, params: RetargetParams | None = None, rotate_base: bool = True) -> None:
        sim, params = self.sim, params or self.params
        if rotate_base and abs(targets.base_rotate_by) > MIN_BASE_ROTATION:
            sim.base.rotate_by(targets.base_rotate_by)
        sim.lift.move_to(targets.lift, v_m=joint_speed("lift", params))
        sim.arm.move_to(targets.arm, v_m=joint_speed("arm", params))
        sim.end_of_arm.wrist_yaw.move_to(targets.wrist_yaw, v_m=joint_speed("wrist_yaw", params))
        sim.end_of_arm.wrist_pitch.move_to(targets.wrist_pitch, v_m=joint_speed("wrist_pitch", params))
        sim.end_of_arm.wrist_roll.move_to(targets.wrist_roll, v_m=joint_speed("wrist_roll", params))
        self.send_gripper(targets.gripper_closed)

    def send_gripper(self, closed: bool) -> None:
        if closed == self._gripper_closed:
            return
        self._gripper_closed = closed
        # Both grippers take the simulator's aperture angle (stretch4_mujoco maps it per tool).
        gripper = self.sim.end_of_arm.parallel_gripper if self.params.use_parallel_gripper else self.sim.end_of_arm.stretch_gripper
        gripper.move_to(SIM_GRIPPER_CLOSED if closed else self.gripper_open)

    def wait(self, command_sim_time: float) -> None:
        if self.params.wait_for_arrival:
            self.sim.wait_command(timeout=STEP_ARRIVAL_TIMEOUT, check_interval=0.02)
        # Never less than one policy period, so the observation history keeps its spacing.
        while self.sim.is_running() and self.sim.pull_status().time < command_sim_time + POLICY_DT:
            time.sleep(0.005)

    def move_to_franka_pose(self, franka_q7=FRANKA_HOME_QPOS, timeout: float = 30.0) -> StretchTargets:
        """
        Put Stretch where the Franka at `franka_q7` would have its tool, gripper open, and wait
        for it. Do this before a rollout, so the policy starts from the state it expects.
        """
        self.retargeter.reset()
        self.retargeter.franka_seed = np.asarray(franka_q7, dtype=float)
        action = np.concatenate([franka_q7, [0.0]])
        targets = self.retargeter.franka_to_stretch(action, self.world_from_footprint(), self.joints())
        if targets is None:
            raise RuntimeError("Stretch 4 cannot reach the Franka's start pose from here")
        # Slowly: this can be a large wrist swing. The guarded contacts can still stop a joint on
        # the effort of it alone; a new command releases one, so send again until it arrives.
        slow = RetargetParams(**{**self.params.__dict__, "slow": True})
        deadline = time.monotonic() + timeout
        for attempt in range(5):
            self._gripper_closed = None
            # The base rotation is relative: send it once.
            self.send(targets, slow, rotate_base=attempt == 0)
            self.sim.wait_command(timeout=max(1.0, deadline - time.monotonic()), check_interval=0.05)
            if self._arrived(targets) or time.monotonic() > deadline:
                break
        self.retargeter.ik_failures = self.retargeter.ik_clamped = 0
        self.reverse_ik_failures = 0
        return targets

    def _arrived(self, targets: StretchTargets, tolerance: float = 0.02) -> bool:
        joints = self.joints()
        return all(
            abs(getattr(joints, name) - getattr(targets, name)) < tolerance
            for name in ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll")
        )

    def reset(self) -> None:
        """Back to `start_q7`, the Franka's home pose unless set (Stretch's base stays where it is)."""
        self.move_to_franka_pose(self.start_q7)

    def stats(self) -> dict[str, float]:
        return {
            "ik_failures": float(self.retargeter.ik_failures),
            "ik_clamped": float(self.retargeter.ik_clamped),
            "reverse_ik_failures": float(self.reverse_ik_failures),
            "furniture_removed": float(len(self.stretch_scene.removed_bodies)),
        }

    # -- scene state, for success checks ----------------------------------------

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.sim.pull_body_poses()[name]

    def body_contacts(self, name: str) -> list[str]:
        return self.sim.pull_body_contacts()[name]

    def close(self) -> None:
        self.mirror.close()
