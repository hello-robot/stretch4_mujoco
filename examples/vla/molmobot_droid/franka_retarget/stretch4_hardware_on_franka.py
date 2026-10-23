"""
Stretch 4's gripper and gripper camera on the Franka DROID, in place of the Robotiq and DROID's
wrist camera: to see whether the policy copes with them on the robot it was trained on
(run_franka's --use_stretch4_gripper and --use_stretch4_gripper_camera).

Both are mounted as on Stretch 4, with the Franka's flange in place of Stretch's wrist roll: the
tool's `quick_connect_interface_link` (its mating face, which all of the tool is in front of) on
the flange, where the Robotiq's `gripper/base` is, and the gripper camera where it is on Stretch 4
relative to the quick connect (the same for either tool, so the same with the Robotiq kept). They
are turned as retargeting turns Stretch's tool to the Franka's TCP (`TCP_ALIGN`): approaching
along the flange's axis, fingers on the Robotiq's finger axis, the camera on the side DROID's is.

The policy's TCP, `gripper/grasp_site`, stays where the Robotiq's is (155 mm from the flange).
The tool's own grasp center ends up further out: 235 mm for the Stretch gripper (SG4), 167 mm
for the parallel gripper (PG4). Retargeting's grasp offset does not apply.

- The gripper: the Robotiq's fingers, linkage and housing go, and its `gripper/base` body stays,
  empty, holding `grasp_site` and DROID's wrist camera. `FrankaStretch4ToolRobotView` drives
  and reads the tool's fingers in the Robotiq's units, so the policy's gripper command and
  state mean what they do on the Robotiq.
- The camera: Stretch's left gripper camera, its housing included, becomes the policy's wrist
  camera, rendered with stretch4_mujoco's intrinsics and without the Franka's arm
  (`Stretch4GripperCameraRenderer`). Its image wants DROID-framing, as retargeting does
  (`stretch4_retarget.wrist_view()`).

The camera's housing reaches 24 mm behind the mating face, beside `fr3_link7`, where on Stretch
the wrist roll is. So none of Stretch's parts collide with the Franka's arm they are mounted on,
and its camera does not see the arm, as on Stretch it would not.
"""

from __future__ import annotations

import functools

import mujoco
import numpy as np
from molmo_spaces.robots.robot_views.abstract import GripperGroup, MJCFFrameMixin, RobotView
from molmo_spaces.robots.robot_views.franka_fr3_view import FrankaFR3ArmGroup, FrankaFR3BaseGroup
from mujoco import MjData

from examples.vla.molmobot_droid.checkpoint import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    ROBOTIQ_DRIVER_CLOSED,
    ROBOTIQ_DRIVER_OPEN,
)
from examples.vla.molmobot_droid.droid import (
    FRANKA_PREFIX,
    GHOST_GEOM_GROUP,
    FrankaKinematics,
    FrankaOverlayView,
    FrankaSpawn,
    OverlayKind,
    StretchCameraRenderer,
    apply_stretch_camera_settings,
    body_transform,
    make_transform,
    mat_to_quat,
    site_transform,
    subtree_geoms,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    PARALLEL_GRIPPER_TOOL,
    STRETCH_GRIPPER_TOOL,
    STRETCH_TCP,
    TCP_ALIGN,
)
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

STRETCH4_GRIPPERS = {"sg4": STRETCH_GRIPPER_TOOL, "pg4": PARALLEL_GRIPPER_TOOL}
"""--use_stretch4_gripper's choices: the Stretch gripper and the parallel gripper."""

STRETCH4_GRIPPER_CAMERA = StretchCameras.cam_gripper_se4_left_rgb
"""The gripper camera that stands in for DROID's wrist camera: retargeting's default."""

WRIST_ROLL = "wrist_roll_link"
"""The body Stretch 4's tool and gripper camera both hang off. Put on the Franka without its own
joint and geometry, it holds whichever of the two are wanted."""

TOOL_ROOT = "tool_attachment_site_link"
"""Everything of the tool's, from where it attaches to the wrist roll. Its own geometry is the
wrist's mounting plate, behind the quick connect, and is left off."""

QUICK_CONNECT = "quick_connect_interface_link"
"""The tool's mating face: its origin is where the tool meets the wrist, and the whole tool is in
front of it (+x). It goes on the Franka's flange."""

CAMERA_ROOT = "gripper_camera_link"
"""The gripper camera's housing, holding its cameras."""

FINGER_ACTUATORS = ("gripper_left_finger", "gripper_right_finger")
"""Either tool's (stretch4_mujoco names the parallel gripper's finger actuators the same)."""

ARM_LINKS = tuple(f"fr3_link{i}" for i in range(8))

ROBOTIQ_FINGER_DIST_RANGE = (0.0, 0.087)
"""m. `RobotIQGripperGroup.inter_finger_dist_range`."""


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@functools.cache
def _robotiq_base_from_grasp_site() -> np.ndarray:
    """The Robotiq's `grasp_site` in its `gripper/base`, from molmospaces' Franka DROID."""
    kinematics = FrankaKinematics()
    gripper = f"{FRANKA_PREFIX}gripper/"
    return np.linalg.inv(body_transform(kinematics.data, f"{gripper}base")) @ site_transform(
        kinematics.data, f"{gripper}grasp_site"
    )


def _stretch4_wrist(tool_name: str, gripper: bool, gripper_camera: bool) -> tuple[mujoco.MjSpec, np.ndarray]:
    """
    Stretch 4's robot spec with `tool_name`, cut down to what goes on the Franka: WRIST_ROLL and
    TOOL_ROOT, without their joint and geometry, holding the tool if `gripper` and the camera if
    `gripper_camera`, with nothing else that names what is gone; and where WRIST_ROLL is
    relative to QUICK_CONNECT, turned to the tool's `grasp_center_link` axes. Compiled, so it
    attaches cleanly.
    """
    from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

    spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_robot_xml_path(tool_name))
    # Its contact pairs are with the scene's floor, which it is loaded without.
    for pair in list(spec.pairs):
        spec.delete(pair)
    model = spec.compile()
    data = MjData(model)
    mujoco.mj_kinematics(model, data)
    quick_connect = body_transform(data, QUICK_CONNECT)
    quick_connect[:3, :3] = body_transform(data, STRETCH_TCP)[:3, :3]
    mount_from_roll = np.linalg.inv(quick_connect) @ body_transform(data, WRIST_ROLL)

    keep_actuators = FINGER_ACTUATORS if gripper else ()
    for element in [*spec.keys, *spec.sensors, *(a for a in spec.actuators if a.name not in keep_actuators)]:
        spec.delete(element)
    roll = spec.body(WRIST_ROLL)
    for element in roll.joints:
        spec.delete(element)
    for body in (roll, spec.body(TOOL_ROOT)):
        for geom in body.geoms:
            spec.delete(geom)
        body.mass, body.inertia = 0.0, [0.0, 0.0, 0.0]
    for name, wanted in ((TOOL_ROOT, gripper), (CAMERA_ROOT, gripper_camera)):
        if not wanted:
            spec.delete(spec.body(name))
    for exclude in list(spec.excludes):
        if spec.body(exclude.bodyname1) is None or spec.body(exclude.bodyname2) is None:
            spec.delete(exclude)
    spec.compile()
    return spec, mount_from_roll


def _remove_robotiq(spec: mujoco.MjSpec, gripper_prefix: str) -> None:
    """Everything of the Robotiq's but its (emptied) `gripper/base`, which holds `grasp_site` and
    DROID's wrist camera."""
    for element in [*spec.actuators, *spec.tendons]:
        if element.name.startswith(gripper_prefix):
            spec.delete(element)
    for equality in list(spec.equalities):
        if equality.name1.startswith(gripper_prefix):
            spec.delete(equality)
    for exclude in list(spec.excludes):
        if exclude.bodyname1.startswith(gripper_prefix):
            spec.delete(exclude)
    base = spec.body(f"{gripper_prefix}base")
    for element in [*base.bodies, *base.geoms]:
        spec.delete(element)
    base.mass, base.inertia = 0.0, [0.0, 0.0, 0.0]


def put_stretch4_hardware_on(
    spec: mujoco.MjSpec, spawn: FrankaSpawn, gripper: str | None, gripper_camera: bool
) -> None:
    """
    Put Stretch 4's gripper (`gripper`: a STRETCH4_GRIPPERS key, or None to keep the Robotiq)
    and, with `gripper_camera`, its gripper camera on the Franka in `spec`, and say so in
    `spawn` (`stretch_tool`, `wrist_camera_name`, `wrist_stretch_camera`) for `FrankaDroidEnv`.
    None of it collides with the Franka's arm it is mounted on.
    """
    if gripper is None and not gripper_camera:
        return
    if gripper is not None and gripper not in STRETCH4_GRIPPERS:
        raise ValueError(f"gripper must be one of {tuple(STRETCH4_GRIPPERS)}, got '{gripper}'")
    tool_name = STRETCH4_GRIPPERS[gripper or "sg4"]
    stretch, mount_from_roll = _stretch4_wrist(tool_name, gripper is not None, gripper_camera)
    # The quick connect on the flange (`gripper/base`'s origin), turned as TCP_ALIGN turns the
    # tool's grasp center to the Robotiq's grasp_site.
    base_from_mount = make_transform(_robotiq_base_from_grasp_site()[:3, :3] @ TCP_ALIGN[:3, :3], [0.0, 0.0, 0.0])
    base_from_roll = base_from_mount @ mount_from_roll

    gripper_prefix = f"{spawn.prefix}gripper/"
    if gripper is not None:
        _remove_robotiq(spec, gripper_prefix)
        spawn.stretch_tool = tool_name
    roll = spec.body(f"{gripper_prefix}base").add_frame().attach_body(stretch.body(WRIST_ROLL), gripper_prefix, "")
    roll.pos = base_from_roll[:3, 3].tolist()
    roll.quat = mat_to_quat(base_from_roll[:3, :3]).tolist()
    for body in [roll, *roll.find_all("body")]:
        if body.geoms:
            for link in ARM_LINKS:
                spec.add_exclude(bodyname1=f"{spawn.prefix}{link}", bodyname2=body.name)
    if gripper_camera:
        spawn.wrist_camera_name = f"{gripper_prefix}{STRETCH4_GRIPPER_CAMERA.camera_name_in_mjcf}"
        spawn.wrist_stretch_camera = STRETCH4_GRIPPER_CAMERA


class Stretch4GripperCameraRenderer(StretchCameraRenderer):
    """
    The Stretch gripper camera on the Franka (`spawn.wrist_stretch_camera`), rendered as
    stretch4_mujoco renders it, with the Franka's arm (`fr3_link0` on, up to the gripper) hidden:
    the camera sits beside `fr3_link7`, where on Stretch its own wrist roll is. `model`'s geom
    groups are changed only while rendering.
    """

    def __init__(self, model: mujoco.MjModel, spawn: FrankaSpawn):
        apply_stretch_camera_settings(model, spawn.wrist_camera_name, spawn.wrist_stretch_camera)
        super().__init__(model, spawn.wrist_camera_name, spawn.wrist_stretch_camera)
        self.model = model
        gripper = set(subtree_geoms(model, f"{spawn.prefix}gripper/base"))
        self.arm = [g for g in subtree_geoms(model, spawn.link0_name) if g not in gripper]

    def render(self, data: mujoco.MjData) -> np.ndarray:
        saved = self.model.geom_group[self.arm].copy()
        self.model.geom_group[self.arm] = GHOST_GEOM_GROUP
        try:
            return super().render(data)
        finally:
            self.model.geom_group[self.arm] = saved


ROBOTIQ_WRIST_CAMERAS: dict[str, OverlayKind] = {
    "franka_wrist": "franka_alone",
    "franka_wrist_with_stretch4": "franka_with_stretch",
}
"""What DROID's wrist camera on the Franka DROID as it comes (`RobotiqGhost`) sees, by Rerun
camera name: without Stretch 4's hardware, and with it in view too."""


class RobotiqGhost:
    """
    The Franka DROID as it comes, Robotiq and DROID's wrist camera, posed as the Franka with
    Stretch 4's hardware on it (`env.ghost`, see `load_custom_scene_franka_droid()`): what its
    wrist camera would see (ROBOTIQ_WRIST_CAMERAS), as `run_stretch4_sim.py` shows for the ghost
    Franka over Stretch. It is posed in a copy of the sim's data, `data`, which leaves the sim
    alone and shows the ghost where the arm is in Rerun.
    """

    def __init__(self, env):
        self.env = env
        franka, ghost = env.spawn, env.ghost
        self.data = mujoco.MjData(env.model)
        self.joints = [(f"{franka.prefix}fr3_joint{i}", f"{ghost.prefix}fr3_joint{i}") for i in range(1, 8)]
        self.views = {
            name: FrankaOverlayView(
                env.model, ghost, ghost.wrist_camera_name, kind,
                hidden_root=franka.base_name if kind == "franka_alone" else None,
            )
            for name, kind in ROBOTIQ_WRIST_CAMERAS.items()
        }

    def pose(self) -> MjData:
        """`data` posed as the sim's, the ghost as the Franka, its Robotiq as open as the tool."""
        sim, data = self.env.data, self.data
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = sim.qpos, sim.mocap_pos, sim.mocap_quat
        for franka, ghost in self.joints:
            data.joint(ghost).qpos = sim.joint(franka).qpos
        next(iter(self.views.values())).pose_fingers(data, self.env.state8[7])
        return data

    def render(self) -> dict[str, np.ndarray]:
        """The wrist camera's views, of `data` as last posed."""
        return {name: view.render(self.data) for name, view in self.views.items()}

    def close(self) -> None:
        for view in self.views.values():
            view.close()


# ---------------------------------------------------------------------------
# Robot view
# ---------------------------------------------------------------------------


def _finger_range(tool_name: str) -> tuple[float, float]:
    """(closed, open) finger joint positions, as stretch4_mujoco commands the tool's fingers."""
    from stretch4_mujoco.config import robot_settings_se4, robot_settings_se4_pg4

    settings = robot_settings_se4_pg4 if tool_name == PARALLEL_GRIPPER_TOOL else robot_settings_se4
    conversion = settings["gripper_conversion"]
    return conversion["urdf_closed_rad"], conversion["urdf_open_rad"]


class Stretch4ToolGripperGroup(MJCFFrameMixin, GripperGroup):
    """
    A Stretch 4 tool on the Franka, in the Robotiq's units, so the policy's gripper output and
    input mean what they do on the Robotiq:

    - `ctrl`: one command, GRIPPER_OPEN .. GRIPPER_CLOSED, driving both fingers from open to
      closed as stretch4_mujoco does (closed is fingertips touching: with something between
      them, the position actuators squeeze it).
    - `joint_pos`: both Robotiq driver angles, at the fraction closed the fingers are on
      average: ROBOTIQ_DRIVER_OPEN .. ROBOTIQ_DRIVER_CLOSED, as retargeting reports Stretch's.
    """

    def __init__(self, mj_data: MjData, base_group: FrankaFR3BaseGroup, namespace: str, tool_name: str) -> None:
        model = mj_data.model
        gripper = f"{namespace}gripper/"
        act_ids = [model.actuator(f"{gripper}{name}").id for name in FINGER_ACTUATORS]
        joint_ids = [int(model.actuator_trnid[a, 0]) for a in act_ids]
        super().__init__(mj_data, joint_ids, act_ids, model.body(f"{gripper}base").id, base_group)
        self._ee_site_id = model.site(f"{gripper}grasp_site").id
        self._finger_closed, self._finger_open = _finger_range(tool_name)

    def _finger(self, open_fraction) -> np.ndarray:
        return self._finger_closed + np.asarray(open_fraction) * (self._finger_open - self._finger_closed)

    def _open_fraction(self, finger) -> float:
        fraction = (np.mean(finger) - self._finger_closed) / (self._finger_open - self._finger_closed)
        return float(np.clip(fraction, 0.0, 1.0))

    @property
    def joint_pos(self) -> np.ndarray:
        closed = 1.0 - self._open_fraction(self.mj_data.qpos[self._joint_posadr])
        return np.full(2, ROBOTIQ_DRIVER_OPEN + closed * (ROBOTIQ_DRIVER_CLOSED - ROBOTIQ_DRIVER_OPEN))

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        closed = (np.asarray(joint_pos, dtype=float) - ROBOTIQ_DRIVER_OPEN) / (ROBOTIQ_DRIVER_CLOSED - ROBOTIQ_DRIVER_OPEN)
        self.mj_data.qpos[self._joint_posadr] = self._finger(1.0 - np.clip(closed, 0.0, 1.0))

    @property
    def ctrl(self) -> np.ndarray:
        closed = 1.0 - self._open_fraction(self.mj_data.ctrl[self._actuator_ids[0]])
        return np.array([GRIPPER_OPEN + closed * (GRIPPER_CLOSED - GRIPPER_OPEN)])

    @ctrl.setter
    def ctrl(self, ctrl) -> None:
        closed = (float(np.asarray(ctrl).reshape(-1)[0]) - GRIPPER_OPEN) / (GRIPPER_CLOSED - GRIPPER_OPEN)
        self.mj_data.ctrl[self._actuator_ids] = self._finger(1.0 - np.clip(closed, 0.0, 1.0))

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = [GRIPPER_OPEN if open else GRIPPER_CLOSED]

    @property
    def inter_finger_dist(self) -> float:
        """The Robotiq's, at the fraction open the fingers are."""
        return self._open_fraction(self.mj_data.qpos[self._joint_posadr]) * ROBOTIQ_FINGER_DIST_RANGE[1]

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return ROBOTIQ_FINGER_DIST_RANGE

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world


class FrankaStretch4ToolRobotView(RobotView):
    """`FrankaDroidRobotView` with a Stretch 4 tool (`Stretch4ToolGripperGroup`) for its gripper."""

    def __init__(self, mj_data: MjData, namespace: str, tool_name: str) -> None:
        self._namespace = namespace
        self._tool_name = tool_name
        base = FrankaFR3BaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "arm": FrankaFR3ArmGroup(mj_data, base, namespace=namespace, grasp_site_name="gripper/grasp_site"),
            "gripper": Stretch4ToolGripperGroup(mj_data, base, namespace, tool_name),
        }
        super().__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}franka_{self._tool_name}"

    @property
    def base(self) -> FrankaFR3BaseGroup:
        return self._move_groups["base"]
