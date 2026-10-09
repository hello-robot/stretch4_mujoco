"""
The Franka DROID robot (FR3 + Robotiq 2F-85) that MolmoBot-DROID was trained on, built from
molmospaces' own robot model and spawn code so it matches training.

- `spawn_franka_droid()` adds it to a scene `MjSpec`, with one of four exo cameras: the
  molmospaces DROID shoulder camera (`droid`), or a Stretch 4 head camera (`left`, `right`,
  `center`) transplanted to the same place relative to the floor under the robot's footprint,
  turning with joint 1 as it would with Stretch's base.
- `add_franka_ghost()` adds a non-colliding, see-through copy for overlaying on Stretch 4.
- `FrankaKinematics` is FK/IK of the arm alone, used for retargeting.
- `FrankaDroidEnv` steps a compiled scene in-process, the way the policy was trained.

Height: molmospaces does not place the Franka relative to the floor. Its mocap base goes at
`target_object_z + ROBOT_OBJECT_Z_OFFSET` (plus noise in datagen), under a 0.58 m pedestal, so
`fr3_link0` sits ~0.17 m below the object. `franka_link0_height_for_object()` reproduces that.
"""

from __future__ import annotations

import contextlib
import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import mujoco
import numpy as np

from examples.vla.molmobot_droid.checkpoint import (
    DROID_IMAGE_SIZE,
    FRANKA_HOME_QPOS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    POLICY_DT,
    ROBOTIQ_DRIVER_OPEN,
)
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

ExoCamera = Literal["droid", "left", "right", "center"]
EXO_CAMERAS: tuple[str, ...] = ("droid", "left", "right", "center")

HEAD_CAMERAS: dict[str, StretchCameras] = {
    "left": StretchCameras.cam_nav_rgb_se4_left,
    "right": StretchCameras.cam_nav_rgb_se4_right,
    # What the simulator renders for the center camera unless asked for full resolution.
    "center": StretchCameras.cam_nav_rgb_se4_center_low_rez,
}

FRANKA_PREFIX = "robot_0/"

# molmospaces `FrankaDroidCameraSystem` / the MolmoBot demo notebook. Quaternions are wxyz.
DROID_EXO_POS = (0.1, 0.57, 0.66)
DROID_EXO_QUAT = (-0.3633, -0.1241, 0.4263, 0.8191)
DROID_EXO_FOVY = 71.0

ROBOT_OBJECT_Z_OFFSET = -0.75
"""molmospaces `robot_object_z_offset`: mocap base z relative to the target object's z."""

FRANKA_PEDESTAL_HEIGHT = 0.58
"""molmospaces `FrankaRobotConfig.base_size[2]`."""

MIN_PEDESTAL_HEIGHT = 0.05

GHOST_RGBA = (0.3, 0.7, 1.0, 0.35)

GHOST_GEOM_GROUP = 5
"""The ghost's geom group. Off by default in MuJoCo, so no robot camera (and so no policy)
sees the ghost; the scene camera and Rerun turn it on, and so can MuJoCo's viewer."""


# ---------------------------------------------------------------------------
# Poses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotPose:
    """A robot's footprint on the floor: x, y (m) and yaw (rad) in the scene frame."""

    x: float
    y: float
    yaw: float = 0.0

    @staticmethod
    def parse(text: str) -> "RobotPose":
        """From "x,y" or "x,y,yaw" (yaw in radians)."""
        values = [float(v) for v in text.split(",")]
        if len(values) not in (2, 3):
            raise ValueError(f"Expected 'x,y[,yaw]', got '{text}'")
        return RobotPose(*values)

    @property
    def quat_wxyz(self) -> list[float]:
        return [math.cos(self.yaw / 2), 0.0, 0.0, math.sin(self.yaw / 2)]

    def matrix(self, z: float = 0.0) -> np.ndarray:
        return make_transform(rotz(self.yaw), [self.x, self.y, z])

    def __str__(self) -> str:
        return f"{self.x:.3f},{self.y:.3f},{self.yaw:.3f}"


def rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def quat_to_mat(quat_wxyz) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=float))
    return mat.reshape(3, 3)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).reshape(9))
    return quat


def pose_to_transform(pos, quat_wxyz) -> np.ndarray:
    return make_transform(quat_to_mat(quat_wxyz), pos)


def body_transform(data: mujoco.MjData, name: str) -> np.ndarray:
    body = data.body(name)
    return make_transform(body.xmat.reshape(3, 3), body.xpos)


def site_transform(data: mujoco.MjData, name: str) -> np.ndarray:
    site = data.site(name)
    return make_transform(site.xmat.reshape(3, 3), site.xpos)


def franka_link0_height_for_object(object_z: float) -> float:
    """Height of `fr3_link0` molmospaces uses for a target object at `object_z` (the mean)."""
    return object_z + ROBOT_OBJECT_Z_OFFSET + FRANKA_PEDESTAL_HEIGHT


# ---------------------------------------------------------------------------
# Stretch 4 head cameras, for transplanting
# ---------------------------------------------------------------------------


@functools.cache
def stretch4_camera_poses() -> dict[str, np.ndarray]:
    """
    Pose of each Stretch 4 camera relative to its footprint on the floor (the `stretch4` root
    body, which is the URDF's `base_footprint`), from the stretch4_mujoco model itself.

    Keyed by the MJCF camera name. The head is rigid on Stretch 4, so the head cameras do not
    depend on the joints.
    """
    from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

    Stretch4MujocoSimulator.get_robot_xml_path()  # (re)generates the model the scene includes
    model = mujoco.MjModel.from_xml_path(Stretch4MujocoSimulator.get_scene_xml_path())
    data = mujoco.MjData(model)
    data.joint("floating_base").qpos = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    footprint_inv = np.linalg.inv(body_transform(data, "stretch4"))
    return {
        model.camera(i).name: footprint_inv
        @ make_transform(data.cam_xmat[i].reshape(3, 3), data.cam_xpos[i])
        for i in range(model.ncam)
    }


def apply_stretch_camera_settings(model: mujoco.MjModel, camera_name: str, camera: StretchCameras):
    """
    Give a camera in `model` the intrinsics stretch4_mujoco gives `camera`
    (`MujocoServerCameraManagerSync.set_camera_params()`), and room in the offscreen buffer.
    """
    cam_id = model.camera(camera_name).id
    settings = camera.initial_camera_settings
    model.cam_fovy[cam_id] = settings.field_of_view_vertical_in_degrees
    model.cam_intrinsic[cam_id] = list(settings.focal) + [0, 0]
    model.cam_resolution[cam_id] = (
        settings.sensor_resolution
        if settings.sensor_resolution is not None
        else (settings.width, settings.height)
    )
    if settings.sensor_size is not None:
        model.cam_sensorsize[cam_id] = settings.sensor_size


class StretchCameraRenderer:
    """
    Renders a camera the way stretch4_mujoco renders `camera` (fisheye included) and returns
    the image the way `StatusStretchCameras.get_camera_data(auto_correct_rgb=False)` does:
    RGB, rotated upright.
    """

    def __init__(self, model: mujoco.MjModel, camera_name: str, camera: StretchCameras):
        self.camera_name = camera_name
        self.camera = camera
        settings = camera.initial_camera_settings
        self.fisheye = camera.create_fisheye_renderer() if camera.is_fisheye else None
        width, height = self.fisheye.render_size if self.fisheye else (settings.width, settings.height)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False  # as the library does

    def render(self, data: mujoco.MjData) -> np.ndarray:
        if self.fisheye is not None:
            self.fisheye.render_views(self.renderer, data, self.camera_name)
            image = self.fisheye.project()
        else:
            self.renderer.update_scene(data, camera=self.camera_name)
            image = self.renderer.render()
        rotations = self.camera.initial_camera_settings.rotate_number_of_times
        return np.ascontiguousarray(np.rot90(image, rotations)) if rotations else image

    def close(self):
        self.renderer.close()


class PinholeRenderer:
    """Renders a plain MuJoCo camera at its own resolution, optionally with extra geom groups on."""

    def __init__(
        self, model: mujoco.MjModel, camera_name: str, size: tuple[int, int], geom_groups_on: tuple[int, ...] = ()
    ):
        self.camera_name = camera_name
        width, height = size
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.scene_option = mujoco.MjvOption()
        for group in geom_groups_on:
            self.scene_option.geomgroup[group] = 1

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera_name, scene_option=self.scene_option)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


@dataclass
class FrankaSpawn:
    """Where a Franka went and what it is called in the scene."""

    prefix: str
    robot_pose: RobotPose
    floor_z: float
    base_z: float
    pedestal_height: float
    exo_camera: str | None
    exo_camera_name: str | None
    wrist_camera_name: str

    @property
    def base_name(self) -> str:
        """The mocap body every Franka body hangs off: the robot's root for contact checks."""
        return f"{self.prefix}base"

    @property
    def link0_name(self) -> str:
        return f"{self.prefix}fr3_link0"

    @property
    def link1_name(self) -> str:
        return f"{self.prefix}fr3_link1"

    @property
    def grasp_site_name(self) -> str:
        return f"{self.prefix}gripper/grasp_site"

    @property
    def world_from_link0(self) -> np.ndarray:
        return self.robot_pose.matrix(self.base_z + self.pedestal_height)

    @property
    def world_from_footprint(self) -> np.ndarray:
        """The floor point under the robot, where Stretch 4's `base_footprint` would be."""
        return self.robot_pose.matrix(self.floor_z)


def _franka_config(pedestal_height: float | None):
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig

    return FrankaRobotConfig(
        base_size=None if pedestal_height is None else [0.5, 0.5, pedestal_height]
    )


def spawn_franka_droid(
    spec: mujoco.MjSpec,
    robot_pose: RobotPose,
    link0_height: float,
    exo_camera: ExoCamera | None = "droid",
    floor_z: float = 0.0,
    standing_on_floor: bool = True,
    prefix: str = FRANKA_PREFIX,
) -> FrankaSpawn:
    """
    Add the Franka DROID to `spec` with molmospaces' `FrankaRobot.add_robot_to_scene()`.

    Args:
        robot_pose: footprint x, y, yaw on the floor.
        link0_height: world z of `fr3_link0`. See `franka_link0_height_for_object()`.
        exo_camera: `droid` adds molmospaces' DROID shoulder camera on `fr3_link0`; `left`,
            `right`, `center` add that Stretch 4 head camera at its pose relative to the floor
            under `robot_pose` (with joint 1 at 0), on `fr3_link1`, so it turns with joint 1
            as it turns with Stretch's base. Named `<prefix>exo_camera_1` either way.
        floor_z: z of the floor under the robot.
        standing_on_floor: size the pedestal to reach the floor. Otherwise use molmospaces'
            0.58 m pedestal under a floating base, as the benchmarks do.
    """
    from molmo_spaces.robots.franka import FrankaRobot

    pedestal = link0_height - floor_z if standing_on_floor else FRANKA_PEDESTAL_HEIGHT
    if pedestal <= MIN_PEDESTAL_HEIGHT:
        raise ValueError(f"fr3_link0 at z={link0_height:.3f} is not above the floor at {floor_z:.3f}")
    base_z = link0_height - pedestal

    config = _franka_config(pedestal)
    config.robot_namespace = prefix
    FrankaRobot.add_robot_to_scene(
        config, spec, prefix=prefix, pos=[robot_pose.x, robot_pose.y, base_z], quat=robot_pose.quat_wxyz
    )
    FrankaRobot.apply_control_overrides(spec, config)

    wrist_camera_name = f"{prefix}gripper/wrist_camera"
    spec.camera(wrist_camera_name).resolution = list(DROID_IMAGE_SIZE)

    spawn = FrankaSpawn(
        prefix=prefix,
        robot_pose=robot_pose,
        floor_z=floor_z,
        base_z=base_z,
        pedestal_height=pedestal,
        exo_camera=exo_camera,
        exo_camera_name=f"{prefix}exo_camera_1" if exo_camera else None,
        wrist_camera_name=wrist_camera_name,
    )

    if exo_camera == "droid":
        spec.body(spawn.link0_name).add_camera(
            name=spawn.exo_camera_name,
            pos=list(DROID_EXO_POS),
            quat=list(DROID_EXO_QUAT),
            fovy=DROID_EXO_FOVY,
            resolution=list(DROID_IMAGE_SIZE),
        )
    elif exo_camera in HEAD_CAMERAS:
        stretch_camera = HEAD_CAMERAS[exo_camera]
        footprint_from_camera = stretch4_camera_poses()[stretch_camera.camera_name_in_mjcf]
        # Express the camera in fr3_link1 as it is with joint 1 at 0. Joint 1 turns about
        # fr3_link0's z, which stands over the footprint, as Stretch's base turns about it.
        link1 = spec.body(spawn.link1_name)
        world_from_link1 = spawn.world_from_link0 @ pose_to_transform(link1.pos, link1.quat)
        link1_from_camera = np.linalg.inv(world_from_link1) @ spawn.world_from_footprint @ footprint_from_camera
        link1.add_camera(
            name=spawn.exo_camera_name,
            pos=link1_from_camera[:3, 3].tolist(),
            quat=mat_to_quat(link1_from_camera[:3, :3]).tolist(),
        )
    elif exo_camera is not None:
        raise ValueError(f"exo_camera must be one of {EXO_CAMERAS}, got '{exo_camera}'")

    return spawn


def add_franka_ghost(
    spec: mujoco.MjSpec,
    robot_pose: RobotPose,
    link0_height: float,
    arm_qpos=FRANKA_HOME_QPOS,
    floor_z: float = 0.0,
    prefix: str = "ghost_franka/",
    rgba=GHOST_RGBA,
) -> FrankaSpawn:
    """
    Add a see-through Franka that nothing collides with, for overlaying on Stretch 4.

    It has no actuators, so a simulator driving its own robot never commands it, and joint
    springs carry it to `arm_qpos` and hold it there. (Starting it there instead would mean
    changing `qpos0`, which MuJoCo also takes as the joints' zero.) Its geoms are in
    GHOST_GEOM_GROUP: in MuJoCo's viewer, turn that group on (key 5, or Rendering > Geom groups)
    to see it.
    """
    spawn = spawn_franka_droid(
        spec, robot_pose, link0_height, exo_camera=None, floor_z=floor_z, prefix=prefix
    )
    # Some molmospaces scenes declare actuators too, so only the ghost's own are removed.
    for actuator in list(spec.actuators):
        if actuator.name.startswith(prefix):
            spec.delete(actuator)
    for sensor in list(spec.sensors):
        if sensor.name.startswith(prefix):
            spec.delete(sensor)

    base = spec.body(spawn.base_name)
    for geom in base.find_all("geom"):
        if geom.contype or geom.conaffinity:
            if geom.group == 3:  # a collision mesh doubling a visual one
                spec.delete(geom)
                continue
            geom.contype = 0
            geom.conaffinity = 0
        geom.group = GHOST_GEOM_GROUP
        geom.rgba = list(rgba)
        geom.material = ""
    for body in base.find_all("body"):
        body.gravcomp = 1.0
    for index, joint_name in enumerate(_arm_joint_names(prefix)):
        joint = spec.joint(joint_name)
        joint.springref = float(arm_qpos[index])
        joint.stiffness = 200.0
        joint.damping = 20.0
    for camera in base.find_all("camera"):
        spec.delete(camera)
    return spawn


def resize_franka_pedestal(model: mujoco.MjModel, spawn: FrankaSpawn, height: float) -> None:
    """
    Raise or lower a Franka's `fr3_link0` to `height` above its base, in `model` and `spawn`,
    resizing the pedestal under it to match. Anything else that should follow is the caller's.
    """
    if height <= MIN_PEDESTAL_HEIGHT:
        raise ValueError(f"The pedestal must be taller than {MIN_PEDESTAL_HEIGHT} m, got {height}")
    # fr3_link0 hangs off the base at the top of the pedestal, the base's (first) box.
    model.body_pos[model.body(spawn.link0_name).id, 2] += height - spawn.pedestal_height
    base = model.body(spawn.base_name).id
    geom = next(
        g for g in range(model.ngeom) if model.geom_bodyid[g] == base and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    )
    model.geom_size[geom, 2] = model.geom_pos[geom, 2] = model.geom_aabb[geom, 5] = height / 2
    model.geom_rbound[geom] = np.linalg.norm(model.geom_size[geom])
    spawn.pedestal_height = height


def _arm_joint_names(prefix: str) -> list[str]:
    return [f"{prefix}fr3_joint{i + 1}" for i in range(7)]


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


class FrankaKinematics:
    """
    FK and IK of the FR3 arm to the Robotiq `grasp_site` (the policy's TCP), in the
    `fr3_link0` frame, on a standalone copy of molmospaces' model.
    """

    def __init__(self):
        from molmo_spaces.robots.franka import FrankaRobot

        spec = mujoco.MjSpec()
        config = _franka_config(None)
        FrankaRobot.add_robot_to_scene(config, spec, prefix=FRANKA_PREFIX, pos=[0, 0, 0], quat=[1, 0, 0, 0])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        joint_ids = [self.model.joint(name).id for name in _arm_joint_names(FRANKA_PREFIX)]
        self._qpos_adr = np.array([self.model.jnt_qposadr[j] for j in joint_ids])
        self._dof_adr = np.array([self.model.jnt_dofadr[j] for j in joint_ids])
        self.lower = self.model.jnt_range[joint_ids, 0].copy()
        self.upper = self.model.jnt_range[joint_ids, 1].copy()
        self._site_id = self.model.site(f"{FRANKA_PREFIX}gripper/grasp_site").id
        mujoco.mj_kinematics(self.model, self.data)
        self._world_from_link0 = body_transform(self.data, f"{FRANKA_PREFIX}fr3_link0")
        self._link0_from_world = np.linalg.inv(self._world_from_link0)

    def _set(self, q7) -> None:
        self.data.qpos[self._qpos_adr] = q7
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q7) -> np.ndarray:
        """4x4 pose of `grasp_site` in `fr3_link0`."""
        self._set(q7)
        return self._link0_from_world @ site_transform(self.data, f"{FRANKA_PREFIX}gripper/grasp_site")

    def ik(
        self,
        target: np.ndarray,
        seed,
        max_iterations: int = 200,
        position_tolerance: float = 1e-4,
        rotation_tolerance: float = 1e-3,
        damping: float = 1e-4,
        posture_gain: float = 0.05,
    ) -> tuple[np.ndarray, bool]:
        """
        Damped least squares IK for a `grasp_site` pose in `fr3_link0`.

        The arm is redundant, so the null space is pulled towards `seed`; seeding with the
        previous solution keeps the elbow from flipping between steps. Returns (q7, converged).
        """
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        q = seed.copy()
        target_world = self._world_from_link0 @ target
        target_quat = mat_to_quat(target_world[:3, :3])
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        current_quat = np.zeros(4)
        error_rot = np.zeros(3)
        for _ in range(max_iterations):
            self._set(q)
            site = self.data.site(self._site_id)
            error_pos = target_world[:3, 3] - site.xpos
            mujoco.mju_mat2Quat(current_quat, site.xmat)
            mujoco.mju_subQuat(error_rot, target_quat, current_quat)
            # mju_subQuat's result is in the site frame; the Jacobian is in the world frame.
            error_rot_world = site.xmat.reshape(3, 3) @ error_rot
            if (
                np.linalg.norm(error_pos) < position_tolerance
                and np.linalg.norm(error_rot_world) < rotation_tolerance
            ):
                return q, True
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._site_id)
            jac = np.vstack([jac_pos, jac_rot])[:, self._dof_adr]
            error = np.concatenate([error_pos, error_rot_world])
            jjt = jac @ jac.T + damping * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, error)
            null = np.eye(7) - jac.T @ np.linalg.solve(jjt, jac)
            dq += null @ (posture_gain * (seed - q))
            q = np.clip(q + dq, self.lower, self.upper)
        self._set(q)
        return q, False


@functools.cache
def robotiq_linkage(steps: int = 26) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    The Robotiq 2F-85's joint positions across its travel: (joint names after the robot's
    prefix, the left driver joint's angle at each of `steps` openings, each opening's joint
    positions in that order). Interpolate on the driver angle to pose the fingers kinematically.

    The fingers are four-bar linkages closed by equality constraints, which only the physics
    solves, so the gripper's own actuator is stepped through its range on a standalone
    molmospaces Franka, arm held at home and gravity off.
    """
    model = FrankaKinematics().model
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    prefix = f"{FRANKA_PREFIX}gripper/"
    names = [model.joint(j).name for j in range(model.njnt) if model.joint(j).name.startswith(prefix)]
    qpos_adr = [model.joint(name).qposadr[0] for name in names]
    driver = model.joint(f"{prefix}left_driver_joint").qposadr[0]
    arm = [model.joint(name).qposadr[0] for name in _arm_joint_names(FRANKA_PREFIX)]
    actuator = model.actuator(f"{prefix}fingers_actuator").id
    data.qpos[arm] = data.ctrl[: len(arm)] = FRANKA_HOME_QPOS
    drivers, table = [], []
    for ctrl in np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, steps):
        data.ctrl[actuator] = ctrl
        for _ in range(1500):
            mujoco.mj_step(model, data)
        drivers.append(data.qpos[driver])
        table.append(data.qpos[qpos_adr].copy())
    return [name[len(FRANKA_PREFIX):] for name in names], np.array(drivers), np.array(table)


@functools.cache
def franka_visual_look() -> tuple[list[str], np.ndarray, list[str | None]]:
    """
    The look `add_franka_ghost()` tints away, for every geom it keeps, in model order: (their
    bodies' names after the robot's prefix, their colours, their materials' names after the
    prefix or None), from a standalone molmospaces Franka DROID.
    """
    spec = mujoco.MjSpec()
    spawn = spawn_franka_droid(spec, RobotPose(0.0, 0.0, 0.0), FRANKA_PEDESTAL_HEIGHT, exo_camera=None)
    model = spec.compile()
    prefix = spawn.prefix
    # The geoms add_franka_ghost() deletes: collision meshes doubling visual ones.
    kept = [
        g for g in subtree_geoms(model, spawn.base_name)
        if not (model.geom_group[g] == 3 and (model.geom_contype[g] or model.geom_conaffinity[g]))
    ]
    bodies = [model.body(model.geom_bodyid[g]).name[len(prefix):] for g in kept]
    rgba = np.array([model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g] for g in kept])
    materials = [model.material(model.geom_matid[g]).name[len(prefix):] if model.geom_matid[g] >= 0 else None for g in kept]
    return bodies, rgba, materials


def subtree_geoms(model: mujoco.MjModel, body_name: str) -> list[int]:
    """The geoms of a body and of every body under it, in model order."""
    root = model.body(body_name).id

    def under(body: int) -> bool:
        while body:
            if body == root:
                return True
            body = int(model.body_parentid[body])
        return False

    return [g for g in range(model.ngeom) if under(int(model.geom_bodyid[g]))]


def robotiq_joint_positions(driver_angle: float) -> dict[str, float]:
    """The Robotiq's joint positions (by name after the robot's prefix) at a driver joint angle."""
    names, drivers, table = robotiq_linkage()
    return {name: float(np.interp(driver_angle, drivers, table[:, i])) for i, name in enumerate(names)}


STRETCH_TOOL_ROOT = "wrist_roll_link"
"""Everything of Stretch's from here on is its tool: the gripper, its cameras and fingers."""

STRETCH_ARM_ROOT = "lift_link"
"""Everything of Stretch's from here on is its arm: the lift carriage, the telescoping arm, the
wrist and the tool. The base, the mast and the head stay."""

OverlayKind = Literal["fingers", "franka"]


class FrankaOverlayView:
    """
    A camera in `model` (Stretch with a ghost Franka from `add_franka_ghost()`), rendered with
    part of Stretch hidden and the ghost shown instead in the Franka's own look, where the
    policy is told its arm and hand are:

      fingers: Stretch's tool hidden, the Robotiq's fingers shown. For Stretch's gripper camera,
          which sits where the Robotiq's housing would be (DROID's wrist camera, beside it, sees
          only the fingers), so the housing and the arm stay hidden.
      franka: Stretch's arm hidden (from the lift carriage on), all of the Franka shown, its
          pedestal too. For the exo views: what the Franka DROID would show there.

    `camera` is one of Stretch's (rendered as stretch4_mujoco renders it, fisheye included,
    with its intrinsics given to `model`) or a camera of `model`'s own, rendered as is. `model`'s
    geom groups, colours and materials are changed only while rendering.
    """

    MASK_COLOUR = (1.0, 0.0, 1.0, 1.0)
    MASK_THRESHOLD = 30
    """Masks are where drawing the geoms in this flat colour changes the image from drawing none of
    them, by more than this in some channel: segmentation cannot go through the fisheye's
    stitched views."""

    def __init__(self, model: mujoco.MjModel, franka: FrankaSpawn, camera: StretchCameras | str, kind: OverlayKind):
        self.model, self.franka, self.kind = model, franka, kind
        if isinstance(camera, StretchCameras):
            name = camera.camera_name_in_mjcf
            apply_stretch_camera_settings(model, name, camera)
            self.renderer = StretchCameraRenderer(model, name, camera)
        else:
            size = tuple(int(v) for v in model.cam_resolution[model.camera(camera).id])
            self.renderer = PinholeRenderer(model, camera, size)
        self.hidden = subtree_geoms(model, STRETCH_TOOL_ROOT if kind == "fingers" else STRETCH_ARM_ROOT)
        self.ghost = subtree_geoms(model, franka.base_name)
        bodies, self.rgba, materials = franka_visual_look()
        ghost_bodies = [model.body(model.geom_bodyid[g]).name[len(franka.prefix):] for g in self.ghost]
        if ghost_bodies != bodies:
            raise RuntimeError("The ghost Franka's geoms do not match molmospaces' Franka DROID's")
        # The ghost's materials are gone from its geoms, but still in the model.
        self.matid = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, franka.prefix + name) if name else -1
            for name in materials
        ])
        if kind == "fingers":
            housing = model.body(f"{franka.prefix}gripper/base").id
            gripper = set(subtree_geoms(model, f"{franka.prefix}gripper/base"))
            self.shown = [g for g in self.ghost if g in gripper and model.geom_bodyid[g] != housing]
        else:
            self.shown = list(self.ghost)

    def pose_fingers(self, data: mujoco.MjData, driver_angle: float) -> None:
        """Open the ghost's Robotiq to `driver_angle` (the Franka state's last value) in `data`."""
        for name, value in robotiq_joint_positions(driver_angle).items():
            data.joint(self.franka.prefix + name).qpos = value
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_camlight(self.model, data)

    @contextlib.contextmanager
    def _showing(self, geoms: list[int], flat: tuple | None = None):
        """
        Of Stretch's hidden part and the ghost, only `geoms` drawn (in a group the renderer
        draws; the rest in the ghost's, which it does not), the ghost in the Franka's look, or
        `geoms` all in the `flat` colour.
        """
        model, regrouped = self.model, self.hidden + self.ghost
        saved = (model.geom_group[regrouped].copy(), model.geom_rgba[regrouped].copy(), model.geom_matid[regrouped].copy())
        model.geom_group[regrouped] = GHOST_GEOM_GROUP
        model.geom_group[geoms] = 0
        model.geom_rgba[self.ghost], model.geom_matid[self.ghost] = self.rgba, self.matid
        if flat is not None:
            model.geom_rgba[geoms], model.geom_matid[geoms] = flat, -1
        try:
            yield
        finally:
            model.geom_group[regrouped], model.geom_rgba[regrouped], model.geom_matid[regrouped] = saved

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """The camera's image, with the Franka in place of Stretch's part."""
        with self._showing(self.shown):
            return self.renderer.render(data)

    def masks(self, data: mujoco.MjData, size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        (where the Franka's shown geoms are, where Stretch's hidden part is) in the image, at
        `size` (width, height) if given: smaller is quicker to compare.
        """
        import cv2

        def render() -> np.ndarray:
            image = self.renderer.render(data)
            return image if size is None else cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        with self._showing([]):
            neither = render()
        masks = []
        for geoms in (self.shown, self.hidden):
            with self._showing(geoms, flat=self.MASK_COLOUR):
                image = render()
            masks.append(cv2.absdiff(image, neither).max(axis=-1) > self.MASK_THRESHOLD)
        return tuple(masks)

    def close(self) -> None:
        self.renderer.close()


# ---------------------------------------------------------------------------
# In-process environment
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    exo_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state8: np.ndarray
    """7 FR3 joint positions + the Robotiq driver joint angle: the policy's state input."""
    extra_cameras: dict[str, np.ndarray] = field(default_factory=dict)


class FrankaDroidEnv:
    """
    A compiled scene with a Franka DROID in it, stepped in this process at the policy's rate.

    `launch_viewer()` opens MuJoCo's passive viewer, kept in sync on every `step()`.
    `prepare_exo`, if set, is applied to the exo image in `observe()` (e.g. a head-camera crop).
    """

    def __init__(self, model: mujoco.MjModel, spawn: FrankaSpawn, scene_camera: str | None = None):
        from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView

        self.model = model
        self.data = mujoco.MjData(model)
        self.spawn = spawn
        self.view = FrankaDroidRobotView(self.data, spawn.prefix)
        self.n_substeps = max(1, round(POLICY_DT / model.opt.timestep))
        self.viewer = None
        self.prepare_exo: Callable[[np.ndarray], np.ndarray] | None = None

        if spawn.exo_camera in HEAD_CAMERAS:
            apply_stretch_camera_settings(model, spawn.exo_camera_name, HEAD_CAMERAS[spawn.exo_camera])
            self.exo_renderer = StretchCameraRenderer(
                model, spawn.exo_camera_name, HEAD_CAMERAS[spawn.exo_camera]
            )
        else:
            self.exo_renderer = PinholeRenderer(model, spawn.exo_camera_name, DROID_IMAGE_SIZE)
        self.wrist_renderer = PinholeRenderer(model, spawn.wrist_camera_name, DROID_IMAGE_SIZE)
        self.scene_renderer = (
            PinholeRenderer(model, scene_camera, (960, 540)) if scene_camera else None
        )
        self.reset()

    def reset(self, arm_qpos=FRANKA_HOME_QPOS) -> None:
        # Resetting also puts the mocap base back where `spawn_franka_droid()` placed it.
        mujoco.mj_resetData(self.model, self.data)
        arm = self.view.get_move_group("arm")
        gripper = self.view.get_move_group("gripper")
        arm.joint_pos = np.asarray(arm_qpos, dtype=float)
        arm.ctrl = np.asarray(arm_qpos, dtype=float)
        gripper.joint_pos = np.array([ROBOTIQ_DRIVER_OPEN, ROBOTIQ_DRIVER_OPEN])
        gripper.ctrl = [0.0]
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()

    def set_pedestal_height(self, height: float) -> None:
        """
        Raise or lower `fr3_link0` by resizing the pedestal under it, in the model, so `reset()`
        keeps it. A head exo camera stays where it was relative to the floor.
        """
        model = self.model
        delta = height - self.spawn.pedestal_height
        resize_franka_pedestal(model, self.spawn, height)
        if self.spawn.exo_camera in HEAD_CAMERAS:
            # It hangs off fr3_link1, whose z is the world's.
            model.cam_pos[model.camera(self.spawn.exo_camera_name).id, 2] -= delta
        mujoco.mj_forward(model, self.data)
        self.sync_viewer()

    @property
    def robot_root(self) -> str:
        """The robot's root body, as contact reports name it."""
        return self.spawn.base_name

    @property
    def state8(self) -> np.ndarray:
        arm = self.view.get_move_group("arm").joint_pos
        gripper = self.view.get_move_group("gripper").joint_pos
        return np.concatenate([arm, gripper[:1]]).astype(float)

    def observe(self) -> Observation:
        exo_rgb = self.exo_renderer.render(self.data)
        return Observation(
            exo_rgb=self.prepare_exo(exo_rgb) if self.prepare_exo is not None else exo_rgb,
            wrist_rgb=self.wrist_renderer.render(self.data),
            state8=self.state8,
        )

    def render_scene(self) -> np.ndarray | None:
        return self.scene_renderer.render(self.data) if self.scene_renderer else None

    def step(self, action8: np.ndarray) -> None:
        """Command the arm joint positions and gripper, then simulate one policy step."""
        self.view.get_move_group("arm").ctrl = np.asarray(action8[:7], dtype=float)
        self.view.get_move_group("gripper").ctrl = [float(action8[7])]
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.sync_viewer()

    def tcp_world(self) -> np.ndarray:
        return site_transform(self.data, self.spawn.grasp_site_name)

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        body = self.data.body(name)
        return body.xpos.copy(), body.xquat.copy()

    def body_contacts(self, name: str) -> list[str]:
        """Root-body names of everything touching `name`'s subtree (as `pull_body_contacts()`)."""
        return contacts_of_subtree(self.model, self.data, self.model.body(name).id)

    def launch_viewer(self) -> None:
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer.cam.lookat[:] = self.spawn.world_from_link0[:3, 3]
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = math.degrees(self.spawn.robot_pose.yaw) + 180
        self.viewer.cam.elevation = -25

    def sync_viewer(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        for renderer in (self.exo_renderer, self.wrist_renderer, self.scene_renderer):
            if renderer is not None:
                renderer.close()


def contacts_of_subtree(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> list[str]:
    """Root-body names of everything touching the subtree under `body_id`, itself excluded."""

    def in_subtree(b: int) -> bool:
        while b != 0:
            if b == body_id:
                return True
            b = int(model.body_parentid[b])
        return body_id == 0

    touching = set()
    for contact in data.contact:
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        for this_body, other_body in ((body1, body2), (body2, body1)):
            if in_subtree(this_body) and not in_subtree(other_body):
                touching.add(model.body(int(model.body_rootid[other_body])).name or "world")
    return sorted(touching)
