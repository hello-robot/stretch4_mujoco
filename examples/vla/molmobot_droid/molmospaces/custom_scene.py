"""
Run MolmoBot-DROID in any MolmoSpaces house, on a chosen object, with either robot.

    scene = load_custom_scene("procthor-10k/val/0", "mug")
    pose = suggest_robot_pose(scene)                       # or RobotPose(x, y, yaw)

    env = load_custom_scene_franka_droid(scene, pose, exo_camera="droid")
    stretch = load_custom_scene_stretch4(scene, pose, include_franka=True)

Scenes, objects and their assets come straight from molmospaces. Stretch 4 is attached with
`examples/molmo_environment.py`'s `add_stretch_to_scene()`, i.e. the stretch4_mujoco MJCF, and
the result is meant for `Stretch4MujocoSimulator(model=...)`.

The virtual Franka is placed the way molmospaces places it in training: `fr3_link0` at
`franka_link0_height_for_object(object z)`, with its footprint at the robot pose. With
`include_franka`, Stretch 4's scene gets a see-through, non-colliding Franka there too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from examples.vla.molmobot_droid.droid import (
    DROID_EXO_FOVY,
    DROID_EXO_POS,
    DROID_EXO_QUAT,
    DROID_IMAGE_SIZE,
    FRANKA_HOME_QPOS,
    GHOST_GEOM_GROUP,
    ExoCamera,
    FrankaDroidEnv,
    FrankaSpawn,
    PinholeRenderer,
    RobotPose,
    add_franka_ghost,
    franka_link0_height_for_object,
    mat_to_quat,
    pose_to_transform,
    spawn_franka_droid,
)

SCENE_CAMERA = "scene_camera"
SCENE_CAMERA_SIZE = (960, 540)
DROID_EXO_IN_STRETCH_SCENE = "droid_exo_camera"
STRETCH_ROOT_BODY = "stretch4"


@dataclass
class CustomScene:
    """A MolmoSpaces house with a target object picked out."""

    scene_id: str
    scene_xml: str
    object_name: str
    """Body name of the target object (a free body)."""
    object_pos: np.ndarray
    object_quat: np.ndarray
    floor_z: float = 0.0
    object_poses: dict[str, list[float]] = field(default_factory=dict)
    """Episode overrides, {body: [x, y, z, qw, qx, qy, qz]}; see `apply_object_poses()`."""
    added_objects: dict[str, str] = field(default_factory=dict)
    """Episode additions, {body path: asset xml relative to molmospaces' ASSETS_DIR}."""
    removed_objects: list[str] = field(default_factory=list)

    @property
    def object_category(self) -> str:
        return object_category(self.object_name)

    def load_spec(self) -> mujoco.MjSpec:
        """A fresh spec of the house with the episode's modifications applied."""
        spec = mujoco.MjSpec.from_file(self.scene_xml)
        remove_objects(spec, self.removed_objects)
        add_objects(spec, self.added_objects, self.object_poses)
        apply_object_poses(spec, self.object_poses)
        return spec


def parse_scene_id(scene_id: str) -> tuple[str, str, int]:
    """"procthor-10k/val/12" -> ("procthor-10k", "val", 12)."""
    try:
        dataset, split, index = scene_id.split("/")
        return dataset, split, int(index)
    except ValueError:
        raise ValueError(f"scene_id must be '<dataset>/<split>/<house_index>', got '{scene_id}'")


def resolve_scene_xml(scene_id: str, variant: str = "base") -> str:
    """Install a house's assets (first use downloads them) and return its scene XML."""
    from examples.molmo_environment import resolve_molmospaces_scene

    dataset, split, index = parse_scene_id(scene_id)
    return resolve_molmospaces_scene(dataset, split, index, variant)


def object_category(body_name: str) -> str:
    """molmospaces names objects `<category>_<asset hash>_<ids>`, e.g. `mug_4697b7..._1_0_2`."""
    return body_name.split("/")[-1].split("_")[0].lower()


def free_bodies(model: mujoco.MjModel) -> list[str]:
    """Bodies with a free joint: the objects that can be picked up."""
    return [
        model.body(model.jnt_bodyid[j]).name
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]


def load_custom_scene(
    scene_id: str, object_type: str, object_index: int = 0, variant: str = "base"
) -> CustomScene:
    """
    Load a MolmoSpaces house and pick the target object.

    Args:
        scene_id: "<dataset>/<split>/<house_index>", e.g. "procthor-10k/val/0".
        object_type: the object's category ("mug", "apple", ...), or an exact body name.
        object_index: which one, when the house has several of that category (sorted by name).
    """
    scene_xml = resolve_scene_xml(scene_id, variant)
    model = mujoco.MjModel.from_xml_path(scene_xml)
    candidates = free_bodies(model)
    if object_type in candidates:
        matches = [object_type]
    else:
        matches = sorted(n for n in candidates if object_category(n) == object_type.lower())
    if not matches:
        categories = sorted({object_category(n) for n in candidates})
        raise ValueError(f"No '{object_type}' in {scene_id}. Movable categories: {categories}")
    if object_index >= len(matches):
        raise ValueError(f"Only {len(matches)} '{object_type}' in {scene_id}: {matches}")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = data.body(matches[object_index])
    return CustomScene(
        scene_id=scene_id,
        scene_xml=scene_xml,
        object_name=body.name,
        object_pos=body.xpos.copy(),
        object_quat=body.xquat.copy(),
    )


# ---------------------------------------------------------------------------
# Scene modifications (molmospaces benchmark episodes)
# ---------------------------------------------------------------------------


def remove_objects(spec: mujoco.MjSpec, names: list[str]) -> None:
    for name in names:
        body = spec.body(name) or spec.body(name.split("/")[-1])
        if body is not None:
            spec.delete(body)


def add_objects(
    spec: mujoco.MjSpec, added_objects: dict[str, str], object_poses: dict[str, list[float]]
) -> None:
    """Add episode objects the way molmospaces' `JsonEvalTaskSampler.add_auxiliary_objects()` does."""
    if not added_objects:
        return
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
    from molmo_spaces.utils.constants.simulation_constants import (
        OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
    )
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    for object_name, object_xml_rel in added_objects.items():
        object_xml = Path(ASSETS_DIR) / object_xml_rel
        if not object_xml.is_file():
            object_xml = Path(install_uid(Path(object_xml_rel).stem))
        object_spec = mujoco.MjSpec.from_file(str(object_xml))
        body = object_spec.worldbody.bodies[0]
        *path, body_name = object_name.split("/")
        body.name = body_name
        if not body.first_joint():
            body.add_joint(
                name="XYZ_jntfree",
                type=mujoco.mjtJoint.mjJNT_FREE,
                damping=OBJAVERSE_FREE_JOINT_DEFAULT_DAMPING,
            )
        pose = object_poses.get(object_name, [0, 0, 0, 1, 0, 0, 0])
        frame = spec.worldbody.add_frame(pos=pose[:3], quat=pose[3:7])
        frame.attach_body(body, "/".join(path) + "/" if path else "", "")


def apply_object_poses(spec: mujoco.MjSpec, object_poses: dict[str, list[float]]) -> None:
    """
    Place free bodies at episode poses. molmospaces writes them into qpos at runtime; here
    they go into the spec, since a free body's spec pose is where it starts. Free bodies are
    always children of the world, so the pose is a world pose.
    """
    for name, pose in object_poses.items():
        body = spec.body(name)
        if body is None or not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints):
            continue
        body.pos = list(pose[:3])
        body.quat = list(pose[3:7])


# ---------------------------------------------------------------------------
# Robot placement
# ---------------------------------------------------------------------------


def suggest_robot_pose(
    scene: CustomScene,
    distances: tuple[float, ...] = (0.6, 0.5, 0.7, 0.8),
    footprint_radius: float = 0.2,
    headings: int = 36,
) -> RobotPose:
    """
    A footprint pose facing the object from one of `distances` away (horizontally, tried in
    order) whose footprint is clear floor and from which the object is in line of sight at
    its own height.

    Robots face +x; both the Franka and Stretch 4 reach forward.
    """
    model = scene.load_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    object_id = model.body(scene.object_name).id
    target = scene.object_pos
    floor_z = scene.floor_z

    def first_hit(origin, direction):
        geom = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geom)
        return dist, int(geom[0])

    candidates = [(d, 2 * math.pi * i / headings) for d in distances for i in range(headings)]
    for distance, angle in candidates:
        x = target[0] - distance * math.cos(angle)
        y = target[1] - distance * math.sin(angle)

        # Every point of the footprint must look straight down onto floor.
        clear = True
        for r, a in [(0, 0)] + [(footprint_radius, k * math.pi / 4) for k in range(8)]:
            origin = np.array([x + r * math.cos(a), y + r * math.sin(a), floor_z + 2.0])
            dist, _ = first_hit(origin, np.array([0, 0, -1.0]))
            if dist < 0 or abs(origin[2] - dist - floor_z) > 0.03:
                clear = False
                break
        if not clear:
            continue

        # Nothing between the robot and the object, at the object's height.
        origin = np.array([x, y, target[2]])
        direction = target - origin
        dist, geom = first_hit(origin, direction / np.linalg.norm(direction))
        if dist >= 0 and model.body_rootid[model.geom_bodyid[geom]] != object_id:
            continue
        return RobotPose(x, y, angle)

    raise ValueError(f"No clear pose {distances} m from {scene.object_name}; pass one explicitly")


def look_at(position, target) -> np.ndarray:
    """wxyz quaternion for a MuJoCo camera (looks down -z, +y up) at `position` facing `target`."""
    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return mat_to_quat(np.column_stack([right, up, -forward]))


def add_scene_camera(
    spec: mujoco.MjSpec, robot_pose: RobotPose, target, name: str = SCENE_CAMERA
) -> None:
    """A fixed third-person camera behind and to the left of the robot, looking at the target."""
    offset = np.array([-1.0, 1.0, 0.0])
    c, s = math.cos(robot_pose.yaw), math.sin(robot_pose.yaw)
    position = np.array(
        [robot_pose.x + c * offset[0] - s * offset[1], robot_pose.y + s * offset[0] + c * offset[1], target[2] + 0.9]
    )
    look_target = (np.array([robot_pose.x, robot_pose.y, target[2]]) + np.asarray(target)) / 2
    spec.worldbody.add_camera(
        name=name,
        pos=position.tolist(),
        quat=look_at(position, look_target).tolist(),
        fovy=60,
        resolution=list(SCENE_CAMERA_SIZE),
    )


def virtual_franka_link0_height(scene: CustomScene) -> float:
    return franka_link0_height_for_object(float(scene.object_pos[2]))


# ---------------------------------------------------------------------------
# Franka DROID
# ---------------------------------------------------------------------------


def load_custom_scene_franka_droid(
    scene: CustomScene,
    robot_pose: RobotPose,
    exo_camera: ExoCamera = "droid",
    link0_height: float | None = None,
    standing_on_floor: bool = True,
) -> FrankaDroidEnv:
    """
    The scene with a Franka DROID at `robot_pose`, ready to step in-process.

    `link0_height` defaults to the training placement for the target object. Benchmarks pass
    their own and `standing_on_floor=False`, to keep molmospaces' floating 0.58 m pedestal.
    """
    spec = scene.load_spec()
    spawn = spawn_franka_droid(
        spec,
        robot_pose,
        link0_height if link0_height is not None else virtual_franka_link0_height(scene),
        exo_camera=exo_camera,
        floor_z=scene.floor_z,
        standing_on_floor=standing_on_floor,
    )
    add_scene_camera(spec, robot_pose, scene.object_pos)
    return FrankaDroidEnv(spec.compile(), spawn, scene_camera=SCENE_CAMERA)


# ---------------------------------------------------------------------------
# Stretch 4
# ---------------------------------------------------------------------------


@dataclass
class Stretch4Scene:
    model: mujoco.MjModel
    scene: CustomScene
    robot_pose: RobotPose
    franka: FrankaSpawn
    """The virtual Franka Stretch 4 is retargeted from. Present in `model` only as the ghost."""
    include_franka: bool
    removed_bodies: list[str] = field(default_factory=list)
    """Furniture taken out because Stretch 4 would have spawned inside it."""

    @property
    def watched_bodies(self) -> list[str]:
        """What to pass to `sim.watch_bodies()`: the robot and the target object."""
        return [STRETCH_ROOT_BODY, self.scene.object_name]


def load_custom_scene_stretch4(
    scene: CustomScene,
    robot_pose: RobotPose,
    include_franka: bool = False,
    tool_name: str | None = None,
    link0_height: float | None = None,
    clear_footprint: bool = True,
) -> Stretch4Scene:
    """
    The scene with Stretch 4 standing at `robot_pose`, compiled for
    `Stretch4MujocoSimulator(model=...)`.

    The retargeting Franka is virtual: its `fr3_link0` is at `robot_pose` and the training
    height for the target object. `include_franka` also puts it in the scene as a ghost (no
    collisions, see-through, held at the home pose).

    `clear_footprint` removes furniture Stretch 4 would spawn inside (see
    `bodies_blocking_stretch()`): benchmark episodes place the Franka's pedestal, which is a
    mocap body that static furniture never pushes on, where chairs can be.
    """
    removed: list[str] = []
    for _ in range(3):
        stretch_scene = _build_stretch4_scene(scene, robot_pose, include_franka, tool_name, link0_height, removed)
        blocking = bodies_blocking_stretch(stretch_scene) if clear_footprint else []
        if not blocking:
            stretch_scene.removed_bodies = removed
            return stretch_scene
        removed += blocking
    raise RuntimeError(f"Stretch 4 still collides with the scene at {robot_pose} after removing {removed}")


def bodies_blocking_stretch(stretch_scene: "Stretch4Scene") -> list[str]:
    """
    Root bodies Stretch 4 touches standing at its home pose: anything but the floor, walls,
    the target object and whatever the target rests on.
    """
    model = stretch_scene.model
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home != -1:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    from examples.vla.molmobot_droid.droid import contacts_of_subtree

    target = stretch_scene.scene.object_name
    keep = {target, "world", *contacts_of_subtree(model, data, model.body(target).id)}
    blocking = []
    for name in contacts_of_subtree(model, data, model.body(STRETCH_ROOT_BODY).id):
        if name in keep or name.startswith(("room", "wall", "floor")) or name.startswith(stretch_scene.franka.prefix):
            continue
        blocking.append(name)
    return blocking


def _build_stretch4_scene(
    scene: CustomScene,
    robot_pose: RobotPose,
    include_franka: bool,
    tool_name: str | None,
    link0_height: float | None,
    removed: list[str],
) -> "Stretch4Scene":
    from examples.molmo_environment import add_stretch_to_scene

    spec = scene.load_spec()
    remove_objects(spec, removed)
    add_stretch_to_scene(
        spec, pos=[robot_pose.x, robot_pose.y, scene.floor_z], quat=robot_pose.quat_wxyz, tool_name=tool_name
    )
    height = link0_height if link0_height is not None else virtual_franka_link0_height(scene)
    if include_franka:
        franka = add_franka_ghost(spec, robot_pose, height, floor_z=scene.floor_z)
    else:
        franka = FrankaSpawn(
            prefix="ghost_franka/",
            robot_pose=robot_pose,
            floor_z=scene.floor_z,
            base_z=scene.floor_z,
            pedestal_height=height - scene.floor_z,
            exo_camera=None,
            exo_camera_name=None,
            wrist_camera_name="",
        )

    # The DROID shoulder camera, where it would be on the virtual Franka. Stretch's simulator
    # does not render it; `SceneMirror` does.
    world_from_camera = franka.world_from_link0 @ pose_to_transform(DROID_EXO_POS, DROID_EXO_QUAT)
    spec.worldbody.add_camera(
        name=DROID_EXO_IN_STRETCH_SCENE,
        pos=world_from_camera[:3, 3].tolist(),
        quat=mat_to_quat(world_from_camera[:3, :3]).tolist(),
        fovy=DROID_EXO_FOVY,
        resolution=list(DROID_IMAGE_SIZE),
    )
    add_scene_camera(spec, robot_pose, scene.object_pos)

    model = spec.compile()
    return Stretch4Scene(model, scene, robot_pose, franka, include_franka)


class SceneMirror:
    """
    A kinematic copy of a Stretch 4 scene in this process, posed from what the simulator
    reports (`pull_status()`, `pull_body_poses()`), for rendering cameras the simulator
    does not have (the scene camera, the DROID exo camera) and for rerun.

    Only watched bodies move; everything else stays where the scene put it.
    """

    def __init__(self, stretch_scene: Stretch4Scene):
        self.stretch_scene = stretch_scene
        self.model = stretch_scene.model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._renderers: dict[tuple[str, bool], PinholeRenderer] = {}
        self._ghost_joints = (
            [f"{stretch_scene.franka.prefix}fr3_joint{i + 1}" for i in range(7)]
            if stretch_scene.include_franka
            else []
        )
        self.set_ghost(FRANKA_HOME_QPOS)

    def update(self, status, body_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Pose Stretch's joints from a `StatusStretchJoints` and bodies from `pull_body_poses()`."""
        data = self.data
        data.joint("lift_joint").qpos = status.lift.pos
        for i in range(1, 5):
            data.joint(f"arm_l{i}_joint").qpos = status.arm.pos / 4
        data.joint("wrist_yaw_joint").qpos = status.wrist_yaw.pos
        data.joint("wrist_pitch_joint").qpos = status.wrist_pitch.pos
        data.joint("wrist_roll_joint").qpos = status.wrist_roll.pos
        for finger in ("left", "right"):
            # The Stretch gripper's finger joints, or the parallel gripper's.
            for joint in (f"gripper_finger_{finger}_joint", f"finger_{finger}_joint"):
                if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint) != -1:
                    data.joint(joint).qpos = getattr(status, f"gripper_{finger}_finger").pos
        for name, (pos, quat) in body_poses.items():
            body = self.model.body(name)
            if body.jntnum[0] and self.model.jnt_type[body.jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[body.jntadr[0]]
                data.qpos[adr : adr + 7] = np.concatenate([pos, quat])
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_camlight(self.model, data)

    def set_ghost(self, arm_qpos) -> None:
        for joint, value in zip(self._ghost_joints, arm_qpos):
            self.data.joint(joint).qpos = value
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def render(self, camera: str, show_ghost: bool = False) -> np.ndarray:
        """
        Render a camera of the mirrored scene. Leave `show_ghost` off for anything the policy
        sees; the ghost is for people.
        """
        key = (camera, show_ghost)
        if key not in self._renderers:
            size = tuple(int(v) for v in self.model.cam_resolution[self.model.camera(camera).id])
            groups = (GHOST_GEOM_GROUP,) if show_ghost else ()
            self._renderers[key] = PinholeRenderer(self.model, camera, size, geom_groups_on=groups)
        return self._renderers[key].render(self.data)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
