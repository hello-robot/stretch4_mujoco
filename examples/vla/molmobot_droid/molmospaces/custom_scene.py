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

import functools
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

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
    FrankaOverlayView,
    OverlayKind,
    FrankaDroidEnv,
    FrankaSpawn,
    PinholeRenderer,
    RendererPool,
    STRETCH_ROOT,
    RobotPose,
    add_franka_ghost,
    franka_link0_height_for_object,
    mat_to_quat,
    pose_to_transform,
    resize_franka_pedestal,
    spawn_franka_droid,
)
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

SCENE_CAMERA = "scene_camera"
SCENE_CAMERA_SIZE = (960, 540)
DROID_EXO_IN_STRETCH_SCENE = "droid_exo_camera"
STRETCH_ROOT_BODY = STRETCH_ROOT

INSERTABLE_OBJECTS = {"tennis_ball": "8e4d2af61f1742ce9e43561f9fdba055"}
"""Objects no MolmoSpaces house has that --object-type can still name, by their Objaverse asset
uid (from molmospaces' val split): `load_custom_scene()` swaps one in for one of the house's
own objects (`in_place_of`), as it can any THOR asset (`find_asset()`)."""
INSERT_DROP_HEIGHT = 0.002
"""m. An inserted object starts this far above where the object it replaces rested."""


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
    object_label: str | None = None
    """What to call the target, when its body name does not say (an `INSERTABLE_OBJECTS` one)."""

    @property
    def object_category(self) -> str:
        return self.object_label or object_category(self.object_name)

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
    scene_id: str, object_type: str, object_index: int = 0, variant: str = "base",
    in_place_of: str | None = None,
) -> CustomScene:
    """
    Load a MolmoSpaces house and pick the target object.

    Args:
        scene_id: "<dataset>/<split>/<house_index>", e.g. "procthor-10k/val/0".
        object_type: the object's category ("mug", "apple", ...), or an exact body name. With
            `in_place_of`, an asset put where that object was instead (`find_asset()`: one of
            INSERTABLE_OBJECTS, or a THOR category or prefab, see `insert_in_place_of()`).
        object_index: which one, when the house has several of that category (sorted by name);
            with `in_place_of`, which of its.
        in_place_of: the house's object (a category or a body name, as `object_type`) that
            `object_type` replaces.
    """
    scene_xml = resolve_scene_xml(scene_id, variant)
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if in_place_of is None and object_type.lower().replace(" ", "_") in INSERTABLE_OBJECTS:
        raise ValueError(
            f"No house has a {object_type}: say which object it goes in place of with --in-place-of "
            f"(movable categories in {scene_id}: {movable_categories(model)})"
        )
    body = data.body(pick_free_body(model, scene_id, in_place_of or object_type, object_index))
    house = CustomScene(
        scene_id=scene_id,
        scene_xml=scene_xml,
        object_name=body.name,
        object_pos=body.xpos.copy(),
        object_quat=body.xquat.copy(),
    )
    return swap_target(house, object_type) if in_place_of is not None else house


def movable_categories(model: mujoco.MjModel) -> list[str]:
    return sorted({object_category(n) for n in free_bodies(model)})


def pick_free_body(model: mujoco.MjModel, scene_id: str, object_type: str, object_index: int) -> str:
    """The `object_index`th free body of category `object_type` (sorted by name), or the one
    named `object_type`."""
    candidates = free_bodies(model)
    if object_type in candidates:
        matches = [object_type]
    else:
        matches = sorted(n for n in candidates if object_category(n) == object_type.lower())
    if not matches:
        raise ValueError(
            f"No '{object_type}' in {scene_id}. Movable categories: {movable_categories(model)}, "
            f"and in place of one of them: {sorted(INSERTABLE_OBJECTS)}"
        )
    if object_index >= len(matches):
        raise ValueError(f"Only {len(matches)} '{object_type}' in {scene_id}: {matches}")
    return matches[object_index]


def world_aabb(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(min, max) corners of the world-aligned box around the colliding geoms of `body_id` and
    its children: what it rests on things by."""
    lows, highs = [], []
    for geom in range(model.ngeom):
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        body = model.geom_bodyid[geom]
        while body not in (body_id, 0):
            body = model.body_parentid[body]
        if body != body_id:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
        half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
        lows.append(center - half)
        highs.append(center + half)
    return np.min(lows, axis=0), np.max(highs, axis=0)


@dataclass(frozen=True)
class Asset:
    """An object to put in a house (`insert_in_place_of()`)."""

    label: str
    """What instructions call it, e.g. "alarm clock"."""
    asset_id: str
    """Its name in its library: an Objaverse uid or a THOR prefab, e.g. "Alarm_Clock_3"."""
    added_path: str
    """Its XML relative to molmospaces' ASSETS_DIR, as `CustomScene.added_objects` holds it."""


THOR_VARIANT_SUFFIXES = ("_mesh", "_prim", "_old")
"""THOR prefab XMLs ending in these are other versions of the one without (collision variants)."""


def thor_category(prefab: str) -> str:
    """A THOR prefab's category: its name, lower-case, without its numbers ("Alarm_Clock_10" and
    "Armchair_205_1" are an "alarm_clock" and an "armchair")."""
    return re.sub(r"(_\d+)+$", "", prefab).lower()


@functools.cache
def thor_prefabs() -> dict[str, list[str]]:
    """THOR's prefabs by category (`thor_category()`), numbered ones in order of number."""
    from molmo_spaces.utils.lazy_loading_utils import get_thor_uid_to_xmls

    by_category: dict[str, list[str]] = {}
    for stem in get_thor_uid_to_xmls():
        if stem.endswith(THOR_VARIANT_SUFFIXES):
            continue
        by_category.setdefault(thor_category(stem), []).append(stem)
    number = lambda stem: int(m.group(1)) if (m := re.search(r"_(\d+)$", stem)) else -1
    return {category: sorted(stems, key=lambda s: (number(s), s)) for category, stems in sorted(by_category.items())}


def find_asset(object_type: str, index: int = 0) -> Asset:
    """
    The asset `object_type` names: one of INSERTABLE_OBJECTS, a THOR prefab ("Apple_3") or a
    THOR category ("apple", "alarm clock"), the `index`th prefab of it. Spaces and underscores
    are the same, and case does not matter.
    """
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
    from molmo_spaces.utils.lazy_loading_utils import get_thor_uid_to_xmls, install_uid

    key = object_type.strip().lower().replace(" ", "_")
    if key in INSERTABLE_OBJECTS:
        uid = INSERTABLE_OBJECTS[key]
        install_uid(uid)  # first use downloads it
        return Asset(key.replace("_", " "), uid, f"objects/objaverse/{uid}/{uid}.xml")
    prefabs = thor_prefabs()
    stems = {stem.lower(): stem for category in prefabs.values() for stem in category}
    if key in stems:
        stem = stems[key]
    elif key in prefabs:
        if index >= len(prefabs[key]):
            raise ValueError(f"Only {len(prefabs[key])} THOR '{key}' prefabs: {', '.join(prefabs[key])}")
        stem = prefabs[key][index]
    else:
        near = [c for c in prefabs if key in c or c in key]
        raise ValueError(
            f"No object '{object_type}': it is neither {' nor '.join(INSERTABLE_OBJECTS)} nor a THOR category or "
            f"prefab" + (f" (did you mean {', '.join(near[:8])}?)" if near else " ('object' alone lists them)")
        )
    # The THOR library is a link into molmospaces' resources: the path goes through it.
    thor = (Path(ASSETS_DIR) / "objects" / "thor").resolve()
    return Asset(thor_category(stem).replace("_", " "), stem, str(Path("objects/thor") / get_thor_uid_to_xmls()[stem].relative_to(thor)))


def swappable_text(scene: CustomScene) -> str:
    """What `object` says: the target, and what can be put in its place."""
    # Not the house's structure (still swappable by name, if anyone wants a window to pick up).
    structure = ("robothor", "_", "countertop", "window", "wall", "door", "floortile", "shower", "sink", "toilet")
    categories = [c for c in thor_prefabs() if not c.startswith(structure)]
    return (
        f"The target is the {scene.object_category} ({scene.object_name}). "
        f"'object <name> [index]' puts another in its place: {', '.join(INSERTABLE_OBJECTS)}, a THOR prefab "
        f"(e.g. Apple_3), or the index-th prefab of a THOR category: {', '.join(c.replace('_', ' ') for c in categories)}."
    )


def swap_target(scene: CustomScene, object_type: str, index: int = 0) -> CustomScene:
    """`scene` with the asset `object_type` names (`find_asset()`) in place of its target."""
    return insert_in_place_of(scene, find_asset(object_type, index), scene.object_name)


def insert_in_place_of(scene: CustomScene, asset: Asset, replaced: str) -> CustomScene:
    """
    `scene` with `asset` as the target, in place of its `replaced` free body as it starts
    (`scene`'s modifications applied, so it can be an asset an earlier swap put in): centred
    where `replaced`'s geoms were, resting where their bottom was (dropped from
    INSERT_DROP_HEIGHT above), upright. The asset's mass is its XML's.
    """
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    model = scene.load_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    asset_model = mujoco.MjModel.from_xml_path(str(Path(ASSETS_DIR) / asset.added_path))
    asset_data = mujoco.MjData(asset_model)
    mujoco.mj_kinematics(asset_model, asset_data)
    asset_low, asset_high = world_aabb(asset_model, asset_data, 1)  # relative to its body at the origin

    low, high = world_aabb(model, data, model.body(replaced).id)
    center = (low + high) / 2
    pos = np.array([
        *(center[:2] - (asset_low[:2] + asset_high[:2]) / 2),
        low[2] - asset_low[2] + INSERT_DROP_HEIGHT,
    ])
    # molmospaces' `<category>_<asset>_<ids>`, the category in one word.
    name = f"{asset.label.replace(' ', '')}_{asset.asset_id.lower()}_0"
    # An object an earlier swap added is taken out of the additions; one of the house's is removed.
    added = {body: xml for body, xml in scene.added_objects.items() if body != replaced}
    poses = {body: pose for body, pose in scene.object_poses.items() if body != replaced}
    removed = list(scene.removed_objects) + ([] if replaced in scene.added_objects else [replaced])
    added[name] = asset.added_path
    poses[name] = [*pos, 1.0, 0.0, 0.0, 0.0]
    return replace(
        scene,
        object_name=name,
        object_pos=pos,
        object_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        object_poses=poses,
        added_objects=added,
        removed_objects=removed,
        object_label=asset.label,
    )


class SwapTarget(Exception):
    """Raised by `object_command()` for the runner to rebuild its scene as `scene`, keeping
    the robot where it is (`carry`, which the runner fills in on the way out)."""

    def __init__(self, scene: CustomScene, replaced_mass: float, mass: float):
        super().__init__(f"Swapping in {scene.object_name}")
        self.scene = scene
        self.replaced_mass = replaced_mass
        self.mass = mass
        self.carry: dict = {}

    def describe(self) -> str:
        return (f"Swapping the {self.scene.object_category} ({self.mass:.2f} kg) in for the target "
                f"({self.replaced_mass:.2f} kg): rebuilding the scene...")


OBJECT_COMMANDS = (
    "'object' to see what can stand in for the target, 'object <category or prefab> [index]' to put it in "
    "the target's place (the scene is rebuilt, the robot kept where it is)"
)


def object_command(text: str, scene: CustomScene, say: Callable[[str], None]) -> bool:
    """
    `object` (what the target is and what can replace it) or `object <name> [index]`: raises
    SwapTarget with `scene`'s target swapped for that asset (`swap_target()`). Returns whether
    `text` was one; an asset that cannot be found is said and counts as one.
    """
    words = text.split()
    if not words or words[0] != "object":
        return False
    if len(words) == 1:
        say(swappable_text(scene))
        return True
    index = int(words.pop()) if len(words) > 2 and words[-1].isdigit() else 0
    try:
        swapped = swap_target(scene, "_".join(words[1:]), index)
    except ValueError as error:
        say(str(error))
        return True
    raise SwapTarget(swapped, subtree_mass(scene, scene.object_name), subtree_mass(swapped, swapped.object_name))


def run_with_swaps(scene: CustomScene, run_scene: Callable[[CustomScene, dict], None],
                   say: Callable[[str, str | None], None]) -> None:
    """
    `run_scene(scene, carry)`, and again with each scene an `object` command swaps in
    (`SwapTarget`), until it returns. `carry` is what the last run put in the SwapTarget's to
    take over (the robot's pose, say); empty the first time.
    """
    carry: dict = {}
    while True:
        try:
            run_scene(scene, carry)
            return
        except SwapTarget as swap:
            say(swap.describe(), "yellow")
            scene, carry = swap.scene, swap.carry


def subtree_mass(scene: CustomScene, body: str) -> float:
    model = scene.load_spec().compile()
    return float(model.body_subtreemass[model.body(body).id])


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
        # The pose goes on the body, not the frame: `apply_object_poses()` sets the body's pose
        # to it too, which on top of the frame's would apply it twice.
        pose = object_poses.get(object_name, [0, 0, 0, 1, 0, 0, 0])
        body.pos = list(pose[:3])
        body.quat = list(pose[3:7])
        frame = spec.worldbody.add_frame()
        # Everything the asset brings (textures, materials, meshes, geoms) is suffixed with the
        # body's name, so it cannot clash with the house's (THOR assets share names with its
        # objects) or another copy's. The body keeps its own name.
        prefix = "/".join(path) + "/" if path else ""
        attached = frame.attach_body(body, prefix, f"_{body_name}")
        attached.name = prefix + body_name


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


def scene_camera_pose(robot_pose: RobotPose, target) -> tuple[np.ndarray, np.ndarray]:
    """
    Where the scene camera is, behind and to the left of the robot, and the point it looks at,
    halfway between the robot and the target at the target's height.
    """
    offset = np.array([-1.0, 1.0, 0.0])
    c, s = math.cos(robot_pose.yaw), math.sin(robot_pose.yaw)
    position = np.array(
        [robot_pose.x + c * offset[0] - s * offset[1], robot_pose.y + s * offset[0] + c * offset[1], target[2] + 0.9]
    )
    look_target = (np.array([robot_pose.x, robot_pose.y, target[2]]) + np.asarray(target)) / 2
    return position, look_target


def add_scene_camera(
    spec: mujoco.MjSpec, robot_pose: RobotPose, target, name: str = SCENE_CAMERA
) -> None:
    """A fixed third-person camera behind and to the left of the robot, looking at the target."""
    position, look_target = scene_camera_pose(robot_pose, target)
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
    stretch4_gripper: str | None = None,
    stretch4_gripper_camera: bool = False,
) -> FrankaDroidEnv:
    """
    The scene with a Franka DROID at `robot_pose`, ready to step in-process.

    `link0_height` defaults to the training placement for the target object. Benchmarks pass
    their own and `standing_on_floor=False`, to keep molmospaces' floating 0.58 m pedestal.
    `stretch4_gripper` ("sg4", "pg4") and `stretch4_gripper_camera` put Stretch 4's gripper and
    gripper camera on it in place of the Robotiq and DROID's wrist camera (see
    `stretch4_hardware_on_franka`), with a ghost Franka DROID as it comes (`env.ghost`, its wrist
    camera kept) standing in the same place, to show what it would see (`RobotiqGhost`).
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
    if stretch4_gripper is not None or stretch4_gripper_camera:
        from examples.vla.molmobot_droid.franka_retarget.stretch4_hardware_on_franka import put_stretch4_hardware_on

        put_stretch4_hardware_on(spec, spawn, stretch4_gripper, stretch4_gripper_camera)
        ghost = add_franka_ghost(
            spec, robot_pose, spawn.world_from_link0[2, 3], floor_z=scene.floor_z, keep_wrist_camera=True
        )
    else:
        ghost = None
    add_scene_camera(spec, robot_pose, scene.object_pos)
    return FrankaDroidEnv(spec.compile(), spawn, scene_camera=SCENE_CAMERA, ghost=ghost)


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
    backed_off: float = 0.0
    """How far (m) `robot_pose` was moved away from the target so Stretch 4 clears what it rests on."""

    @property
    def watched_bodies(self) -> list[str]:
        """What to pass to `sim.watch_bodies()`: the robot and the target object."""
        return [STRETCH_ROOT_BODY, self.scene.object_name]

    def set_franka_pedestal_height(self, height: float) -> None:
        """
        Raise or lower the virtual Franka to `fr3_link0` `height` above the floor: where it is
        retargeted from, and in `model` (so `SceneMirror` renders it) the ghost and the DROID
        exo camera on it. The simulator's own copy of the ghost stays where it was.
        """
        model, delta = self.model, height - self.franka.pedestal_height
        if self.include_franka:
            resize_franka_pedestal(model, self.franka, height)
        else:
            self.franka.pedestal_height = height
        model.cam_pos[model.camera(DROID_EXO_IN_STRETCH_SCENE).id, 2] += delta


def load_custom_scene_stretch4(
    scene: CustomScene,
    robot_pose: RobotPose,
    include_franka: bool = False,
    tool_name: str | None = None,
    link0_height: float | None = None,
    clear_footprint: bool = True,
    stretch_yaw: float = 0.0,
) -> Stretch4Scene:
    """
    The scene with Stretch 4 standing at `robot_pose`, compiled for
    `Stretch4MujocoSimulator(model=...)`. Stretch faces `stretch_yaw` (rad) to the left of
    `robot_pose`'s heading, the Franka along it: see `franka_yaw_for_start()`.

    The retargeting Franka is virtual: its `fr3_link0` is at `robot_pose` and the training
    height for the target object. `include_franka` also puts it in the scene as a ghost (no
    collisions, see-through, held at the home pose).

    `clear_footprint` removes furniture Stretch 4 would spawn inside (see
    `bodies_blocking_stretch()`): benchmark episodes place the Franka's pedestal, which is a
    mocap body that static furniture never pushes on, where chairs can be. What the target
    rests on cannot be removed, so if Stretch would spawn inside that (its footprint and
    stowed arm are wider than the Franka's pedestal), `robot_pose`, the Franka with it, is
    moved straight away from the target until it clears (see `support_back_off()`).
    """
    removed: list[str] = []
    pose, backed_off = robot_pose, 0.0
    for _ in range(4):
        stretch_scene = _build_stretch4_scene(scene, pose, include_franka, tool_name, link0_height, removed, stretch_yaw)
        if not clear_footprint:
            return stretch_scene
        blocking = bodies_blocking_stretch(stretch_scene)
        if blocking:
            removed += blocking
            continue
        back_off = support_back_off(stretch_scene)
        if back_off == 0.0:
            stretch_scene.removed_bodies = removed
            stretch_scene.backed_off = backed_off
            return stretch_scene
        pose, backed_off = moved_away_from(pose, scene.object_pos, back_off), backed_off + back_off
    raise RuntimeError(f"Stretch 4 still collides with the scene at {pose} after removing {removed}")


def moved_away_from(pose: RobotPose, target, distance: float) -> RobotPose:
    """`pose` moved `distance` (m) horizontally straight away from `target`, still facing the same way."""
    away = np.array([pose.x - target[0], pose.y - target[1]])
    norm = np.linalg.norm(away)
    away = away / norm if norm > 1e-6 else -np.array([math.cos(pose.yaw), math.sin(pose.yaw)])
    return RobotPose(pose.x + distance * away[0], pose.y + distance * away[1], pose.yaw)


def support_back_off(stretch_scene: "Stretch4Scene", step: float = 0.01, max_distance: float = 0.3) -> float:
    """
    How far (m, in `step`s) Stretch 4 has to move straight away from the target so that,
    standing at its home pose, it touches neither the target nor what the target rests on:
    0 if it already does not.
    """
    model = stretch_scene.model
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home != -1:
        mujoco.mj_resetDataKeyframe(model, data, home)
    mujoco.mj_forward(model, data)

    from examples.vla.molmobot_droid.droid import contacts_of_subtree

    target = stretch_scene.scene.object_name
    supports = {target, *contacts_of_subtree(model, data, model.body(target).id)} - {"world"}
    root = model.body(STRETCH_ROOT_BODY)
    adr = model.jnt_qposadr[root.jntadr[0]]
    start = data.qpos[adr : adr + 2].copy()
    pose = stretch_scene.robot_pose
    for k in range(int(round(max_distance / step)) + 1):
        moved = moved_away_from(pose, stretch_scene.scene.object_pos, k * step)
        data.qpos[adr : adr + 2] = start + [moved.x - pose.x, moved.y - pose.y]
        mujoco.mj_forward(model, data)
        if not supports.intersection(contacts_of_subtree(model, data, root.id)):
            return k * step
    raise RuntimeError(
        f"Stretch 4 at {pose} still touches {sorted(supports)} {max_distance} m further from the target"
    )


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
    stretch_yaw: float = 0.0,
) -> "Stretch4Scene":
    from examples.molmo_environment import add_stretch_to_scene

    spec = scene.load_spec()
    remove_objects(spec, removed)
    stretch_pose = RobotPose(robot_pose.x, robot_pose.y, robot_pose.yaw + stretch_yaw)
    add_stretch_to_scene(
        spec, pos=[robot_pose.x, robot_pose.y, scene.floor_z], quat=stretch_pose.quat_wxyz, tool_name=tool_name
    )
    height = link0_height if link0_height is not None else virtual_franka_link0_height(scene)
    if include_franka:
        franka = add_franka_ghost(spec, robot_pose, height, floor_z=scene.floor_z, keep_wrist_camera=True)
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

    Only watched bodies move; everything else stays where the scene put it. Every camera is
    rendered with one renderer (`RendererPool`): one each would hold a copy of the house's
    textures on the GPU apiece.
    """

    def __init__(self, stretch_scene: Stretch4Scene):
        self.stretch_scene = stretch_scene
        self.model = stretch_scene.model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.pool = RendererPool(self.model)
        self._renderers: dict[tuple[str, bool], PinholeRenderer] = {}
        self._franka_views: dict[tuple, FrankaOverlayView] = {}
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

    def render_with_franka(self, camera: StretchCameras | str, driver_angle: float, kind: OverlayKind) -> np.ndarray:
        """
        `camera` (one of Stretch's, as the simulator renders it, or one of the scene's own, like
        DROID_EXO_IN_STRETCH_SCENE) with the ghost Franka, its Robotiq open to `driver_angle`, in
        the Franka's own look in place of part of Stretch (`FrankaOverlayView`): the Robotiq's
        fingers for Stretch's tool (`kind` "fingers", the gripper camera), or the whole Franka
        for Stretch's arm ("franka", the exo views), beside all of Stretch
        ("franka_with_stretch") or in place of all of it ("franka_alone"). Needs the ghost
        (`include_franka`), posed by `set_ghost()`.
        """
        if not self._ghost_joints:
            raise RuntimeError("Overlaying the Franka needs the ghost Franka (include_franka)")
        key = (camera, kind)
        if key not in self._franka_views:
            # The simulator runs its own copy of the model, so this changes only what is rendered here.
            self._franka_views[key] = FrankaOverlayView(self.model, self.stretch_scene.franka, camera, kind, pool=self.pool)
        view = self._franka_views[key]
        view.pose_fingers(self.data, driver_angle)
        return view.render(self.data)

    def render(self, camera: str, show_ghost: bool = False) -> np.ndarray:
        """
        Render a camera of the mirrored scene. Leave `show_ghost` off for anything the policy
        sees; the ghost is for people.
        """
        key = (camera, show_ghost)
        if key not in self._renderers:
            size = tuple(int(v) for v in self.model.cam_resolution[self.model.camera(camera).id])
            groups = (GHOST_GEOM_GROUP,) if show_ghost else ()
            self._renderers[key] = PinholeRenderer(self.model, camera, size, geom_groups_on=groups, pool=self.pool)
        return self._renderers[key].render(self.data)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        for view in self._franka_views.values():
            view.close()
        self._franka_views.clear()
        self.pool.close()
