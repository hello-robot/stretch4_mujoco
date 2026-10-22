"""
Pick benchmarks whose scenes are generated here rather than by molmospaces' datagen, for
`--benchmark random|one_object|five_objects` (`run_benchmark_follow_franka.py`):

- `random`: a random object of RANDOM_OBJECTS (a random THOR prefab of it) to pick up.
- `one_object`: the same object every episode, `--object_type` (`find_asset()`: a category's
  first prefab, or the prefab or INSERTABLE_OBJECTS one named).
- `five_objects`: FIVE_OBJECTS together, one of them, at random, the one to pick up.

Each episode puts them on a random surface (molmospaces' RECEPTACLE_TYPES_THOR) in a random room
of a random procthor-10k house, at random spots and headings, and lets them settle; the house's
own objects of the same kinds are taken out, so the instruction names only one. The Franka is
placed as molmospaces' pick datagen places it (`PickTaskSamplerConfig`, and as the
FrankaPickDroidMiniBench episodes are): 0.35-0.7 m from the target, facing it to within 15
degrees, its base ROBOT_OBJECT_Z_OFFSET below it, its arm at home with up to 0.05 rad of noise.
Targets start between MIN_TARGET_HEIGHT and Stretch 4's mast, so every episode is reachable.

A benchmark is a molmospaces benchmark directory (`benchmark.json`, a list of `EpisodeSpec`s,
and `benchmark_metadata.json`) under BENCHMARKS_DIR: written the first time it is asked for
(`ensure_benchmark()`), read from there after, so runs are repeatable. Episode i is generated
from its own seed (the benchmark's name and i), so writing one again gives the same episodes.

    python -m examples.vla.molmobot_droid.molmospaces.generated_benchmarks five_objects --workers 8
    python -m examples.vla.molmobot_droid.molmospaces.generated_benchmarks one_object --object_type potato
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import click
import mujoco
import numpy as np

from examples.vla.molmobot_droid.droid import FRANKA_HOME_QPOS, ROBOT_OBJECT_Z_OFFSET, contacts_of_subtree
from examples.vla.molmobot_droid.molmospaces.benchmark import SCENE_VARIANT, STRETCH_MAST_HEIGHT
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    INSERT_DROP_HEIGHT,
    Asset,
    CustomScene,
    added_object_name,
    find_asset,
    resolve_scene_xml,
    robot_footprint_clear,
    thor_prefabs,
    world_aabb,
)

KINDS = ("random", "one_object", "five_objects")
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
DEFAULT_EPISODES = 3000
DEFAULT_OBJECT_TYPE = "potato"

FIVE_OBJECTS = ("apple", "salt_shaker", "bowl", "tennis_ball", "pen")
RANDOM_OBJECTS = (
    "alarm_clock", "apple", "bowl", "box", "bread", "candle", "cup", "dish_sponge", "egg", "fork", "kettle", "knife",
    "ladle", "mug", "pen", "pencil", "pepper_shaker", "plate", "pot", "potato", "salt_shaker", "soap_bar",
    "soap_bottle", "spatula", "spoon", "spray_bottle", "tennis_ball", "tissue_box", "toilet_paper", "tomato", "watch",
    "wine_bottle",
)
"""THOR's pickupable categories (molmospaces' ALL_PICKUP_TYPES_THOR) that have prefabs and are small
enough for a parallel gripper, the MiniBench's own among them, and the tennis ball."""
MAX_RANDOM_OBJECT_RADIUS = 0.15
"""m. A random prefab whose footprint (half its horizontal diagonal) is wider than this is drawn again."""

SCENE_DATASET, SCENE_SPLIT = "procthor-10k", "val"
"""The houses the molmospaces benchmarks are built on (and that come installed with them)."""

ROBOT_DISTANCE = (0.35, 0.7)
"""m. How far from the target the Franka stands: the MiniBench's 5th to 95th percentiles."""
ROBOT_YAW_NOISE = math.radians(15)
ARM_QPOS_NOISE = 0.05
"""rad. Uniform noise on the Franka's home pose (molmospaces' `init_qpos_noise_range`)."""
GRIPPER_QPOS = [0.00296, 0.00296]
IMG_RESOLUTION = (624, 352)
TASK_HORIZON_SEC = 20

FIVE_OBJECTS_SPREAD = 0.35
"""m. How far from the target the other four objects can be."""
OBJECT_GAP = 0.02
"""m. The least room between two objects' footprints."""
SETTLE_SECONDS = 1.0
MAX_SETTLE_DRIFT = 0.03
"""m. How far an object may move as it settles; more means it slid or fell."""
MIN_TARGET_HEIGHT = 0.6
"""m. About the MiniBench's lowest targets (its 5th percentile, 0.65 m; median 0.95 m): no sofa seats."""
MAX_ATTEMPTS = 200
"""Houses tried for one episode before giving up."""


class Rejected(Exception):
    """This house, room or surface did not work out; the episode tries another."""


def benchmark_name(kind: str, object_type: str = DEFAULT_OBJECT_TYPE) -> str:
    """e.g. "five_objects", "one_object_potato"."""
    if kind not in KINDS:
        raise ValueError(f"Generated benchmarks are {', '.join(KINDS)}, not '{kind}'")
    return f"one_object_{object_type.strip().lower().replace(' ', '_')}" if kind == "one_object" else kind


def benchmark_dir(kind: str, object_type: str = DEFAULT_OBJECT_TYPE) -> Path:
    return BENCHMARKS_DIR / benchmark_name(kind, object_type)


def ensure_benchmark(
    kind: str, object_type: str = DEFAULT_OBJECT_TYPE, episodes: int = DEFAULT_EPISODES, workers: int | None = None
) -> Path:
    """The benchmark's directory, generating it first if it is not there yet."""
    directory = benchmark_dir(kind, object_type)
    if not (directory / "benchmark.json").exists():
        generate_benchmark(kind, object_type, episodes, workers)
    return directory


def generate_benchmark(kind: str, object_type: str, episodes: int, workers: int | None = None) -> Path:
    """Generate `episodes` episodes (in parallel over `workers` processes) and write the benchmark."""
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    name = benchmark_name(kind, object_type)
    directory = BENCHMARKS_DIR / name
    if kind == "one_object":
        find_asset(object_type)  # a misspelt --object_type fails here, not in every worker
    click.secho(f"Generating {episodes} episodes of {name} -> {directory}", fg="yellow")
    started = time.time()
    jobs = [(kind, object_type, index) for index in range(episodes)]
    specs = []
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        for spec in pool.map(_generate_episode_job, jobs, chunksize=4):
            specs.append(EpisodeSpec.model_validate(spec).model_dump())
            if len(specs) % 50 == 0 or len(specs) == episodes:
                rate = (time.time() - started) / len(specs)
                click.echo(f"    {len(specs)}/{episodes} episodes, {rate * (episodes - len(specs)) / 60:.0f} min left")

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "benchmark.json").write_text(json.dumps(specs))
    (directory / "benchmark_metadata.json").write_text(json.dumps(metadata(kind, object_type, specs), indent=2))
    click.secho(f"Wrote {directory} in {(time.time() - started) / 60:.1f} min", fg="green")
    return directory


def metadata(kind: str, object_type: str, specs: list[dict]) -> dict:
    """`benchmark_metadata.json` (molmospaces' `BenchmarkMetadata`, with how the episodes were made)."""
    return {
        "description": f"Generated by {Path(__file__).name}: {benchmark_name(kind, object_type)}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_episodes": len(specs),
        "num_houses": len({s["house_index"] for s in specs}),
        "task_cls_counts": dict(Counter(s["task"]["task_cls"] for s in specs)),
        "object_category_counts": dict(Counter(s["language"]["referral_expressions"]["pickup_obj_name"] for s in specs)),
        "robot_counts": dict(Counter(s["robot"]["robot_name"] for s in specs)),
        "room_type_counts": dict(Counter(s["task"]["room_type"] for s in specs)),
        "generator": {"kind": kind, "object_type": object_type if kind == "one_object" else None,
                      "scene_dataset": SCENE_DATASET, "data_split": SCENE_SPLIT, "scene_variant": SCENE_VARIANT},
    }


def _generate_episode_job(job: tuple[str, str, int]) -> dict:
    return generate_episode(*job)


def generate_episode(kind: str, object_type: str, index: int) -> dict:
    """Episode `index` of a generated benchmark, as an `EpisodeSpec` dict: the same every time."""
    name = benchmark_name(kind, object_type)
    seed = zlib.crc32(f"{name}/{index}".encode())
    rng = np.random.default_rng(seed)
    last_error = None
    for _ in range(MAX_ATTEMPTS):
        assets = episode_objects(kind, object_type, rng)
        house = int(rng.choice(available_houses()))
        try:
            return episode_in_house(house, assets, int(rng.integers(len(assets))), rng, seed)
        except Rejected as error:
            last_error = error
    raise RuntimeError(f"No scene for {name} episode {index} in {MAX_ATTEMPTS} houses; last: {last_error}")


def episode_objects(kind: str, object_type: str, rng: np.random.Generator) -> list[Asset]:
    if kind == "one_object":
        return [find_asset(object_type)]
    if kind == "five_objects":
        return [find_asset(name) for name in FIVE_OBJECTS]
    while True:
        category = str(rng.choice(RANDOM_OBJECTS))
        asset = find_asset(category, int(rng.integers(len(thor_prefabs().get(category, [category])))))
        if asset_extents(asset)[2] <= MAX_RANDOM_OBJECT_RADIUS:
            return [asset]


@functools.cache
def available_houses() -> list[int]:
    from molmo_spaces.molmo_spaces_constants import get_scenes

    houses = get_scenes(SCENE_DATASET, split=SCENE_SPLIT)[SCENE_SPLIT]
    return sorted(i for i, house in houses.items() if house is not None)


@functools.cache
def asset_extents(asset: Asset) -> tuple[np.ndarray, np.ndarray, float]:
    """The asset's colliding geoms' box (low, high corners) about its body, upright, and its
    footprint radius: half the box's horizontal diagonal, which covers it at any heading."""
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    model = mujoco.MjModel.from_xml_path(str(Path(ASSETS_DIR) / asset.added_path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    low, high = world_aabb(model, data, 1)
    return low, high, float(np.linalg.norm(high[:2] - low[:2]) / 2)


def episode_in_house(house: int, assets: list[Asset], target: int, rng: np.random.Generator, seed: int) -> dict:
    """`assets` on a random surface of a random room of `house`, `assets[target]` to pick up."""
    from molmo_spaces.utils.constants.object_constants import RECEPTACLE_TYPES_THOR

    scene_id = f"{SCENE_DATASET}/{SCENE_SPLIT}/{house}"
    with contextlib.redirect_stdout(io.StringIO()):  # it announces every install, already done or not
        scene_xml = resolve_scene_xml(scene_id, SCENE_VARIANT)
    stem = scene_xml.removesuffix(f"_{SCENE_VARIANT}.xml")
    objects = json.loads(Path(f"{stem}_metadata.json").read_text())["objects"]
    room_types = {int(room["id"].split("|")[-1]): room["roomType"] for room in json.loads(Path(f"{stem}.json").read_text())["rooms"]}

    # The house's own objects of the kinds being added would make the instruction ambiguous.
    kinds = {asset.label.replace(" ", "").lower() for asset in assets}
    removed = sorted(name for name, o in objects.items() if not o["is_static"] and o["category"].lower() in kinds)
    house_scene = CustomScene(scene_id, scene_xml, "", np.zeros(3), np.array([1.0, 0, 0, 0]), removed_objects=removed)
    model = house_scene.load_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    surfaces_by_room: dict[int, list[str]] = {}
    for name, o in objects.items():
        if o["category"] in RECEPTACLE_TYPES_THOR and o.get("room_id") in room_types:
            top = world_aabb(model, data, model.body(name).id)[1][2]  # tall ones may have lower shelves
            if top >= MIN_TARGET_HEIGHT - 0.05:
                surfaces_by_room.setdefault(o["room_id"], []).append(name)
    if not surfaces_by_room:
        raise Rejected(f"{scene_id} has no surfaces high enough")
    room = int(rng.choice(sorted(surfaces_by_room)))
    surface = str(rng.choice(sorted(surfaces_by_room[room])))
    poses = place_on_surface(model, data, surface, assets, target, rng)

    names = [added_object_name(asset) for asset in assets]
    scene = replace(
        house_scene,
        object_name=names[target],
        added_objects={name: asset.added_path for name, asset in zip(names, assets)},
        object_poses=dict(zip(names, poses)),
    )
    model, data, settled = settle(scene, surface)
    target_pose = settled[names[target]]
    if not MIN_TARGET_HEIGHT <= target_pose[2] <= STRETCH_MAST_HEIGHT:
        raise Rejected(f"{surface} puts the target at {target_pose[2]:.2f} m")
    robot_base_pose = place_franka(model, data, names[target], target_pose, rng)

    label = assets[target].label
    arm = np.asarray(FRANKA_HOME_QPOS) + rng.uniform(-ARM_QPOS_NOISE, ARM_QPOS_NOISE, len(FRANKA_HOME_QPOS))
    return {
        "source": None,
        "house_index": house,
        "scene_dataset": SCENE_DATASET,
        "data_split": SCENE_SPLIT,
        "seed": seed,
        "robot": {"robot_name": "franka_droid", "init_qpos": {"base": [], "arm": arm.tolist(), "gripper": GRIPPER_QPOS}},
        "img_resolution": IMG_RESOLUTION,
        "cameras": [],
        "scene_modifications": {
            "added_objects": scene.added_objects,
            "object_poses": settled,
            "removed_objects": removed,
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            "task_type": "pick",
            "robot_base_pose": robot_base_pose,
            "pickup_obj_name": names[target],
            "pickup_obj_start_pose": target_pose,
            "pickup_obj_goal_pose": [*target_pose[:2], target_pose[2] + 0.05, *target_pose[3:]],
            "receptacle_name": surface,
            "room_type": room_types[room],
            "succ_pos_threshold": 0.01,
            "task_horizon_sec": TASK_HORIZON_SEC,
        },
        "task_relevant_objects": [names[target]],
        "language": {"task_description": f"Pick up the {label}", "referral_expressions": {"pickup_obj_name": label}},
    }


def place_on_surface(
    model: mujoco.MjModel, data: mujoco.MjData, surface: str, assets: list[Asset], target: int,
    rng: np.random.Generator, tries: int = 40,
) -> list[list[float]]:
    """
    Poses ([x, y, z, qw, qx, qy, qz]) for `assets` resting on `surface` at random spots and
    headings, the target first anywhere on it, the others within FIVE_OBJECTS_SPREAD of it, each
    where the surface is flat and bare under its whole footprint and clear of the others'.
    """
    surface_id = model.body(surface).id
    low, high = world_aabb(model, data, surface_id)
    order = [target, *(i for i in range(len(assets)) if i != target)]
    placed: dict[int, tuple[np.ndarray, float]] = {}  # index -> (xy, footprint radius)
    poses: dict[int, list[float]] = {}
    for i in order:
        asset_low, asset_high, radius = asset_extents(assets[i])
        for _ in range(tries):
            if not placed:
                if np.any(high[:2] - low[:2] < 2 * radius):
                    raise Rejected(f"{surface} is too small for the {assets[i].label}")
                xy = rng.uniform(low[:2] + radius, high[:2] - radius)
            else:
                angle, distance = rng.uniform(0, 2 * math.pi), FIVE_OBJECTS_SPREAD * math.sqrt(rng.uniform())
                xy = placed[target][0] + distance * np.array([math.cos(angle), math.sin(angle)])
            if any(np.linalg.norm(xy - other) < radius + r + OBJECT_GAP for other, r in placed.values()):
                continue
            top = surface_height(model, data, surface_id, xy, radius, high[2] + 0.05)
            if top is None:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            c, s = math.cos(yaw), math.sin(yaw)
            center = (asset_low[:2] + asset_high[:2]) / 2  # the box's centre goes on xy
            body_xy = xy - np.array([c * center[0] - s * center[1], s * center[0] + c * center[1]])
            z = top - asset_low[2] + INSERT_DROP_HEIGHT
            poses[i] = [*map(float, body_xy), float(z), math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            placed[i] = (xy, radius)
            break
        else:
            raise Rejected(f"No room on {surface} for the {assets[i].label}")
    return [poses[i] for i in range(len(assets))]


def surface_height(
    model: mujoco.MjModel, data: mujoco.MjData, surface_id: int, xy: np.ndarray, radius: float, from_z: float,
    flatness: float = 0.01,
) -> float | None:
    """
    The height of `surface_id`'s top under a footprint of `radius` at `xy`, looking down from
    `from_z`: None unless every ray finds the surface first (nothing on it, and not past its
    edge) and all within `flatness` of each other.
    """
    heights = []
    for r, a in [(0, 0)] + [(radius, k * math.pi / 4) for k in range(8)]:
        origin = np.array([xy[0] + r * math.cos(a), xy[1] + r * math.sin(a), from_z])
        geom = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(model, data, origin, np.array([0, 0, -1.0]), None, 1, -1, geom)
        if dist < 0 or not in_subtree(model, int(model.geom_bodyid[geom[0]]), surface_id):
            return None
        heights.append(from_z - dist)
    return max(heights) if max(heights) - min(heights) <= flatness else None


def in_subtree(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body != 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return root == 0


def settle(scene: CustomScene, surface: str) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, list[float]]]:
    """
    `scene` with its added objects let fall for SETTLE_SECONDS, and where they came to rest.
    Rejected if one starts inside something, slides or falls more than MAX_SETTLE_DRIFT, or
    ends up off `surface`.
    """
    model = scene.load_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = {name: model.body(name).id for name in scene.added_objects}
    for contact in data.contact[: data.ncon]:
        if contact.dist < -0.005 and any(
            in_subtree(model, int(model.geom_bodyid[g]), b) for g in (contact.geom1, contact.geom2) for b in bodies.values()
        ):
            raise Rejected("an object starts inside something")
    start = {name: data.xpos[b].copy() for name, b in bodies.items()}
    for _ in range(round(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
    settled = {}
    for name, b in bodies.items():
        if np.linalg.norm(data.xpos[b] - start[name]) > MAX_SETTLE_DRIFT:
            raise Rejected(f"{name} slid or fell")
        if surface not in contacts_of_subtree(model, data, b):
            raise Rejected(f"{name} is not on {surface}")
        settled[name] = [*map(float, data.xpos[b]), *map(float, data.xquat[b])]
    return model, data, settled


def place_franka(
    model: mujoco.MjModel, data: mujoco.MjData, target: str, target_pose: list[float], rng: np.random.Generator,
    tries: int = 100,
) -> list[float]:
    """The Franka's `robot_base_pose` (x, y, z, qw, qx, qy, qz): ROBOT_DISTANCE from the target,
    on clear floor, the target in sight, facing it to within ROBOT_YAW_NOISE."""
    target_id = model.body(target).id
    low, high = world_aabb(model, data, target_id)
    sight = (low + high) / 2  # its body's origin can be at its bottom, level with the surface
    for _ in range(tries):
        angle, distance = rng.uniform(0, 2 * math.pi), rng.uniform(*ROBOT_DISTANCE)
        x, y = target_pose[0] - distance * math.cos(angle), target_pose[1] - distance * math.sin(angle)
        if robot_footprint_clear(model, data, target_id, sight, x, y, floor_z=0.0):
            yaw = angle + rng.uniform(-ROBOT_YAW_NOISE, ROBOT_YAW_NOISE)
            return [x, y, target_pose[2] + ROBOT_OBJECT_Z_OFFSET, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    raise Rejected(f"No clear floor for the Franka near {target}")


@click.command()
@click.argument("kind", type=click.Choice(KINDS))
@click.option("--object_type", default=DEFAULT_OBJECT_TYPE, show_default=True, help="The object, for one_object.")
@click.option("--episodes", type=int, default=DEFAULT_EPISODES, show_default=True)
@click.option("--workers", type=int, default=None, help="Processes. Default: one per CPU.")
@click.option("--overwrite", is_flag=True, help="Generate it again even if it is already there.")
def main(kind, object_type, episodes, workers, overwrite):
    """Generate (or regenerate) a benchmark of generated scenes."""
    if (benchmark_dir(kind, object_type) / "benchmark.json").exists() and not overwrite:
        sys.exit(f"{benchmark_dir(kind, object_type)} is already there; --overwrite to generate it again")
    generate_benchmark(kind, object_type, episodes, workers)


if __name__ == "__main__":
    main()
