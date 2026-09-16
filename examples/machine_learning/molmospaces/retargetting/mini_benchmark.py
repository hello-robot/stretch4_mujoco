"""
The four-episode benchmark every trial is scored on: one kitchen, one pose, four objects.

A parameter search needs the smallest scene that can still tell a good setting
from a bad one, run identically every time. The released `pick` benchmark is the
opposite of that -- a thousand episodes in a thousand houses, each with its own
lighting, clutter and robot pose -- so a difference of two successes between two
camera settings there says nothing at all.

So this builds a benchmark by hand:

* **One house.** ProcTHOR-10k val 0, the kitchen `demo_droid_on_stretch.py`
  works in and the one its base pose, receptacle position and task strings were
  all chosen against. Its counter runs along y = 10.2 at z = 0.938.
* **One robot pose.** The demo's (6.73, 9.7), which leaves a 34cm chassis on
  open floor 0.62m from the target -- inside Stretch's reach band and inside the
  Franka's, so both robots work from the same spot and the setups stay
  comparable. Yaw and base height are set per robot by the episode overrides in
  `setups.py`.
* **One spot on the counter,** (7.1, 10.2), with the four objects put there one
  at a time. Every episode is therefore the same episode apart from the thing
  being picked up, which is what makes "this setup grasps the knife but not the
  salt shaker" a statement about the setup.
* **Four objects,** chosen to span what makes a grasp hard: a bowl is wide and
  low, a potato is small and round, a salt shaker is narrow and tall, a knife is
  thin and long. Their duplicates already standing on that counter are removed,
  so "pick up the knife" names exactly one thing in the room.

Build it with `python -m ...retargetting.mini_benchmark`, or let
`params_search.py` build it on demand.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

log = logging.getLogger(__name__)

HOUSE_SPLIT = "val"
HOUSE_INDEX = 0
SCENE_DATASET = "procthor-10k"

TARGET_XY = (7.1, 10.2)
"""Where on the counter every target object stands. See the module docstring."""

ROBOT_BASE_XY = (6.73, 9.7)
"""Where the robot stands. Yaw and z are decided per robot by the episode override."""

SURFACE_CLEARANCE_M = 0.01
"""How far above the counter an object is released before it is settled.

Small and positive: releasing an object intersecting the counter makes MuJoCo
push it out on the first step, which is a bounce rather than a settle. Five
millimetres is a drop too short to bounce and too tall to interpenetrate.
"""

SETTLE_SECONDS = 3.0
"""
How long the object is left to fall over before its pose is recorded.

Three seconds of simulated time, which is about ten times what any of these four
take to stop moving -- `_settle_pose` stops early once the object is still, so
the number only bounds the pathological case.
"""

SETTLE_STILL_SPEED = 1e-3
"""Linear speed, in m/s, below which an object counts as having come to rest."""

REST_QUAT: tuple[float, float, float, float] = (
    float(np.cos(np.pi / 4)),
    float(np.sin(np.pi / 4)),
    0.0,
    0.0,
)
"""
A quarter turn about +x, as [w, x, y, z]: the orientation these prefabs stand in.

THOR authors its prefabs y-up. Measured on the four here, the short axis is y
for the bowl and the salt shaker and the long axis is z for the knife, so
dropped at the identity quaternion the bowl lies on its side and the knife
balances on its tip -- and, being nearly massless until the repair below, stays
there. This is the same correction `demo_droid_on_stretch.py` applies when it
stands its Bowl_3 receptacle on the counter.

It is a starting orientation, not the authored one: `settled_pose` drops the
object from here and records wherever it actually comes to rest.
"""

TASK_CLS = "molmo_spaces.tasks.pick_task.PickTask"
TASK_HORIZON_SEC = 20.0
SUCCESS_LIFT_M = 0.01

IMG_RESOLUTION = (656, 400)
"""
Recorded offscreen buffer. Overwritten per trial by the episode overrides, which
size it to whatever the trial's cameras render at; this is only what the file
says, and it has to be big enough not to clip them if it is ever replayed
without an override.
"""


PLACEHOLDER_CAMERAS = [
    {
        "name": "exo_camera_1",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/fr3_link0"],
        "camera_offset": [0.1, 0.57, 0.66],
        "lookat_offset": [0.0, 0.0, 0.08],
        "camera_quaternion": [-0.3633, -0.1241, 0.4263, 0.8191],
        "fov": 71.0,
    },
    {
        "name": "wrist_camera",
        "type": "robot_mounted",
        "reference_body_names": ["robot_0/gripper/base"],
        "camera_offset": [0.043, 0.079, 0.02],
        "lookat_offset": [0.0, 0.0, 0.08],
        "camera_quaternion": [-0.0303, -0.0338, 0.9927, 0.1119],
        "fov": 56.74,
    },
]
PLACEHOLDER_INIT_QPOS = {
    "arm": [0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.0],
    "gripper": [0.00296, 0.00296],
}
"""`FrankaRobotConfig.init_qpos`, for the same reason as `PLACEHOLDER_CAMERAS`."""

"""
What the episode file records under `cameras`, and what nothing renders.

`FrankaDroidCameraSystem`'s two, written out rather than imported so the
benchmark file stays a self-contained description of an episode -- which is the
whole premise of the JSON benchmark format. See the comment at the use site.
"""


@dataclass(frozen=True)
class Target:
    """One object to pick up, and the words the instruction names it with."""

    key: str
    uid: str
    referral: str

    @property
    def instruction(self) -> str:
        return f"Pick up the {self.referral}"

    @property
    def body_name(self) -> str:
        """Name the object is added to the scene under.

        The `pickup/` prefix is the namespace `build_benchmark.py` and the
        released benchmarks use for a staged pickup object, and
        `added_pickup_repair` keys its asset-mass correction off exactly this
        shape of name -- without which an added object can weigh 20kg and no
        policy lifts anything.
        """
        return f"pickup/0_0/{self.uid}"


TARGETS: tuple[Target, ...] = (
    Target(key="bowl", uid="Bowl_3", referral="bowl"),
    Target(key="potato", uid="Potato_9", referral="potato"),
    Target(key="salt_shaker", uid="Salt_Shaker_1", referral="salt shaker"),
    Target(key="knife", uid="Knife_1", referral="knife"),
)

TARGET_KEYS = tuple(target.key for target in TARGETS)

DISTRACTOR_PREFIXES = (
    "bowl",
    "potato",
    "irishpotato",
    "saltshaker",
    "salt_shaker",
    "peppershaker",
    "knife",
    "butterknife",
)
"""
Scene objects removed from every episode, matched case-insensitively on the body
name's leading category.

This kitchen already has a knife, a salt shaker and a potato on the same
counter. Leaving them in would make "pick up the knife" ambiguous, and a
language-conditioned policy reaching for the wrong knife would be scored as a
camera problem. `peppershaker` goes too: it is a salt shaker in everything but
the label.
"""


# =============================================================================
# Where an object comes to rest
# =============================================================================


def house_scene_path() -> Path:
    """The base MJCF of the house, with its objects and grasps installed."""
    from molmo_spaces.molmo_spaces_constants import get_procthor_10k_houses
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    houses = get_procthor_10k_houses(split=HOUSE_SPLIT)
    path = Path(houses[HOUSE_SPLIT][HOUSE_INDEX]["base"])
    install_scene_with_objects_and_grasps_from_path(str(path))
    return path


def surface_height(model: mujoco.MjModel, data: mujoco.MjData, xy: tuple[float, float]) -> float:
    """Height of whatever is under `xy`, by casting a ray straight down from 2m."""
    origin = np.array([xy[0], xy[1], 2.0])
    direction = np.array([0.0, 0.0, -1.0])
    geom_id = np.zeros(1, np.int32)
    distance = mujoco.mj_ray(model, data, origin, direction, None, 1, -1, geom_id)
    if distance < 0:
        raise ValueError(f"Nothing under {xy} in this house; the counter is not where it was.")
    return float(origin[2] - distance)


def object_base_offset(object_xml: Path, quat_wxyz: tuple[float, float, float, float]) -> float:
    """How far the object's lowest point sits below its body origin, in metres.

    Measured with `quat_wxyz` applied, since that is the orientation the object
    is released at. Only used to decide where to *release* it in `settled_pose`;
    where it ends up is decided by physics. The bound comes from each geom's
    axis-aligned box rotated into the body frame, which is loose for a mesh and
    does not need to be tight for that purpose.
    """
    from scipy.spatial.transform import Rotation as Rot

    body_rotation = Rot.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True).as_matrix()
    model = mujoco.MjModel.from_xml_path(str(object_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lowest = 0.0
    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float
    )
    for geom_id in range(model.ngeom):
        centre = model.geom_aabb[geom_id, :3]
        half = model.geom_aabb[geom_id, 3:]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        position = data.geom_xpos[geom_id]
        points = (body_rotation @ (position + (rotation @ (centre + corners * half).T).T).T).T
        lowest = min(lowest, float(points[:, 2].min()))
    return -lowest


def settled_pose(scene_path: Path, object_xml: Path, release_z: float, body_name: str) -> list[float]:
    """Drop the object on the counter and return the pose it comes to rest in.

    An asset's canonical orientation is not a resting one -- `Knife_1` is
    authored standing on its tip, and released upright it would be authored into
    an episode balanced on end. What the benchmark wants is the pose the object
    is actually found in, and the cheapest correct way to get that is to let the
    simulator decide: attach the object to the house, release it a few
    millimetres above the counter, and read the free joint back once it has
    stopped moving.

    Done per object rather than once for all four because they have to settle at
    the *same* spot on the counter, and four objects dropped on one spot land in
    a pile.
    """
    from mujoco import MjSpec

    from examples.machine_learning.molmospaces.added_pickup_repair import (
        repair_added_pickup_masses,
    )

    spec = MjSpec.from_file(str(scene_path))
    object_spec = MjSpec.from_file(str(object_xml))
    body = object_spec.worldbody.bodies[0]
    if not body.first_joint():
        # Same fix `JsonEvalTaskSampler.add_auxiliary_objects` applies: a staged
        # object with no joint is welded to the world and cannot fall at all.
        body.add_joint(name="XYZ_jntfree", type=mujoco.mjtJoint.mjJNT_FREE)
    # Read before attaching: `attach_body` renames the body in place, so
    # afterwards `body.name` is already the prefixed one.
    prefix = body_name.rpartition("/")[0] + "/"
    attached_name = f"{prefix}{body.name}"
    frame = spec.worldbody.add_frame(
        pos=[TARGET_XY[0], TARGET_XY[1], release_z], quat=list(REST_QUAT)
    )
    frame.attach_body(body, prefix, "")
    # The same correction evaluation applies to this object. Without it a THOR
    # prefab attached straight from `install_uid` weighs tens of kilograms --
    # its visual mesh carries `inertia="shell"` and no mass, so MuJoCo derives
    # one from surface area at 1000 kg/m^3 -- and an object that heavy neither
    # topples into a natural pose nor can be lifted. See `added_pickup_repair`.
    repair_added_pickup_masses(spec, [body_name])

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = model.body(attached_name).id

    steps = int(SETTLE_SECONDS / model.opt.timestep)
    for step in range(steps):
        mujoco.mj_step(model, data)
        # Not before it has had time to start falling: at step 0 the object is
        # released from rest, so "still" is true and meaningless.
        if step > 100 and step % 50 == 0 and float(np.linalg.norm(data.cvel[body_id, 3:])) < SETTLE_STILL_SPEED:
            break

    position = np.asarray(data.xpos[body_id], dtype=float)
    orientation = np.asarray(data.xquat[body_id], dtype=float)
    return [*position.tolist(), *orientation.tolist()]


def asset_relative_path(uid: str) -> str:
    """`uid`'s XML as a path relative to `ASSETS_DIR`, installing it if need be.

    `JsonEvalTaskSampler.add_auxiliary_objects` resolves `added_objects` as
    `ASSETS_DIR / <recorded path>` and only falls back to `install_uid` when
    that misses -- and then insists the two agree -- so an absolute path here
    would fail at episode load rather than at build time. The installed path
    carries a version directory (`objects/thor/20251117/...`) that the assets
    tree presents unversioned, hence the two candidates.
    """
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    installed = Path(install_uid(uid))
    parts = list(installed.parts)
    if "objects" not in parts:
        raise ValueError(f"{uid} installed outside the object tree, at {installed}")
    tail = parts[parts.index("objects") :]

    candidates = [Path(*tail)]
    if len(tail) > 2:
        candidates.append(Path(*tail[:2], *tail[3:]))  # drop the version segment
    for candidate in candidates:
        if (Path(ASSETS_DIR) / candidate).is_file():
            return candidate.as_posix()
    raise FileNotFoundError(
        f"{uid} is installed at {installed} but not reachable under {ASSETS_DIR}; "
        f"tried {[c.as_posix() for c in candidates]}"
    )


def distractor_bodies(model: mujoco.MjModel) -> list[str]:
    """Scene bodies whose category collides with one of the four targets."""
    names = []
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if not name:
            continue
        category = name.split("_")[0].lower()
        if category in DISTRACTOR_PREFIXES:
            names.append(name)
    return sorted(names)


# =============================================================================
# Building the file
# =============================================================================


def build_episodes() -> list[dict]:
    """One episode spec per target, as JSON-ready dicts."""
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    scene_path = house_scene_path()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    counter_z = surface_height(model, data, TARGET_XY)
    removed = distractor_bodies(model)
    log.info(f"[benchmark] counter top at z={counter_z:.4f}, removing {len(removed)} distractors")

    today = datetime.date.today().isoformat()
    episodes = []
    for target in TARGETS:
        object_xml = Path(install_uid(target.uid))
        release_z = counter_z + object_base_offset(object_xml, REST_QUAT) + SURFACE_CLEARANCE_M
        start_pose = settled_pose(scene_path, object_xml, release_z, target.body_name)
        goal_pose = list(start_pose)
        goal_pose[2] += 0.05

        spec = EpisodeSpec(
            source={
                "h5_file": f"(authored by {__name__})",
                "traj_key": target.key,
                "camera_system_class": "RetargetExoCameraSystem",
                "benchmark_created_date": today,
            },
            house_index=HOUSE_INDEX,
            scene_dataset=SCENE_DATASET,
            data_split=HOUSE_SPLIT,
            seed=0,
            robot={
                # Informational, and overwritten by whichever episode override
                # in `setups.py` runs -- but `JsonEvalTaskSampler` validates the
                # field before any override is called, so it cannot be empty.
                # The Franka's home pose, to match `PLACEHOLDER_CAMERAS`.
                "robot_name": "franka_droid",
                "init_qpos": PLACEHOLDER_INIT_QPOS,
            },
            img_resolution=IMG_RESOLUTION,
            # Every setup installs its own cameras in its episode override, so
            # these are never the ones rendered -- but `JsonEvalTaskSampler`
            # rejects an episode with no cameras at all, so the defaults of
            # setup 1 stand in. Anything recorded here that a setup does not
            # replace would be a camera nobody chose, which is why there are
            # exactly the two every setup does replace.
            cameras=PLACEHOLDER_CAMERAS,
            scene_modifications={
                "added_objects": {target.body_name: asset_relative_path(target.uid)},
                "object_poses": {target.body_name: start_pose},
                "removed_objects": removed,
            },
            task={
                "task_cls": TASK_CLS,
                "task_type": "pick",
                "robot_base_pose": [
                    ROBOT_BASE_XY[0],
                    ROBOT_BASE_XY[1],
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "pickup_obj_name": target.body_name,
                "pickup_obj_start_pose": start_pose,
                "pickup_obj_goal_pose": goal_pose,
                "succ_pos_threshold": SUCCESS_LIFT_M,
                "task_horizon_sec": TASK_HORIZON_SEC,
            },
            task_relevant_objects=[target.body_name],
            language={
                "task_description": target.instruction,
                "referral_expressions": {"pickup_obj_name": target.referral},
            },
        )
        episodes.append(spec.model_dump(mode="json"))
        log.info(f"[benchmark] {target.key}: {target.instruction!r}, resting at {np.round(start_pose, 4).tolist()}")

    return episodes


def build(output_dir: Path, force: bool = False) -> Path:
    """Write `benchmark.json` and `benchmark_metadata.json`, and return the directory.

    Idempotent unless `force`: the file depends only on the house and the four
    objects, none of which change between runs, and building it costs a house
    compile.
    """
    output_dir = Path(output_dir)
    benchmark_path = output_dir / "benchmark.json"
    if benchmark_path.is_file() and not force:
        log.info(f"[benchmark] reusing {benchmark_path}")
        return output_dir

    episodes = build_episodes()
    output_dir.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w") as handle:
        json.dump(episodes, handle, indent=1)
    with (output_dir / "benchmark_metadata.json").open("w") as handle:
        json.dump(
            {
                "name": "retarget-params",
                "description": (
                    "One ProcTHOR-10k val house, one robot pose, four pickup objects. "
                    "Built by retargetting/mini_benchmark.py for the camera parameter search."
                ),
                "num_episodes": len(episodes),
                "house_indices": [HOUSE_INDEX],
                "objects": [target.key for target in TARGETS],
                "created": datetime.date.today().isoformat(),
            },
            handle,
            indent=1,
        )
    log.info(f"[benchmark] wrote {benchmark_path} ({len(episodes)} episodes)")
    return output_dir


def episode_target(index: int) -> Target:
    """The target of episode `index`, in the order `build_episodes` writes them."""
    return TARGETS[index % len(TARGETS)]


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_output") / "retarget_params" / "benchmark",
    help="Where to write benchmark.json.",
)
@click.option("--force", is_flag=True, help="Rebuild even if the file is already there.")
def main(output_dir: Path, force: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    directory = build(output_dir, force=force)
    click.secho(f"Benchmark at {directory}", fg="green")


if __name__ == "__main__":
    main()
