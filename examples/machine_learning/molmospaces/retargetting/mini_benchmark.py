"""
The benchmark every trial is scored on: a few scenes, one robot pose each, four objects.

A parameter search needs the smallest benchmark that can still tell a good
setting from a bad one, run identically every time. The released `pick`
benchmark is the opposite of that -- a thousand episodes in a thousand houses,
each with its own lighting, clutter and robot pose -- so a difference of two
successes between two camera settings there says nothing at all. One house is
the opposite error: a setting that suits one kitchen looks like a setting that
works.

So this builds a small benchmark, `DEFAULT_SCENE_COUNT` scenes of four objects:

* **Scene 0 is hand-tuned.** ProcTHOR-10k val 0, the kitchen
  `demo_droid_on_stretch.py` works in and the one its base pose, receptacle
  position and task strings were all chosen against. Its counter runs along
  y = 10.2 at z = 0.938, the robot stands at (6.73, 9.7), and every object goes
  at (7.1, 10.2). It is kept first and kept fixed so older numbers stay
  comparable.
* **The rest are borrowed and then checked.** Each takes a (house, robot base
  pose, spot on a surface) triple from a released `pick` episode -- validated
  upstream, an expert solved it -- and then has to pass `validate_scene`.
  Borrowing alone is not enough: that triple was validated for *its* object and
  *its* task, and the first two candidates the benchmark offers put the spot on
  a toilet cistern and on the episode's own pickup object. So the supporting
  body must be a work surface, the spot must be clear, and the standoff must be
  in reach of both robots.
* **The same four objects everywhere,** chosen to span what makes a grasp hard:
  a bowl is wide and low, a potato is small and round, a salt shaker is narrow
  and tall, a knife is thin and long. Same-category duplicates already in the
  room are removed, as is the borrowed episode's own target, so "pick up the
  knife" names exactly one thing.

Every episode is therefore the same episode apart from the scene and the thing
being picked up, which is what makes "this setup grasps the knife but not the
salt shaker" a statement about the setup.

Build it with `python -m ...retargetting.mini_benchmark --scenes N`, or let
`params_search.py` build it on demand.
"""

from __future__ import annotations

import dataclasses
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
"""Where on the counter every target object stands, in the hand-tuned scene."""

ROBOT_BASE_XY = (6.73, 9.7)
"""Where the robot stands there. Yaw and z are decided per robot by the episode override."""

DEFAULT_SCENE_COUNT = 3
"""
How many scenes the benchmark holds by default.

One scene cannot tell a setting that works from a setting that happens to suit
one kitchen, and the sweep is long enough that the difference matters. Three is
the smallest number where a parameter winning everywhere is worth believing, and
it divides evenly across workers -- MolmoSpaces parallelises by *house*, so the
scene count is also the useful worker count.
"""

PICK_BENCHMARK = "pick"
"""
Where the extra scenes come from: the released MB-Pick benchmark.

Their houses, robot base poses and pickup locations were authored and validated
upstream -- an expert solved them -- so borrowing a (house, base pose, spot on a
surface) triple is far safer than inventing one. This benchmark then keeps the
triple and swaps in its own four objects, so the scenes differ but the task does
not.
"""


WORK_SURFACE_CATEGORIES = frozenset(
    {
        "countertop",
        "counter",
        "diningtable",
        "coffeetable",
        "sidetable",
        "table",
        "desk",
        "shelf",
        "dresser",
        "chestofdrawers",
        "nightstand",
        "stand",
        "cabinet",
    }
)
"""
Bodies a target object may be placed on.

A borrowed episode's pickup spot is only validated for *its* object and *its*
task. Measured on the first two candidates the benchmark offers, one sat on a
toilet cistern and one on a cabinet hinge -- fine for the sandal the episode was
authored around, useless as a bench for "pick up the bowl", and in the first case
physically unreachable for the arm. So the supporting body has to be something a
person would actually put a bowl down on, and that is a list rather than a
heuristic because the alternative is discovering the exception in a rollout
video.

Matched on the body name's leading category, with objaverse's `obja` prefix
stripped: `countertop_<hash>_1_1_2` and `objadiningtable_<hash>_1_0_2` both pass.
"""

CLEARANCE_RADIUS_M = 0.22
"""
How much free space a target spot needs around it, in metres.

Enough for the gripper to come down around an object without fouling whatever
else is on the surface. The second failure the borrowed scenes produced was an
object dropped onto a spot with a medical mask 10cm away and a collar 11cm away,
which is not a grasp anyone can make.
"""

CLEARANCE_HEIGHT_M = 0.25
"""How far above the surface the clearance check looks, in metres."""

REACH_BAND_M = (0.45, 0.85)
"""
Standoff from the robot base a target has to sit in, in metres.

`stretch.episode_overrides.REACH_BAND_M` is (0.55, 0.90) for Stretch and the
Franka's usable envelope is a little shorter; this is the intersection, so a
scene accepted here is workable for both robots without either of them having to
move.
"""


@dataclass(frozen=True)
class Scene:
    """One kitchen, one robot pose, one spot to put objects on."""

    house_index: int
    scene_dataset: str
    data_split: str
    robot_base_xy: tuple[float, float]
    target_xy: tuple[float, float]
    source: str
    removed_objects: tuple[str, ...] = ()
    """
    Scene bodies to delete before anything else happens.

    For a borrowed scene this is the episode's *own* pickup object. It has to go:
    it is sitting exactly where this benchmark wants to put its own object, so
    left in place the new object settles on top of it -- which is what a rollout
    video showed, a bowl balanced on a sandal.
    """
    surface_z: float = float("nan")
    """Height of the supporting surface, measured with `removed_objects` ignored."""

    @property
    def key(self) -> str:
        return f"{self.scene_dataset}_{self.data_split}_{self.house_index}"


HAND_TUNED_SCENE = Scene(
    house_index=HOUSE_INDEX,
    scene_dataset=SCENE_DATASET,
    data_split=HOUSE_SPLIT,
    robot_base_xy=ROBOT_BASE_XY,
    target_xy=TARGET_XY,
    source="hand-tuned (demo_droid_on_stretch.py's kitchen)",
)
"""
Scene 0, kept first and kept fixed.

Everything measured before this benchmark grew past one scene was measured here,
so keeping it in place -- and first -- is what makes the older numbers still
comparable.
"""


def house_paths(scene_dataset: str, split: str) -> dict:
    """The scene index for a dataset, as `{house_index: {"base": path, ...}}`."""
    from molmo_spaces.molmo_spaces_constants import (
        get_procthor_10k_houses,
        get_procthor_objaverse_houses,
    )

    resolver = {
        "procthor-10k": get_procthor_10k_houses,
        "procthor-objaverse": get_procthor_objaverse_houses,
    }.get(scene_dataset)
    if resolver is None:
        raise ValueError(f"No scene resolver for dataset {scene_dataset!r}")
    return resolver(split=split)[split]


def _category(body_name: str) -> str:
    """The leading category of a scene body name, objaverse prefix stripped."""
    category = body_name.split("_")[0].lower()
    return category[4:] if category.startswith("obja") else category


def surface_under(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    xy: tuple[float, float],
    ignore: set[str],
) -> tuple[float, str]:
    """`(height, supporting body)` under `xy`, skipping bodies named in `ignore`.

    A plain downward ray finds whatever is topmost, which for a borrowed scene is
    the episode's own pickup object -- the thing this benchmark is about to
    replace. Skipping it means walking the ray down past each ignored hit and
    casting again, because `mj_ray` can exclude only one body per call.
    """
    height = 2.5
    for _ in range(12):
        origin = np.array([xy[0], xy[1], height], dtype=float)
        geom_id = np.zeros(1, np.int32)
        distance = mujoco.mj_ray(model, data, origin, np.array([0.0, 0.0, -1.0]), None, 1, -1, geom_id)
        if distance < 0 or geom_id[0] < 0:
            return float("nan"), ""
        hit = model.body(model.geom_bodyid[geom_id[0]]).name
        z = float(origin[2] - distance)
        if hit not in ignore:
            return z, hit
        # Just below the hit, so the next cast cannot return the same surface.
        height = z - 1e-3
    return float("nan"), ""


def validate_scene(scene: Scene) -> tuple[Scene | None, str]:
    """Accept a candidate scene, or say why not.

    Three checks, each of which a borrowed spot failed in practice:

    * **a surface worth putting something on** -- not the episode's own target
      object, not a cabinet hinge, not a toilet cistern;
    * **room around it** -- nothing else within `CLEARANCE_RADIUS_M` of the spot
      and above the surface, so the gripper can come down around the object;
    * **within reach** -- the standoff from the recorded base pose lands in
      `REACH_BAND_M`, which both robots can work in without moving.

    Returns the scene with its measured `surface_z` filled in, or `(None, why)`.
    """
    path = house_scene_path(scene)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ignore = set(scene.removed_objects)
    surface_z, support = surface_under(model, data, scene.target_xy, ignore)
    if not support:
        return None, "nothing under the target spot"
    if _category(support) not in WORK_SURFACE_CATEGORIES:
        return None, f"supported by {support[:40]} ({_category(support)}), not a work surface"

    standoff = float(np.hypot(*(np.asarray(scene.target_xy) - np.asarray(scene.robot_base_xy))))
    if not REACH_BAND_M[0] <= standoff <= REACH_BAND_M[1]:
        return None, f"standoff {standoff:.2f}m outside {REACH_BAND_M}"

    for body_id in range(model.nbody):
        name = model.body(body_id).name
        if not name or name in ignore or name == support:
            continue
        if _category(name) in WORK_SURFACE_CATEGORIES or name.startswith(("wall", "room")):
            continue
        position = data.xpos[body_id]
        planar = float(np.hypot(position[0] - scene.target_xy[0], position[1] - scene.target_xy[1]))
        if planar < CLEARANCE_RADIUS_M and surface_z - 0.05 < position[2] < surface_z + CLEARANCE_HEIGHT_M:
            return None, f"{name[:36]} is {planar * 100:.0f}cm from the spot"

    return dataclasses.replace(scene, surface_z=surface_z), ""


def scenes_from_pick_benchmark(count: int, max_candidates: int = 200) -> list[Scene]:
    """`count` further scenes, borrowed from the released pick benchmark and validated.

    Taken in the benchmark's own order so the selection is deterministic, but
    every candidate is checked before it is accepted -- see `validate_scene`. The
    episode's own pickup object is removed, both because it occupies the spot and
    because leaving a second graspable object 0cm away makes the instruction
    ambiguous.

    Compiling a house costs a few seconds, so `max_candidates` bounds how long a
    hunt can run before giving up rather than scanning all thousand episodes.
    """
    from examples.machine_learning.molmospaces.benchmarks import resolve_benchmark_dir

    if count <= 0:
        return []
    episodes = json.loads((resolve_benchmark_dir(PICK_BENCHMARK) / "benchmark.json").read_text())

    scenes: list[Scene] = []
    seen = {HAND_TUNED_SCENE.key}
    rejected = 0
    for episode in episodes[:max_candidates]:
        task = episode["task"]
        candidate = Scene(
            house_index=int(episode["house_index"]),
            scene_dataset=episode["scene_dataset"],
            data_split=episode["data_split"],
            robot_base_xy=(task["robot_base_pose"][0], task["robot_base_pose"][1]),
            target_xy=(task["pickup_obj_start_pose"][0], task["pickup_obj_start_pose"][1]),
            source=f"{PICK_BENCHMARK} benchmark, {episode['source']['traj_key']}",
            removed_objects=(task["pickup_obj_name"],),
        )
        if candidate.key in seen:
            continue
        try:
            if not house_paths(candidate.scene_dataset, candidate.data_split).get(
                candidate.house_index
            ):
                continue
        except (ValueError, KeyError):
            continue
        seen.add(candidate.key)

        validated, why = validate_scene(candidate)
        if validated is None:
            rejected += 1
            log.info(f"[benchmark] skipping house {candidate.house_index}: {why}")
            continue

        log.info(
            f"[benchmark] accepted house {validated.house_index} "
            f"(surface z={validated.surface_z:.3f})"
        )
        scenes.append(validated)
        if len(scenes) == count:
            break

    if len(scenes) < count:
        raise RuntimeError(
            f"Only {len(scenes)} of the {count} extra scenes passed validation after "
            f"{rejected} rejections; raise max_candidates or relax WORK_SURFACE_CATEGORIES."
        )
    return scenes


def build_scenes(count: int = DEFAULT_SCENE_COUNT) -> list[Scene]:
    """The hand-tuned scene, then as many borrowed ones as asked for."""
    return [HAND_TUNED_SCENE, *scenes_from_pick_benchmark(count - 1)]

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

SETTLE_MAX_DRIFT_M = 0.12
"""
How far an object may travel from the spot while settling, in metres.

More than this and it has rolled off the edge or slid down something rather than
come to rest where the episode says it is -- and `pickup_obj_start_pose`, which
the success threshold is measured against, would then describe somewhere the
object is not.
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


def house_scene_path(scene: Scene = HAND_TUNED_SCENE) -> Path:
    """The base MJCF of a scene's house, with its objects and grasps installed."""
    from molmo_spaces.utils.lazy_loading_utils import (
        install_scene_with_objects_and_grasps_from_path,
    )

    path = Path(house_paths(scene.scene_dataset, scene.data_split)[scene.house_index]["base"])
    install_scene_with_objects_and_grasps_from_path(str(path))
    return path


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


def _delete_bodies(spec, names: list[str]) -> None:
    """Remove named bodies from a spec, matching the way the eval sampler does.

    `JsonEvalTaskSampler.add_auxiliary_objects` accepts an exact name or a
    trailing `/name`, and quietly ignores a miss; this mirrors that, so a body
    deleted here is one that will also be deleted at episode load.
    """
    wanted = set(names)
    if not wanted:
        return
    body = spec.worldbody.first_body()
    doomed = []
    while body is not None:
        if body.name in wanted or any(body.name.endswith(f"/{n}") for n in wanted):
            doomed.append(body)
        body = spec.worldbody.next_body(body)
    for body in doomed:
        try:
            spec.delete(body)
        except Exception as error:  # noqa: BLE001 - a miss is not fatal
            log.debug(f"[benchmark] could not delete {body.name}: {error}")


def settled_pose(
    scene_path: Path,
    object_xml: Path,
    release_z: float,
    body_name: str,
    target_xy: tuple[float, float],
    removed: list[str] | None = None,
) -> list[float]:
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
    # The same bodies the episode deletes at load time. Settling against a scene
    # that still holds them would record a pose resting on something that is not
    # there when the rollout runs -- which is how an object ends up hovering, or
    # falling at the first step.
    _delete_bodies(spec, removed or [])
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
        pos=[target_xy[0], target_xy[1], release_z], quat=list(REST_QUAT)
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

    # Did it come to rest where it was meant to? Three ways a settle goes wrong,
    # all seen: the object rolls off the surface, it lands on something else that
    # was standing there, or it never stops moving. A benchmark that ships any of
    # those scores every setup badly for a reason that has nothing to do with the
    # parameters, so this is checked rather than assumed.
    drift = float(np.hypot(position[0] - target_xy[0], position[1] - target_xy[1]))
    speed = float(np.linalg.norm(data.cvel[body_id, 3:]))
    resting_on = ""
    lowest = position[2]
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            lowest = min(lowest, float(data.geom_xpos[geom_id][2]))
    for contact in data.contact[: data.ncon]:
        roots = [model.body_rootid[model.geom_bodyid[g]] for g in (contact.geom1, contact.geom2)]
        if body_id in roots:
            other = roots[0] if roots[1] == body_id else roots[1]
            resting_on = model.body(other).name
            break

    problems = []
    if drift > SETTLE_MAX_DRIFT_M:
        problems.append(f"drifted {drift * 100:.0f}cm from the spot")
    if speed > SETTLE_STILL_SPEED * 20:
        problems.append(f"still moving at {speed:.3f} m/s")
    if not resting_on:
        problems.append("came to rest touching nothing")
    return [*position.tolist(), *orientation.tolist()], resting_on, problems


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


def build_episodes(scenes: list[Scene] | None = None) -> list[dict]:
    """One episode spec per (scene, target), as JSON-ready dicts.

    Scene-major order -- all four objects of scene 0, then all four of scene 1 --
    which is the order the evaluation runs them in, because MolmoSpaces takes a
    *house* as its unit of work. That is also what makes `--num-workers` worth
    anything here: the scenes go to different workers and the four objects within
    one scene stay together on the same compiled model.
    """
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.utils.lazy_loading_utils import install_uid

    scenes = scenes if scenes is not None else build_scenes()
    today = datetime.date.today().isoformat()
    episodes = []
    failures: list[str] = []

    for scene in scenes:
        scene_path = house_scene_path(scene)
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        # Measured with the scene's own removals ignored, so it is the surface
        # the objects will actually land on rather than whatever is standing on
        # it -- `validate_scene` already did this, and repeating it here keeps
        # the two from drifting if a scene is built without validation.
        surface_z, support = surface_under(model, data, scene.target_xy, set(scene.removed_objects))
        removed = sorted(set(distractor_bodies(model)) | set(scene.removed_objects))
        log.info(
            f"[benchmark] house {scene.house_index} ({scene.scene_dataset}): surface at "
            f"z={surface_z:.4f} on {support[:34]}, removing {len(removed)} bodies "
            f"[{scene.source}]"
        )

        for target in TARGETS:
            object_xml = Path(install_uid(target.uid))
            release_z = surface_z + object_base_offset(object_xml, REST_QUAT) + SURFACE_CLEARANCE_M
            start_pose, resting_on, problems = settled_pose(
                scene_path,
                object_xml,
                release_z,
                target.body_name,
                scene.target_xy,
                removed,
            )
            if problems:
                failures.append(
                    f"house {scene.house_index} / {target.key}: " + "; ".join(problems)
                )
            goal_pose = list(start_pose)
            goal_pose[2] += 0.05

            spec = EpisodeSpec(
                source={
                    "h5_file": f"(authored by {__name__})",
                    "traj_key": f"{scene.key}_{target.key}",
                    "camera_system_class": "RetargetExoCameraSystem",
                    "benchmark_created_date": today,
                },
                house_index=scene.house_index,
                scene_dataset=scene.scene_dataset,
                data_split=scene.data_split,
                seed=0,
                robot={
                    # Informational, and overwritten by whichever episode
                    # override in `setups.py` runs -- but `JsonEvalTaskSampler`
                    # validates the field before any override is called, so it
                    # cannot be empty. The Franka's home pose, to match
                    # `PLACEHOLDER_CAMERAS`.
                    "robot_name": "franka_droid",
                    "init_qpos": PLACEHOLDER_INIT_QPOS,
                },
                img_resolution=IMG_RESOLUTION,
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
                        scene.robot_base_xy[0],
                        scene.robot_base_xy[1],
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
            note = f" on {resting_on[:34]}" if resting_on else ""
            log.info(
                f"[benchmark]   {target.key}: {target.instruction!r}, "
                f"z={start_pose[2]:.4f}{note}"
                + (f"  !! {'; '.join(problems)}" if problems else "")
            )

    if failures:
        raise RuntimeError(
            "Objects did not spawn correctly in "
            f"{len({f.split(' / ')[0] for f in failures})} scene(s):\n  "
            + "\n  ".join(failures)
            + "\nThe scene is unsuitable rather than the object -- tighten "
            "WORK_SURFACE_CATEGORIES or CLEARANCE_RADIUS_M so validate_scene rejects it."
        )
    return episodes


def build(output_dir: Path, force: bool = False, scene_count: int = DEFAULT_SCENE_COUNT) -> Path:
    """Write `benchmark.json` and `benchmark_metadata.json`, and return the directory.

    Idempotent unless `force` or the scene count has changed: the file depends
    only on the scenes and the four objects, and building it costs a house
    compile per scene plus a settle per object.
    """
    output_dir = Path(output_dir)
    benchmark_path = output_dir / "benchmark.json"
    if benchmark_path.is_file() and not force:
        existing = json.loads(benchmark_path.read_text())
        houses = len({episode["house_index"] for episode in existing})
        if houses == scene_count:
            log.info(f"[benchmark] reusing {benchmark_path} ({len(existing)} episodes)")
            return output_dir
        log.info(
            f"[benchmark] {benchmark_path} has {houses} scenes but {scene_count} were asked "
            "for; rebuilding"
        )

    scenes = build_scenes(scene_count)
    episodes = build_episodes(scenes)
    output_dir.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w") as handle:
        json.dump(episodes, handle, indent=1)
    with (output_dir / "benchmark_metadata.json").open("w") as handle:
        json.dump(
            {
                "name": "retarget-params",
                "description": (
                    f"{len(scenes)} scenes, one robot pose each, the same four pickup "
                    "objects in every one. Built by retargetting/mini_benchmark.py for the "
                    "camera parameter search."
                ),
                "num_episodes": len(episodes),
                "num_scenes": len(scenes),
                "scenes": [
                    {
                        "house_index": scene.house_index,
                        "scene_dataset": scene.scene_dataset,
                        "data_split": scene.data_split,
                        "source": scene.source,
                    }
                    for scene in scenes
                ],
                "house_indices": [scene.house_index for scene in scenes],
                "objects": [target.key for target in TARGETS],
                "created": datetime.date.today().isoformat(),
            },
            handle,
            indent=1,
        )
    log.info(f"[benchmark] wrote {benchmark_path} ({len(episodes)} episodes)")
    return output_dir


def episode_target(index: int) -> Target:
    """The target of episode `index`, in the order `build_episodes` writes them.

    Scene-major, so the objects cycle within each scene. Only a fallback: the
    search matches an episode to its object by instruction text, which survives
    the episodes arriving out of order.
    """
    return TARGETS[index % len(TARGETS)]


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_output") / "retarget_params" / "benchmark",
    help="Where to write benchmark.json.",
)
@click.option("--force", is_flag=True, help="Rebuild even if the file is already there.")
@click.option(
    "--scenes",
    "scene_count",
    type=int,
    default=DEFAULT_SCENE_COUNT,
    help="How many scenes to build. The first is always the hand-tuned kitchen; the rest "
    "are borrowed from the released pick benchmark.",
)
def main(output_dir: Path, force: bool, scene_count: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    directory = build(output_dir, force=force, scene_count=scene_count)
    click.secho(f"Benchmark at {directory}", fg="green")


if __name__ == "__main__":
    main()
