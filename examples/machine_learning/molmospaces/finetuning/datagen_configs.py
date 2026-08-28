"""
Stretch 4 configs for MolmoSpaces' own data generation pipeline.

A released benchmark's 1000-2000 episodes are *fixed*, and they are the test
set: cloning the simple_ik expert over them measures memorisation, and there are
only a few thousand of them either way.

MolmoSpaces' data generation pipeline is the source both learners here use
instead -- `training/` for behaviour cloning, `finetuning/` for a VLA. It samples
tasks procedurally -- pick a house, pick an object, place the robot, plan, roll out --
so it is unbounded, drawn from the training splits, and can be pointed at
`procthor-10k`, `procthor-objaverse` or `holodeck-objaverse`. Its entry point
resolves config classes out of a registry by name:

    python -m molmo_spaces.data_generation.main \\
        examples.machine_learning.molmospaces.finetuning.datagen_configs:StretchPickDataGenConfig

which is what this module populates. `generate_dataset.py` is a friendlier way
in, and does the same thing.

Each config is a MolmoSpaces datagen config with three substitutions -- Stretch's
robot, Stretch's cameras, Stretch's simple_ik expert -- plus the one thing that
genuinely has to change beyond that: **where the robot gets placed.** The
samplers put a Franka within 0.7m of the target because that is the Franka's
reach; Stretch's tool cannot come closer than 0.39m to its own base axis and
cannot go past 0.99m, so a Franka standoff is frequently a pose Stretch cannot
work from at all. `STRETCH_PLACEMENT` widens those constraints to the same band
`stretch/episode_overrides.py` retargets benchmark episodes into, so
generated and retargeted episodes present the robot with the same geometry.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import cast

from examples.machine_learning.molmospaces.added_pickup_repair import repair_added_pickup_masses
from examples.machine_learning.molmospaces.stretch.episode_overrides import REACH_BAND_M
from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicyConfig
from examples.machine_learning.molmospaces.stretch.config import (
    Stretch4CameraSystem,
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.robot import CHASE_CAMERA
from molmo_spaces.configs.base_open_task_configs import ClosingBaseConfig, OpeningBaseConfig
from molmo_spaces.configs.base_pick_and_place_color_configs import PickAndPlaceColorDataGenConfig
from molmo_spaces.configs.base_pick_and_place_configs import PickAndPlaceDataGenConfig
from molmo_spaces.configs.base_pick_and_place_next_to_configs import (
    PickAndPlaceNextToDataGenConfig,
)
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

log = logging.getLogger(__name__)

STRETCH_BASE_SAFETY_RADIUS_M = 0.3
"""
Clearance to keep around Stretch's base when the sampler places it.

The object-manipulation samplers default to 0.15m, sized for a Franka on a
pedestal. Stretch's mobile base is roughly 0.33m across and its mast leans over
it; 0.3m is what `NavToObjTaskSamplerConfig` already uses for the same robot
footprint problem, so it is the value with precedent rather than a new guess.
"""

STRETCH_PLACEMENT_ROTATION_RANGE_RAD = 0.25
"""
How far the sampler may randomise the base yaw after aiming it at the target.

The object-manipulation samplers default to 45 degrees, which costs a Franka on
a pedestal nothing. Stretch's arm extends along its base's +x axis, so a base
turned 45 degrees away from the object is a base whose arm points 45 degrees
away from it -- and while the reach solver can now turn the base back (see
`policies/kinematics.py`), starting it aimed elsewhere just spends the episode's
step budget undoing the sampler's randomisation.

0.25 rather than zero: with no yaw freedom at all the sampler has one fewer
dimension to search for a collision-free pose, and a cluttered house then fails
placement outright -- measured, a test house was abandoned after ten consecutive
`RobotPlacementError`s. 0.25 is the value MolmoSpaces itself uses for its tighter
samplers (`task_sampler_configs.py`).
"""

STRETCH_PLACEMENT: dict[str, object] = {
    "base_pose_sampling_radius_range": REACH_BAND_M,
    "max_robot_to_obj_dist": REACH_BAND_M[1],
    "max_robot_to_target_dist": REACH_BAND_M[1],
    "robot_safety_radius": STRETCH_BASE_SAFETY_RADIUS_M,
    "robot_placement_rotation_range_rad": STRETCH_PLACEMENT_ROTATION_RANGE_RAD,
}
"""
Task-sampler fields to overwrite so the robot is placed where Stretch can work.

Applied by name and only where the field exists, rather than as typed fields on
a config class: the samplers do not share one class, and a field one of them
carries (`max_robot_to_target_dist`, say) is absent from the next. Setting them
by name keeps a single definition of Stretch's standoff for all of them and
degrades quietly for a sampler that has never heard of it.

Note what is *not* in here: `robot_object_z_offset`. The samplers use it to lift
a Franka's base to a height from which the object is reachable, which for
Stretch would be meaningless -- its base is on the floor and its lift covers the
vertical range instead. Harmlessly so, as it happens:
`HoloJointsRobotBaseGroup.pose` only reads x, y and yaw, so the sampler's z is
discarded rather than obeyed.
"""

DATAGEN_ROOT = Path(ASSETS_DIR) / "experiment_output" / "datagen"
"""Where MolmoSpaces' own configs write, so Stretch's runs sit beside them."""

POTATO_SYNSET = "potato.n.01"
"""
WordNet synset every potato asset in the object metadata is annotated with.

The right key to select on rather than the category string: `category` is
`"potato"` for the iTHOR prefabs and `"ObjaPotato"`/`"potato"` for objaverse
assets depending on the pack, while the synset is normalised across both. It is
also what `_pickupable_synset_to_uids()` groups by upstream, so the set this
picks out is the same set MolmoSpaces would call the potato class.
"""

POTATO_PICKUPS_PER_HOUSE = 10
"""
How many potatoes to add to each house's scene.

There are only ~29 potato assets with grasps, so this is not a sample of a large
pool -- it is how much of the pool each house sees. Every added pickupable is
compiled into the scene MJCF whether or not it is used, so the whole pool in
every house would pay model-compile time for assets most episodes never touch;
ten is enough that a house's episodes are not all the same potato, and the
sampler's own random pre-selection differs per house, so the run as a whole
still covers the pool.

Only one is placed in the room at a time -- see `episodes_per_added_pickup`
below and `PickTaskSampler._prepare_added_pickupable`. The rest stay on the
staging platform the sampler parks them on, 25m above the house.
"""


@lru_cache(maxsize=1)
def potato_pickup_uids() -> tuple[str, ...]:
    """Asset UIDs for every potato that can actually be picked up.

    Two filters, both necessary. The synset selects potatoes; `has_valid_pickup_grasps`
    drops the ones with no grasp annotation, which is most of them -- of the 56
    potato assets in the metadata, 29 have grasps, and they are the iTHOR
    `Potato_*` prefabs rather than the objaverse scans. An asset without grasps
    is not merely lower quality: `PickTaskSampler` raises `ValueError` from
    `get_pickup_grasps()` on it and burns an episode attempt per house.

    Not `synset_utils._pickupable_class_ranking()`, which is the upstream route
    to the same thing (`added_pickup_class_rank`): it reads
    `BENCHMARK_BLACKLIST_UIDS_PATH`, an absolute `/weka/...` path that only
    exists inside AI2, so it raises `FileNotFoundError` on any other machine.

    Cached because it walks all ~131k annotation records.
    """
    from molmo_spaces.utils.grasps import has_valid_pickup_grasps
    from molmo_spaces.utils.object_metadata import ObjectMeta

    # Called with no arguments this returns the whole annotation container, a
    # mapping; the signature's `list | dict | None` covers its by-uid overloads.
    annotations = cast(Mapping[str, dict], ObjectMeta.annotation())
    potatoes = sorted(
        uid for uid, anno in annotations.items() if anno.get("synset") == POTATO_SYNSET
    )
    graspable = tuple(uid for uid in potatoes if has_valid_pickup_grasps(uid))

    if not graspable:
        raise RuntimeError(
            f"No potato assets with pickup grasps found: {len(potatoes)} assets are "
            f"annotated {POTATO_SYNSET} but none has a grasp file. The 'objects' and "
            "'grasps' asset packages are probably not installed; generate any other "
            "task first, which installs them on demand."
        )

    log.info(
        f"[stretch-datagen] potato pool: {len(graspable)} graspable of {len(potatoes)} "
        f"{POTATO_SYNSET} assets"
    )
    return graspable


class StretchDataGenMixin:
    """Substitutes Stretch's robot, cameras, expert and standoff into a datagen config.

    A mixin rather than a base class because the seven task families each have
    their own MolmoSpaces datagen base -- `PickBaseConfig`, `OpeningBaseConfig`
    and so on -- carrying the task config, the sampler class and the success
    criteria. Those are the parts that must *not* change: swapping the robot is
    the whole point, and swapping the task would make the data something else.
    """

    robot_config: BaseRobotConfig = Stretch4RobotConfig()

    # Widened from the Franka-specific type on the datagen bases, the same way
    # `Stretch4BenchmarkEvalConfig` widens it -- those bases annotate this field
    # with a concrete Franka camera system, which pydantic would reject a
    # Stretch camera system against.
    camera_config: CameraSystemConfig = Stretch4CameraSystem()

    policy_config: BasePolicyConfig = StretchSimpleIKPolicyConfig()

    # Where `generate_dataset.py --visualize` points the passive viewer. The same
    # camera `Stretch4BenchmarkEvalConfig` uses, and for the same reason:
    # `setup_viewer` in MolmoSpaces' rollout pipeline only accepts a *fixed* MJCF
    # camera here, and the datagen bases leave this as free-camera parameters it
    # ignores -- which frames the whole model, i.e. a sealed procthor house seen
    # from 70m away with the robot invisible inside it. `Stretch4Robot` mounts
    # this camera on the base for exactly this purpose; see `_add_chase_camera()`.
    # Swap in "robot_0/camera_center_link" for the robot's own egocentric view.
    viewer_cam_dict: dict = {"camera": f"{Stretch4RobotConfig().robot_namespace}{CHASE_CAMERA}"}

    apply_stretch_placement: bool = True
    """
    Whether to overwrite the sampler's robot-placement fields with `STRETCH_PLACEMENT`.

    True for every config here, because all of their samplers derive from
    `PickTaskSamplerConfig` and inherit its Franka-reach standoff -- opening and
    closing included, which is easy to get wrong: it is `DoorOpeningTaskSamplerConfig`
    that stands the robot 1.0-1.5m back to leave a swinging door room, not the
    `OpenTaskSamplerConfig` these use.

    The switch exists for the samplers where the distance describes the *task*
    rather than the arm -- door opening, and navigation, which is supposed to
    start out of reach -- so that adding a config for one of those cannot
    silently inherit a manipulation standoff.
    """

    def _init_policy_config(self) -> BasePolicyConfig:
        """Keep the Stretch expert rather than the base's Franka planner.

        The datagen bases build their policy config here (see
        `PickBaseConfig.model_post_init`), and most of them return a cuRobo or
        planner config wired to a Franka. Returning the field as set is what
        makes the class attribute above actually take effect.
        """
        return self.policy_config

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)

        # The simple_ik expert is a demonstrator, not a policy under test: noise on
        # its actions is noise on the demonstrations. MolmoSpaces disables it on
        # every path where a planner drives the robot (`eval_main.py`,
        # `evaluation_configs.py`, `json_eval_task_sampler.py`) but not on the
        # datagen path, where it defaults to on -- up to 2cm of TCP noise against
        # this expert's 3cm arrival tolerance.
        if self.robot_config.action_noise_config is not None:
            self.robot_config.action_noise_config.enabled = False

        if not self.apply_stretch_placement:
            return
        sampler_config = self.task_sampler_config
        for field, value in STRETCH_PLACEMENT.items():
            if hasattr(sampler_config, field):
                setattr(sampler_config, field, value)
                log.debug(f"[stretch-datagen] {type(sampler_config).__name__}.{field} -> {value}")


@register_config("StretchPickDataGenConfig", strict=False)
class StretchPickDataGenConfig(StretchDataGenMixin, PickBaseConfig):
    """Stretch picking objects off surfaces. The workhorse for fine-tuning data."""

    scene_dataset: str = "procthor-objaverse"
    output_dir: Path = DATAGEN_ROOT / "stretch_pick_v1"

    @property
    def tag(self) -> str:
        return "stretch_pick_datagen"


class StretchAddedPickupTaskSampler(PickTaskSampler):
    """`PickTaskSampler` that repairs the mass of the pickupables it adds to a scene.

    Without this, an added THOR prefab weighs 16-40 kg rather than ~30 g, which
    makes it both unliftable and prone to rolling away. See
    `added_pickup_repair` for the measurements and the reason.

    Not potato-specific -- it applies to any THOR prefab used as an added
    pickupable, which is why it selects by name rather than by asset.
    """

    def add_auxiliary_objects(self, spec) -> None:
        super().add_auxiliary_objects(spec)

        if self.config.task_sampler_config.added_pickup_objects is None:
            return

        # `self.added_objects` is every pickupable attached to this scene, before
        # `_select_pickup_object` trims `task_config.added_objects` to the one
        # this episode uses. All of them need repairing: an unused potato still
        # sits on the staging platform as a 20 kg body the solver has to carry.
        added = list(getattr(self, "added_objects", {}) or {})
        repaired = repair_added_pickup_masses(spec, added)

        if repaired:
            log.debug(f"[stretch-datagen] repaired {repaired} added-pickupable visual geoms")
        else:
            # Loud, because a silent no-op here reappears as an unexplained
            # collapse in success rate rather than as an error.
            log.warning(
                f"[stretch-datagen] repaired nothing on {len(added)} added pickupables. "
                "If the asset layout changed upstream they may weigh tens of kilograms; "
                "see added_pickup_repair."
            )


@register_config("StretchPotatoPickDataGenConfig", strict=False)
class StretchPotatoPickDataGenConfig(StretchPickDataGenConfig):
    """Stretch picking up potatoes, and only potatoes.

    Same task, same expert and same houses as `StretchPickDataGenConfig`; the one
    change is *what* gets picked up. Rather than filtering the scene's own
    objects down to potatoes -- procthor kitchens rarely have one, so almost every
    house would be abandoned with `HouseInvalidForTask` -- this uses the sampler's
    pick-from-set mode: `added_pickup_objects` puts a set of potatoes into the
    scene and makes them the pickup targets, while the scene's own objects serve
    only as position anchors to place a potato next to on a cluttered surface.
    So every house that can host a pick episode at all hosts a potato pick.

    Note what is *not* set: `pickup_types`. It is tempting as a "only potatoes"
    filter, but in pick-from-set mode it selects the *reference* objects a potato
    is placed beside, not the pickup target -- restricting those to potatoes
    would put us straight back to the empty-candidate-list failure above.
    """

    output_dir: Path = DATAGEN_ROOT / "stretch_potato_v1"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)

        # Cast only for the type checker: the mixin widens this field's declared
        # type, while every config here builds a pick-family sampler config.
        sampler_config = cast(PickTaskSamplerConfig, self.task_sampler_config)
        sampler_config.added_pickup_objects = list(potato_pickup_uids())
        sampler_config.num_added_pickups = POTATO_PICKUPS_PER_HOUSE

        # Without this the potatoes weigh 16-40 kg each; see the class docstring.
        sampler_config.task_sampler_class = StretchAddedPickupTaskSampler

        # One episode per potato before advancing to the next one in the house's
        # set, so a house's episodes vary the asset rather than repeating one.
        sampler_config.episodes_per_added_pickup = 1

        # `PickTaskSamplerConfig.model_post_init` does this for a config that
        # declares `added_pickup_objects` as a field default; ours is set here,
        # after that has already run. Oversampling exists to raise the share of
        # objaverse assets among *pickup* candidates, and in pick-from-set mode
        # the pickup candidates are the potatoes we added, so it now only skews
        # which scene objects get used as anchors.
        sampler_config.objaverse_oversampling_factor = 1

    @property
    def tag(self) -> str:
        return "stretch_potato_pick_datagen"


@register_config("StretchPickAndPlaceDataGenConfig", strict=False)
class StretchPickAndPlaceDataGenConfig(StretchDataGenMixin, PickAndPlaceDataGenConfig):
    """Stretch picking an object and placing it on a named receptacle."""

    scene_dataset: str = "procthor-objaverse"
    output_dir: Path = DATAGEN_ROOT / "stretch_pick_and_place_v1"

    @property
    def tag(self) -> str:
        return "stretch_pick_and_place_datagen"


@register_config("StretchPickAndPlaceNextToDataGenConfig", strict=False)
class StretchPickAndPlaceNextToDataGenConfig(StretchDataGenMixin, PickAndPlaceNextToDataGenConfig):
    """Stretch placing an object next to a named reference object."""

    scene_dataset: str = "procthor-objaverse"
    output_dir: Path = DATAGEN_ROOT / "stretch_pnp_next_to_v1"

    @property
    def tag(self) -> str:
        return "stretch_pnp_next_to_datagen"


@register_config("StretchPickAndPlaceColorDataGenConfig", strict=False)
class StretchPickAndPlaceColorDataGenConfig(StretchDataGenMixin, PickAndPlaceColorDataGenConfig):
    """Stretch placing an object on a receptacle picked out by colour."""

    scene_dataset: str = "procthor-objaverse"
    output_dir: Path = DATAGEN_ROOT / "stretch_pnp_color_v1"

    @property
    def tag(self) -> str:
        return "stretch_pnp_color_datagen"


@register_config("StretchOpenDataGenConfig", strict=False)
class StretchOpenDataGenConfig(StretchDataGenMixin, OpeningBaseConfig):
    """Stretch opening drawers and cabinets."""

    output_dir: Path = DATAGEN_ROOT / "stretch_open_v1"

    @property
    def tag(self) -> str:
        return "stretch_open_datagen"


@register_config("StretchCloseDataGenConfig", strict=False)
class StretchCloseDataGenConfig(StretchDataGenMixin, ClosingBaseConfig):
    """Stretch closing drawers and cabinets."""

    output_dir: Path = DATAGEN_ROOT / "stretch_close_v1"

    @property
    def tag(self) -> str:
        return "stretch_close_datagen"


@register_config("StretchPickDebugDataGenConfig", strict=False)
class StretchPickDebugDataGenConfig(StretchPickDataGenConfig):
    """Two episodes in one house, on the small scene dataset. Use this first.

    A generation run that turns out to be misconfigured is expensive to discover
    at episode 900, and the failure modes are all visible in the first two: the
    expert plans or it does not, the cameras render or they do not, the robot is
    placed within reach or it is not.
    """

    scene_dataset: str = "procthor-10k"
    num_workers: int = 1
    output_dir: Path = DATAGEN_ROOT / "stretch_pick_debug"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.task_sampler_config.house_inds = [0]
        self.task_sampler_config.samples_per_house = 2
        self.task_sampler_config.max_tasks = 2

    @property
    def tag(self) -> str:
        return "stretch_pick_debug_datagen"


DATAGEN_CONFIGS: dict[str, str] = {
    "pick": "StretchPickDataGenConfig",
    "potato": "StretchPotatoPickDataGenConfig",
    "pnp": "StretchPickAndPlaceDataGenConfig",
    "pnp_next_to": "StretchPickAndPlaceNextToDataGenConfig",
    "pnp_color": "StretchPickAndPlaceColorDataGenConfig",
    "open": "StretchOpenDataGenConfig",
    "close": "StretchCloseDataGenConfig",
    "debug": "StretchPickDebugDataGenConfig",
}
"""
Task family -> registered config name.

Keys match `benchmarks.BENCHMARKS` where a family has both a benchmark and a
datagen config, so the same `--task pick` selects the matching pair. `door_opening`
and `nav_to_obj` have no entry: neither has a MolmoSpaces datagen base that a
robot substitution alone makes work -- door opening's task class is tied to its
authoring robot upstream (see `Benchmark.supported`), and navigation
demonstrations come from the A* planner rather than the manipulation expert.
"""

CONFIG_MODULE = "examples.machine_learning.molmospaces.finetuning.datagen_configs"


def qualified_config_name(task: str) -> str:
    """ "module:Class" for a task family, the form `data_generation.main` takes."""
    if task not in DATAGEN_CONFIGS:
        raise KeyError(
            f"No Stretch datagen config for {task!r}. Available: {sorted(DATAGEN_CONFIGS)}"
        )
    return f"{CONFIG_MODULE}:{DATAGEN_CONFIGS[task]}"
