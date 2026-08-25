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
from pathlib import Path

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
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

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
