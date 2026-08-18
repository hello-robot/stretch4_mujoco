"""
Evaluation configs that run Stretch 4 on the MolmoSpaces benchmarks.

Following the recipe in `molmo_spaces/evaluation/README.md`, an external repo
needs three things to evaluate on a JSON benchmark: a policy class, a policy
config, and an eval config extending `JsonBenchmarkEvalConfig`. The policies live
in `policies/`; this module supplies the eval configs and wires the Stretch
episode retargeting into MolmoSpaces' override registry.

Every config here is addressable from MolmoSpaces' own entry point, e.g.

    python molmo_spaces/evaluation/eval_main.py \\
        examples.machine_learning.molmospaces.configs:StretchScriptedEvalConfig \\
        --benchmark_dir <dir> --no_wandb

though `run_benchmarks.py` is the more convenient way in.
"""

from __future__ import annotations

import logging

from examples.machine_learning.molmospaces.policies.bc_policy import StretchBCPolicyConfig
from examples.machine_learning.molmospaces.policies.scripted import StretchScriptedPolicyConfig
from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    WRIST_CAMERA,
    Stretch4CameraSystem,
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.episode_overrides import (
    stretch_episode_override,
)
from molmo_spaces.configs.camera_configs import CameraSystemConfig
from molmo_spaces.configs.policy_configs import AStarNavToObjPolicyConfig, DummyPolicyConfig
from molmo_spaces.configs.task_configs import NavToObjTaskConfig
from molmo_spaces.configs.task_sampler_configs import NavToObjTaskSamplerConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.evaluation.robot_eval_overrides import (
    ROBOT_OVERRIDE_REGISTRY,
    register_robot_override,
)
from molmo_spaces.tasks.nav_task import NavToObjTask
from molmo_spaces.tasks.nav_task_sampler import NavToObjTaskSampler

log = logging.getLogger(__name__)


def register_stretch_episode_override() -> None:
    """Teach `JsonEvalRunner.adjust_robot()` how to retarget episodes for Stretch.

    `register_robot_override` raises on a duplicate registration, and this module
    is imported once per evaluation worker process *and* re-imported when an eval
    config is resolved from its "module:Class" string, so the guard is load-bearing
    rather than defensive.
    """
    if Stretch4RobotConfig not in ROBOT_OVERRIDE_REGISTRY:
        register_robot_override(Stretch4RobotConfig, stretch_episode_override)


register_stretch_episode_override()


class Stretch4BenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    """Shared settings for every Stretch benchmark evaluation.

    Subclasses supply only a `policy_config`. Everything episode-specific --
    scene, object poses, instruction, success criteria -- still comes from the
    benchmark JSON, and everything robot-specific is handled by
    `stretch_episode_override`.
    """

    robot_config: Stretch4RobotConfig = Stretch4RobotConfig()

    # Declared here because `JsonBenchmarkEvalConfig` narrows this field's type to
    # `None` (it expects the benchmark's recorded cameras to fill it in). Stretch
    # replaces those cameras rather than replaying them, so the field has to
    # accept a camera system again.
    camera_config: CameraSystemConfig | None = Stretch4CameraSystem()

    # 15Hz. Fast enough that the position controllers track a moving waypoint
    # smoothly, slow enough that a benchmark's `task_horizon_sec` still converts
    # to a workable number of steps (20s of MB-Pick becomes ~300).
    policy_dt_ms: float = 66.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0

    end_on_success: bool = True

    @property
    def tag(self) -> str:
        return "stretch4_benchmark_eval"


class StretchScriptedEvalConfig(Stretch4BenchmarkEvalConfig):
    """The scripted expert: baseline scores, and the teacher for BC data."""

    policy_config: StretchScriptedPolicyConfig = StretchScriptedPolicyConfig()

    @property
    def tag(self) -> str:
        return "stretch4_scripted"


class StretchScriptedTopDownEvalConfig(StretchScriptedEvalConfig):
    """The scripted expert reaching straight down instead of side-on.

    Worth running as an ablation on the pick-family benchmarks: top-down clears
    clutter better, but Stretch has to creep its base to correct lateral error
    (see `policies/kinematics.py`), so it is slower and reaches less far.
    """

    policy_config: StretchScriptedPolicyConfig = StretchScriptedPolicyConfig(grasp_style="top_down")

    @property
    def tag(self) -> str:
        return "stretch4_scripted_top_down"


class StretchNavEvalConfig(Stretch4BenchmarkEvalConfig):
    """Navigation baseline, using MolmoSpaces' own A* planner.

    `AStarSmoothPlannerPolicy` only ever emits `{"base": [x, y, theta]}`, and
    Stretch's holonomic base takes exactly that command in exactly that mode, so
    it runs against Stretch unmodified. It plans on the scene's occupancy map and
    samples its goal with a `NavGoalSampler` pointed at a camera called
    "head_camera" -- which is what `Stretch4CameraSystem` calls Stretch's own
    head camera, so the goal it picks is one Stretch can actually see the target
    from.
    """

    policy_config: AStarNavToObjPolicyConfig = AStarNavToObjPolicyConfig()

    # `JsonBenchmarkEvalConfig` leaves a bare `BaseMujocoTaskSamplerConfig` here
    # as a stub, on the grounds that `JsonEvalTaskSampler` replaces the sampler
    # per episode anyway. That holds for the manipulation policies, but
    # `AStarPlannerPolicy.__init__` reads `task_sampler_config.robot_safety_radius`
    # to size its planner inflation, and the stub has no such field -- the
    # evaluation dies with an AttributeError before the first step. The nav
    # sampler config supplies it (0.3m).
    task_sampler_config: NavToObjTaskSamplerConfig = NavToObjTaskSamplerConfig(
        task_sampler_class=NavToObjTaskSampler,
        house_inds=[0],  # overwritten by JsonEvalRunner from the benchmark
        samples_per_house=1,
        task_batch_size=1,
        max_tasks=10000,
    )
    task_config: NavToObjTaskConfig = NavToObjTaskConfig(task_cls=NavToObjTask)

    # Navigation episodes run 100s; at 15Hz that is 1500 steps of waypoint
    # following, which the A* policy paces comfortably.
    policy_dt_ms: float = 66.0

    @property
    def tag(self) -> str:
        return "stretch4_astar_nav"


class StretchBCEvalConfig(Stretch4BenchmarkEvalConfig):
    """A behaviour-cloned policy trained by `training/train_bc.py`.

    `checkpoint_path` is normally supplied on the command line
    (`--checkpoint_path`), which `run_evaluation` writes over this config's value.
    """

    policy_config: StretchBCPolicyConfig = StretchBCPolicyConfig(
        camera_names=[HEAD_CAMERA, WRIST_CAMERA]
    )

    @property
    def tag(self) -> str:
        return "stretch4_bc"


class StretchDummyEvalConfig(Stretch4BenchmarkEvalConfig):
    """No-op policy. Use it to check that a benchmark loads and renders at all."""

    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    @property
    def tag(self) -> str:
        return "stretch4_dummy"


# Which eval config is the sensible default for each benchmark key. Navigation
# needs a path planner rather than a reach planner, so it does not share the
# manipulation baseline.
DEFAULT_BASELINE_CONFIGS: dict[str, str] = {
    "pick": "StretchScriptedEvalConfig",
    "pnp": "StretchScriptedEvalConfig",
    "pnp_next_to": "StretchScriptedEvalConfig",
    "pnp_color": "StretchScriptedEvalConfig",
    "open": "StretchScriptedEvalConfig",
    "close": "StretchScriptedEvalConfig",
    "door_opening": "StretchScriptedEvalConfig",
    "nav_to_obj": "StretchNavEvalConfig",
}

CONFIG_MODULE = "examples.machine_learning.molmospaces.configs"


def qualified_config_name(class_name: str) -> str:
    """ "module:Class" string, the form `run_evaluation` resolves config names in."""
    return f"{CONFIG_MODULE}:{class_name}"
