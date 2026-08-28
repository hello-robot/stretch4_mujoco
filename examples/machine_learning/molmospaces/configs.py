"""
Evaluation configs that run Stretch 4 on the MolmoSpaces benchmarks.

Following the recipe in `molmo_spaces/evaluation/README.md`, an external repo
needs three things to evaluate on a JSON benchmark: a policy class, a policy
config, and an eval config extending `JsonBenchmarkEvalConfig`. The policies live
in `policies/`; this module supplies the eval configs and wires the Stretch
episode retargeting into MolmoSpaces' override registry.

Every config here is addressable from MolmoSpaces' own entry point, e.g.

    python molmo_spaces/evaluation/eval_main.py \\
        examples.machine_learning.molmospaces.configs:StretchSimpleIKEvalConfig \\
        --benchmark_dir <dir> --no_wandb

though `run_benchmarks.py` is the more convenient way in.
"""

from __future__ import annotations

import logging
import os

from examples.machine_learning.molmospaces.added_pickup_repair import install_eval_repair
from examples.machine_learning.molmospaces.policies.bc_policy import StretchBCPolicyConfig
from examples.machine_learning.molmospaces.policies.molmobot_policy import (
    StretchMolmoBotPolicyConfig,
)
from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicyConfig
from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    WRIST_CAMERA_RIGHT,
    Stretch4CameraSystem,
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.robot import CHASE_CAMERA
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


VIEWER_ENV_VAR = "STRETCH_MOLMOSPACES_VIEWER"
"""
Set to "1" to launch MuJoCo's passive viewer during evaluation.

An environment variable rather than an argument because `run_evaluation()`
constructs the experiment config itself, from a class, and exposes no hook for
overriding a field on it -- so the config has to read the request from
somewhere the caller can reach. `run_benchmarks.py --viewer` sets this. Workers
inherit it, which is why that flag also forces single-worker.
"""


MOLMOBOT_ACTION_TYPE_ENV_VAR = "STRETCH_MOLMOSPACES_MOLMOBOT_ACTION_TYPE"
"""
Which action type a `--policy molmobot` checkpoint was trained with.

Same injection route as the viewer, and for the same reason:
`run_evaluation()` builds the config from a class. Unlike those, this one has a
correct-by-default value (`joint_pos_rel`, MolmoBot's own default), so it only
needs setting for a checkpoint trained the other way.
"""


def viewer_requested() -> bool:
    return os.environ.get(VIEWER_ENV_VAR, "") not in ("", "0", "false", "False")


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

# A benchmark whose target object is *added* to the scene (the locally built
# `potato` one) needs the same asset-mass correction data generation applies, or
# every policy scores 0% on a 20kg potato. See `added_pickup_repair`; measured,
# a benchmark built from episodes the expert had just solved replayed at 0/3
# without this. A no-op for the released benchmarks, whose target objects come
# with the scene and whose `added_objects` is empty.
install_eval_repair()


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

    # What `--viewer` points the camera at. `setup_viewer` in MolmoSpaces'
    # rollout pipeline only accepts a fixed MJCF camera here, and without one it
    # leaves Mujoco's default framing of the whole model -- which for a benchmark
    # house, loaded in its "ceiling" variant, is a sealed building seen from 70m
    # away with the robot invisible inside. `Stretch4Robot` mounts this camera on
    # the base for exactly this purpose; see `_add_chase_camera()`. Swap in
    # "robot_0/camera_center_link" for the robot's own egocentric view.
    viewer_cam_dict: dict = {"camera": f"{Stretch4RobotConfig().robot_namespace}{CHASE_CAMERA}"}

    @property
    def tag(self) -> str:
        return "stretch4_benchmark_eval"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.use_passive_viewer = viewer_requested()


class StretchSimpleIKEvalConfig(Stretch4BenchmarkEvalConfig):
    """The simple ik expert: baseline scores, and the teacher for BC data.

    Grasps at the pose the asset's own grasp library authors, falling back to a
    horizontal reach when it cannot reach one -- see
    `StretchSimpleIKPolicyConfig.use_authored_grasps`.
    """

    policy_config: StretchSimpleIKPolicyConfig = StretchSimpleIKPolicyConfig()

    @property
    def tag(self) -> str:
        return "stretch4_simple_ik"


class StretchSimpleIKTopDownEvalConfig(StretchSimpleIKEvalConfig):
    """The simple ik expert reaching straight down instead of side-on.

    Worth running as an ablation on the pick-family benchmarks: top-down clears
    clutter better, but Stretch has to creep its base to correct lateral error
    (see `policies/kinematics.py`), so it is slower and reaches less far.

    Authored grasps are off here, which is what makes this an ablation of the
    hand-written styles rather than a run that quietly ignores `grasp_style` --
    with them on, the style is only what the ~5% of objects with no reachable
    authored grasp fall back to, and the two configs would score almost alike.
    """

    policy_config: StretchSimpleIKPolicyConfig = StretchSimpleIKPolicyConfig(
        grasp_style="top_down", use_authored_grasps=False
    )

    @property
    def tag(self) -> str:
        return "stretch4_simple_ik_top_down"


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
        camera_names=[HEAD_CAMERA, WRIST_CAMERA_RIGHT]
    )

    @property
    def tag(self) -> str:
        return "stretch4_bc"


class StretchMolmoBotEvalConfig(Stretch4BenchmarkEvalConfig):
    """A MolmoBot checkpoint driving Stretch natively.

    For a checkpoint fine-tuned on Stretch's own move groups -- see
    `finetuning/finetune.py --trainer molmobot`. MolmoBot's action space is
    configured by move group, so the checkpoint emits Stretch's own ten numbers;
    `policies/molmobot_policy.py` supplies Stretch's spec and delegates to
    MolmoBot's `SynthVLAPolicy`.

    The *released* `allenai/MolmoBot-DROID` is a different case and is not
    runnable here: it was trained on the `franka_joint` action spec, so it emits
    seven Franka arm joints, which nothing in this repo translates. Fine-tune on
    Stretch data instead -- see `finetuning/README.md`.
    """

    policy_config: StretchMolmoBotPolicyConfig = StretchMolmoBotPolicyConfig()

    @property
    def tag(self) -> str:
        return "stretch4_molmobot"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        action_type = os.environ.get(MOLMOBOT_ACTION_TYPE_ENV_VAR)
        if action_type:
            self.policy_config.action_type = action_type


class StretchDummyEvalConfig(Stretch4BenchmarkEvalConfig):
    """No-op policy. Use it to check that a benchmark loads and renders at all."""

    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    @property
    def tag(self) -> str:
        return "stretch4_dummy"


# Which eval config is the sensible default for each benchmark key. Navigation
# needs a path planner rather than a reach planner, so it does not share the
# manipulation baseline.
#
# The pick family used to default to the top-down config, on the grounds that a
# fixed downward tilt cleared clutter better than a fixed side-on one. That
# choice is obsolete: the tilt is no longer fixed, because the expert now grasps
# at the pose the asset's own grasp library authors and only falls back to a
# hand-written style when it cannot reach one. `StretchSimpleIKTopDownEvalConfig`
# is the ablation of that fallback now, not the baseline.
DEFAULT_BASELINE_CONFIGS: dict[str, str] = {
    "pick": "StretchSimpleIKEvalConfig",
    "potato": "StretchSimpleIKEvalConfig",
    "pnp": "StretchSimpleIKEvalConfig",
    "pnp_next_to": "StretchSimpleIKEvalConfig",
    "pnp_color": "StretchSimpleIKEvalConfig",
    "open": "StretchSimpleIKEvalConfig",
    "close": "StretchSimpleIKEvalConfig",
    "door_opening": "StretchSimpleIKEvalConfig",
    "nav_to_obj": "StretchNavEvalConfig",
}

CONFIG_MODULE = "examples.machine_learning.molmospaces.configs"


def qualified_config_name(class_name: str) -> str:
    """ "module:Class" string, the form `run_evaluation` resolves config names in."""
    return f"{CONFIG_MODULE}:{class_name}"
