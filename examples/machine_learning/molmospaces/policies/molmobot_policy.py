"""
Run a MolmoBot checkpoint on Stretch natively.

A policy trained on a Franka emits Franka joint angles, and nothing here
translates those. MolmoBot does not have to: its action space is configured *by
move group* (`MolmoBot/olmo/data/synthmanip_presets.py` -- `franka_joint` is
`arm(7), gripper(1)`, `RBY1_full` is seven groups totalling 29), and its
MolmoSpaces evaluation policy hands back an action dict keyed by move group,
which is exactly the shape Stretch's controllers already take.

So a MolmoBot checkpoint fine-tuned on Stretch's own move groups drives Stretch
directly, in the numbers it was trained on.

    # a checkpoint fine-tuned with finetuning/finetune.py --trainer molmobot
    python -m examples.machine_learning.molmospaces.run_benchmarks \\
        --policy molmobot --checkpoint /path/to/checkpoint --benchmark pick

This module is a thin adapter, not a reimplementation. MolmoBot's own
`SynthVLAPolicy` does the work: it buffers an `action_horizon`-step prediction,
executes `execute_horizon` of it before re-querying, reads cameras by name out of
the MolmoSpaces observation, and unpacks its output vector across the configured
move groups. All this does is construct it with Stretch's spec and delegate.

**MolmoBot is not a dependency of this repository**, so the import is lazy and
the field names below are taken from its published source rather than validated
against an installed copy. If its constructor has moved, the error names exactly
which fields were passed, which is the information needed to fix it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    HEAD_CAMERA_LEFT,
    HEAD_CAMERA_RIGHT,
    WRIST_CAMERA_LEFT,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

STRETCH_ACTION_SPEC: dict[str, int] = {
    "base": 3,
    "lift": 1,
    "arm": 1,
    "wrist": 3,
    "gripper": 2,
}
"""
Stretch's move groups and their widths -- MolmoBot's `action_spec`.

Ten numbers over five groups, from `Stretch4RobotView`. Must be identical to
`finetuning/finetune.STRETCH_ACTION_SPEC`, which is what a checkpoint gets
trained against; a mismatch would unpack the model's output vector into the
wrong joints, and every joint after the first wrong group would be silently
misassigned. Asserted equal in `tests/test_stretch_finetuning.py`.
"""

MOLMOBOT_MODULES = (
    "olmo.eval.configure_molmo_spaces",
    "MolmoBot.olmo.eval.configure_molmo_spaces",
    "molmobot.olmo.eval.configure_molmo_spaces",
)
"""
Where to look for `SynthVLAPolicy`, in order.

MolmoBot's repository root holds a `MolmoBot/` package directory whose own
top-level package is `olmo`, so how it imports depends on which directory is on
`PYTHONPATH` -- its README runs `PYTHONPATH=. python launch_scripts/...` from
inside `MolmoBot/MolmoBot`, which makes it `olmo.*`. All three spellings are
tried rather than documenting one, because getting it wrong produces an
ImportError that looks like "MolmoBot is not installed" when it is.
"""


class StretchMolmoBotPolicyConfig(BasePolicyConfig):
    """Configuration for `StretchMolmoBotPolicy`."""

    policy_type: str = "learned"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    checkpoint_path: str | None = None
    """
    The fine-tuned MolmoBot checkpoint. Usually supplied as `--checkpoint`, which
    `run_evaluation` writes over whatever is set here.
    """

    action_type: str = "joint_pos_rel"
    """
    `joint_pos_rel` or `joint_pos`, matching how the checkpoint was trained.

    MolmoBot's dataset prefers the relative key and its trainer defaults to it,
    so that is the default here too. Getting it wrong is not subtle for long but
    it is subtle at first: absolute targets applied as deltas make the arm creep
    away from wherever it started.
    """

    action_horizon: int = 16
    """Steps the model predicts per query. `SynthVLAPolicyConfig`'s own default."""

    execute_horizon: int = 8
    """Steps executed before re-querying. Also MolmoBot's default."""

    camera_names: list[str] = [
        HEAD_CAMERA,
        WRIST_CAMERA_LEFT,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
    ]
    """
    Cameras to feed, in order.

    Defaults to all four Stretch 4 camera viewpoints (head center, wrist, head left fisheye,
    head right fisheye).
    """

    action_move_group_names: list[str] = list(STRETCH_ACTION_SPEC)
    action_spec: dict[str, int] = dict(STRETCH_ACTION_SPEC)

    extra_policy_kwargs: dict = {}
    """
    Anything else to pass to `SynthVLAPolicyConfig`.

    An escape hatch, because MolmoBot's config carries fields this integration
    has no opinion about (`cameras_to_warp`, `use_point_prompts`,
    `point_prompt_camera`, ...) and its defaults for them are the right ones
    until someone needs otherwise. Point prompts in particular already default
    to `head_camera`, which is what Stretch's head camera is called.
    """

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if set(self.action_move_group_names) != set(self.action_spec):
            raise ValueError(
                "action_move_group_names and action_spec must cover the same groups; got "
                f"{sorted(self.action_move_group_names)} and {sorted(self.action_spec)}."
            )
        if self.policy_cls is None:
            self.policy_cls = StretchMolmoBotPolicy
            self.policy_factory = make_lenient(StretchMolmoBotPolicy)


class StretchMolmoBotPolicy(BasePolicy):
    """MolmoBot's `SynthVLAPolicy`, configured for Stretch and delegated to.

    Subclasses `BasePolicy` rather than `InferencePolicy` because there is no
    obs -> input -> inference -> action template to fill in here: the wrapped
    policy owns all four steps, including the action buffering that makes a
    chunked model usable at 15Hz. Wrapping rather than subclassing `SynthVLAPolicy`
    keeps this file importable when MolmoBot is not installed, which is what
    lets `configs.py` register the eval config unconditionally.
    """

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._inner = self._build_inner_policy()

    # =========================================================================
    # Construction
    # =========================================================================

    def _build_inner_policy(self) -> BasePolicy:
        policy_config = self.config.policy_config
        if not policy_config.checkpoint_path:
            raise ValueError(
                "StretchMolmoBotPolicy needs a checkpoint. Pass --checkpoint, or set "
                "checkpoint_path on the policy config. Fine-tune one with\n"
                "  python -m examples.machine_learning.molmospaces.finetuning.finetune "
                "--rollouts <run> --trainer molmobot"
            )

        module = self._import_molmobot()
        fields = dict(
            checkpoint_path=policy_config.checkpoint_path,
            camera_names=list(policy_config.camera_names),
            action_move_group_names=list(policy_config.action_move_group_names),
            action_spec=dict(policy_config.action_spec),
            action_type=policy_config.action_type,
            action_horizon=policy_config.action_horizon,
            execute_horizon=policy_config.execute_horizon,
            **policy_config.extra_policy_kwargs,
        )
        log.info(
            f"[molmobot] {policy_config.action_type} over "
            f"{policy_config.action_move_group_names} "
            f"({sum(policy_config.action_spec.values())} dims), cameras "
            f"{policy_config.camera_names}, checkpoint {policy_config.checkpoint_path}"
        )
        try:
            inner_config = module.SynthVLAPolicyConfig(**fields)
        except TypeError as error:
            raise TypeError(
                f"MolmoBot's SynthVLAPolicyConfig rejected these fields: {sorted(fields)}. "
                "Its constructor has probably moved since this adapter was written -- "
                f"the underlying error was: {error}. Fix the field names in "
                "policies/molmobot_policy.py rather than working around it here."
            ) from error
        return module.SynthVLAPolicy(inner_config, self.task)

    @staticmethod
    def _import_molmobot():
        """Import MolmoBot's MolmoSpaces evaluation module, or explain how to get it."""
        import importlib

        errors = {}
        for module_name in MOLMOBOT_MODULES:
            try:
                return importlib.import_module(module_name)
            except ImportError as error:
                errors[module_name] = str(error)

        raise ImportError(
            "MolmoBot is not importable, so --policy molmobot cannot run. Clone it and put "
            "it on PYTHONPATH:\n"
            "  git clone https://github.com/allenai/MolmoBot\n"
            "  export PYTHONPATH=$PYTHONPATH:/path/to/MolmoBot/MolmoBot\n"
            "Tried: " + "; ".join(f"{name} ({error})" for name, error in errors.items())
        )

    # =========================================================================
    # BasePolicy, delegated
    # =========================================================================

    def reset(self) -> None:
        # The wrapped policy is built against `self.task`, which
        # `BaseMujocoTask.register_policy()` sets after construction -- so on the
        # first episode it may have been built with None. Keep it in step.
        self._inner.task = self.task
        self._inner.reset()

    def get_action(self, observation) -> dict[str, Any]:
        return self._inner.get_action(observation)

    def get_action_chunk(self, observation) -> list[dict[str, Any]] | None:
        return self._inner.get_action_chunk(observation)

    def create_policy_sensors(self):
        return self._inner.create_policy_sensors()

    def get_info(self) -> dict:
        info = dict(self._inner.get_info())
        policy_config = self.config.policy_config
        info.update(
            {
                "policy": "molmobot_native",
                "checkpoint_path": policy_config.checkpoint_path,
                "action_type": policy_config.action_type,
                "action_move_groups": list(policy_config.action_move_group_names),
                "action_dim": sum(policy_config.action_spec.values()),
                "camera_names": list(policy_config.camera_names),
                "remapped": False,
            }
        )
        return info
