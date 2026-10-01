"""
The released `allenai/MolmoBot-DROID` checkpoint driving the Franka it was trained on.

This is the control condition for the whole study. `policies/molmobot_droid_policy.py`
runs the same checkpoint on Stretch, and everything it has to do -- the virtual
Franka, the IK, the Robotiq-to-fingers mapping, the height offset -- is
retargeting, which can only lose. Without a number for the same checkpoint on
the same task with none of that in the way, a Stretch score is unreadable: a bad
rollout could be the retargeting, the camera, or the policy simply not being
able to do this task in this kitchen.

So there is deliberately almost nothing in this file. MolmoSpaces' Franka Droid
robot view already speaks DROID's interface:

    move group "arm"      seven fr3 joint angles, commanded as joint positions
    move group "gripper"  the Robotiq's two driver joints, commanded 0-255

which is exactly what `RealRobotVLAPolicy` reads and emits. The action dict goes
straight through, unit for unit, and the only translation left is which
observation key each camera arrives under.

`StretchMolmoBotDroidPolicy`'s construction helpers are reused rather than
copied -- the import search, the checkpoint resolution, the per-process model
cache -- so the two paths cannot drift into driving the same checkpoint
differently, which would make the comparison this file exists for meaningless.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from examples.machine_learning.molmospaces.policies.molmobot_droid_policy import (
    DROID_ACTION_SPEC,
    DROID_EXO_CAMERA_KEY,
    DROID_HF_REPO,
    DROID_WRIST_CAMERA_KEY,
    StretchMolmoBotDroidPolicy,
    _model_reusing_policy_cls,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

FRANKA_EXO_CAMERA = "exo_camera_1"
FRANKA_WRIST_CAMERA = "wrist_camera"
"""
The camera names `setups.py` registers on the Franka camera system.

They happen to be spelled the same as the observation keys the checkpoint was
trained with, which is why this file has no camera renaming in it. They are kept
as two separate constants anyway, because the setups that put a Stretch camera
on a Franka change what is behind the exo name without changing the name.
"""


class FrankaMolmoBotDroidPolicyConfig(BasePolicyConfig):
    """Configuration for `FrankaMolmoBotDroidPolicy`.

    A near-copy of `StretchMolmoBotDroidPolicyConfig` minus everything about
    retargeting, which is the point: the fields that remain are the ones that
    describe the checkpoint rather than the robot, and both configs have to set
    them the same way for the comparison to hold.
    """

    policy_type: str = "learned"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    checkpoint_path: str | None = None
    """A local checkpoint, or None to fetch the released one from the Hub."""

    hf_repo: str = DROID_HF_REPO

    action_type: str = "joint_pos"
    """Absolute joint targets, which is how MolmoBot's own DROID demo drives this checkpoint."""

    exo_camera: str = FRANKA_EXO_CAMERA
    wrist_camera: str = FRANKA_WRIST_CAMERA

    action_horizon: int = 16
    execute_horizon: int = 8
    max_relative_arm_delta: float = 0.2

    reuse_loaded_model: bool = True
    """Load the checkpoint once per process rather than once per episode."""

    extra_policy_kwargs: dict = {}

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = FrankaMolmoBotDroidPolicy
            self.policy_factory = make_lenient(FrankaMolmoBotDroidPolicy)


class FrankaMolmoBotDroidPolicy(BasePolicy):
    """MolmoBot's DROID policy, applied to a Franka Droid without translation."""

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._inner = self._build_inner_policy()

    def _build_inner_policy(self) -> Any:
        policy_config = self.config.policy_config
        module = StretchMolmoBotDroidPolicy._import_molmobot()
        checkpoint_path = StretchMolmoBotDroidPolicy._resolve_checkpoint(policy_config)

        fields = dict(
            checkpoint_path=checkpoint_path,
            # Required by this repository's `molmo_spaces` on every policy
            # config and not declared by MolmoBot's class; never used, because
            # this adapter builds the inner policy itself. Same story as in
            # `molmobot_droid_policy._build_inner_policy`.
            policy_cls=module.RealRobotVLAPolicy,
            policy_factory=make_lenient(module.RealRobotVLAPolicy),
            camera_names=[DROID_EXO_CAMERA_KEY, DROID_WRIST_CAMERA_KEY],
            action_move_group_names=list(DROID_ACTION_SPEC),
            action_spec=dict(DROID_ACTION_SPEC),
            action_type=policy_config.action_type,
            action_keys={"arm": policy_config.action_type, "gripper": "joint_pos"},
            action_horizon=policy_config.action_horizon,
            execute_horizon=policy_config.execute_horizon,
            relative_max_joint_delta=[policy_config.max_relative_arm_delta]
            * DROID_ACTION_SPEC["arm"],
            **policy_config.extra_policy_kwargs,
        )
        log.info(
            f"[droid/franka] {policy_config.action_type} over {list(DROID_ACTION_SPEC)}, "
            f"un-retargeted; cameras {policy_config.exo_camera} -> {DROID_EXO_CAMERA_KEY}, "
            f"{policy_config.wrist_camera} -> {DROID_WRIST_CAMERA_KEY}; "
            f"checkpoint {checkpoint_path}"
        )

        try:
            inner_config = module.RealRobotVLAPolicyConfig(**fields)
        except (TypeError, ValidationError) as error:
            raise TypeError(
                f"MolmoBot's RealRobotVLAPolicyConfig rejected these fields: {sorted(fields)}. "
                f"The underlying error was: {error}. Fix the field names here and in "
                "policies/molmobot_droid_policy.py together."
            ) from error

        inner_exp_config = self.config.model_copy(update={"policy_config": inner_config})
        policy_cls = module.RealRobotVLAPolicy
        if policy_config.reuse_loaded_model:
            policy_cls = _model_reusing_policy_cls(policy_cls)
        return policy_cls(inner_exp_config, self.task)

    @property
    def camera_names(self) -> list[str]:
        """The cameras this policy reads, for the MP4 recorder and the Rerun stream."""
        return [self.config.policy_config.exo_camera, self.config.policy_config.wrist_camera]

    def reset(self) -> None:
        """Drop the action chunk and observation history of the previous episode."""
        self._inner.reset()

    def get_action(self, observation) -> dict[str, Any]:
        obs = observation[0] if isinstance(observation, list) else observation
        robot_view = self.task.env.current_robot.robot_view
        policy_config = self.config.policy_config

        inner_obs = {
            "task": self.task.get_task_description(),
            "qpos": {
                "arm": robot_view.get_move_group("arm").joint_pos,
                "gripper": robot_view.get_move_group("gripper").joint_pos,
            },
            DROID_EXO_CAMERA_KEY: _camera(obs, policy_config.exo_camera),
            DROID_WRIST_CAMERA_KEY: _camera(obs, policy_config.wrist_camera),
        }
        franka_action = self._inner.get_action(inner_obs)

        # Every group, so that one the policy does not command holds still
        # rather than keeping a stale target. The Franka's base is mocap and its
        # noop is its current pose.
        action = robot_view.get_noop_ctrl_dict()
        action["arm"] = np.asarray(franka_action["arm"], dtype=float).reshape(-1)[
            : DROID_ACTION_SPEC["arm"]
        ]
        action["gripper"] = np.asarray(franka_action["gripper"], dtype=float).reshape(-1)[:1]
        return action


def _camera(obs: dict, name: str) -> np.ndarray:
    """One camera frame, as an array `torch.from_numpy` will accept.

    The copy is required, not tidiness: MolmoSpaces hands over rendered frames
    as flipped views with negative strides, which `torch.from_numpy` refuses on
    the first step of every episode.
    """
    if name not in obs:
        raise KeyError(
            f"Camera {name!r} is not in the observation. Available: "
            f"{sorted(k for k in obs if not k.startswith('_'))}."
        )
    return np.ascontiguousarray(obs[name])
