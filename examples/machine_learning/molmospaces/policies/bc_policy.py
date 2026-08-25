"""
The behaviour-cloned Stretch policy, as a MolmoSpaces `InferencePolicy`.

This is the "learned policy" half of the benchmark story: `training/collect.py`
records the scripted expert, `training/train_bc.py` fits `StretchBCNet` to it,
and this class runs the resulting checkpoint inside the same evaluation harness
the expert was scored in.

Everything about loading and running the checkpoint lives in
`policies/checkpoint.py`, which knows nothing about MolmoSpaces. This class is
only the adapter: it pulls images and proprioception out of a MolmoSpaces
observation and robot view, and hands back per-move-group targets. `live_policy.py`
is the same adapter written against `Stretch4MujocoSimulator` instead, and the
two share `TrainedPolicy` precisely so a checkpoint cannot behave differently in
the benchmark than it does in the live sim.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from examples.machine_learning.molmospaces.policies.checkpoint import TrainedPolicy
from examples.machine_learning.molmospaces.policies.networks import decode_action, encode_state
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import InferencePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)


class StretchBCPolicyConfig(BasePolicyConfig):
    """Configuration for `StretchBCPolicy`."""

    policy_type: str = "learned"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    checkpoint_path: str | None = None
    """
    Path to a checkpoint written by `training/train_bc.py`. Usually supplied on
    the command line instead; `run_evaluation(checkpoint_path=...)` overwrites
    whatever is set here.
    """

    camera_names: list[str] = []
    """
    Cameras to feed the network, in order. Left empty by default so the
    checkpoint's own list wins; set it only to deliberately override.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    execute_chunk_steps: int | None = None
    """
    How many steps of each predicted chunk to execute before re-querying the
    network. Defaults to the whole chunk. Executing fewer trades inference cost
    for responsiveness.
    """

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = StretchBCPolicy
            self.policy_factory = make_lenient(StretchBCPolicy)


class StretchBCPolicy(InferencePolicy):
    """Runs a trained `StretchBCNet`, one action chunk at a time."""

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._policy: TrainedPolicy | None = None
        self.prepare_model(config.policy_config.checkpoint_path)

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def prepare_model(self, model_name: str | None = None) -> None:
        policy_config = self.config.policy_config
        checkpoint_path = model_name or policy_config.checkpoint_path
        if not checkpoint_path:
            raise ValueError(
                "StretchBCPolicy needs a checkpoint. Pass --checkpoint_path, or set "
                "checkpoint_path on the policy config."
            )
        self._policy = TrainedPolicy.load(
            checkpoint_path,
            device=policy_config.device,
            camera_names=list(policy_config.camera_names) or None,
            execute_chunk_steps=policy_config.execute_chunk_steps,
        )

    def reset(self) -> None:
        self._policy.reset()

    # =========================================================================
    # InferencePolicy
    # =========================================================================

    def obs_to_model_input(self, obs) -> tuple[dict[str, np.ndarray], np.ndarray]:
        observation = obs[0] if isinstance(obs, list) else obs
        robot_view = self.task.env.current_robot.robot_view
        images = {name: observation[name] for name in self._policy.camera_names}
        return images, encode_state(robot_view.get_qpos_dict())

    def inference_model(self, model_input) -> np.ndarray:
        images, state = model_input
        return self._policy.predict_chunk(images, state)

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, Any]:
        return decode_action(model_output[0], self._base_xytheta())

    def get_action(self, observation) -> dict[str, Any]:
        """One step, served from the current action chunk.

        This overrides `InferencePolicy.get_action`'s
        obs -> input -> inference -> action template rather than filling it in,
        because the template runs the network every step and the whole point of
        predicting a chunk is not to. `TrainedPolicy.act` owns that bookkeeping
        (and does it identically for the live sim); building the observation is
        cheap enough to do on every step regardless of whether it gets used.
        """
        images, state = self.obs_to_model_input(observation)
        return self._policy.act(images, state, self._base_xytheta())

    def _base_xytheta(self) -> np.ndarray:
        base_pose = self.task.env.current_robot.robot_view.base.pose
        return np.array(
            [base_pose[0, 3], base_pose[1, 3], np.arctan2(base_pose[1, 0], base_pose[0, 0])]
        )

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy"] = "stretch_bc"
        info["checkpoint_path"] = str(self._policy.checkpoint_path)
        info["chunk_size"] = self._policy.chunk_size
        return info
