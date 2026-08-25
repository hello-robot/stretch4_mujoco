"""
Run a Franka-space VLA on Stretch, through the remapping.

MolmoSpaces' released manipulation baselines are not checkpoints you import;
they are inference *servers* you connect to. `pi_policy.py`, `dreamzero_policy.py`
and `cap_policy.py` are all clients, and the first two speak the same protocol:
a websocket carrying msgpack-numpy dicts, one `infer` call returning a chunk of
actions. That protocol is what this module speaks, so any of them -- or any
openpi-style server, or the one `finetuning/` produces -- can drive Stretch
without changes on the model side.

What the model sees and says is fixed by its training, and is Franka-shaped:

    observation/exterior_image_0_left    an exocentric view          (180, 320)
    observation/exterior_image_1_left    a second exocentric view    (180, 320)
    observation/wrist_image_left         the wrist view              (180, 320)
    observation/joint_position           7 Franka arm joints
    observation/gripper_position         1, 0 open .. 1 closed
    observation/cartesian_position       6, zeros -- unused by the reference clients
    prompt                               the task's language instruction

    action[:7]                           7 Franka arm joints
    action[7]                            gripper, 0 open .. 1 closed

`FrankaActionRemapper` supplies the two proprioception fields from Stretch's
real state and consumes the action; this class is the adapter around it, plus
the two things that are specific to running a *Franka-trained* model on Stretch:

- **Cameras.** Stretch has a head camera and a wrist camera, not two shoulder
  cameras and a wrist camera. The head camera stands in for both exocentric
  views, which is a real substitution and not a neutral one -- see
  `CAMERA_SUBSTITUTION_NOTE`.
- **Handover.** Stretch spawns stowed, a configuration no Franka pose maps to.
  The first steps of an episode drive to the authoring arm's start pose without
  consulting the model at all, so the model's first observation is one it could
  plausibly have been trained on.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from examples.machine_learning.molmospaces.franka_remapping.action_remap import (
    FrankaActionRemapper,
)
from examples.machine_learning.molmospaces.franka_remapping.pose_solver import (
    EXACT_ORIENTATION_DOFS,
    FREE_AZIMUTH_DOFS,
    TRANSLATING_DOFS,
)
from examples.machine_learning.molmospaces.stretch.config import HEAD_CAMERA, WRIST_CAMERA
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import InferencePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

CAMERA_SUBSTITUTION_NOTE = """
Stretch's head camera is fed to both `exterior_image_0_left` and
`exterior_image_1_left`. A DROID-trained model expects two *fixed, off-robot*
shoulder views and gets one egocentric view that moves with the robot, twice.
This is the largest single mismatch in the whole retarget, and it is the one
that cannot be fixed kinematically: the arm can be remapped exactly, the
viewpoint cannot. Fine-tuning on Stretch's own cameras is the fix; see
`finetuning/`.
""".strip()

SOLVER_MODES = {
    "exact": EXACT_ORIENTATION_DOFS,
    "free_azimuth": FREE_AZIMUTH_DOFS,
    "translating": TRANSLATING_DOFS,
}
"""
Which degrees of freedom the pose solve may move, by name.

Measured on the pick benchmark's grasp trajectories (see `pose_solver.py`),
`exact` puts the tool within 5mm of the requested grasp pose at the median and
26mm at the 90th percentile with the orientation matched to 0.01 rad, and it is
the default. `free_azimuth` avoids turning the base but mis-orients the gripper
by more than a radian through the approach. `translating` tracks the *approach*
better but ends the grasp worse, because the base wanders off the standoff the
episode chose.
"""


class FrankaVLAPolicyConfig(BasePolicyConfig):
    """Configuration for `FrankaVLAPolicy`."""

    policy_type: str = "learned"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    remote_config: dict = dict(host="localhost", port=8000, include_endpoint=False)
    """
    Where the inference server is, and how to address it.

    Same shape as `DreamZeroPolicyConfig.remote_config`, plus `include_endpoint`.
    `run_benchmarks.py --policy vla` fills the host and port from
    `--vla-host`/`--vla-port` and the protocol from `--vla-protocol`.

    `include_endpoint` defaults to False, the openpi-style protocol: MolmoBot's
    `olmo/eval/websocket_server.py` and openpi's own server both pass the whole
    unpacked request into the policy, where an unexpected `endpoint` key can be a
    hard error. DreamZero's server routes on that key instead -- see
    `vla_client.VLAWebsocketClient`, which also has to reset differently for the
    two.
    """

    checkpoint_path: str | None = None
    """Reported in `get_info()` only. The server owns the weights."""

    chunk_size: int = 8
    """
    How many actions of each returned chunk to execute before re-querying.

    Not necessarily the chunk the server produces -- the reference clients slice
    whatever they are given to this length -- so it is a client-side
    responsiveness setting.
    """

    action_space: str = "joint_position"
    """`joint_position` or `joint_velocity`; see `action_remap.ACTION_SPACES`."""

    gripper_mapping: str = "normalized"
    """`normalized` or `aperture`; see `action_remap.GRIPPER_MAPPINGS`."""

    frame_source: str = "episode"
    """
    `episode` or `mast`; see `action_remap.FRAME_SOURCES`.

    Set this to `mast` when running a checkpoint fine-tuned on a dataset from
    `finetuning/lerobot_export.py --action-space franka`, which encodes in the
    mast frame. Leave it at `episode` for a pretrained Franka checkpoint.
    """

    grasping_type: str = "binary"
    """
    `binary`, `semi_binary` or `continuous`, matching the reference clients.

    A diffusion policy's gripper channel rarely saturates, and a gripper that
    closes to 0.7 holds nothing -- so the baselines threshold it. Kept here so a
    checkpoint tuned against one of those conventions behaves the same.
    """

    grasping_threshold: float = 0.5

    solver_mode: str = "exact"
    """Which entry of `SOLVER_MODES` the pose solve uses."""

    max_base_rotation: float = float(np.pi)
    """Radians the retarget may turn the base over a whole episode."""

    max_base_translation: float = 0.0
    """Metres the retarget may drive the base over a whole episode."""

    handover_steps: int = 30
    """
    Steps allowed to drive Stretch from stowed to the authoring arm's start pose
    before the model is first queried. At 15Hz that is two seconds.
    """

    handover_tolerance_m: float = 0.05
    """Tool position error that counts as "arrived" and ends the handover early."""

    camera_names: list[str] = [HEAD_CAMERA, WRIST_CAMERA]
    """Stretch cameras to read, as `[exocentric, wrist]`. See `CAMERA_SUBSTITUTION_NOTE`."""

    image_size: tuple[int, int] = (180, 320)
    """`(height, width)` the images are letterboxed to, as the DROID clients do."""

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.solver_mode not in SOLVER_MODES:
            raise ValueError(
                f"solver_mode must be one of {tuple(SOLVER_MODES)}, got {self.solver_mode!r}"
            )
        if self.policy_cls is None:
            self.policy_cls = FrankaVLAPolicy
            self.policy_factory = make_lenient(FrankaVLAPolicy)


class FrankaVLAPolicy(InferencePolicy):
    """A Franka-space VLA server, driving Stretch through `FrankaActionRemapper`."""

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        policy_config = config.policy_config
        self._remapper = FrankaActionRemapper(
            robot_config=config.robot_config,
            action_space=policy_config.action_space,
            gripper_mapping=policy_config.gripper_mapping,
            frame_source=policy_config.frame_source,
            dofs=SOLVER_MODES[policy_config.solver_mode],
            max_base_rotation=policy_config.max_base_rotation,
            max_base_translation=policy_config.max_base_translation,
            velocity_dt=config.policy_dt_ms / 1000.0,
        )
        self._client = None
        self._chunk: np.ndarray | None = None
        self._chunk_index = 0
        self._handover_step = 0
        self._session_id: str | None = None
        self._started_at: float | None = None
        self._queries = 0
        log.info(f"[franka-vla] {CAMERA_SUBSTITUTION_NOTE}")

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def prepare_model(self, model_name: str | None = None) -> None:
        """Connect to the inference server.

        Deferred to the first `get_action` rather than done in `__init__`: the
        rollout runner builds the policy once per worker and the handover phase
        does not need the model, so a server that is still starting up has the
        first two seconds of every episode to come up.
        """
        from examples.machine_learning.molmospaces.franka_remapping.vla_client import (
            VLAWebsocketClient,
        )

        remote = self.config.policy_config.remote_config or {}
        self._client = VLAWebsocketClient(
            host=remote.get("host", "localhost"),
            port=remote.get("port", 8000),
            include_endpoint=remote.get("include_endpoint", True),
        )

    def reset(self) -> None:
        self._remapper.reset(self._base_pose())
        self._chunk = None
        self._chunk_index = 0
        self._handover_step = 0
        self._session_id = str(uuid.uuid4())
        self._started_at = None
        self._queries = 0
        if self._client is not None:
            self._client.reset({"session_id": self._session_id})

    # =========================================================================
    # InferencePolicy
    # =========================================================================

    def get_action(self, observation) -> dict[str, Any]:
        """One step: the handover, or a remapped action from the model's chunk.

        Overrides the `obs -> input -> inference -> action` template rather than
        filling it in, because two of this policy's steps do not query the model
        at all -- the handover ones -- and the rest are served from a chunk. The
        template's pieces are still implemented, so the base class's contract
        holds for anything that calls them directly.
        """
        observation = observation[0] if isinstance(observation, list) else observation
        if self._started_at is None:
            self._started_at = time.time()

        handover = self._handover_action()
        if handover is not None:
            return handover

        model_input = self.obs_to_model_input(observation)
        model_output = self.inference_model(model_input)
        return self.model_output_to_action(model_output)

    def obs_to_model_input(self, obs) -> dict[str, Any]:
        """Stretch's cameras and state, in the field names the server expects."""
        from examples.machine_learning.molmospaces.franka_remapping.vla_client import letterbox

        obs = obs[0] if isinstance(obs, list) else obs
        policy_config = self.config.policy_config
        height, width = policy_config.image_size
        exocentric_name, wrist_name = policy_config.camera_names
        exocentric = letterbox(obs[exocentric_name], height, width)
        wrist = letterbox(obs[wrist_name], height, width)

        robot_view = self.task.env.current_robot.robot_view
        qpos = robot_view.get_qpos_dict()
        state = self._remapper.observation(
            tool_pose_world=robot_view.get_move_group("gripper").leaf_frame_to_world,
            gripper_closedness=FrankaActionRemapper.stretch_gripper_closedness(qpos["gripper"]),
        )

        return {
            "observation/exterior_image_0_left": exocentric,
            "observation/exterior_image_1_left": exocentric,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": state["joint_position"].astype(np.float64),
            "observation/cartesian_position": np.zeros(6, dtype=np.float64),
            "observation/gripper_position": np.array(
                state["gripper_position"], dtype=np.float64
            ).reshape(1),
            "prompt": self.task.get_task_description(),
            "session_id": self._session_id,
        }

    def inference_model(self, model_input) -> np.ndarray:
        """The next action from the current chunk, querying the server if it is spent."""
        if self._client is None:
            self.prepare_model()
        if self._chunk is None or self._chunk_index >= min(
            len(self._chunk), self.config.policy_config.chunk_size
        ):
            self._chunk = np.asarray(self._client.infer(model_input)["actions"])
            self._chunk_index = 0
            self._queries += 1
        action = self._chunk[self._chunk_index]
        self._chunk_index += 1
        return np.asarray(action, dtype=float).reshape(-1)

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, Any]:
        """A Franka-space action -> Stretch's move-group targets."""
        model_output = np.asarray(model_output, dtype=float).reshape(-1).copy()
        model_output[7] = self._grasping(model_output[7])
        robot_view = self.task.env.current_robot.robot_view
        return self._remapper.action(
            model_output, base_pose=robot_view.base.pose, current_qpos=robot_view.get_qpos_dict()
        )

    # =========================================================================
    # Handover
    # =========================================================================

    def _handover_action(self) -> dict[str, Any] | None:
        """Drive toward the authoring arm's start pose, or None once we are there.

        Ends on arrival rather than always burning the full budget, so an
        episode whose start pose Stretch can reach quickly does not spend two
        seconds of its horizon on it. A start pose Stretch *cannot* reach -- the
        Franka's home posture is often inside Stretch's minimum reach -- runs
        the budget out and hands over from the nearest pose, which is the right
        answer rather than a failure.
        """
        policy_config = self.config.policy_config
        if self._handover_step >= policy_config.handover_steps:
            return None

        robot_view = self.task.env.current_robot.robot_view
        achieved = robot_view.get_move_group("gripper").leaf_frame_to_world
        target = self._remapper.handover_tool_pose
        if (
            self._handover_step > 0
            and float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
            < policy_config.handover_tolerance_m
        ):
            log.info(f"[franka-vla] handover complete after {self._handover_step} steps")
            self._handover_step = policy_config.handover_steps
            return None

        self._handover_step += 1
        action = self._remapper.tool_pose_action(
            target, base_pose=robot_view.base.pose, current_qpos=robot_view.get_qpos_dict()
        )
        # Open, because every benchmark task starts by reaching for something.
        action["gripper"] = self._remapper.gripper_action(0.0)
        return action

    # =========================================================================
    # Reporting
    # =========================================================================

    def _grasping(self, channel: float) -> float:
        """Apply the configured gripper thresholding to the model's raw channel."""
        policy_config = self.config.policy_config
        channel = float(np.clip(channel, 0.0, 1.0))
        if policy_config.grasping_type == "continuous":
            return channel
        if policy_config.grasping_type == "binary":
            return 1.0 if channel >= policy_config.grasping_threshold else 0.0
        if policy_config.grasping_type == "semi_binary":
            return 1.0 if channel > policy_config.grasping_threshold else channel
        raise ValueError(f"Unknown grasping_type {policy_config.grasping_type!r}")

    def _base_pose(self) -> np.ndarray:
        return self.task.env.current_robot.robot_view.base.pose

    def get_info(self) -> dict:
        info = super().get_info()
        policy_config = self.config.policy_config
        info.update(
            {
                "policy": "franka_vla_remapped",
                "policy_checkpoint": policy_config.checkpoint_path,
                "remote": policy_config.remote_config,
                "action_space": policy_config.action_space,
                "gripper_mapping": policy_config.gripper_mapping,
                "frame_source": policy_config.frame_source,
                "solver_mode": policy_config.solver_mode,
                "chunk_size": policy_config.chunk_size,
                "server_queries": self._queries,
                "session_id": self._session_id,
                "prompt": self.task.get_task_description() if self.task else None,
                "time_spent": time.time() - self._started_at if self._started_at else None,
            }
        )
        info.update(self._remapper.telemetry.as_dict())
        return info
