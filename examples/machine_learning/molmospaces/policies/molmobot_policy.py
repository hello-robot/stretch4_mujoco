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
        --policy molmobot --checkpoint /path/to/step9500_bestfit --benchmark pick

This module is a thin adapter, not a reimplementation. MolmoBot's own
`SynthVLAPolicy` does the work: it buffers an `action_horizon`-step prediction,
executes `execute_horizon` of it before re-querying, reads cameras by name out of
the MolmoSpaces observation, and unpacks its output vector across the configured
move groups. All this does is construct it with Stretch's spec, fix up the
places where its defaults are shaped for a Franka, and delegate.

**MolmoBot is not a dependency of this repository**, so the import is lazy and
the field names below are taken from its published source rather than validated
against an installed copy. If its constructor has moved, the error names exactly
which fields were passed, which is the information needed to fix it.

Three of its defaults do not survive contact with Stretch, and each is set here
rather than left to the caller, because each fails quietly:

- **`clamp_gripper`** rounds the gripper command to 0 or 255 around a threshold
  of 128, which is a sensible normalisation of a Franka gripper's 0-255 command
  and is nonsense for Stretch's, measured in radians. Everything it produces
  lands on 0. Off, here.
- **`relative_max_joint_delta`** is a seven-element clamp applied to
  `action["arm"][:7]`, because a Franka arm is seven joints. Stretch's `arm` move
  group is one number -- the telescoping extension -- so the shipped default
  makes the *scaling* branch raise `could not broadcast input array from shape
  (7,) into shape (1,)`, and only once a predicted delta is large enough to need
  scaling, which is to say in the middle of a rollout rather than at startup.
  Resized to the arm's real width from `max_relative_arm_delta`.
- **the action frame.** See `_to_absolute_targets`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
    MolmoBotSetupError,
    ensure_importable,
    inference_requirements_message,
    missing_inference_requirements,
)
from examples.machine_learning.molmospaces.policies.molmobot_checkpoint import (
    TrainingArgs,
    read_training_args,
    read_training_state,
)
from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    HEAD_CAMERA_LEFT,
    HEAD_CAMERA_RIGHT,
    WRIST_CAMERA_LEFT,
)
from examples.machine_learning.molmospaces.stretch.robot_view import JointTargetClipper
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

`ensure_importable` puts the first spelling within reach without anyone exporting
anything; the other two remain for a copy installed some other way.
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

    The directory that holds `config.yaml`, which is the step directory
    (`.../step9500_bestfit`), *not* the `model_and_optim/` inside it: MolmoBot's
    loader reads the config from the path it is given and appends
    `model_and_optim` itself.
    """

    action_type: str = "joint_pos_rel"
    """
    `joint_pos_rel` or `joint_pos`, matching how the checkpoint was trained.

    MolmoBot's dataset prefers the relative key and its trainer defaults to it,
    so that is the default here too. Getting it wrong is not subtle for long but
    it is subtle at first: absolute targets applied as deltas make the arm creep
    away from wherever it started. Normally there is nothing to get wrong --
    `configure_from_checkpoint` reads the answer off the checkpoint.
    """

    action_type_explicit: bool = False
    """
    True when the caller named `action_type` themselves (`--molmobot-action-type`).

    An explicit choice outranks the checkpoint's own record of how it was
    trained -- that is what makes the flag useful for investigating a checkpoint
    whose `config.yaml` is wrong -- but the disagreement is logged rather than
    passed over.
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

    Only a fallback in practice: a checkpoint records the cameras it was trained
    on and `configure_from_checkpoint` uses those, because serving a VLA a
    different set -- or the same set in a different order -- is not an error
    anywhere in the stack, just a policy acting on a scene it cannot see.
    """

    configure_from_checkpoint: bool = True
    """
    Take `camera_names`, `action_type` and `states_mode` from the checkpoint.

    `train_molmobot.py` records its whole launch line under `runtime_data.args`
    in the `config.yaml` it saves, so the settings an evaluation has to match are
    already on disk -- see `policies/molmobot_checkpoint.py`. Set False to run a
    checkpoint strictly as this config describes it, which is worth having for
    the case where the recorded arguments are the thing under suspicion.
    """

    max_relative_arm_delta: float = 0.2
    """
    Largest arm delta, in metres, MolmoBot's own scaling will let through in one step.

    Only reaches `action["arm"]`, because that is all MolmoBot's clamp touches;
    every other group is bounded by `JointTargetClipper` against the model's real
    limits. 0.2 is MolmoBot's shipped per-joint value, kept so this is a guard
    against a garbage prediction rather than a speed limit -- Stretch's arm
    covers its whole 0.5m of travel in far more than one 66ms step.
    """

    action_move_group_names: list[str] = list(STRETCH_ACTION_SPEC)
    action_spec: dict[str, int] = dict(STRETCH_ACTION_SPEC)

    extra_policy_kwargs: dict = {}
    """
    Anything else to pass to `SynthVLAPolicyConfig`, and the last word on all of it.

    An escape hatch, because MolmoBot's config carries fields this integration
    has no opinion about (`cameras_to_warp`, `use_point_prompts`,
    `point_prompt_camera`, ...) and its defaults for them are the right ones
    until someone needs otherwise. Point prompts in particular already default
    to `head_camera`, which is what Stretch's head camera is called. Applied
    last, so it can also override the fields this module does have an opinion
    about.
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
        self._clipper: JointTargetClipper | None = None
        self._previous_qpos: dict[str, np.ndarray] | None = None
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
        training = read_training_args(policy_config.checkpoint_path)
        self._check_action_width(training)
        camera_names = self._resolve_camera_names(training)
        action_type = self._resolve_action_type(training)

        fields = dict(
            checkpoint_path=policy_config.checkpoint_path,
            camera_names=camera_names,
            action_move_group_names=list(policy_config.action_move_group_names),
            action_spec=dict(policy_config.action_spec),
            action_type=action_type,
            action_horizon=policy_config.action_horizon,
            execute_horizon=policy_config.execute_horizon,
            # Both of these are Franka-shaped defaults that fail quietly on
            # Stretch; see this module's docstring.
            clamp_gripper=False,
            relative_max_joint_delta=[policy_config.max_relative_arm_delta]
            * policy_config.action_spec.get("arm", 1),
            **policy_config.extra_policy_kwargs,
        )
        if policy_config.configure_from_checkpoint and training.states_mode:
            # Not passed unconditionally: MolmoBot's default overrides whatever
            # the checkpoint's own model config says, so the *only* safe value
            # to send is the one the checkpoint was built with.
            fields.setdefault("states_mode", training.states_mode)

        log.info(
            f"[molmobot] {action_type} over {policy_config.action_move_group_names} "
            f"({sum(policy_config.action_spec.values())} dims), cameras {camera_names}, "
            f"checkpoint {policy_config.checkpoint_path}"
        )
        # What the run had got to when these weights were written -- the
        # difference between "this policy is bad" and "this policy is early",
        # which is otherwise invisible from an evaluation's side.
        state = read_training_state(policy_config.checkpoint_path)
        if state:
            log.info(f"[molmobot] checkpoint saved at {state.summary()}")
        try:
            inner_config = module.SynthVLAPolicyConfig(**fields)
        except TypeError as error:
            raise TypeError(
                f"MolmoBot's SynthVLAPolicyConfig rejected these fields: {sorted(fields)}. "
                "Its constructor has probably moved since this adapter was written -- "
                f"the underlying error was: {error}. Fix the field names in "
                "policies/molmobot_policy.py rather than working around it here."
            ) from error

        # Kept on the policy rather than read back off the inner config, because
        # `_to_absolute_targets` runs on every step and this is the value that
        # decides what those numbers mean.
        self._action_type = action_type

        # `SynthVLAPolicy` takes the *experiment* config, not the policy config:
        # it reads `config.policy_config.*` throughout, and its base class reads
        # `force_enable_depth` off the same path. So it gets this experiment with
        # its policy config swapped for MolmoBot's -- a shallow copy, so the
        # robot, cameras and task settings are the very same objects this
        # evaluation is running with.
        inner_exp_config = self.config.model_copy(update={"policy_config": inner_config})
        try:
            return module.SynthVLAPolicy(inner_exp_config, self.task)
        except ModuleNotFoundError as error:
            # The model is built and the checkpoint loaded inside this
            # constructor, which is where MolmoBot's own runtime dependencies
            # are first imported -- and this runs in a rollout worker, so a bare
            # "No module named 'cached_path'" arrives with no hint of whose
            # dependency that is.
            raise ModuleNotFoundError(
                inference_requirements_message(missing_inference_requirements() or [error.name])
            ) from error

    def _resolve_camera_names(self, training: TrainingArgs) -> list[str]:
        """The cameras to serve: the checkpoint's, if it recorded any.

        A camera set that disagrees with training is the failure this whole
        detour exists to prevent, because it does not surface as an error --
        the model simply attends to the wrong pictures. So the checkpoint wins,
        loudly.
        """
        policy_config = self.config.policy_config
        configured = list(policy_config.camera_names)
        if not policy_config.configure_from_checkpoint or not training.camera_names:
            return configured

        if training.camera_names != configured:
            log.warning(
                f"[molmobot] serving the cameras this checkpoint was trained on, "
                f"{training.camera_names}, not the configured {configured} "
                f"(from {training.source}). Set configure_from_checkpoint=False to "
                "override."
            )
        return list(training.camera_names)

    def _resolve_action_type(self, training: TrainingArgs) -> str:
        """`joint_pos_rel` or `joint_pos`, preferring an explicit choice, then the checkpoint."""
        policy_config = self.config.policy_config
        configured = policy_config.action_type
        if not policy_config.configure_from_checkpoint or not training.action_type:
            return configured
        if training.action_type == configured:
            return configured

        if policy_config.action_type_explicit:
            log.warning(
                f"[molmobot] running as {configured} though {training.source} records this "
                f"checkpoint as trained with --action_type {training.action_type}."
            )
            return configured
        log.info(
            f"[molmobot] this checkpoint was trained with --action_type "
            f"{training.action_type}; using that rather than the configured {configured}."
        )
        return training.action_type

    def _check_action_width(self, training: TrainingArgs) -> None:
        """Refuse a checkpoint whose action vector is a different width than Stretch's.

        Unlike the camera set, this one is cheap to detect and catastrophic to
        miss: the output vector is unpacked across move groups in order, so a
        width mismatch does not misread one number, it shifts every joint after
        the first disagreement. The released `allenai/MolmoBot-DROID` is exactly
        this case -- seven Franka arm joints plus a gripper against Stretch's ten.
        """
        expected = sum(self.config.policy_config.action_spec.values())
        if training.action_dim is None or training.action_dim == expected:
            return
        raise ValueError(
            f"This checkpoint emits {training.action_dim} numbers per step "
            f"({training.source} says action_dim={training.action_dim}), but Stretch's move "
            f"groups need {expected}: {self.config.policy_config.action_spec}. Unpacking it "
            "anyway would misassign every joint after the first group that disagrees. "
            "Fine-tune on Stretch data -- see finetuning/README.md -- or, if the action "
            "spec is genuinely different, set action_spec and action_move_group_names on "
            "the policy config to match the checkpoint."
        )

    @staticmethod
    def _import_molmobot():
        """Import MolmoBot's MolmoSpaces evaluation module, or explain how to get it."""
        import importlib

        # The checkout is not installed, so put it on the path first. Done here
        # rather than only in `run_benchmarks.py` because this also runs inside
        # rollout workers, which may be fresh interpreters, and because a policy
        # constructed from a test or a notebook deserves to work too.
        setup_error = None
        try:
            ensure_importable()
        except MolmoBotSetupError as error:
            setup_error = error

        errors = {}
        for module_name in MOLMOBOT_MODULES:
            try:
                return importlib.import_module(module_name)
            except ImportError as error:
                errors[module_name] = str(error)

        raise ImportError(
            "MolmoBot is not importable, so --policy molmobot cannot run. "
            + (
                f"{setup_error}\n"
                if setup_error is not None
                else "It is on the path but its evaluation module would not import.\n"
            )
            + "Tried: "
            + "; ".join(f"{name} ({error})" for name, error in errors.items())
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
        # The model is recompiled per episode, so the limits are re-read with it.
        self._clipper = None
        # And the frame the first delta of the next episode is measured from is
        # that episode's first observation, not the last one of this episode.
        self._previous_qpos = None

    def get_action(self, observation) -> dict[str, Any]:
        observation = self._prepare_observation(observation)
        qpos = self._qpos()
        # The delta is measured from the *previous* step's state -- see
        # `_to_absolute_targets`. On the first step of an episode there is no
        # previous one, which is the step whose recorded action is zeros anyway.
        reference = self._previous_qpos if self._previous_qpos is not None else qpos
        action = self._to_absolute_targets(self._inner.get_action(observation), reference)
        self._previous_qpos = qpos
        return self._clip_to_limits(action)

    def get_action_chunk(self, observation) -> list[dict[str, Any]] | None:
        chunk = self._inner.get_action_chunk(self._prepare_observation(observation))
        if chunk is None:
            return None

        # Only the first two entries of a chunk have a measured state to
        # difference against: this step's action is relative to the previous
        # observation and the next one is relative to this one. Past that the
        # states are in the future, so each converted target stands in for the
        # measurement that has not happened -- which assumes the robot arrives
        # where it was sent, the same assumption the chunk makes by not looking.
        qpos = self._qpos()
        previous = self._previous_qpos if self._previous_qpos is not None else qpos
        self._previous_qpos = qpos

        actions: list[dict[str, Any]] = []
        for index, action in enumerate(chunk):
            if index == 0:
                reference = previous
            elif index == 1:
                reference = qpos
            else:
                reference = {**qpos, **{k: v for k, v in actions[-2].items() if k in qpos}}
            absolute = self._clip_to_limits(self._to_absolute_targets(action, reference))
            actions.append(absolute)
        return actions

    # =========================================================================
    # Observations, deltas, limits
    # =========================================================================

    def _prepare_observation(self, observation):
        """Make one MolmoSpaces observation into the shape MolmoBot expects.

        Two adjustments, both version skew rather than disagreement: MolmoBot
        evaluates against the MolmoSpaces commit its `pyproject.toml` pins, and
        the installed one hands out a different observation.

        **`robot_state`** is where it reads proprioception.

        `SynthVLAPolicy._populate_action_buffer` builds its state vector from
        `obs["robot_state"]["qpos"][<move group>]`, which is a sensor in the
        MolmoSpaces commit MolmoBot pins (`pyproject.toml`'s `eval` extra names
        one) and is in no sensor suite the installed MolmoSpaces assembles for
        Stretch -- `RobotStateSensor` exists there but nothing registers it, and
        the one place that would is commented out. The result is a `KeyError:
        'robot_state'` on the first inference, several minutes into a benchmark.

        Supplying it here rather than registering the sensor is deliberate.
        `RobotStateSensor` pads every group to a Franka's widths (`arm` to 7,
        `base` to 3) and JSON-encodes the result for HDF5, which is a shape
        Stretch's ten numbers would have to be unpacked back out of; the move
        groups are already right here, in the same `get_qpos_dict()` the training
        data was recorded from. The observation is copied rather than mutated
        because it is the pipeline's, and it keeps it.

        **The camera frames arrive flipped**, as a view with a negative stride --
        MuJoCo renders bottom-up and MolmoSpaces turns the image the right way up
        without copying it. Torch refuses such an array outright ("tensors with
        negative strides are not currently supported"), from inside the image
        preprocessor, on the first inference. `np.ascontiguousarray` is the copy
        the error message asks for, and it is made here, on the images this
        policy is actually going to look at, rather than for every camera the
        episode renders.
        """
        # The pipeline hands out a list, one observation per batched environment,
        # and MolmoBot reads the first of them; a bare dict arrives from
        # `get_action` callers that are not the rollout loop. Both are handled
        # rather than assumed, since the difference is invisible until an episode
        # is already running.
        if isinstance(observation, (list, tuple)):
            return type(observation)(self._prepare_observation(entry) for entry in observation)
        if not isinstance(observation, dict):
            return observation

        prepared = dict(observation)
        if "robot_state" not in prepared:
            prepared["robot_state"] = {"qpos": self._qpos()}
        for camera in self._inner.camera_names:
            frame = prepared.get(camera)
            if isinstance(frame, np.ndarray) and any(stride < 0 for stride in frame.strides):
                prepared[camera] = np.ascontiguousarray(frame)
        return prepared

    def _to_absolute_targets(self, action: dict[str, Any], reference: dict[str, Any]) -> dict:
        """Turn a `joint_pos_rel` action into the absolute targets Stretch is commanded with.

        MolmoBot's relative actions and Stretch's controllers do not meet on
        their own. `LastCommandedRelativeJointPosSensor` -- which is what wrote
        the `actions/joint_pos_rel` the checkpoint was trained on -- computes

            delta[t] = commanded joint position[t] - measured qpos[t - 1]

        keeping the qpos it saw on its *previous* call and differencing against
        that, so the inverse is `previous qpos + delta`. The off-by-one is not a
        detail: over a real pick episode, differencing against this step's qpos
        instead misses the recorded command by up to 0.23 rad of lift and 1.17
        rad of wrist, while the previous step's reproduces it exactly. (Checked
        against `actions/joint_pos`, which the same sensor suite records
        alongside: `qpos[t-1] + rel[t] == joint_pos[t]` to the last decimal on
        every move group.)

        Every Stretch move group is commanded absolutely
        (`Stretch4RobotConfig.command_mode`), so doing the addition here is both
        the faithful inversion and the one that leaves the gripper alone: a
        relative finger command cannot hold a grasp, because the fingers stop on
        the object and the target stops with them, while `qpos + delta`
        reproduces the squeeze the expert commanded.

        Left alone: `done`, and any other key that is not a move group or whose
        width does not match the state's -- a width disagreement is an
        action-spec mismatch, which `_check_action_width` reports properly rather
        than half-applying here.
        """
        if self._action_type != "joint_pos_rel":
            return action

        absolute = {}
        for group, value in action.items():
            current = reference.get(group)
            delta = np.asarray(value, dtype=float).reshape(-1)
            if current is None or delta.shape != np.asarray(current, dtype=float).shape:
                absolute[group] = value
                continue
            absolute[group] = np.asarray(current, dtype=float) + delta
        return absolute

    def _qpos(self) -> dict[str, np.ndarray]:
        """This step's measured joint positions, per move group."""
        return {
            group: np.asarray(value, dtype=float).reshape(-1)
            for group, value in self.task.env.current_robot.robot_view.get_qpos_dict().items()
        }

    def _clip_to_limits(self, action: dict[str, Any]) -> dict[str, Any]:
        """Bring one of the VLA's actions inside what the model says is possible.

        The checkpoint is trained, not constrained: nothing in it knows where
        Stretch's joints stop, and a target past a limit is a standing position
        error the actuator holds against a joint that cannot follow.
        `JointTargetClipper` reads the bounds off the compiled MJCF, and leaves
        alone the keys that are not move groups -- `done`, and whatever else the
        wrapped policy reports alongside its targets.
        """
        if self._clipper is None:
            self._clipper = JointTargetClipper(self.task.env.current_robot.robot_view)
        return self._clipper.clip_action(action)

    def create_policy_sensors(self):
        return self._inner.create_policy_sensors()

    def get_info(self) -> dict:
        info = dict(self._inner.get_info())
        policy_config = self.config.policy_config
        info.update(
            {
                "policy": "molmobot_native",
                "checkpoint_path": policy_config.checkpoint_path,
                "action_type": self._action_type,
                "action_move_groups": list(policy_config.action_move_group_names),
                "action_dim": sum(policy_config.action_spec.values()),
                "camera_names": list(self._inner.camera_names),
                "remapped": False,
            }
        )
        return info
