"""
Run the *released* `allenai/MolmoBot-DROID` checkpoint on Stretch, by retargeting.

This is the counterpart to `molmobot_policy.py`, and the difference between them
is the whole reason both exist:

* `molmobot_policy.py` runs a checkpoint **fine-tuned on Stretch's own move
  groups**. MolmoBot's action space is configured by move group, so such a
  checkpoint emits Stretch's own ten numbers and needs no translation at all.
* this module runs the **released DROID checkpoint**, which was trained on the
  `franka_joint` action spec: seven Franka arm joints plus a Robotiq 2F-85
  gripper command, with seven Franka joint angles as proprioception. Stretch has
  no seven-jointed arm, so every action and every observation is translated --
  see `policies/franka_retarget.py`, which does the kinematics.

So this is a zero-shot cross-embodiment baseline: what a Franka policy scores on
Stretch with only the action interface bridged. Read it as a floor, not as a
comparison with the fine-tuned numbers -- retargeting fixes the interface, and
leaves the visual domain gap exactly where it was. The policy is looking at a
Stretch arm through a Stretch camera, having been trained on neither.

    python -m examples.machine_learning.molmospaces.run_benchmarks \\
        --policy molmobot_droid --benchmark pick --episodes 20

`--checkpoint` is optional here, unlike for the other learned policies: with none
given the released checkpoint is fetched from the Hub (`allenai/MolmoBot-DROID`),
which is the only checkpoint this adapter is for.

**MolmoBot is not a dependency of this repository**, so the import is lazy and
the field names below are taken from its published source rather than validated
against an installed copy. The class this wraps is
`olmo.eval.configure_real_robot.RealRobotVLAPolicy` -- the same one MolmoBot's own
DROID demo notebook drives -- and it is wrapped rather than reimplemented: it
owns the action chunking, the observation history, the image preprocessing and
the relative-delta scaling, none of which change under retargeting.

Three of its defaults are worth knowing about, because they are Franka-shaped and
this module sets them deliberately:

- **`action_type`** defaults to `joint_pos_rel` on MolmoBot's config, but the
  released DROID checkpoint is driven as `joint_pos` (absolute targets) in
  MolmoBot's own demo, and that is the default here. Getting it wrong is not
  subtle for long but it is subtle at first: absolute targets applied as deltas
  make the arm creep away from wherever it started. Override with
  `--molmobot-action-type`.
- **`relative_max_joint_delta`** caps how far the seven Franka joints may move in
  one step. It is kept at MolmoBot's own 0.2 rad: a guard against a garbage
  prediction, applied in the Franka's joint space before anything reaches
  Stretch.
- **`clamp_gripper`** rounds the gripper command to 0 or 255 around a threshold
  of 128. That *is* the right normalisation here -- unlike in the fine-tuned
  case, this gripper command really is a Robotiq 0-255 one -- and
  `RealRobotVLAPolicy` hard-codes it on regardless of config, so it is simply
  noted rather than set.
"""

from __future__ import annotations

import gc
import importlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
    MolmoBotSetupError,
    ensure_importable,
    inference_requirements_message,
    missing_inference_requirements,
)
from examples.machine_learning.molmospaces.policies.franka_retarget import (
    FrankaOnStretchView,
    VirtualFranka,
    franka_mount_pose_from_base,
)
from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    HEAD_CAMERA_RIGHT,
    WRIST_CAMERA_RIGHT,
)
from examples.machine_learning.molmospaces.stretch.robot_view import JointTargetClipper
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask

log = logging.getLogger(__name__)

DROID_HF_REPO = "allenai/MolmoBot-DROID"
"""The released checkpoint. Fetched from the Hub when no `--checkpoint` is given."""

DROID_ACTION_SPEC: dict[str, int] = {"arm": 7, "gripper": 1}
"""
What a DROID checkpoint emits per step: seven Franka arm joints, one gripper
command. MolmoBot's `franka_joint` preset, and `RealRobotVLAPolicyConfig`'s own
default -- restated because it is what makes the retargeting necessary, and
because a checkpoint whose width disagrees is one this adapter cannot drive.
"""

DROID_EXO_CAMERA_KEY = "exo_camera_1"
DROID_WRIST_CAMERA_KEY = "wrist_camera"
"""
The observation keys `RealRobotVLAPolicy` reads images out of, in order.

These are the names the checkpoint was trained with, so they are kept exactly --
the images behind them are Stretch's, chosen by `exo_camera` / `wrist_camera` on
the config below. Feeding a VLA a different set, or the same set in a different
order, is not an error anywhere in the stack: the model simply attends to the
wrong pictures.
"""

MOLMOBOT_MODULES = (
    "olmo.eval.configure_real_robot",
    "MolmoBot.olmo.eval.configure_real_robot",
    "molmobot.olmo.eval.configure_real_robot",
)
"""
Where to look for `RealRobotVLAPolicy`, in order. See `molmobot_policy.MOLMOBOT_MODULES`
for why there are three spellings.
"""

_LOADED_MODELS: dict[str, Any] = {}
"""
The checkpoint this process has loaded, by path. See `molmobot_policy._LOADED_MODELS`:
MolmoSpaces builds a policy per *episode*, so without this every episode reads
the weights off disk again and, for the length of that read, holds two copies on
the GPU.
"""

_MODEL_REUSING_CLASSES: dict[type, type] = {}
"""`_model_reusing_policy_cls`'s subclass per base class, so it makes each once."""


def _cuda_allocated_gib() -> float | None:
    """How much VRAM this process has allocated, or None without a GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.memory_allocated() / 1024**3
    except Exception:  # noqa: BLE001 - a number for a log line is not worth an exception
        return None


def _model_reusing_policy_cls(base: type) -> type:
    """`base`, but loading its inference wrapper through `_LOADED_MODELS`.

    A subclass rather than a wrapper because the load happens *inside*
    `RealRobotVLAPolicy.__init__`, which calls its own `prepare_model()` -- there
    is no moment between construction and loading at which an already-loaded
    model could be handed over from outside.
    """
    reusing = _MODEL_REUSING_CLASSES.get(base)
    if reusing is not None:
        return reusing

    class ModelReusingRealRobotVLAPolicy(base):  # type: ignore[misc, valid-type]
        """MolmoBot's DROID policy, with the loaded checkpoint kept between episodes."""

        def prepare_model(self) -> None:
            key = str(self.config.policy_config.checkpoint_path)
            loaded = _LOADED_MODELS.get(key)
            if loaded is not None:
                self.agent = loaded
                self._prepared = True
                log.debug(f"[droid] reusing the checkpoint already loaded from {key}")
                return

            # Dropped before the load, not after: a key that misses is a
            # different checkpoint, so the one held here is dead weight, and
            # freeing it first is the difference between one model on the GPU
            # and two.
            _LOADED_MODELS.clear()
            gc.collect()
            super().prepare_model()
            _LOADED_MODELS[key] = self.agent
            allocated = _cuda_allocated_gib()
            log.info(
                f"[droid] loaded {key}"
                + (f" ({allocated:.1f} GiB of VRAM allocated)" if allocated is not None else "")
                + "; kept for the rest of this process"
            )

    _MODEL_REUSING_CLASSES[base] = ModelReusingRealRobotVLAPolicy
    return ModelReusingRealRobotVLAPolicy


class StretchMolmoBotDroidPolicyConfig(BasePolicyConfig):
    """Configuration for `StretchMolmoBotDroidPolicy`."""

    policy_type: str = "learned"
    policy_cls: type | None = None
    policy_factory: PolicyFactory | None = None

    checkpoint_path: str | None = None
    """
    The DROID checkpoint. Unlike the other learned policies here this may be left
    unset, in which case `hf_repo` is fetched from the Hub -- the released
    checkpoint is the only one this adapter is for, so requiring it to be named
    would be ceremony. `--checkpoint` still overrides it, for a local copy.
    """

    hf_repo: str = DROID_HF_REPO
    """Hub repository to fall back to when `checkpoint_path` is unset."""

    action_type: str = "joint_pos"
    """
    `joint_pos` or `joint_pos_rel`, matching how the checkpoint is driven.

    Absolute by default, which is how MolmoBot's own DROID demo drives the
    released checkpoint, and unlike the fine-tuned path there is no
    `config.yaml` recording a per-checkpoint answer to read instead. Override
    with `--molmobot-action-type`.
    """

    exo_camera: str = HEAD_CAMERA_RIGHT
    """
    Which Stretch camera stands in for DROID's third-person view.

    The head centre camera by default: forward-and-down, upright, and the closest
    thing Stretch has to the pinhole exo view the policy was trained on.
    `head_camera_right` is the other reasonable choice -- it is where the view
    physically sits on the hardware, and it is a 123-degree fisheye mounted
    sideways, so it is both more honest and further from the training
    distribution. Which one is "right" depends on whether the question is what
    the robot sees or what the policy can cope with.
    """

    wrist_camera: str = WRIST_CAMERA_RIGHT
    """Which Stretch camera stands in for DROID's wrist view."""

    action_horizon: int = 16
    """Steps the model predicts per query. `RealRobotVLAPolicyConfig`'s own default."""

    execute_horizon: int = 8
    """Steps executed before re-querying. Also MolmoBot's default."""

    max_relative_arm_delta: float = 0.2
    """
    Largest Franka joint delta, in radians, MolmoBot's scaling will let through in one step.

    Applied by `RealRobotVLAPolicy` in the *Franka's* joint space, before
    anything is retargeted, which is the only place it means anything: the same
    bound on Stretch's joints would be a speed limit on a different robot.
    MolmoBot's shipped per-joint value, kept so this is a guard against a garbage
    prediction rather than a throttle.
    """

    include_base: bool = True
    """
    Let the holonomic base join the IK.

    Stretch's lift, arm and wrist reach a corridor roughly 0.2m either side of
    the arm's line, which is not enough for benchmark tasks that put the target
    anywhere on a counter. The base joins on a leash and at a cost; see
    `franka_retarget.StretchArmIK`. Set False to score what the arm alone can do.
    """

    target_z_offset: float = 0.0
    """
    Metres to raise every retargeted target by. Zero, and deliberately so.

    Stretch's tool centre comes out lower than the Franka's wherever the lift
    runs out of travel, and this exists to buy that clearance back. It used to
    default to a *measured* correction -- `measure_tool_height_offset()` at
    episode start, scaled by a fraction -- and that is now removed, because the
    measurement is sound and the way it was used was not.

    `measure_tool_height_offset()` solves for the Franka's **home** pose, which
    is above Stretch's lift ceiling, and returns the residual there: about 103mm.
    That number describes one pose the robot cannot reach at all. Applying a
    fraction of it to *every* target raised grasps that needed no raising -- at a
    counter-height grasp the lift solves around 0.95m with 0.25m of headroom and
    a residual of zero -- while buying nothing at the pose it was measured at,
    where the lift is already at its stop and the offset only grows the miss
    (`FrankaOnStretchView._warn_if_lift_saturated` says so out loud now).

    So it is a hand-set number again, and 0 is the default. `setups.py` had
    already searched its way to `0.0` for the study's Stretch setups, which is
    the same conclusion from the other direction. Raise it if a specific task
    wants clearance, and see `FrankaOnStretchView` for what it does not fix.
    """

    jaw_mode: str = "auto"
    """
    Which way round Stretch holds its jaw: "auto", "flipped" or "upright".

    A parallel jaw grasps the same object the same way either way round, so this
    is free reach rather than a trade in grasp quality -- what it changes is which
    poses Stretch's wrist can hold. "auto" is the default and the only mode that
    gives up nothing, at two IK solves per step; "flipped" is one solve and better
    on position everywhere, at the price of 0.43 rad at large tool yaws. See
    `franka_retarget.JAW_MODES`, which records the measurements.
    """

    match_robotiq_aperture: bool = True
    """
    Open Stretch's jaw only as wide as the Robotiq's, rather than as wide as it goes.

    Stretch's hand opens to 188mm and the Robotiq 2F-85 to 87mm, so an un-narrowed
    "open" command spreads the fingers more than twice as far as any hand in the
    checkpoint's training data -- a domain gap on the channel a grasping policy
    reads most closely, and a visibly different gripper in the wrist camera.

    On by default because matching is the fairer comparison, but it is a real
    change in what Stretch can do: a jaw capped at 87mm cannot go around an object
    wider than that, which its own 188mm could. Turn it off to get the old
    behaviour back and to measure how much of a Stretch/Franka difference was the
    aperture. See `franka_retarget.ROBOTIQ_MAX_APERTURE_M`.
    """

    snap_to_franka_home: bool = True
    """
    Start each episode at the Franka's home tool pose.

    The two robots have unrelated home configurations, so without this the
    policy's first observation reads as an arm two-and-a-bit radians from where
    it expects to be -- a large apparent jump before it has acted at all, which
    its first chunk then spends correcting. This writes `qpos` on the first step
    of an episode, which is a visible teleport of the arm; see
    `FrankaOnStretchView.snap_to_franka_joint_pos`.
    """

    reuse_loaded_model: bool = True
    """Load the checkpoint once per process rather than once per episode. See `_LOADED_MODELS`."""

    extra_policy_kwargs: dict = {}
    """
    Anything else to pass to `RealRobotVLAPolicyConfig`, and the last word on all of it.

    An escape hatch, because MolmoBot's config carries fields this integration
    has no opinion about and its defaults for them are the right ones until
    someone needs otherwise. Applied last, so it can also override the fields
    this module does have an opinion about.
    """

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = StretchMolmoBotDroidPolicy
            self.policy_factory = make_lenient(StretchMolmoBotDroidPolicy)


class StretchMolmoBotDroidPolicy(BasePolicy):
    """MolmoBot's DROID policy, with its Franka interface retargeted onto Stretch.

    Subclasses `BasePolicy` rather than `InferencePolicy` because there is no
    obs -> input -> inference -> action template to fill in here: the wrapped
    policy owns all four steps, including the action buffering that makes a
    chunked model usable at 15Hz. Wrapping rather than subclassing
    `RealRobotVLAPolicy` keeps this file importable when MolmoBot is not
    installed, which is what lets `configs.py` register the eval config
    unconditionally.

    The per-step round trip is:

        Stretch's cameras and its gripper's actual pose
          -> `FrankaOnStretchView` (as seven Franka joint angles)
          -> `RealRobotVLAPolicy` (as seven Franka joint targets)
          -> `FrankaOnStretchView` (as base / lift / arm / wrist / gripper targets)
          -> the action dict MolmoSpaces applies

    so the policy reads where Stretch's gripper actually is rather than an echo
    of its own last command, and writes targets Stretch's own controllers take.
    """

    def __init__(self, config: "MlSpacesExpConfig", task: "BaseMujocoTask" = None) -> None:
        super().__init__(config, task)
        self._proxy: FrankaOnStretchView | None = None
        self._clipper: JointTargetClipper | None = None
        self._residuals: list[tuple[float, float]] = []
        self._inner = self._build_inner_policy()

    # =========================================================================
    # Construction
    # =========================================================================

    def _build_inner_policy(self) -> Any:
        policy_config = self.config.policy_config
        module = self._import_molmobot()
        checkpoint_path = self._resolve_checkpoint(policy_config)

        fields = dict(
            checkpoint_path=checkpoint_path,
            # This repository's `molmo_spaces` requires both of these on every
            # `BasePolicyConfig`. MolmoBot's class was written against an older
            # one: it sets `policy_cls` in `model_post_init`, which runs after
            # validation, and never declares `policy_factory` at all -- so
            # constructing it here fails validation unless they are passed.
            # Neither is ever used, because this adapter builds the policy
            # itself rather than letting MolmoSpaces build it from the config.
            # (Its `SynthVLAPolicyConfig` does declare them, which is why
            # `molmobot_policy.py` needs none of this.)
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
            f"[droid] {policy_config.action_type} over {list(DROID_ACTION_SPEC)} "
            f"({sum(DROID_ACTION_SPEC.values())} dims), retargeted onto Stretch; "
            f"cameras {policy_config.exo_camera} -> {DROID_EXO_CAMERA_KEY}, "
            f"{policy_config.wrist_camera} -> {DROID_WRIST_CAMERA_KEY}; "
            f"checkpoint {checkpoint_path}"
        )

        try:
            inner_config = module.RealRobotVLAPolicyConfig(**fields)
        except (TypeError, ValidationError) as error:
            raise TypeError(
                f"MolmoBot's RealRobotVLAPolicyConfig rejected these fields: {sorted(fields)}. "
                "Its constructor has probably moved since this adapter was written -- "
                f"the underlying error was: {error}. Fix the field names in "
                "policies/molmobot_droid_policy.py rather than working around it here."
            ) from error

        # `RealRobotVLAPolicy` takes the *experiment* config, not the policy
        # config: it reads `config.policy_config.*` throughout. So it gets this
        # experiment with its policy config swapped for MolmoBot's -- a shallow
        # copy, so the robot, cameras and task settings are the very same objects
        # this evaluation is running with.
        inner_exp_config = self.config.model_copy(update={"policy_config": inner_config})
        policy_cls = module.RealRobotVLAPolicy
        if policy_config.reuse_loaded_model:
            policy_cls = _model_reusing_policy_cls(policy_cls)
        try:
            return policy_cls(inner_exp_config, self.task)
        except ModuleNotFoundError as error:
            # The model is built and the checkpoint loaded inside this
            # constructor, which is where MolmoBot's own runtime dependencies are
            # first imported -- and this runs in a rollout worker, so a bare "No
            # module named 'cached_path'" arrives with no hint of whose
            # dependency that is.
            raise ModuleNotFoundError(
                inference_requirements_message(missing_inference_requirements() or [error.name])
            ) from error

    @staticmethod
    def _resolve_checkpoint(policy_config: Any) -> str:
        """The local checkpoint directory, fetching the released one if none was named."""
        if policy_config.checkpoint_path:
            return str(policy_config.checkpoint_path)

        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ModuleNotFoundError(
                "No --checkpoint was given, so the released checkpoint would be fetched "
                f"from {policy_config.hf_repo}, which needs `huggingface_hub`. Install it, "
                "or pass --checkpoint with a local copy."
            ) from error

        log.info(f"[droid] no checkpoint given; fetching {policy_config.hf_repo} from the Hub")
        return snapshot_download(policy_config.hf_repo)

    @staticmethod
    def _import_molmobot():
        """Import MolmoBot's real-robot evaluation module, or explain how to get it."""
        # The checkout is not installed, so put it on the path first. Done here
        # rather than only in `run_benchmarks.py` because this also runs inside
        # rollout workers, which may be fresh interpreters.
        try:
            ensure_importable()
        except MolmoBotSetupError as error:
            log.warning(f"[droid] {error}")

        errors = {}
        for name in MOLMOBOT_MODULES:
            try:
                return importlib.import_module(name)
            except ImportError as error:
                errors[name] = error
        raise MolmoBotSetupError(
            "Could not import MolmoBot's RealRobotVLAPolicy. Tried "
            + ", ".join(f"{name} ({errors[name]})" for name in MOLMOBOT_MODULES)
            + ". Clone it with `python -m examples.machine_learning.molmospaces.finetuning"
            ".molmobot_repo` or see finetuning/README.md."
        )

    # =========================================================================
    # BasePolicy
    # =========================================================================

    @property
    def camera_names(self) -> list[str]:
        """The Stretch cameras this policy reads, for `--visualize` and `--export-to-mp4`.

        Declared here because `visualize.policy_camera_names` would otherwise
        find it on the wrapped policy, where it is the *DROID* key list
        (`exo_camera_1`, `wrist_camera`) -- names no Stretch observation has, so
        every recorded frame would come back empty. This property is checked
        first, and reports the two Stretch cameras whose images are actually
        being fed through those keys.
        """
        return [self.config.policy_config.exo_camera, self.config.policy_config.wrist_camera]

    def reset(self) -> None:
        """Drop everything that describes the previous episode's robot.

        The proxy view and the limit clipper are both built against one episode's
        `MjData` -- the MJCF is recompiled per episode -- so they are rebuilt on
        the next step rather than reset in place. The inner policy carries an
        action chunk and an observation history, and the ones left over belong to
        the last episode.
        """
        self._inner.reset()
        self._proxy = None
        self._clipper = None
        self._residuals = []

    def get_action(self, observation) -> dict[str, Any]:
        obs = observation[0] if isinstance(observation, list) else observation
        robot_view = self.task.env.current_robot.robot_view
        proxy = self._proxy or self._build_proxy(robot_view)

        franka_action = self._inner.get_action(self._inner_observation(obs, proxy))

        # Every group, not only the ones the IK solves for: an action dict that
        # omits a group leaves it holding whatever it was last told, and with
        # `include_base` off that would be the base's stale target rather than a
        # command to stand still.
        action = robot_view.get_noop_ctrl_dict()
        action.update(proxy.retarget_franka_joint_pos(franka_action["arm"]))
        action["gripper"] = proxy.retarget_robotiq_ctrl(franka_action["gripper"])
        # Recorded after the retarget, so a residual describes the action being
        # returned rather than the one before it.
        self._residuals.append((proxy.last_position_error, proxy.last_orientation_error))
        return self._clip_to_limits(robot_view, action)

    def get_info(self) -> dict:
        """Adds how far the commanded tool poses fell outside Stretch's reach.

        The number that says whether a failed episode failed at retargeting or at
        the policy: a few centimetres is the retargeting working, a persistent
        10cm+ means the policy is asking for somewhere this robot cannot go from
        where it is standing.
        """
        info = super().get_info()
        if not self._residuals:
            return info
        position, orientation = np.array(self._residuals).T
        info["retarget_position_error_mean_m"] = float(position.mean())
        info["retarget_position_error_max_m"] = float(position.max())
        info["retarget_orientation_error_mean_rad"] = float(orientation.mean())
        info["retarget_orientation_error_max_rad"] = float(orientation.max())
        if self._proxy is not None:
            # Steps the policy asked for somewhere the lift could not go. Distinct
            # from the residual above, which averages every step's compromise:
            # this counts only the ones where the arm was against a travel limit
            # and still short, which is the difference between "the retargeting
            # approximated" and "the robot cannot go there". See
            # `FrankaOnStretchView._warn_if_lift_saturated`.
            info["retarget_unreachable_steps"] = int(self._proxy.unreachable_steps)
        return info

    # =========================================================================
    # The retargeting, per episode and per step
    # =========================================================================

    def _build_proxy(self, robot_view) -> FrankaOnStretchView:
        """Stand the virtual Franka at this episode's robot and start Stretch at its home pose.

        Built on the first step rather than in `__init__` or `reset` because this
        reads where the robot is standing, and only by the first `get_action` is
        the episode's scene compiled, the robot spawned and the reset settled.
        """
        policy_config = self.config.policy_config
        base_xytheta = robot_view.get_move_group("base").joint_pos
        proxy = FrankaOnStretchView(
            robot_view,
            self.config.robot_config.robot_namespace,
            franka_mount_pose_from_base(base_xytheta),
            include_base=policy_config.include_base,
            match_robotiq_aperture=policy_config.match_robotiq_aperture,
            jaw_mode=policy_config.jaw_mode,
        )

        # Measured before the snap and from where the robot is standing, so it
        # measures this scene's shortfall rather than one left over from the
        # pose the snap puts the robot in.
        # Taken as given, never measured. See `target_z_offset`.
        proxy.target_z_offset = float(policy_config.target_z_offset)

        # Before the snap, not after, and `replay.replay_episode` does the same.
        # `reset()` calls `StretchArmIK.releash()`, which re-centres the base's
        # leash on wherever the robot is standing *at that moment*. With
        # `include_base` -- the default -- the opening snap is a whole-body solve
        # that can drive the base to reach the Franka's home pose, so re-leashing
        # afterwards makes the snap's excursion the new centre and the episode
        # gets that yaw for free on top of its own leash, with nothing pulling the
        # base back towards where the episode meant to stand it. Leashing first
        # bounds the snap itself, which is what `_ScenePanel` already assumes when
        # it frames a replay camera on the episode's base pose rather than the
        # robot's.
        proxy.reset()
        if policy_config.snap_to_franka_home:
            residual = proxy.snap_to_franka_joint_pos()
            log.debug(
                f"[droid] snapped to the Franka home pose, residual "
                f"{np.round(residual, 4).tolist()}"
            )

        log.info(
            f"[droid] virtual Franka at {np.round(proxy.franka_mount_pose[:3, 3], 3).tolist()}, "
            f"target z offset {proxy.target_z_offset:+.4f}m, "
            f"jaw {proxy.jaw_mode}, opens to {proxy.finger_open:.4f} rad, "
            f"base {'in' if policy_config.include_base else 'out of'} the IK"
        )
        self._proxy = proxy
        return proxy

    def _inner_observation(self, obs: dict, proxy: FrankaOnStretchView) -> dict:
        """The observation `RealRobotVLAPolicy` expects, built from a Stretch one.

        Two images under the names the checkpoint was trained with, the arm and
        gripper state in Franka DROID units, and the episode's instruction. The
        instruction comes from the task rather than the observation because
        MolmoSpaces carries it on the task (`get_task_description`) and not in the
        per-step observation dict.
        """
        policy_config = self.config.policy_config
        return {
            "task": self.task.get_task_description(),
            "qpos": {
                "arm": proxy.get_move_group("arm").joint_pos,
                "gripper": proxy.get_move_group("gripper").joint_pos,
            },
            DROID_EXO_CAMERA_KEY: self._camera(obs, policy_config.exo_camera),
            DROID_WRIST_CAMERA_KEY: self._wrist_camera(
                obs, policy_config.wrist_camera, proxy
            ),
        }

    @staticmethod
    def _wrist_camera(obs: dict, name: str, proxy: FrankaOnStretchView) -> np.ndarray:
        """The wrist frame, turned back upright when the wrist is held half over.

        `JAW_FLIP` is a half turn about the tool's *approach* axis, and Stretch's
        wrist camera looks along that axis -- so holding the flipped branch rolls
        the camera 180 degrees about its own optical axis. Measured on the
        compiled model at four tool poses: 180.000 degrees, with the axis of that
        rotation 0.003 degrees off the camera's own -z. The viewpoint moves about
        110mm with it, the camera crossing to the other side of the approach
        axis.

        It does *not* buy a better view of the grasp, which an earlier version of
        this note claimed. The flip is a symmetry of a parallel jaw, so the hand's
        own appearance is invariant under it: projected into the wrist frame, both
        fingertips and the grasp centre land in the same place on either branch
        (v = -0.265 and -0.243). What changes is which side of the world the
        camera sees past the hand.

        What is not wanted is the roll. The DROID checkpoint reads the wrist view
        more closely than any other channel (see `cameras.RetargetParams.
        wrist_fov_deg`), and it was trained on a Franka whose hand is not turned
        over: hand it an upside-down frame and its corrections come back
        inverted, which looks exactly like an arm driving away from the object it
        is reaching for. Undoing the roll in image space returns the orientation
        the checkpoint expects -- the same trick `ExoCameraParams.quarter_turns`
        plays for the head camera, which is bolted on sideways for reasons
        equally uninteresting to a policy.

        **And it costs the near field.** Because the flip moves the camera across
        the axis as well as rolling it, turning the image back lands the gripper
        at the *top* of the frame (v = +0.265) where the Franka's sits at the
        bottom (v = -0.19), and puts the grasp centre above the optical centre
        rather than below it. So the turn is right for the scene and wrong for
        the hand -- and the scene wins decisively. Measured, not argued:
        `PoseConventions.keep_flipped_wrist_camera_frame` skips the turn and the
        arm stops arriving at the object at all, because an unrotated frame
        inverts the sign of every lateral correction while the turn only biases
        where the loop settles. Read that flag for the measurement, and for what
        the turn still leaves wrong.

        Keyed on `jaw_flipped` rather than on the pose convention because that
        flag is the physical truth: under `jaw_mode="auto"` the branch can change
        mid-episode, and the compensation has to change with it.

        Exact only where the branch is: at a tool yaw that runs the wrist into
        its roll limit the arm settles short of the half turn (158.9 degrees at
        one of the four poses measured; see `JAW_MODES`), and a full turn back
        then over-corrects by the shortfall. Still much nearer upright than
        leaving it, and `retarget_orientation_error_mean_rad` already reports
        when the branch is not being held.
        """
        frame = StretchMolmoBotDroidPolicy._camera(obs, name)
        if not proxy.jaw_flipped:
            return frame
        if proxy.pose_conventions.keep_flipped_wrist_camera_frame:
            # Asked for the frame as the camera produced it. The turn below fixes
            # the roll and breaks where the gripper sits in the frame, and which
            # of the two a checkpoint prefers is a rollout question; see
            # `PoseConventions.keep_flipped_wrist_camera_frame` for the
            # projections.
            return frame
        # `ascontiguousarray` again, for `_camera`'s reason: `rot90` returns a
        # view with negative strides and `torch.from_numpy` refuses those.
        return np.ascontiguousarray(np.rot90(frame, 2))

    @staticmethod
    def _camera(obs: dict, name: str) -> np.ndarray:
        """One camera frame, as an array `torch.from_numpy` will accept.

        The copy is not defensive tidiness, it is required. MolmoSpaces hands
        over rendered frames as flipped *views* -- negative strides -- and
        MolmoBot's preprocessor puts them straight into `torch.from_numpy`,
        which refuses them: "At least one stride in the given numpy array is
        negative". That surfaces as a rollout error on the first step of every
        episode, so the whole benchmark is skipped. `ascontiguousarray` copies
        only when the array is not already contiguous.
        """
        if name not in obs:
            raise KeyError(
                f"Camera {name!r} is not in the observation. Available: "
                f"{sorted(k for k in obs if not k.startswith('_'))}. Set exo_camera / "
                "wrist_camera on the policy config to cameras the eval config renders."
            )
        return np.ascontiguousarray(obs[name])

    def _clip_to_limits(self, robot_view, action: dict[str, Any]) -> dict[str, Any]:
        """Bring the retargeted targets inside what the model says the robot can do.

        The IK already clips to `commandable_limits`, so this is belt and braces
        for the groups it does not solve for -- the gripper -- and the guarantee
        that what leaves here is something `JointPosController` can command
        without clipping it again. An actuator held past its limit is a standing
        position error driven into a mechanical stop.
        """
        if self._clipper is None:
            self._clipper = JointTargetClipper(robot_view)
        return self._clipper.clip_action(action)


# Re-exported so a caller can check a checkpoint's width against what this
# adapter drives without importing the retargeting module too.
FRANKA_ARM_JOINTS = VirtualFranka.N_JOINTS
