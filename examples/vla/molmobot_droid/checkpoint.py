"""
Download, load and release the MolmoBot-DROID checkpoint (https://huggingface.co/allenai/MolmoBot-DROID).

The Hugging Face repo holds only the weights (`model.pt`, a ~19 GB raw state dict) and an OLMo
`config.yaml`; the network itself is defined in MolmoBot's `olmo` package. MolmoBot pins
Python 3.11 while this repo is 3.12+, so instead of being pip-installed its code is cloned at a
pinned commit and imported by path (see `ensure_molmobot_code()`). Its runtime dependencies are
in this repo's `molmobot-droid` extra.

Usage:

    from examples.vla.molmobot_droid.checkpoint import load_policy, build_instruction

    with load_policy() as policy:
        policy.reset()
        policy.add_observation(exo_rgb, wrist_rgb)
        chunk = policy.predict_chunk(state8, build_instruction("pick", "red mug"))

See README.md in this directory for what the inputs and outputs mean.
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

HF_REPO = "allenai/MolmoBot-DROID"
HF_REVISION = "cbe6ec358958d07ddfb20d3aa54e560e9e1b18c9"

MOLMOBOT_REPO_URL = "https://github.com/allenai/MolmoBot.git"
MOLMOBOT_COMMIT = "33c0ca77bf6062a23d60ffd4a6859334c4a46d30"
MOLMOBOT_CACHE_DIR = Path.home() / ".cache" / "molmobot"
MOLMOBOT_PATH_ENV = "MOLMOBOT_PATH"

DROID_IMAGE_SIZE = (640, 368)
"""(width, height) the training data was rendered at (molmospaces `FrankaDroidCameraSystem`)."""

ACTION_HORIZON = 16
"""Actions per predicted chunk."""

POLICY_HZ = 15
"""Control rate the policy was trained at (molmospaces `policy_dt_ms=66`)."""

POLICY_DT = 0.066

N_OBS_STEPS = 2
OBS_STEP_DELTA = 8
"""Each query sees the current frame and the one 8 control steps before it, per camera."""

RELATIVE_MAX_JOINT_DELTA = 0.2
"""rad. DROID's IK solver limit, applied per step by MolmoBot's `RealRobotVLAPolicy`."""

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 255.0
"""Action units for the Robotiq 2F-85 `fingers_actuator`; the policy's output is thresholded at 128."""

FRANKA_HOME_QPOS = (0.0, -0.7853, 0.0, -2.35619, 0.0, 1.57079, 0.0)
"""`FrankaRobotConfig.init_qpos["arm"]` in molmospaces."""

ROBOTIQ_DRIVER_OPEN = 0.00296
ROBOTIQ_DRIVER_CLOSED = 0.824
"""Range of `gripper/left_driver_joint`, which is the 8th state element the policy reads."""


@dataclass(frozen=True)
class TaskSpec:
    prompt: str
    """Instruction template. Fields: {object}, {receptacle}."""
    molmospaces_task_cls: str
    """The molmospaces task whose success criterion applies."""


TASKS: dict[str, TaskSpec] = {
    # Phrased like the molmospaces benchmark instructions.
    "pick": TaskSpec("Pick up the {object}", "molmo_spaces.tasks.pick_task.PickTask"),
    "pick_and_place": TaskSpec(
        "Pick up the {object} and place it in or on the {receptacle}",
        "molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
    ),
    "pick_and_place_next_to": TaskSpec(
        "Pick up the {object} and place it next to the {receptacle}",
        "molmo_spaces.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask",
    ),
    "pick_and_place_color": TaskSpec(
        "Pick up the {object} and place it in or on the {receptacle}",
        "molmo_spaces.tasks.pick_and_place_color_task.PickAndPlaceColorTask",
    ),
    # Opening and closing are the same task class, told apart by `task_type`.
    "open": TaskSpec("Open the {object}", "molmo_spaces.tasks.opening_tasks.OpeningTask"),
    "close": TaskSpec("Close the {object}", "molmo_spaces.tasks.opening_tasks.OpeningTask"),
}


def build_instruction(task: str, object_name: str, receptacle: str | None = None) -> str:
    """The language instruction for `task`, e.g. `build_instruction("pick", "red mug")`."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choose from {sorted(TASKS)}")
    template = TASKS[task].prompt
    if "{receptacle}" in template and receptacle is None:
        raise ValueError(f"Task '{task}' needs a receptacle")
    return template.format(object=object_name, receptacle=receptacle)


def ensure_molmobot_code() -> Path:
    """
    Make MolmoBot's `olmo` package importable and return the directory that contains it.

    Uses $MOLMOBOT_PATH if set (either the allenai/MolmoBot checkout or its `MolmoBot/`
    subdirectory), otherwise clones MOLMOBOT_COMMIT into ~/.cache/molmobot/<commit> on first use.
    """
    override = os.environ.get(MOLMOBOT_PATH_ENV)
    if override:
        root = Path(override).expanduser()
    else:
        root = MOLMOBOT_CACHE_DIR / MOLMOBOT_COMMIT
        if not (root / ".git").exists():
            log.warning(f"Cloning {MOLMOBOT_REPO_URL}@{MOLMOBOT_COMMIT[:7]} into {root}")
            root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--quiet", MOLMOBOT_REPO_URL, str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "checkout", "--quiet", MOLMOBOT_COMMIT], check=True
            )

    # The repo is a monorepo; the model code lives in MolmoBot/olmo.
    for candidate in (root, root / "MolmoBot"):
        if (candidate / "olmo").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    raise FileNotFoundError(
        f"No `olmo` package under {root} or {root / 'MolmoBot'}. "
        f"Point ${MOLMOBOT_PATH_ENV} at a checkout of {MOLMOBOT_REPO_URL}."
    )


def download_checkpoint(revision: str | None = HF_REVISION) -> Path:
    """Download (or find in the HF cache) the checkpoint and return its local directory."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, revision=revision))


def resize_to_droid(image_rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB image to DROID_IMAGE_SIZE (no cropping; see `crop_to_aspect()` for that)."""
    if (image_rgb.shape[1], image_rgb.shape[0]) == DROID_IMAGE_SIZE:
        return image_rgb
    return cv2.resize(image_rgb, DROID_IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def clamp_gripper(gripper: float) -> float:
    """MolmoBot's `clamp_gripper`: anything over 128 is closed."""
    return GRIPPER_CLOSED if gripper > 128 else GRIPPER_OPEN


def limit_joint_delta(arm_target: np.ndarray, arm_state: np.ndarray) -> np.ndarray:
    """
    Scale the step from `arm_state` to `arm_target` so no joint moves more than
    RELATIVE_MAX_JOINT_DELTA, keeping its direction (MolmoBot's `relative_max_joint_delta`).
    """
    delta = np.asarray(arm_target, dtype=float) - np.asarray(arm_state, dtype=float)
    scale = np.max(np.abs(delta)) / RELATIVE_MAX_JOINT_DELTA
    if scale > 1:
        delta = delta / scale
    return np.asarray(arm_state, dtype=float) + delta


class MolmoBotDroidPolicy:
    """
    MolmoBot-DROID with its observation history.

    Call `add_observation()` once per *executed* control step (every 1/15 s of robot time),
    and `predict_chunk()` whenever a new chunk is needed. The history is what makes
    `obs_step_delta` mean "8 control steps ago", as in MolmoBot's `RealRobotVLAPolicy`.
    """

    def __init__(self, checkpoint_dir: str | Path | None = None, device: str = "cuda"):
        ensure_molmobot_code()
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else download_checkpoint()
        log.warning(f"Loading MolmoBot-DROID from {self.checkpoint_dir} on {device}...")
        self._wrapper = SynthManipMolmoInferenceWrapper(
            checkpoint_path=str(self.checkpoint_dir), device=device, use_bfloat16=True
        )
        self.n_obs_steps = getattr(self._wrapper.model_config, "n_obs_steps", N_OBS_STEPS)
        self.obs_step_delta = getattr(self._wrapper.model_config, "obs_step_delta", OBS_STEP_DELTA)
        self.action_horizon = self._wrapper.action_horizon
        # Only the frames a query can reach are kept, so the history never grows.
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=(self.n_obs_steps - 1) * self.obs_step_delta + 1
        )

    @property
    def is_loaded(self) -> bool:
        return self._wrapper is not None

    def reset(self) -> None:
        """Forget the observation history. Call at the start of every episode."""
        self._history.clear()

    def add_observation(self, exo_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
        """Append this control step's exo and wrist images (RGB, any size)."""
        self._history.append((resize_to_droid(exo_rgb), resize_to_droid(wrist_rgb)))

    def predict_chunk(self, state8: np.ndarray, instruction: str) -> np.ndarray:
        """
        Predict an [action_horizon, 8] chunk: 7 absolute FR3 joint positions (rad) and the
        gripper command, already clamped to GRIPPER_OPEN/GRIPPER_CLOSED.

        `state8` is the 7 FR3 joint positions and the Robotiq driver joint angle.
        Joint-delta limiting is per executed step, against the state at that step; see
        `limit_joint_delta()`.
        """
        if self._wrapper is None:
            raise RuntimeError("The policy has been closed")
        if not self._history:
            raise RuntimeError("Call add_observation() before predict_chunk()")

        # Per camera, frames [t - delta*(n-1), ..., t], skipping any before the episode began.
        # The camera order (exo, then wrist) is the training order.
        current = len(self._history) - 1
        indices = [
            current - (self.n_obs_steps - 1 - i) * self.obs_step_delta
            for i in range(self.n_obs_steps)
        ]
        indices = [i for i in indices if i >= 0]
        images = [self._history[i][0] for i in indices] + [self._history[i][1] for i in indices]

        chunk = self._wrapper.get_action_chunk(
            images=images,
            task_description=instruction,
            state=np.asarray(state8, dtype=np.float32),
        )
        chunk = np.array(chunk, dtype=float)
        chunk[:, 7] = [clamp_gripper(g) for g in chunk[:, 7]]
        return chunk

    def close(self) -> None:
        """Free the model's GPU and CPU memory. Safe to call more than once."""
        if self._wrapper is None:
            return
        self._history.clear()
        self._wrapper.model = None
        self._wrapper = None
        _release_torch_memory()

    def __enter__(self) -> "MolmoBotDroidPolicy":
        return self

    def __exit__(self, *exc) -> None:
        close_policy(self)


def _release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


# A single slot: the model needs ~11 GB of VRAM, so two never live at once.
_loaded: MolmoBotDroidPolicy | None = None


def load_policy(
    checkpoint_dir: str | Path | None = None, device: str = "cuda"
) -> MolmoBotDroidPolicy:
    """
    Load the policy, reusing the one already loaded if it came from the same checkpoint and
    closing it otherwise. Release it with `close_policy()` (or a `with` block).
    """
    global _loaded
    wanted = Path(checkpoint_dir) if checkpoint_dir else download_checkpoint()
    if _loaded is not None and _loaded.is_loaded and _loaded.checkpoint_dir == wanted:
        _loaded.reset()
        return _loaded
    unload_policy()
    _loaded = MolmoBotDroidPolicy(wanted, device=device)
    return _loaded


def close_policy(policy: MolmoBotDroidPolicy) -> None:
    global _loaded
    policy.close()
    if _loaded is policy:
        _loaded = None


def unload_policy() -> None:
    """Close the cached policy, if any."""
    global _loaded
    if _loaded is not None:
        _loaded.close()
        _loaded = None
