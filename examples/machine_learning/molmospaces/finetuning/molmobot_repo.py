"""
The MolmoBot checkout that `finetune.py` hands the training off to.

Training happens in MolmoBot's own repository, not this one: the model, its
torch stack and its checkpoint format live there, and none of them is a
dependency here. This module is the part of that handoff that can be automated
without installing anything -- it puts the checkout on disk and collects the
scripts the documented workflow needs, so `finetune.py` can emit a shell script
whose paths all resolve.

Two things about MolmoBot's layout are worth knowing before reading the code,
because both are easy to get wrong from the README alone:

**The package is one level down.** `git clone` gives a repository whose top
level holds `MolmoBot/`, `MolmoBot-Pi0/`, `MolmoBot-SPOC/` and `robot_eval/`.
The trainer is `MolmoBot/launch_scripts/train_molmobot.py`, and `pyproject.toml`
and the virtualenv live beside it -- so every command runs from
`<checkout>/MolmoBot`, which is what `PACKAGE_SUBDIR` names.

**The two postprocessing scripts are not in the repository.**
`validate_trajectories.py` and `calculate_stats.py` are referenced by MolmoBot's
README as though they sat next to the trainer, but they ship with the
*HuggingFace dataset* (`allenai/molmobot-data`) instead. Cloning the git repo
and looking for them is the obvious first move and it finds nothing, so this
module downloads them into `DATA_SCRIPTS_SUBDIR` -- a directory that is ours,
outside the package, so it can never collide with a file MolmoBot later adds.

Nothing here installs anything. Creating the virtualenv is `uv sync --extra
train`, which pulls torch and the rest, and that belongs in the generated script
where it is visible and the user runs it deliberately.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

MOLMOBOT_GIT_URL = "https://github.com/allenai/MolmoBot.git"

DEFAULT_CHECKOUT = Path("third_party/MolmoBot")
"""
Where the checkout goes when `--trainer-repo` is not given.

Beside `robocasa` and `robosuite`. Those two are git submodules and this is a
plain clone, so `.gitignore` carries an entry for it.
"""

REPO_ROOT = Path(__file__).resolve().parents[4]
"""
This repository's root, so a checkout can be found from any working directory.

`DEFAULT_CHECKOUT` is relative, which is what `finetune.py` wants -- it is run
from the repository root and its generated script records absolute paths. But
`ensure_importable` is called from `run_benchmarks.py`, which is run from
wherever the evaluation output is going, and a relative clone path there resolves
to a directory that does not exist.
"""

PACKAGE_SUBDIR = "MolmoBot"
"""The python package inside the checkout: `pyproject.toml`, `launch_scripts/`, `.venv/`."""

TRAINER_SCRIPT = "launch_scripts/train_molmobot.py"
"""Relative to the package directory. Its presence is what makes a directory a MolmoBot checkout."""

DATA_SCRIPTS_SUBDIR = "data_scripts"
"""Where the HuggingFace postprocessing scripts are put, relative to the checkout root."""

HF_DATASET_REPO = "allenai/molmobot-data"

POSTPROCESSING_SCRIPTS: dict[str, str] = {
    "validate_trajectories.py": (
        f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/validate_trajectories.py"
    ),
    "calculate_stats.py": (
        f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/calculate_stats.py"
    ),
}
"""
MolmoBot's data postprocessing, which lives with the dataset rather than the code.

`validate_trajectories.py` writes the `valid_trajectory_index.json` that
`SynthmanipDataset` refuses to start without, plus a `valid_traj_mask` in each
HDF5. `calculate_stats.py` writes a per-trajectory `stats` group and an
`aggregated_stats.json`; see `finetune.py`'s generated script for when that one
actually matters.
"""

DOWNLOAD_TIMEOUT_SECONDS = 60


PRESETS_MODULE = "olmo/data/synthmanip_presets.py"
"""Where MolmoBot keeps the robot presets, relative to the package directory."""

OPTIM_MODULE = "olmo/train/optim.py"
"""Where MolmoBot builds the optimizer, relative to the package directory."""

EVAL_MODULE = "olmo/eval/configure_molmo_spaces.py"
"""Where MolmoBot keeps its MolmoSpaces evaluation policy, relative to the package directory."""

TRAINER_MODULE = "olmo/train/trainer.py"
"""Where MolmoBot's training loop lives, relative to the package directory."""

INFERENCE_REQUIREMENTS: dict[str, str] = {
    "cached_path": "cached_path",
    "cached_property": "cached_property",
    "fiddle": "fiddle",
    "rich": "rich",
    "tokenizers": "tokenizers",
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "einops": "einops",
    "einops_exts": "einops-exts",
    "accelerate": "accelerate",
    "av": "av",
}
"""
What MolmoBot's model has to import, mapped from module name to what installs it.

Training happens in MolmoBot's own virtualenv, which `uv sync --extra train`
builds inside the checkout from its `pyproject.toml`. *Evaluation does not.*
`run_benchmarks.py --policy molmobot` runs MolmoSpaces and MolmoBot in the same
interpreter -- this repository's -- because a rollout is a MolmoSpaces
simulation with MolmoBot's weights in it, and nothing can bridge two
interpreters mid-episode. So MolmoBot's runtime dependencies have to exist here
too, and they are not in this repository's own dependency list because MolmoBot
is not a dependency of it.

Only what the inference path touches, not MolmoBot's whole training stack: the
model, its tokenizer, and the checkpoint loader. Missing ones surface deep
inside `SynthManipMolmoInferenceWrapper` as a bare ModuleNotFoundError, several
minutes into a benchmark, once per worker -- so `missing_inference_requirements`
is called at the command line instead, before anything loads.
"""

METRICS_ENV_VAR = "MOLMOBOT_METRICS_JSONL"
"""
Where `ensure_metrics_log`'s patch writes, and the switch that turns it on.

An environment variable rather than a flag because the patched function is
MolmoBot's, called from inside its training loop, and the generated run script is
the only thing that needs to set it. Unset, the patch does nothing at all.
"""

ADAMW8BIT = "adamw8bit"
"""
The `optimizer.name` the generated script passes by default.

MolmoBot ships `lionw` and `adamw`, both keeping optimizer state in fp32. At
`TRAINABLE=vision` that is 918M trainable parameters (vision tower + action
expert) at 8 bytes each -- 6.8GiB of moments on top of 18.6GiB of fp32 master
weights, which does not fit beside activations on a 32GiB card. torchao's
block-wise 8-bit AdamW is the same update rule with the moments quantized, ~2
bytes/param, and `ensure_adamw8bit` teaches MolmoBot to build it.
"""

_ADAMW8BIT_ENUM_MEMBER = '''    adamw8bit = "adamw8bit"
    """
    AdamW with both moments held in 8 bits instead of fp32, from torchao.

    Same update rule as `adamw`; only the state is quantized, block-wise with a
    per-block scale, so the optimizer costs ~2 bytes/param rather than 8. On a
    single 32GiB card that is the difference between fitting the vision tower
    and not: fp32 moments for 918M trainable params (ViT + action expert) are
    6.8GiB against 1.7GiB here.

    torchao rather than bitsandbytes because this trainer runs FSDP2
    (`FSDPConfig.fsdp2`), so parameters are DTensors -- torchao's optimizers
    unwrap to the local shard and rewrap with `DTensor.from_local`, which
    bitsandbytes has no equivalent of.

    Kept out of `pyproject.toml` on purpose: this is a vendored checkout, and
    resolving torchao as a project dependency drags torch back to a version
    without Blackwell (sm_120) wheels. The generated launch script installs it
    into the venv on its own.
    """
'''

_ADAMW8BIT_BUILDER = '''

def _build_adamw8bit(param_groups, **kwargs) -> Optimizer:
    """torchao's 8-bit AdamW, adapted to the learning rate this trainer hands it.

    torchao pins each group's ``lr`` to a tensor -- it is passed as an argument
    into a ``torch.compile``d step -- and raises if anything replaces it with a
    float. :meth:`Trainer.train_step` assigns the scheduled lr as a plain float
    every step, so the value is rewrapped here just before stepping.

    Rewrapped *in place*: one tensor per group, reused with ``fill_`` rather than
    rebuilt, so the compiled step keeps seeing the same object and does not
    re-guard on a new one every step.
    """
    try:
        from torchao.optim import AdamW8bit
    except ImportError as e:
        raise ImportError(
            "optimizer.name=adamw8bit needs torchao, which is not installed. "
            "Install it into this venv without adding it to pyproject.toml "
            "(as a project dependency it pulls torch back off the cu128 wheels "
            "that Blackwell needs):\\n"
            "    uv pip install --no-deps torchao"
        ) from e

    class AdamW8bitScheduledLr(AdamW8bit):
        def step(self, closure=None):
            holders = self.__dict__.setdefault("_lr_holders", {})
            for i, group in enumerate(self.param_groups):
                lr = group["lr"]
                if not torch.is_tensor(lr):
                    holder = holders.get(i)
                    if holder is None:
                        holder = holders[i] = torch.tensor(float(lr), dtype=torch.float32)
                    else:
                        holder.fill_(float(lr))
                    group["lr"] = holder
            return super().step(closure)

    return AdamW8bitScheduledLr(param_groups, **kwargs)
'''

_ADAMW8BIT_BRANCH = '''        elif self.name == OptimizerType.adamw8bit:
            log.info("Using 8-bit AdamW; optimizer state is ~2 bytes/param instead of 8")
            return _build_adamw8bit(
                param_groups,
                lr=self.learning_rate,
                betas=self.betas,
                weight_decay=self.weight_decay,
                eps=self.eps,
            )
'''

STRETCH_PRESET_NAMES: dict[str, str] = {
    "joint_pos": "stretch_joint",
    "joint_pos_rel": "stretch_jointdelta",
}
"""
Action type -> the preset name `--action_preset` takes.

MolmoBot names presets after the robot and the action encoding
(`franka_joint`, `franka_jointdelta`), and reads the matching trajectory key out
of `ACTION_DATASET_KEYS`, so there is one preset per action type rather than one
per robot.
"""

GRIPPER_STATE_WIDTH = 1
"""
How many of the gripper's qpos values go into the *state* vector.

Stretch's gripper qpos is two numbers (the two finger joints) and the action
spec keeps both, but MolmoBot truncates the gripper to
`SynthmanipDatasetConfig.gripper_representation_count` when it builds a state
vector -- in `synthmanip_dataset._get_state`, in the statistics pass that writes
`synthmanip_norm_stats.yaml`, and again at inference in
`configure_molmo_spaces`. That field defaults to 1 and nothing here overrides
it, so the state is nine wide where the action is ten.

`state_dim` is otherwise `sum(action_spec)`, which makes MolmoBot reject every
example with

    ValueError: State shape (9,) != expected 10 for action_spec {...}

`STATE_SPECS` is the mechanism it provides for exactly this, so this is the
number that goes there. It has to keep matching the inference default: the state
layout the model is trained on is the one it has to be served, and a silent
disagreement shifts every value after the gripper.
"""

GENERATED_MARKER = "# --- generated by stretch4_mujoco finetuning/molmobot_repo.py ---"


class MolmoBotSetupError(RuntimeError):
    """The checkout could not be produced or is not a MolmoBot checkout."""


@dataclass
class MolmoBotCheckout:
    """A MolmoBot checkout on disk, and what had to be done to get it there."""

    root: Path
    """The clone itself."""

    package_dir: Path
    """`<root>/MolmoBot`, where every command runs from."""

    data_scripts_dir: Path
    """`<root>/data_scripts`, holding the two downloaded postprocessing scripts."""

    cloned: bool = False
    """True if this call created the checkout rather than finding it."""

    fetched_scripts: list[str] = field(default_factory=list)
    """Postprocessing scripts downloaded by this call; already-present ones are not re-fetched."""

    @property
    def venv_python(self) -> Path:
        """The interpreter `uv sync` creates, which the generated script calls directly.

        Calling it by path rather than sourcing `.venv/bin/activate` keeps the
        generated script compatible with `set -u`: activation scripts read
        `$PS1`, which is unset in a non-interactive shell.
        """
        return self.package_dir / ".venv" / "bin" / "python"

    @property
    def has_venv(self) -> bool:
        return self.venv_python.exists()

    def script(self, name: str) -> Path:
        return self.data_scripts_dir / name


def ensure_checkout(
    root: Path | str = DEFAULT_CHECKOUT,
    clone: bool = True,
    fetch_scripts: bool = True,
) -> MolmoBotCheckout:
    """Put a usable MolmoBot checkout at `root` and describe it.

    Idempotent: an existing checkout is validated and its missing postprocessing
    scripts topped up, never re-cloned or overwritten.

    Args:
        root: where the clone lives, or should live.
        clone: clone when `root` is absent. False turns a missing checkout into
            an error instead, for callers that would rather not touch the disk.
        fetch_scripts: download the postprocessing scripts that are missing.
            False skips the network entirely once the clone is there.

    Raises:
        MolmoBotSetupError: the clone failed, or `root` exists but is something
            other than a MolmoBot checkout.
    """
    root = Path(root)
    checkout = MolmoBotCheckout(
        root=root,
        package_dir=root / PACKAGE_SUBDIR,
        data_scripts_dir=root / DATA_SCRIPTS_SUBDIR,
    )

    if not (root.exists() and any(root.iterdir())):
        if not clone:
            raise MolmoBotSetupError(
                f"No MolmoBot checkout at {root}. Clone it with\n"
                f"  git clone --depth 1 {MOLMOBOT_GIT_URL} {root}\n"
                "or re-run without --no-clone to have this script do it."
            )
        _clone(root)
        checkout.cloned = True

    if not (checkout.package_dir / TRAINER_SCRIPT).exists():
        raise MolmoBotSetupError(
            f"{root} exists but has no {PACKAGE_SUBDIR}/{TRAINER_SCRIPT}, so it is not a "
            "MolmoBot checkout. Point --trainer-repo somewhere else, or remove that "
            "directory and let this script clone into it."
        )

    if fetch_scripts:
        checkout.fetched_scripts = _fetch_postprocessing_scripts(checkout.data_scripts_dir)

    return checkout


def ensure_stretch_presets(checkout: MolmoBotCheckout, action_spec: dict[str, int]) -> list[str]:
    """Register Stretch in MolmoBot's preset table, and say what was added.

    MolmoBot cannot be told an action spec from the command line. `--action_dim`
    and `--action_move_groups` give it the width and the names, but the *per
    group* widths come only from `ACTION_SPECS[args.action_preset]`, and with no
    preset matched `train_molmobot.py` raises

        ValueError: Action spec must be specified via --action_preset.

    after the data paths have been validated. There is no Stretch preset
    upstream, so this writes one -- `stretch_joint` and `stretch_jointdelta`,
    differing only in which `actions/` key they read -- generated from
    `action_spec` so the widths cannot drift from the ones this repository
    trains and evaluates against.

    It writes `STATE_SPECS` alongside them, because the state vector is narrower
    than the action vector by the gripper -- see `GRIPPER_STATE_WIDTH`, which is
    where that reasoning lives.

    Editing a dependency's source is not something to do lightly, and this is the
    narrowest form of it available: two entries per dictionary, in a gitignored
    clone that `ensure_checkout` will recreate from scratch if it is deleted. The
    alternative is patching the `raise` in `train_molmobot.py`, which means owning
    a diff against a function that moves between versions -- with the preset in
    place that branch is never reached, so the trainer runs unmodified.

    Idempotent per table, and deliberately conservative: a table that already
    mentions `stretch_joint` -- a hand-applied patch, or an earlier run of this --
    is left exactly as it is. Per table rather than per file because the tables
    were not all written at once: a checkout carrying only the `ACTION_SPECS`
    entries from an earlier version of this function still needs `STATE_SPECS`.

    Returns:
        The preset names, empty if every table already had them.
    """
    path = checkout.package_dir / PRESETS_MODULE
    if not path.exists():
        raise MolmoBotSetupError(
            f"{path} not found, so this MolmoBot checkout does not have the preset table "
            "this expects. It may be a version whose layout has moved; re-clone, or add "
            f"the Stretch entries to its ACTION_SPECS by hand: {dict(action_spec)}"
        )

    source = path.read_text()
    names = sorted(set(STRETCH_PRESET_NAMES.values()))
    state_spec = {
        group: (GRIPPER_STATE_WIDTH if "gripper" in group else width)
        for group, width in action_spec.items()
    }

    blocks = {
        "ACTION_SPECS": _spec_block(names, action_spec),
        "ACTION_DATASET_KEYS": f"    {GENERATED_MARKER}\n"
        + "".join(
            f'    "{name}": "{action_type}",\n'
            for action_type, name in sorted(STRETCH_PRESET_NAMES.items())
        ),
        "STATE_SPECS": _spec_block(names, state_spec),
    }

    written = []
    for table, block in blocks.items():
        body = _table_body(source, table, path)
        if all(f'"{name}"' in body for name in names):
            continue
        source = _insert_after_dict_opening(source, table, block, path)
        written.append(table)

    if not written:
        return []

    path.write_text(source)
    # Which tables were touched goes to the log, not the return value: callers
    # care that the presets are now there, and a checkout that already had
    # ACTION_SPECS but not STATE_SPECS should still report the presets it gained.
    log.info(f"[molmobot] registered {', '.join(names)} in {', '.join(written)} in {path}")
    return names


def _spec_block(names: list[str], spec: dict[str, int]) -> str:
    """One `"<preset>": {<group>: <width>, ...}` entry per name, as source text."""
    return "".join(
        f"    {GENERATED_MARKER}\n"
        f'    "{name}": {{\n'
        + "".join(f'        "{group}": {width},\n' for group, width in spec.items())
        + "    },\n"
        for name in names
    )


def _find_dict_opening(source: str, name: str, path: Path) -> re.Match:
    """Locate the module-level `name = {` line, or say why it could not be found.

    Anchored on the assignment rather than parsed, because the file is a plain
    table of dict literals and rewriting it through an AST would reformat the
    rest of it into a diff nobody wants to read.
    """
    match = re.search(rf"^{name}\s*(?::[^=]+)?=\s*\{{[ \t]*$", source, re.MULTILINE)
    if match is None:
        # STATE_SPECS gets its own explanation: without it the dataset falls back
        # to ACTION_SPECS for the state width, which for Stretch is 10 against a
        # 9-wide qpos, and every example then fails the state shape guard. Better
        # to say so here than to let it surface as a per-example ValueError.
        detail = (
            "Stretch's state is narrower than its action -- see GRIPPER_STATE_WIDTH -- "
            "and STATE_SPECS is the only way to tell MolmoBot so. Without it the "
            "trainer rejects every example with `State shape (9,) != expected 10`. "
            "Add the table, or the Stretch entries in it, by hand."
            if name == "STATE_SPECS"
            else "MolmoBot's preset file has changed shape; add them by hand, or pass "
            "--action-preset once a real one exists upstream."
        )
        raise MolmoBotSetupError(
            f"Could not find the `{name} = {{` table in {path}, so the Stretch presets "
            f"cannot be registered. {detail}"
        )
    return match


def _table_body(source: str, name: str, path: Path) -> str:
    """The text between the braces of the module-level `name = {` literal.

    Scoped to the one table so the "is it already registered?" check cannot be
    satisfied by the preset appearing in a *different* table -- which is exactly
    the case a checkout with `ACTION_SPECS` but no `STATE_SPECS` presents.
    Terminated on the closing brace in column zero, the only place a module-level
    dict literal in this file ends.
    """
    start = _find_dict_opening(source, name, path).end()
    end = re.compile(r"^\}", re.MULTILINE).search(source, start)
    if end is None:
        raise MolmoBotSetupError(
            f"The `{name} = {{` table in {path} is never closed by a `}}` in column zero, "
            "so its extent cannot be determined. MolmoBot's preset file has changed shape; "
            "add the Stretch entries by hand."
        )
    return source[start : end.start()]


def _insert_after_dict_opening(source: str, name: str, block: str, path: Path) -> str:
    """Put `block` immediately inside the `name = {` literal at module level."""
    cut = _find_dict_opening(source, name, path).end() + 1  # past the newline ending that line
    return source[:cut] + block + source[cut:]


def ensure_adamw8bit(checkout: MolmoBotCheckout) -> bool:
    """Teach MolmoBot's optimizer factory to build torchao's 8-bit AdamW.

    See `ADAMW8BIT` for why: fp32 optimizer moments do not fit beside the fp32
    master weights on one 32GiB card once the vision tower is trainable.

    Three insertions into `olmo/train/optim.py` -- a member on the `OptimizerType`
    enum, a `_build_adamw8bit` factory, and the branch in
    `OptimizerConfig.build_optimizer` that calls it -- in the same gitignored
    clone, and for the same reason, as `ensure_stretch_presets`: the generated
    script *defaults* to `--optimizer.name=adamw8bit`, so without this a freshly
    recreated checkout would reject its own launch script. torchao itself is not
    touched here; the script installs it into the venv.

    Idempotent: a checkout already mentioning `adamw8bit` is left alone.

    Returns:
        True if this call patched the file, False if it was already patched.
    """
    path = checkout.package_dir / OPTIM_MODULE
    if not path.exists():
        raise MolmoBotSetupError(
            f"{path} not found, so this MolmoBot checkout does not keep its optimizer "
            "where this expects. It may be a version whose layout has moved; re-clone, "
            "or run with OPTIMIZER=adamw and accept the fp32 optimizer state."
        )

    source = path.read_text()
    if ADAMW8BIT in source:
        return False

    # The enum member goes after the last existing member, found by anchoring on
    # `adamw` inside the `class OptimizerType` block rather than on the class line,
    # so a reordered enum still lands somewhere valid.
    enum_anchor = re.search(
        r"^class OptimizerType\(StrEnum\):\n(?:[ \t]+\w+ = \"[^\"]+\"\n)+", source, re.MULTILINE
    )
    if enum_anchor is None:
        raise MolmoBotSetupError(
            f"Could not find the `class OptimizerType(StrEnum)` block in {path}, so "
            f"{ADAMW8BIT} cannot be registered. Run with OPTIMIZER=adamw, or add the "
            "member and the matching `build_optimizer` branch by hand."
        )

    # The branch goes immediately before build_optimizer's fallthrough, so it is
    # reached only after the names MolmoBot ships have had their turn.
    branch_anchor = re.search(
        r"^([ \t]+)else:\n[ \t]+raise NotImplementedError\n", source[enum_anchor.end() :], re.MULTILINE
    )
    if branch_anchor is None:
        raise MolmoBotSetupError(
            f"Could not find `build_optimizer`'s `else: raise NotImplementedError` in {path}, "
            f"so the {ADAMW8BIT} branch has nowhere to go. Run with OPTIMIZER=adamw, or add "
            "the branch by hand."
        )
    cut = enum_anchor.end() + branch_anchor.start()

    # The factory goes at module level, after the last helper defined before
    # OptimizerConfig, so it is in scope by the time build_optimizer runs.
    helper_anchor = re.search(
        r"^def _clean_param_name\(name: str\) -> str:\n(?:[ \t]+.*\n)+", source, re.MULTILINE
    )
    if helper_anchor is None:
        raise MolmoBotSetupError(
            f"Could not find `_clean_param_name` in {path}, so there is no anchor for the "
            f"{ADAMW8BIT} factory. Run with OPTIMIZER=adamw, or add it by hand."
        )
    if helper_anchor.end() > cut:
        raise MolmoBotSetupError(
            f"`_clean_param_name` sits after `build_optimizer` in {path}, so inserting the "
            f"{ADAMW8BIT} factory there would not put it in scope. Add it by hand."
        )

    # Applied back to front so each insertion cannot shift the offsets of the
    # ones still to come.
    source = source[:cut] + _ADAMW8BIT_BRANCH + source[cut:]
    source = source[: helper_anchor.end()] + _ADAMW8BIT_BUILDER + source[helper_anchor.end() :]
    source = source[: enum_anchor.end()] + _ADAMW8BIT_ENUM_MEMBER + source[enum_anchor.end() :]
    path.write_text(source)
    log.info(f"[molmobot] registered {ADAMW8BIT} in {path}")
    return True


_POLICY_FACTORY_FIELD = f'''    {GENERATED_MARKER}
    # The installed molmo_spaces declares `policy_factory` on `BasePolicyConfig`
    # with no default, and `SynthVLAPolicyConfig` never sets one, so *importing*
    # this module fails while evaluating its own `FrankaState8ClampConfig` class
    # body:
    #
    #     ValidationError: 1 validation error for SynthVLAPolicyConfig
    #     policy_factory  Field required
    #
    # which reads like "MolmoBot is not installed" when it is. Annotated `object`
    # rather than `PolicyFactory` so no import has to be added to a file this
    # repository does not own; nothing ever calls it, because
    # `policies/molmobot_policy.py` constructs `SynthVLAPolicy` itself rather
    # than letting MolmoSpaces build one from this config.
    policy_factory: object = None
'''


def resolve_checkout_root(root: Path | str | None = None) -> Path:
    """Where the checkout is, for a caller that has not been told.

    Prefers `third_party/MolmoBot` under the current directory -- the path
    `finetune.py` clones into and the one a second checkout would be put at --
    and falls back to the same path under this repository, so an evaluation run
    from an output directory still finds the clone the fine-tune produced.
    """
    if root is not None:
        return Path(root)
    local = Path.cwd() / DEFAULT_CHECKOUT
    if (local / PACKAGE_SUBDIR / TRAINER_SCRIPT).exists():
        return local
    return REPO_ROOT / DEFAULT_CHECKOUT


def ensure_importable(root: Path | str | None = None, patch: bool = True) -> Path:
    """Make `import olmo` work from the checkout, and return the directory that does it.

    MolmoBot is not installed -- it is a clone, whose package directory holds a
    top-level `olmo` package -- so something has to put that directory on the
    import path. Doing it here rather than asking for `PYTHONPATH=...` on every
    command is the difference between an evaluation that runs and one that dies
    inside a rollout worker with an ImportError.

    Both mechanisms are set, because the workers are not this process:
    `sys.path` covers this interpreter and the `forkserver` children that
    inherit its memory, and `PYTHONPATH` covers `spawn`, which starts a fresh
    interpreter that reads only the environment.

    Args:
        root: the checkout, or None to find it with `resolve_checkout_root`.
        patch: apply `ensure_policy_factory_default` first. False skips the
            write for a caller that only wants the path.

    Returns:
        The package directory now on the path.

    Raises:
        MolmoBotSetupError: there is no MolmoBot checkout at `root`.
    """
    root = resolve_checkout_root(root)
    package_dir = (root / PACKAGE_SUBDIR).resolve()
    if not (package_dir / TRAINER_SCRIPT).exists():
        raise MolmoBotSetupError(
            f"No MolmoBot checkout at {root}, so `import olmo` cannot be made to work.\n"
            f"  git clone --depth 1 {MOLMOBOT_GIT_URL} {root}\n"
            "or run the fine-tuning setup, which clones it:\n"
            "  python -m examples.machine_learning.molmospaces.finetuning.finetune "
            "--rollouts <run> --trainer molmobot"
        )

    if patch:
        ensure_policy_factory_default(
            MolmoBotCheckout(
                root=root,
                package_dir=package_dir,
                data_scripts_dir=root / DATA_SCRIPTS_SUBDIR,
            )
        )

    entry = str(package_dir)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    inherited = os.environ.get("PYTHONPATH", "")
    if entry not in inherited.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [entry, inherited]))
    return package_dir


def ensure_policy_factory_default(checkout: MolmoBotCheckout) -> bool:
    """Give MolmoBot's policy configs the `policy_factory` default their base class requires.

    See `_POLICY_FACTORY_FIELD` for what breaks without it. This is version skew
    between the installed `molmo_spaces` and the MolmoBot checkout, not a
    mistake in either, so it is fixed the same way as the presets and the
    optimizer: insertions in a gitignored clone that `ensure_checkout` recreates
    from scratch if it is deleted.

    Every `BasePolicyConfig` subclass in the file is patched, not just the one
    this repository instantiates: the module builds several of its own eval
    configs at import time (`FrankaState8ClampConfig`,
    `RBY1DoorOpeningEvalConfig`, ...), and *any* of them missing the field
    fails the import for all of them.

    Idempotent: a class that already declares `policy_factory` is left alone,
    which is also what makes this safe against a MolmoBot that grows its own.

    Returns:
        True if this call patched the file, False if it was already fine.
    """
    path = checkout.package_dir / EVAL_MODULE
    if not path.exists():
        raise MolmoBotSetupError(
            f"{path} not found, so this MolmoBot checkout does not keep its MolmoSpaces "
            "evaluation policy where this expects. It may be a version whose layout has "
            "moved; re-clone, or add `policy_factory: object = None` to its "
            "BasePolicyConfig subclasses by hand."
        )

    source = path.read_text()
    headers = list(
        re.finditer(
            r'^class (\w+)\(BasePolicyConfig\):\n(?:[ \t]*"""(?:.|\n)*?"""\n)?',
            source,
            re.MULTILINE,
        )
    )
    if not headers:
        raise MolmoBotSetupError(
            f"Could not find any `class ...(BasePolicyConfig)` in {path}, so the missing "
            "`policy_factory` default has nowhere to go. MolmoBot's evaluation module has "
            "changed shape; add the field by hand."
        )

    # Back to front, so each insertion cannot shift the offsets of the ones
    # still to come. Each class body is scoped to the next `class` line: the
    # name appears in molmo_spaces' own configs elsewhere in the file, and
    # finding one of those would report a patch that was never applied.
    patched = []
    for header in reversed(headers):
        end = re.compile(r"^class ", re.MULTILINE).search(source, header.end())
        body = source[header.end() : end.start() if end else len(source)]
        if "policy_factory" in body:
            continue
        source = source[: header.end()] + _POLICY_FACTORY_FIELD + source[header.end() :]
        patched.append(header.group(1))

    if not patched:
        return False

    _write_atomic(path, source)
    log.info(
        f"[molmobot] gave {', '.join(reversed(patched))} a policy_factory default in {path}"
    )
    return True


_METRICS_CALL = "        _stretch4_record_metrics(self, prefix, metrics)\n"

CHECKPOINT_METRICS_FILENAME = "training_metrics.json"
"""
What the patch writes inside every checkpoint directory the trainer saves.

Beside the weights rather than only in the run's `metrics.jsonl`, because a
checkpoint outlives the terminal it was produced in. `step9500_bestfit/` on its
own says which step it came from and nothing else -- not what the validation
loss was there, not whether training had converged or was still improving, not
what the learning rates had decayed to. Six weeks later, pointing `--checkpoint`
at it, that is exactly what you want to know, and this travels with the
directory when it is copied to another machine.
"""

_METRICS_RECORDER_TEMPLATE = '''

@@MARKER@@
_stretch4_last_metrics = {"train": None, "eval": {}}
"""The most recent metrics dump of each kind, for `_stretch4_checkpoint_metrics`."""


def _stretch4_record_metrics(trainer, prefix, metrics):
    """Remember, and optionally log, one metrics dump.

    Called from the top of `Trainer.log_metrics_to_console`, which is the one
    place every metric this run computes passes through with its full numeric
    value: the console formats them and drops most of `optim/`, and W&B is off
    unless a project and entity are exported, so without this a run that is not
    using W&B keeps no record of its own loss curve.

    Two destinations, on separate switches. `$@@ENV@@` gets
    a JSON line per dump, which is the whole curve. `_stretch4_last_metrics`
    keeps the latest of each kind in memory whether or not that variable is set,
    because the checkpoint summary is written from it and a checkpoint should
    describe itself even when nobody asked for a log.

    Deliberately incapable of failing a training run: an unwritable path or an
    exotic metric value costs a line of the log, not eleven hours of fine-tune.
    """
    import datetime
    import json
    import os

    try:
        is_train = prefix.lstrip().startswith("[step=")
        record = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "step": int(getattr(trainer, "global_step", 0) or 0),
            "max_steps": int(getattr(trainer, "max_steps", 0) or 0),
            "split": "train" if is_train else "eval",
            "label": None if is_train else prefix.strip(),
            "metrics": {
                name: number
                for name, value in metrics.items()
                if (number := _stretch4_number(value)) is not None
            },
        }
        if is_train:
            _stretch4_last_metrics["train"] = record
        else:
            _stretch4_last_metrics["eval"][record["label"]] = record
    except Exception:  # noqa: BLE001 - bookkeeping must never kill a run
        return

    path = os.environ.get("@@ENV@@")
    if not path:
        return
    # `loss_eval` logs from every rank; only one of them should write.
    if int(os.environ.get("RANK", "0") or 0) != 0:
        return
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\\n")
    except Exception:  # noqa: BLE001
        pass


def _stretch4_number(value):
    """`value` as a float, if it is one -- including a scalar tensor.

    The tensor case is not exotic here, it is the learning rates: torchao's
    8-bit optimizer replaces `group["lr"]` with a 0-dim `torch.Tensor` so its
    compiled update can read it, and `LRMonitor.check()` hands those straight
    into the metrics dict. MolmoBot's own console filter is an `isinstance`
    check against `(int, float)`, which is why a run with this optimizer shows
    gradient norms and no learning rates anywhere -- the numbers are computed,
    they just never survive being logged.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if hasattr(value, "numel") and value.numel() != 1:
            return None
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _stretch4_finite(value):
    """`value` as a JSON-safe float, or None -- `best_eval_loss` starts at inf."""
    import math

    number = _stretch4_number(value)
    return number if number is not None and math.isfinite(number) else None


def _stretch4_checkpoint_metrics(trainer, kind):
    """The state training was in at the moment a checkpoint was written."""
    import datetime
    import os
    import time

    payload = {
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "step": int(getattr(trainer, "global_step", 0) or 0),
        "max_steps": int(getattr(trainer, "max_steps", 0) or 0),
        "epoch": getattr(trainer, "epoch", None),
        "tokens_seen": getattr(trainer, "global_train_tokens_seen", None),
        "train": {
            "loss": _stretch4_finite(getattr(trainer, "cur_train_loss", None)),
            "min_loss": _stretch4_finite(getattr(trainer, "min_train_loss", None)),
            "last_logged": _stretch4_last_metrics["train"],
        },
        "eval": _stretch4_last_metrics["eval"],
        "metrics_file": os.environ.get("@@ENV@@") or None,
    }

    # The best-fit bookkeeping, when this checkout has it: what the best
    # validation loss was, where, and how long it has been since it moved --
    # which is what says whether this checkpoint is the converged one or just
    # the newest one.
    best_loss = _stretch4_finite(getattr(trainer, "best_eval_loss", None))
    if best_loss is not None or hasattr(trainer, "best_eval_step"):
        payload["best"] = {
            "metric": os.environ.get("MOLMOBOT_BESTFIT_METRIC", "action_flow_loss"),
            "loss": best_loss,
            "step": getattr(trainer, "best_eval_step", None),
            "evals_since_improvement": getattr(trainer, "evals_since_improvement", None),
        }

    # Read off the optimizer rather than from the metrics: `LRMonitor` reports
    # nothing under some optimizers (the 8-bit one included), and the decayed
    # rate at the moment of the save is the number that explains the loss.
    try:
        rates = {}
        for group in trainer.optim.param_groups:
            name = group.get("group_name", "all")
            rate = _stretch4_finite(group.get("lr"))
            if rate is not None:
                rates.setdefault(name, rate)
        if rates:
            payload["learning_rates"] = rates
    except Exception:  # noqa: BLE001
        pass

    started = getattr(trainer, "_train_start_time", None)
    if started is not None:
        payload["elapsed_seconds"] = round(time.monotonic() - started, 1)
    return payload


def _stretch4_write_checkpoint_metrics(trainer, checkpoint_dir, kind):
    """Write `@@SUMMARY@@` into a checkpoint directory that has just been saved."""
    import json
    import os

    if int(os.environ.get("RANK", "0") or 0) != 0:
        return
    try:
        directory = str(checkpoint_dir)
        # Remote checkpoints (s3://, gs://) are written through a client this
        # has no handle on; a local directory is the case worth covering.
        if not os.path.isdir(directory):
            return
        payload = _stretch4_checkpoint_metrics(trainer, kind)
        with open(os.path.join(directory, "@@SUMMARY@@"), "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except Exception:  # noqa: BLE001 - a checkpoint that saved must not fail here
        pass


def _stretch4_summarising(original, kind):
    """Wrap a Trainer save method so the summary lands beside the weights.

    Wrapping the two save methods rather than editing inside them: the summary
    is then written after the save has returned -- past its barriers, with the
    directory certainly on disk -- and one insertion covers `step<N>/`,
    `step<N>_bestfit/` and the final checkpoint alike.
    """
    import functools

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # `save_checkpoint` returns (checkpoint_dir, unsharded_dir);
        # `save_bestfit_checkpoint` returns the directory itself.
        directory = result[0] if isinstance(result, tuple) else result
        _stretch4_write_checkpoint_metrics(self, directory, kind)
        return result

    return wrapper


Trainer.save_checkpoint = _stretch4_summarising(Trainer.save_checkpoint, "periodic")
if hasattr(Trainer, "save_bestfit_checkpoint"):
    Trainer.save_bestfit_checkpoint = _stretch4_summarising(
        Trainer.save_bestfit_checkpoint, "bestfit"
    )
'''

_METRICS_RECORDER = (
    _METRICS_RECORDER_TEMPLATE.replace("@@MARKER@@", GENERATED_MARKER)
    .replace("@@ENV@@", METRICS_ENV_VAR)
    .replace("@@SUMMARY@@", CHECKPOINT_METRICS_FILENAME)
)
"""
The recorder, appended at module level to MolmoBot's `trainer.py`.

Built by substitution rather than as an f-string because the block is mostly
JSON-shaped dict literals and docstrings; doubling every brace to survive
`str.format` would make it unreadable in the one place it has to stay readable.
"""


def missing_inference_requirements() -> list[str]:
    """Which of `INFERENCE_REQUIREMENTS` this interpreter cannot import.

    Returns the *installable* names, in the order they would go on a pip
    command line, so a caller can quote the fix rather than describe it.
    """
    import importlib.util

    missing = []
    for module, package in INFERENCE_REQUIREMENTS.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    return missing


def inference_requirements_message(missing: list[str]) -> str:
    """How to fix a missing inference requirement, as one paragraph."""
    return (
        "MolmoBot's model cannot be loaded in this environment: "
        + ", ".join(missing)
        + " missing. Evaluation runs MolmoSpaces and MolmoBot in one interpreter -- this "
        "repository's, not the training venv inside the checkout -- so MolmoBot's runtime "
        "dependencies have to be installed here:\n"
        "  uv pip install " + " ".join(missing)
    )


def ensure_metrics_log(checkout: MolmoBotCheckout) -> bool:
    """Teach MolmoBot to write its own metrics to a file, and into its checkpoints.

    MolmoBot computes everything a fine-tune needs to be tuned -- the flow
    loss, the per-group learning rates and gradient norms, throughput, and the
    validation loss from each evaluator -- and then throws most of it away
    unless Weights & Biases is configured: `log_metrics_to_console` filters out
    `optim/*` (there are too many for a terminal) and the console is the only
    other destination. `finetuning/README.md` explains why W&B is off by
    default here, and "turn on W&B" is a poor answer to "what learning rate
    should I use" anyway, because it does not work offline and leaves nothing on
    disk beside the checkpoint it describes.

    So one call is inserted at the top of that method, where the *unfiltered*
    metrics dict is still in hand, and a block at the end of the module does two
    things with it: writes a JSON line per dump to `METRICS_ENV_VAR`, which
    `training_report.py` reads, and keeps the latest dump of each kind so that
    every checkpoint the trainer saves gets a `CHECKPOINT_METRICS_FILENAME`
    written inside it. The second is the one that survives the run -- see that
    constant.

    Same terms as `ensure_stretch_presets` and `ensure_adamw8bit`: insertions in
    a gitignored clone, and inert -- the recorder returns immediately, the
    summary is still written -- when the environment variable is not set.

    Idempotent, and self-upgrading: a checkout carrying an older version of the
    block has it replaced rather than left alone, because the block is generated
    from this module and a stale copy is a bug that hides in a directory nobody
    reads.

    Returns:
        True if this call wrote to the file, False if it was already current.
    """
    path = checkout.package_dir / TRAINER_MODULE
    if not path.exists():
        raise MolmoBotSetupError(
            f"{path} not found, so this MolmoBot checkout does not keep its trainer where "
            "this expects. It may be a version whose layout has moved; re-clone, or run "
            "with WANDB=on and read the metrics there."
        )

    source = path.read_text()
    # Everything before the marker is MolmoBot's own file plus the one call
    # line; everything from it on is this module's, and is regenerated wholesale.
    body, marker, _ = source.partition(GENERATED_MARKER)
    if not marker:
        body = source

    if _METRICS_CALL not in body:
        signature = re.search(
            r"^[ \t]+def log_metrics_to_console\(self[^)]*\):[ \t]*\n", body, re.MULTILINE
        )
        if signature is None:
            raise MolmoBotSetupError(
                f"Could not find `def log_metrics_to_console` in {path}, so there is nowhere "
                "to record metrics from. MolmoBot's trainer has changed shape; run with "
                "WANDB=on instead, or add the call by hand."
            )
        body = body[: signature.end()] + _METRICS_CALL + body[signature.end() :]

    # The recorder goes at module level at the end of the file rather than
    # beside the method: names in a function body are resolved when it runs, so
    # ordering does not matter, and appending cannot disturb a line of the
    # trainer itself.
    desired = body.rstrip("\n") + "\n" + _METRICS_RECORDER
    if desired == source:
        return False

    _write_atomic(path, desired)
    log.info(
        f"[molmobot] metrics logging and checkpoint summaries "
        f"{'updated in' if marker else 'patched into'} {path}"
    )
    return True


def _write_atomic(path: Path, text: str) -> None:
    """Replace `path`'s contents in one step.

    `ensure_policy_factory_default` runs from `ensure_importable`, which runs in
    every rollout worker as well as the process that started them -- so on a
    fresh checkout several processes can decide to patch the same file at the
    same moment. A half-written module is a syntax error in a dependency, which
    is a bad afternoon; a rename is atomic, so the losers of that race overwrite
    with an identical file instead.
    """
    temporary = path.with_name(f"{path.name}.stretch4-{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _clone(root: Path) -> None:
    """Shallow-clone MolmoBot into `root`.

    `--depth 1` because nothing downstream reads the history and the repository
    carries its paper assets; the working tree is all that is wanted.
    """
    root.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"[molmobot] cloning {MOLMOBOT_GIT_URL} into {root}")
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", MOLMOBOT_GIT_URL, str(root)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise MolmoBotSetupError(
            f"git clone of {MOLMOBOT_GIT_URL} into {root} failed:\n{completed.stderr.strip()}"
        )


def _fetch_postprocessing_scripts(destination: Path) -> list[str]:
    """Download the postprocessing scripts that are not already in `destination`.

    Returns the names actually downloaded, so the caller can say so rather than
    reporting work it did not do.
    """
    destination.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    for name, url in POSTPROCESSING_SCRIPTS.items():
        path = destination / name
        if path.exists():
            continue
        log.info(f"[molmobot] downloading {name} from {HF_DATASET_REPO}")
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as e:
            raise MolmoBotSetupError(
                f"Could not download {name} from {url}: {e}\n"
                f"It is not in the git repository -- it ships with the {HF_DATASET_REPO} "
                f"dataset -- so fetch it by hand into {destination} and re-run."
            ) from e
        # Written whole rather than streamed: a half-written script that still
        # imports is worse than no script, and these are a few kilobytes.
        path.write_bytes(payload)
        fetched.append(name)
    return fetched
