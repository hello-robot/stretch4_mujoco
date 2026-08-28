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
import re
import subprocess
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
        The table names written to, empty if all of them were already there.
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
    log.info(f"[molmobot] registered {', '.join(names)} in {', '.join(written)} in {path}")
    return written


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
        raise MolmoBotSetupError(
            f"Could not find the `{name} = {{` table in {path}, so the Stretch presets "
            "cannot be registered. MolmoBot's preset file has changed shape; add them by "
            "hand, or pass --action-preset once a real one exists upstream."
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
