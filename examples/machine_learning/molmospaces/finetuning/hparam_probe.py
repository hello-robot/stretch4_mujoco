"""
Short runs that answer a hyperparameter question, written as one script.

The honest way to choose a learning rate for a fine-tune is to try a few and
look at the validation loss. Everything else -- rules of thumb, what MolmoBot's
README used for DROID, what worked on the last dataset -- is a starting point
for the first try, and this repository already has those as the defaults in the
generated `run_molmobot.sh`. What it did not have is a cheap way to run the
comparison.

    python -m examples.machine_learning.molmospaces.finetuning.hparam_probe \\
        --script data/stretch_pick/rollouts/molmobot/pick/run_molmobot.sh \\
        --values 3e-5,1e-4,3e-4 --steps 600

That writes `run_probe.sh` beside the runs it will produce: three fine-tunes of
600 steps each, identical but for `ACTION_EXPERT_LR`, each into its own save
folder with its own `metrics.jsonl`, followed by `training_report.py --compare`
over all three. Nothing is launched from here, for the same reason `finetune.py`
launches nothing: the script is a thing to read before it spends hours of GPU.

It drives the *existing* generated script rather than reimplementing the
trainer command, so a probe cannot drift from the real run -- it is the real
run, with `MAX_STEPS` shortened and one variable changed. Three of that script's
switches make this cheap:

- `PREPARE=off` after the first run skips the venv sync, the trajectory index
  and the statistics pass, all of which depend only on the data.
- `STATS_PATH` is shared, so the normalisation statistics are computed once.
- `ASSUME_YES=1` answers the preflight questions, since a probe deliberately
  runs far fewer steps than a real fine-tune and would otherwise be asked about
  it every time.

**What a short probe can and cannot tell you.** `MAX_STEPS` is the
learning-rate horizon as well as the stopping point, so each probe run is a
complete, miniature schedule -- warmup, cosine decay and all -- rather than the
first tenth of a long one. That makes the runs comparable to each other. It does
not make them comparable to a full fine-tune: a rate that wins over 600 steps is
often a little high for 10,000, because the long run has time to make progress
the short one has to rush. Read the ordering, not the absolute numbers, and
prefer the lower of two rates that tie.

**Before any of this, check that the model can fit the data at all.** A probe
compares learning rates; it cannot tell you that the trajectories are
mislabelled. The check for that is to overfit something tiny on purpose --
built out of the data you already have, four houses of it:

    mkdir -p data/stretch_overfit/rollouts/tiny
    for h in $(ls data/stretch_pick/rollouts/pick | grep '^house_' | head -4); do
        ln -sfn "$PWD/data/stretch_pick/rollouts/pick/$h" \\
            "data/stretch_overfit/rollouts/tiny/$h"
    done
    python -m examples.machine_learning.molmospaces.finetuning.finetune \\
        --rollouts data/stretch_overfit/rollouts/tiny --trainer molmobot \\
        --cameras "head_camera_right,wrist_camera_right" \\
        --steps 300 --batch-size 8 --val-fraction 0.25
    bash data/stretch_overfit/rollouts/molmobot/tiny/run_molmobot.sh

Sixteen or so demonstrations, three hundred steps: the training loss should fall
to nearly nothing, because the model is being asked to memorise them. If it does
not, no learning rate will help -- the problem is in the data, the action spec or
the camera names, and `training_report.py` will say which of those it looks
like.

Four houses rather than the `--task debug` dataset because a split needs
somewhere to hold a house out: `arrange_train_val_split` holds out none when
there is only one, and MolmoBot's dataloader raises on a `val/` with no
trajectory index in it.
"""

from __future__ import annotations

import shlex
import stat
import sys
from pathlib import Path

import click

DEFAULT_VARIABLE = "ACTION_EXPERT_LR"
"""
What a probe varies unless told otherwise.

The action expert is the part being trained -- it is unfrozen in every
`TRAINABLE` tier and carries the highest learning rate of the three -- so it is
both the most consequential number and the one with the least prior art behind
it: MolmoBot's default was chosen for its own data, not for Stretch's.
"""

DEFAULT_VALUES: dict[str, str] = {
    "ACTION_EXPERT_LR": "3e-5,1e-4,3e-4",
    "VIT_LR": "1e-6,5e-6,2e-5",
    "LLM_LR": "3e-6,1e-5,3e-5",
    "TRAINABLE": "action_expert,vision",
    "GLOBAL_BATCH": "16,32,64",
    "SEQ_LEN": "528,1024",
}
"""
A default grid per variable: half and roughly three times the shipped default.

Wide enough that a wrong default shows up as a clear ordering rather than
noise, and short enough to finish. Learning rates go in thirds of a decade
because that is the resolution at which a fine-tune's validation loss actually
differs; anything finer measures the seed.
"""

PROBE_SCRIPT_NAME = "run_probe.sh"


def probe_script(
    script: Path,
    variable: str,
    values: list[str],
    steps: int,
    eval_interval: int,
    output_dir: Path,
    report: Path,
    python: str = "python",
) -> str:
    """The sweep script: one run of `script` per value, then a comparison."""
    runs = [(_slug(variable, value), value) for value in values]
    shared_stats = output_dir / "synthmanip_norm_stats.yaml"

    lines = [
        "#!/usr/bin/env bash",
        "#",
        f"# {len(runs)} short fine-tunes of {steps} steps, varying {variable}.",
        "#",
        "# Generated by examples/machine_learning/molmospaces/finetuning/hparam_probe.py",
        f"# from {script}",
        "#",
        "# Each run is that script with MAX_STEPS shortened and one variable changed, so",
        "# what is measured here is the real training path and not an approximation of",
        "# it. MAX_STEPS is also the learning-rate horizon, so each run is a complete",
        "# short schedule rather than a truncated long one -- which makes these",
        "# comparable to each other, but not to a full-length run. Read the ordering.",
        "set -euo pipefail",
        "",
        f"SCRIPT={shlex.quote(str(script.resolve()))}",
        f"PROBE_ROOT={shlex.quote(str(output_dir.resolve()))}",
        f"REPORT={shlex.quote(str(report))}",
        f"PYTHON={shlex.quote(python)}",
        "",
        "# Shared across the runs: the data preparation and the normalisation",
        "# statistics depend on the dataset, which no probe varies.",
        f"SHARED_STATS={shlex.quote(str(shared_stats.resolve()))}",
        "",
        f'STEPS="${{STEPS:-{steps}}}"',
        f'EVAL_INTERVAL="${{EVAL_INTERVAL:-{eval_interval}}}"',
        "",
        'mkdir -p "$PROBE_ROOT"',
        "",
    ]

    for index, (name, value) in enumerate(runs):
        lines += [
            "# --------------------------------------------------------------------------",
            f"# {index + 1}/{len(runs)}: {variable}={value}",
            "# --------------------------------------------------------------------------",
            f'echo "=== {variable}={value} ({index + 1}/{len(runs)}) ==="',
            f'SAVE_FOLDER="$PROBE_ROOT"/{name} \\',
            f'METRICS="$PROBE_ROOT"/{name}/metrics.jsonl \\',
            '    STATS_PATH="$SHARED_STATS" \\',
            f"    PREPARE={'on' if index == 0 else 'off'} \\",
            '    MAX_STEPS="$STEPS" \\',
            '    EVAL_INTERVAL="$EVAL_INTERVAL" \\',
            "    ASSUME_YES=1 \\",
            f"    {variable}={shlex.quote(value)} \\",
            '    bash "$SCRIPT"',
            "",
        ]

    lines += [
        "# --------------------------------------------------------------------------",
        "# The comparison. Sorted by best validation loss; a run whose best is at its",
        "# last step has not finished improving, so its row understates it.",
        "# --------------------------------------------------------------------------",
        '"$PYTHON" "$REPORT" '
        + " ".join(f'"$PROBE_ROOT"/{name}' for name, _ in runs)
        + " --compare",
        '"$PYTHON" "$REPORT" '
        + " ".join(f'"$PROBE_ROOT"/{name}' for name, _ in runs)
        + ' --plot "$PROBE_ROOT"/probe.png || true',
        "",
    ]
    return "\n".join(lines) + "\n"


def _slug(variable: str, value: str) -> str:
    """`ACTION_EXPERT_LR`, `3e-5` -> `action_expert_lr_3e-5`, safe as a directory name."""
    cleaned = "".join(character if character.isalnum() else "-" for character in value)
    return f"{variable.lower()}_{cleaned}"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--script",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The run_molmobot.sh a probe drives. Written by finetune.py --trainer molmobot.",
)
@click.option(
    "--vary",
    "variable",
    default=DEFAULT_VARIABLE,
    help="Which of the generated script's variables to sweep. Any of its "
    "${NAME:-default} knobs works; the defaults below cover the usual ones.",
)
@click.option(
    "--values",
    default=None,
    help="Comma-separated values to try. Defaults to a grid around the shipped default "
    "for the known variables.",
)
@click.option(
    "--steps",
    type=int,
    default=600,
    help="Steps per run. Also the learning-rate horizon, so each run is a complete short "
    "schedule.",
)
@click.option(
    "--eval-interval",
    type=int,
    default=100,
    help="Steps between validation passes. Short runs need a short interval or they "
    "produce two points and nothing to compare.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where the probe runs go. Defaults to a `probe/` beside the script's own runs.",
)
def main(
    script: Path,
    variable: str,
    values: str | None,
    steps: int,
    eval_interval: int,
    output_dir: Path | None,
) -> None:
    """Write a sweep script over one of the generated run script's variables."""
    from examples.machine_learning.molmospaces.finetuning.finetune import TRAINING_REPORT

    variable = variable.upper()
    raw = values or DEFAULT_VALUES.get(variable)
    if not raw:
        raise click.UsageError(
            f"No default grid for {variable}, so --values is required. It is passed to "
            f"{script.name} as an environment variable, so any value that script accepts "
            "works."
        )
    parsed = [value.strip() for value in raw.split(",") if value.strip()]
    if len(parsed) < 2:
        raise click.UsageError("A probe needs at least two values to compare.")

    if not _mentions(script, variable):
        raise click.UsageError(
            f"{script} has no ${{{variable}:-...}} knob, so setting it in the environment "
            "would do nothing. Check the tuning block at the top of that script for the "
            "names it honours."
        )
    if not _mentions(script, "METRICS"):
        # Without the metrics file there is nothing to compare, so the probe
        # would spend hours of GPU and produce a scrollback.
        raise click.UsageError(
            f"{script} predates the metrics recording this compares runs with -- it has no "
            "${METRICS:-...} knob. Regenerate it:\n"
            "  python -m examples.machine_learning.molmospaces.finetuning.finetune "
            "--rollouts <run> --trainer molmobot\n"
            "(with the same arguments as before; it rewrites the script, not the data or "
            "the checkpoints)."
        )
    for knob, cost in (
        ("PREPARE", "the venv sync, trajectory index and statistics pass run again for "
         "every value"),
        ("STATS_PATH", "the normalisation statistics are recomputed for every value"),
    ):
        if not _mentions(script, knob):
            click.secho(
                f"Note: {script.name} has no ${{{knob}:-...}}, so {cost}. The probe still "
                "works; regenerating the script makes it faster.",
                fg="yellow",
            )

    output_dir = output_dir or script.parent.parent / "probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROBE_SCRIPT_NAME
    path.write_text(
        probe_script(
            script=script,
            variable=variable,
            values=parsed,
            steps=steps,
            eval_interval=eval_interval,
            output_dir=output_dir,
            report=TRAINING_REPORT,
            # This interpreter, not `python`: the report is read from a shell
            # that may have no virtualenv active, and matplotlib is here.
            python=sys.executable,
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    click.echo("")
    click.secho(f"Wrote {path}", fg="green")
    click.echo("")
    click.echo(
        f"  {len(parsed)} runs x {steps} steps, varying {variable} over "
        f"{', '.join(parsed)}\n"
        f"  into {output_dir}, compared with training_report.py when they finish."
    )
    click.echo("")
    click.secho("Read it, then run it:", bold=True)
    click.echo(f"  bash {path}")
    click.echo("")
    click.echo(
        "If the model has not been shown to fit a tiny dataset yet, do that first --\n"
        "a sweep cannot tell a bad learning rate from mislabelled data. The recipe is\n"
        "in this module's docstring: generate --task debug, fine-tune for 300 steps,\n"
        "and watch the training loss go to nearly zero."
    )


def _mentions(script: Path, variable: str) -> bool:
    """Whether `script` reads `variable` from the environment.

    Matched on the `${NAME:-` expansion rather than the whole assignment,
    because the generated script quotes the right-hand side (`NAME="${NAME:-x}"`)
    and a future one may not.
    """
    return f"${{{variable}:-" in script.read_text(errors="replace")


if __name__ == "__main__":
    main()
