"""
Turn a finished sweep into the tables that say which parameters won.

`report.md` ranks every trial and lists what happened in each. That is the right
thing when there are seven trials and the wrong thing when there are 184: the
question stops being "what did this trial do" and becomes "what does this
*parameter* do, averaged over everything else that varied". This aggregates.

    python -m examples.machine_learning.molmospaces.retargetting.analyze
    python -m ...analyze --run-dir eval_output/retarget_params

It writes `analysis.md` beside the run's own files and prints the same thing, and
drops a tidy one-row-per-trial `analysis.csv` for anything else you want to ask.

Ranking, and why not by score
-----------------------------
`report.md` sorts by score, and over a large sweep that is misleading at the top.
The score gives a failed episode up to 1.0 of partial credit for approaching,
touching and nearly lifting (see `scoring.py`), so a trial that misses by a
millimetre on two objects can outscore one that actually picks up three. Measured
over this sweep: trials with 2 of 4 picked ran as high as 0.985, while trials
with 3 of 4 picked started at 0.864 -- the ranges overlap.

So everything here ranks by **(objects picked, then score)**: the count is what
the benchmark is actually asking, and the score breaks its ties. The score is
still the right thing for a *search* to climb, because it is continuous and the
count is not; it is the wrong thing to read a winner off.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

RUN_DIR = Path("eval_output") / "retarget_params"

OBJECT_COLUMNS = ("bowl", "potato", "salt_shaker", "knife")


def load_many(run_dirs: tuple[Path, ...]) -> pd.DataFrame:
    """Several runs pooled into one table, later runs winning.

    A sweep is expensive enough that when one setup's data is invalidated -- a
    mount measured wrong, say -- the cheap fix is to re-run that setup alone and
    pool the two directories rather than spend another nine hours on the rest.
    Where a setup appears in more than one run the *last* one given wins
    outright, so the order of `--run-dir` is the order of precedence and a
    correction cannot be silently averaged with the thing it corrects.
    """
    frames = []
    for run_dir in run_dirs:
        frame = load(run_dir)
        frame["run"] = run_dir.name
        frames.append(frame)

    # The last run that contains a setup owns every row for it.
    owner = {
        setup: run_dir.name
        for run_dir, frame in zip(run_dirs, frames)
        for setup in frame.setup.unique()
    }
    pooled = pd.concat(frames, ignore_index=True)
    kept = pooled[pooled.run == pooled.setup.map(owner)].reset_index(drop=True)

    dropped = len(pooled) - len(kept)
    if dropped:
        superseded = sorted({s for s, r in owner.items() if (pooled.setup == s).sum() > (kept.setup == s).sum()})
        print(f"[analyze] {dropped} superseded trials dropped for: {', '.join(superseded)}")
    return kept


def load(run_dir: Path) -> pd.DataFrame:
    """One row per trial: its parameters, its score, and which objects it picked up.

    Joined from the two files the search writes, by position: `trials.jsonl`
    carries the parameters as structured values (the CSV has only the
    human-readable description) and `episodes.csv` carries the per-object
    outcome. Both are written in the same order, four episodes per trial, by the
    same function -- `params_search._write_outputs` rewrites all of them together
    after every trial, so they cannot be out of step with each other.
    """
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines()]
    summary = pd.read_csv(run_dir / "trials.csv")
    episodes = pd.read_csv(run_dir / "episodes.csv")

    if len(summary) != len(trials):
        raise ValueError(
            f"{run_dir} has {len(trials)} trials in trials.jsonl and {len(summary)} in "
            "trials.csv; they are written together, so one of them is from another run."
        )

    rows = []
    for index, (trial, (_, row)) in enumerate(zip(trials, summary.iterrows())):
        params = trial["params"]["params"]
        exo = params["exo"]
        # Episodes are grouped four to a trial, in the benchmark's own order.
        window = episodes.iloc[index * 4 : (index + 1) * 4]
        picked = {f"picked_{r.target}": bool(r.success) for r in window.itertuples()}
        rows.append(
            {
                "setup": trial["setup"],
                "robot": "stretch" if trial["setup"].startswith("stretch") else "franka",
                "lens": trial["setup"].split("_", 1)[1],
                # The sweep names its trials "<stage>_<index>_<dims>"; the stage
                # is the only part worth grouping on.
                "stage": "gripper" if "__gripper" in str(row["output_dir"]) else "camera",
                "score": float(trial["score"]),
                "picked": int(row["successes"]),
                "pitch_deg": exo["pitch_deg"],
                "fovy": exo["fovy"],
                "grasp_offset_m": params["grasp_offset_m"],
                "wrist_tilt_deg": params["wrist_tilt_deg"],
                "z_offset_fraction": params["z_offset_fraction"],
                "error": str(row["error"]) if isinstance(row["error"], str) else "",
                **picked,
            }
        )
    return pd.DataFrame(rows)


def best_per_setup(df: pd.DataFrame) -> pd.DataFrame:
    """The winning trial for each setup, ranked by objects picked then score."""
    ranked = df[df.error == ""].sort_values(["picked", "score"], ascending=False)
    return ranked.groupby("setup", sort=False).head(1).sort_values(
        ["picked", "score"], ascending=False
    )


def _markdown(frame: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """A markdown table, without taking a dependency on `tabulate`."""
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda v: floatfmt.format(v))
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    rule = "|" + "|".join("---" for _ in formatted.columns) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |" for row in formatted.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def _pivot_markdown(frame: pd.DataFrame, label: str, fmt: str = "{:.2f}") -> str:
    """A pivot table as markdown, with its index column named."""
    out = frame.reset_index()
    out.columns = [label if i == 0 else str(c) for i, c in enumerate(out.columns)]
    return _markdown(out, floatfmt=fmt)


def analyse(df: pd.DataFrame) -> str:
    """Every table, as one markdown document."""
    usable = df[df.error == ""]
    camera = usable[usable.stage == "camera"]
    gripper = usable[usable.stage == "gripper"]
    object_cols = [f"picked_{name}" for name in OBJECT_COLUMNS]

    parts: list[str] = [
        "# Sweep analysis",
        "",
        f"{len(df)} trials, {len(df) * 4} rollouts"
        + (f", {int((df.error != '').sum())} errored" if (df.error != "").any() else "")
        + ".",
        "",
        "Ranked by **objects picked, then score** — see this module's docstring for why "
        "not by score alone.",
        "",
        "## Best parameters per setup",
        "",
        _markdown(
            best_per_setup(usable)[
                [
                    "setup",
                    "picked",
                    "score",
                    "pitch_deg",
                    "fovy",
                    "grasp_offset_m",
                    "z_offset_fraction",
                    "wrist_tilt_deg",
                ]
            ]
        ),
        "",
        "## Which objects are winnable",
        "",
        "Success rate per object over every trial in the sweep — this is a property of "
        "the objects, not of any setting.",
        "",
        _markdown(
            pd.DataFrame(
                {
                    "object": list(OBJECT_COLUMNS),
                    "picked in % of trials": [usable[c].mean() * 100 for c in object_cols],
                }
            ),
            floatfmt="{:.1f}",
        ),
        "",
        "## Camera: pitch × field of view",
        "",
        "Mean objects picked, per robot. Both cameras sit 1.5432 m above the floor "
        "(`setups.FRANKA_STRETCHCAM_HEIGHT` derives the Franka offset so they match), so "
        "a difference between these two tables is the robot under the camera, not the "
        "viewpoint.",
        "",
    ]

    for robot, group in camera.groupby("robot"):
        parts += [
            f"**{robot}**",
            "",
            _pivot_markdown(
                group.pivot_table(index="pitch_deg", columns="fovy", values="picked", aggfunc="mean"),
                "pitch_deg",
            ),
            "",
        ]

    parts += [
        "## Field of view interacts with the lens",
        "",
        "Mean score by `fovy`, per lens. A pinhole wants a narrow field and a rectified "
        "fisheye wants a wide one — rectification has nothing to straighten if the "
        "source was not wide to begin with.",
        "",
        _pivot_markdown(
            camera.pivot_table(index="lens", columns="fovy", values="score", aggfunc="mean"),
            "lens",
            fmt="{:.3f}",
        ),
        "",
    ]

    if not gripper.empty:
        parts += [
            "## Gripper: grasp offset × height offset",
            "",
            "Mean objects picked, over the Stretch setups. These two interact — at "
            "`grasp_offset_m = 0` the height offset changes nothing, because the depth "
            "error is already losing every grasp.",
            "",
            _pivot_markdown(
                gripper.pivot_table(
                    index="grasp_offset_m",
                    columns="z_offset_fraction",
                    values="picked",
                    aggfunc="mean",
                ),
                "grasp_offset_m",
            ),
            "",
            "## Gripper: wrist tilt",
            "",
            _markdown(
                gripper.groupby("wrist_tilt_deg")[["picked", "score"]]
                .mean()
                .reset_index()
                .rename(columns={"picked": "mean picked", "score": "mean score"}),
            ),
            "",
        ]

    return "\n".join(parts) + "\n"


@click.command()
@click.option(
    "--run-dir",
    "run_dirs",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    help="A directory `params_search` wrote: trials.jsonl, trials.csv, episodes.csv. "
    "Repeatable — later directories supersede earlier ones for any setup they share, "
    "so a re-run of one setup can be pooled with the sweep it corrects.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where analysis.md and analysis.csv go. Defaults to the last --run-dir.",
)
def main(run_dirs: tuple[Path, ...], out_dir: Path | None) -> None:
    dirs = run_dirs or (RUN_DIR,)
    frame = load_many(dirs)
    document = analyse(frame)

    destination = out_dir or dirs[-1]
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "analysis.csv", index=False)
    (destination / "analysis.md").write_text(document)
    click.echo(document)
    if len(dirs) > 1:
        click.secho(
            "Pooled " + ", ".join(d.name for d in dirs) + " (later wins per setup).", fg="yellow"
        )
    click.secho(f"Wrote {destination / 'analysis.md'} and {destination / 'analysis.csv'}", fg="green")


if __name__ == "__main__":
    main()
