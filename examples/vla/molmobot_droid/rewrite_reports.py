"""
Write benchmark runs' `report.md` (and `results.json`) again from their `results.json`, without
rerunning anything: e.g. after the report gains a score or a column.

Usage:
    python -m examples.vla.molmobot_droid.rewrite_reports outputs/molmobot_droid/<run> [<run> ...]
    # Every run in a directory:
    python -m examples.vla.molmobot_droid.rewrite_reports --all outputs/molmobot_droid
"""

from __future__ import annotations

from pathlib import Path

import click

from examples.vla.molmobot_droid.molmospaces.benchmark import rewrite_report


@click.command()
@click.argument("run_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option("--all", "all_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Every run in this directory's subfolders.")
def main(run_dirs, all_dir):
    dirs = [Path(d) for d in run_dirs]
    if all_dir:
        dirs += [results.parent for results in sorted(Path(all_dir).glob("*/results.json"))]
    if not dirs:
        raise click.UsageError("Give run directories, or --all")
    for run_dir in dirs:
        click.echo(f"wrote {rewrite_report(run_dir)}")


if __name__ == "__main__":
    main()
