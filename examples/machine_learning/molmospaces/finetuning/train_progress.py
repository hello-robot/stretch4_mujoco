"""
A progress bar for a MolmoBot fine-tune, drawn from the trainer's own log lines.

MolmoBot already knows how far along it is: `Trainer.fit` emits a header of the
form `[step=120/30000, eta=2 hours, 5 minutes]` every `console_log_interval`
steps, computed by `Trainer.get_eta()`. What it does not do is make that easy to
find -- the header is the first line of a multi-line metrics dump, so on a long
run the one number you want scrolls past between screenfuls of loss terms.

This filter sits on the end of the training pipeline in the generated
`run_<trainer>.sh`. It passes every line through untouched, so the log is
exactly what it would have been, and draws a bar each time a step header goes
by:

    [##############------------------] 43.2%  step 12960/30000
    2.41 it/s  elapsed 1:29:38  eta 1 hour, 58 minutes  (trainer's own estimate)

Two estimates appear because they answer different questions. The trainer's
`eta=` is measured from the start of *its* run and is the one to trust for a
finish time. The rate here is measured between the headers this process has
actually seen, so it reflects the last stretch rather than the average -- which
is what tells you a run has slowed down.

Deliberately stdlib-only and deliberately append-only: no cursor movement, no
carriage returns, no terminal detection. The output of a training run is
something you scroll back through, pipe to a file and paste into a bug report,
and a bar that repaints in place turns all three into a mess of escape codes.
One tidy block every log interval is legible in a terminal and in a log file.
"""

from __future__ import annotations

import re
import sys
import time

STEP_HEADER = re.compile(r"\[step=(\d+)/(\d+)(?:,\s*eta=([^\]]*))?\]")
"""
`Trainer.fit`'s console header, as it appears inside a log line.

Searched rather than matched: the logging handler puts a timestamp, level and
module in front of it, and may wrap the whole line in ANSI colour.
"""

BAR_WIDTH = 32


def render_bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """`0.43` -> `##############------------------`."""
    fraction = min(max(fraction, 0.0), 1.0)
    filled = round(fraction * width)
    return "#" * filled + "-" * (width - filled)


def format_elapsed(seconds: float) -> str:
    """Seconds -> `H:MM:SS`, which sorts and subtracts more easily than prose."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def progress_block(
    step: int,
    total: int,
    trainer_eta: str,
    rate: float | None,
    elapsed: float,
) -> str:
    """The two-line block drawn under each step header."""
    fraction = step / total if total else 0.0
    rate_text = f"{rate:.2f} it/s" if rate else "-- it/s"
    eta_text = f"{trainer_eta}  (trainer's own estimate)" if trainer_eta else "unknown"
    return (
        f"\n  [{render_bar(fraction)}] {100 * fraction:5.1f}%  step {step:,}/{total:,}\n"
        f"  {rate_text}  elapsed {format_elapsed(elapsed)}  eta {eta_text}\n"
    )


def main() -> int:
    """Stream stdin to stdout, drawing a bar at every step header.

    Reads with `readline` rather than iterating the file object: iteration does
    read-ahead buffering, which on a pipe holds lines back until a block fills
    and would make the trainer look hung for minutes at a time.
    """
    first_step: int | None = None
    started = time.monotonic()

    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        sys.stdout.write(line)

        match = STEP_HEADER.search(line)
        if match is None:
            sys.stdout.flush()
            continue

        step, total = int(match.group(1)), int(match.group(2))
        trainer_eta = (match.group(3) or "").strip()

        # The first header is the baseline: the run has already spent time on
        # the checkpoint load and the normalisation statistics before step 1, and
        # counting that would report a rate the run never sustains.
        if first_step is None:
            first_step, started = step, time.monotonic()

        elapsed = time.monotonic() - started
        steps_done = step - first_step
        rate = steps_done / elapsed if elapsed > 0 and steps_done > 0 else None

        sys.stdout.write(progress_block(step, total, trainer_eta, rate, elapsed))
        sys.stdout.flush()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BrokenPipeError, KeyboardInterrupt):
        # The trainer died or the user interrupted; the shell reports the real
        # exit status through `set -o pipefail`, so say nothing and get out of
        # the way rather than stacking a traceback on top of the actual error.
        raise SystemExit(0) from None
