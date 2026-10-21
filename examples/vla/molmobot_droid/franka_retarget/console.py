"""
The terminal control loop of the Stretch 4 runners (`run_stretch4_real.py`, `run_stretch4_sim.py`):
type an instruction and press Enter, and the robot works on it until told otherwise (KEYS_HELP).
"""

from __future__ import annotations

import contextlib
import os
import queue
import select
import sys
import threading
from typing import Callable

import click

from examples.vla.molmobot_droid.franka_retarget.start_pose_editor import StretchStartPoseEditor
from examples.vla.molmobot_droid.rollout import StepInfo, run_rollout

KEYS_HELP = (
    "  Type an instruction + Enter   run it (typing another while one runs switches to it)\n"
    "  Enter or Space                stop the robot where it is\n"
    "  'home' or 'reset' + Enter       go back to the start pose (the real robot raises its lift first)\n"
    "  'jog' + Enter                   move the virtual Franka's joints and pedestal with the keys\n"
    "  'set q1 ... q7' + Enter         go to the Franka at those joints (rad)\n"
    "  'pose' / 'height <m>' + Enter   print the Franka's joints and pedestal / set the pedestal\n"
    "  'offset x y z [r yw p]' + Enter set the grasp offset (mm, deg); 'offset' prints it\n"
    "  'quit' + Enter, or Ctrl+C       stop and exit"
)


class Console:
    """
    The terminal while the robot runs: a status line at the bottom with what is being typed
    after it, events printed above it, and commands queued for the control loop as they are
    typed -- ("stop", None) the moment Enter or Space is pressed on an empty line, ("text",
    line) when a line is entered. Falls back to plain lines when stdin is not a terminal.

    Inside `keys()`, every key press goes to `read()` instead (the `Keys` jogging takes).
    """

    PROMPT = click.style("› ", fg="cyan", bold=True)

    def __init__(self):
        self.queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._keys: queue.Queue[str] | None = None
        self._lock = threading.Lock()
        self._status = ""
        self._buffer = ""
        self._terminal = None
        if sys.stdin.isatty():
            import termios
            import tty

            self._terminal = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)  # keys arrive one at a time; Ctrl+C still interrupts
        threading.Thread(target=self._read, daemon=True).start()
        self._render()

    # -- output -------------------------------------------------------------

    def _render(self) -> None:
        if self._terminal is None:
            return
        status = f"{self._status}  " if self._status else ""
        sys.stdout.write(f"\r\x1b[2K{status}{self.PROMPT}{self._buffer}")
        sys.stdout.flush()

    def event(self, message: str, color: str | None = None) -> None:
        """A line above the status line."""
        with self._lock:
            if self._terminal is not None:
                sys.stdout.write("\r\x1b[2K")
            click.secho(message, fg=color)
            self._render()

    def status(self, text: str) -> None:
        with self._lock:
            self._status = text
            self._render()

    # -- input --------------------------------------------------------------

    def _read(self) -> None:
        if self._terminal is None:
            for line in sys.stdin:
                text = line.strip()
                self.queue.put(("text", text) if text else ("stop", None))
            self.queue.put(("text", "quit"))
            return
        while True:
            # Woken now and then to see whether close() was called, so the thread ends with it.
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                if self._terminal is None:
                    return
                continue
            # A key press at a time: an arrow key's escape sequence arrives in one read.
            chunk = os.read(sys.stdin.fileno(), 8).decode(errors="ignore")
            with self._lock:
                if self._terminal is None:
                    return
                if not chunk:
                    self.queue.put(("text", "quit"))
                    return
                if self._keys is not None:
                    self._keys.put(chunk)
                    continue
                for char in chunk:
                    if char in ("\n", "\r") or (char == " " and not self._buffer):
                        text, self._buffer = self._buffer.strip(), ""
                        self.queue.put(("text", text) if text else ("stop", None))
                    elif char in ("\x7f", "\b"):
                        self._buffer = self._buffer[:-1]
                    elif char.isprintable():
                        self._buffer += char
                self._render()

    def pending(self) -> bool:
        return not self.queue.empty()

    def get(self) -> tuple[str, str | None]:
        return self.queue.get()

    @contextlib.contextmanager
    def keys(self):
        """Every key press to `read()` until left, and `show()` on the status line."""
        if self._terminal is None:
            raise RuntimeError("Jogging needs a terminal")
        with self._lock:
            self._keys = queue.Queue()
        try:
            yield self
        finally:
            with self._lock:
                self._keys = None
            self.status("")

    def read(self) -> str:
        return self._keys.get()

    def show(self, text: str) -> None:
        self.status(text)

    def close(self) -> None:
        if self._terminal is not None:
            import termios

            with self._lock:
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._terminal)
                self._terminal = None


def start_pose_command(editor: StretchStartPoseEditor, text: str, console: Console) -> bool:
    """Run `text` if it is a start pose command (the editor's), and say whether it was."""
    try:
        return editor.handle_command(text)
    except RuntimeError as error:  # e.g. jogging without a terminal
        console.event(f"◆ {error}", "red")
        return True
    finally:
        console.status("")


def continuous_session(
    env,
    policy,
    params,
    max_steps: int,
    on_step,
    console: Console,
    editor: StretchStartPoseEditor,
    reset: Callable[[], None],
) -> None:
    """
    The control loop: runs the current instruction until a key says otherwise (KEYS_HELP),
    indefinitely unless `max_steps` > 0, or until `on_step` returns True (a success check, say).
    The start pose commands go to `editor`; `reset` goes back to the start pose.

    `env` is a `RealStretch4Env` or a `Stretch4SimEnv`, with `stop_motion()`, `last_miss` and
    `tool_error_m()`; its `interrupted` is pointed at the keys, and its `say` at the console.

    Every action Stretch cannot follow is printed, and the first of a run of twisted ones.
    """
    env.interrupted = console.pending
    env.say = console.event
    command = console.get()
    while True:
        kind, text = command
        if kind == "stop":
            env.stop_motion()
            console.status("")
            console.event("■ stopped", "yellow")
        elif text in ("quit", "exit", "q"):
            return
        elif text == "reset":
            console.status(click.style("◆ going to the start pose", fg="yellow"))
            try:
                reset()
                console.event("◆ start pose" if not console.pending() else "◆ start pose (interrupted)", "yellow")
            except RuntimeError as error:
                console.event(f"◆ {error}", "red")
            editor.on_move()  # the ghost and Rerun follow
            console.status("")
        elif start_pose_command(editor, text, console):
            pass
        else:
            console.event(f"▶ {text}", "green")
            twisting = False

            def step(info: StepInfo) -> bool | None:
                nonlocal twisting
                done = on_step(info)
                miss = env.last_miss
                if miss is not None and (miss.kind == "unreachable" or not twisting):
                    color = "red" if miss.kind == "unreachable" else "yellow"
                    console.event(f"  step {info.step}: {miss.describe()}", color)
                twisting = miss is not None and miss.kind == "twisted"
                error = env.tool_error_m()
                retargeter = env.retargeter
                console.status(
                    click.style(f"▶ {text}", fg="green")
                    + click.style(
                        f"  step {info.step}"
                        + (f" · tool {error * 1000:.0f} mm off" if error is not None else "")
                        + (f" · {retargeter.ik_twisted} twisted" if retargeter.ik_twisted else "")
                        + (f" · {retargeter.ik_failures} unreachable" if retargeter.ik_failures else "")
                        + (f" · {env.recoveries} recovered" if getattr(env, "recoveries", 0) else "")
                        + (f" · {env.stale_observations} stale" if env.stale_observations else "")
                        + "  · Enter/Space stops",
                        dim=True,
                    )
                )
                return done

            result = run_rollout(
                env, policy, text, params.execute_horizon, params.execute_first_n,
                max_steps if max_steps > 0 else 10**12, step, should_stop=console.pending,
            )
            if not console.pending():
                env.stop_motion()
                console.status("")
                console.event(f"■ done after {result.steps} steps", "yellow")
        command = console.get()
