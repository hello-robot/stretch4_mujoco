"""
Start pose commands for the Stretch 4 runners, like `run_franka.StartPoseEditor`'s for the
Franka. They move the virtual Franka (its joints, its pedestal and its gripper), and Stretch
follows it through the retargeting, one `env.step()` at a time, as it follows the policy.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, ContextManager, Protocol

import click
import numpy as np

from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, GRIPPER_CLOSED


class Keys(Protocol):
    """One key press at a time (an arrow key is one), and a status line to show while jogging."""

    def read(self) -> str: ...

    def show(self, text: str) -> None: ...


class TerminalKeys:
    """`Keys` straight from the terminal, for runners that leave stdin alone between prompts."""

    def __enter__(self) -> TerminalKeys:
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # Ctrl+C still interrupts
        return self

    def __exit__(self, *exc) -> None:
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        print()

    def read(self) -> str:
        return os.read(self._fd, 8).decode(errors="ignore")

    def show(self, text: str) -> None:
        sys.stdout.write(f"\r\x1b[K{text}")
        sys.stdout.flush()


class StretchStartPoseEditor:
    """
    `handle_command()` takes the start pose commands (COMMANDS) and returns True for them, so
    it plugs into `interactive_session()` and the real robot's control loop.

    `env` is a `Stretch4SimEnv` or a `RealStretch4Env`. `go_to(q7)` moves Stretch to where the
    Franka at `q7` has its tool, and waits; `set_pedestal_height(m)` raises or lowers the virtual
    Franka to `fr3_link0` that far above the floor; `keys()` gives the `Keys` to jog with;
    `say()` prints a line; `on_move()` is called after Stretch moves, to log it.
    """

    COMMANDS = (
        "'jog' to move the virtual Franka and its pedestal with the arrow keys (Stretch follows), "
        "'home' to send it home, 'pose', 'set q1 ... q7' (rad), 'height <m>' (pedestal)"
    )
    PEDESTAL_RANGE = (0.1, 1.5)

    def __init__(
        self,
        env,
        go_to: Callable[[np.ndarray], None],
        set_pedestal_height: Callable[[float], None],
        keys: Callable[[], ContextManager[Keys]] = TerminalKeys,
        say: Callable[[str], None] = click.echo,
        on_move: Callable[[], None] | None = None,
    ):
        self.env = env
        self.go_to = go_to
        self._set_pedestal_height = set_pedestal_height
        self.keys = keys
        self.say = say
        self.on_move = on_move or (lambda: None)
        kinematics = env.retargeter.franka_kinematics
        self.lower, self.upper = kinematics.lower, kinematics.upper

    # -- the virtual Franka --------------------------------------------------

    @property
    def pedestal_height(self) -> float:
        return self.env.retargeter.franka.pedestal_height

    def franka_pose(self) -> np.ndarray:
        """The Franka's joints with its tool where Stretch's is now: what the policy would see."""
        return np.array(self.env.observe().state8[:7])

    def pose_text(self, q7=None) -> str:
        q7 = self.franka_pose() if q7 is None else q7
        return "set " + " ".join(f"{q:.4f}" for q in q7) + f"\nheight {self.pedestal_height:.3f}"

    def move_to(self, q7) -> None:
        """Stretch to the Franka at `q7`, the way it goes to the start pose."""
        try:
            self.go_to(np.clip(np.asarray(q7, dtype=float), self.lower, self.upper))
        except RuntimeError:  # the env's error names the start pose, which this need not be
            self.say("Stretch 4 cannot reach that Franka pose from here")
        self.on_move()

    def follow(self, q7, gripper_closed: bool) -> str:
        """One step of Stretch towards the Franka at `q7`, as for a policy action; how it went."""
        action = np.append(q7, GRIPPER_CLOSED if gripper_closed else 0.0)
        targets = self.env.step(action)
        self.on_move()
        if targets is None:
            return "out of Stretch's reach"
        return f"Stretch {targets.tool_error_m * 1000:.0f} mm off" if targets.clamped else "Stretch on it"

    def set_pedestal_height(self, height: float) -> float:
        height = float(np.clip(height, *self.PEDESTAL_RANGE))
        self._set_pedestal_height(height)
        return height

    # -- commands ------------------------------------------------------------

    def handle_command(self, text: str) -> bool:
        words = text.replace(",", " ").split()
        if not words:
            return False
        if words[0] == "pose" and len(words) == 1:
            self.say(self.pose_text())
        elif words[0] == "home" and len(words) == 1:
            self.move_to(FRANKA_HOME_QPOS)
        elif words[0] == "set" and len(words) == 8:
            try:
                q7 = [float(w) for w in words[1:]]
            except ValueError:
                return False
            self.move_to(q7)
            self.say(self.pose_text())
        elif words[0] == "height" and len(words) == 2:
            try:
                height = float(words[1])
            except ValueError:
                return False
            q7 = self.franka_pose()
            height = self.set_pedestal_height(height)
            self.say(f"height {height:.3f}: {self.follow(q7, bool(self.env._gripper_closed))}")
        elif words[0] == "jog" and len(words) == 1:
            self.jog()
        else:
            return False
        return True

    # -- keyboard jogging ----------------------------------------------------

    def jog(self) -> None:
        """Move one Franka joint, or the pedestal, at a time from the keys until Enter, q or Esc."""
        self.say(
            "1-7 or left/right: joint (8: pedestal)   up/down or +/-: move   "
            "[ ]: step size (deg, or cm for the pedestal)   g: gripper   h: home   Enter/q/Esc: done"
        )
        pedestal = 7  # the "joint" after the arm's 7
        joint, step = 0, 5.0
        target = self.franka_pose()
        closed = bool(self.env._gripper_closed)
        result = ""
        try:
            with self.keys() as keys:
                while True:
                    q_deg = "  ".join(
                        (f"[{math.degrees(q):7.1f}]" if i == joint else f" {math.degrees(q):7.1f} ")
                        for i, q in enumerate(target)
                    )
                    height = f"{self.pedestal_height:.2f} m"
                    height = f"[{height}]" if joint == pedestal else f" {height} "
                    name = "pedestal" if joint == pedestal else f"joint {joint + 1}"
                    keys.show(f"{name}  step {step:g}  target {q_deg}  pedestal {height}  {result}")

                    key = keys.read()
                    if key in ("", "\n", "\r", "q", "\x1b"):
                        break
                    if len(key) == 1 and key in "12345678":
                        joint = int(key) - 1
                        continue
                    if key == "\x1b[C":
                        joint = (joint + 1) % 8
                        continue
                    if key == "\x1b[D":
                        joint = (joint - 1) % 8
                        continue
                    if key == "]":
                        step = min(step * 2, 45.0)
                        continue
                    if key == "[":
                        step = max(step / 2, 0.25)
                        continue
                    if key in ("\x1b[A", "+", "=", "\x1b[B", "-", "_"):
                        sign = 1 if key in ("\x1b[A", "+", "=") else -1
                        if joint == pedestal:
                            self.set_pedestal_height(self.pedestal_height + sign * step / 100)
                        else:
                            target[joint] += sign * math.radians(step)
                    elif key == "h":
                        target = np.array(FRANKA_HOME_QPOS, dtype=float)
                    elif key == "g":
                        closed = not closed
                    else:
                        continue
                    target = np.clip(target, self.lower, self.upper)
                    result = self.follow(target, closed)
        except KeyboardInterrupt:
            pass
        self.say(self.pose_text(target))
