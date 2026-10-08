"""
The policy loop shared by every runner: query MolmoBot-DROID, keep the first
`--execute-horizon` actions of the chunk, execute the first
`--execute-horizon-do-only-first-n-steps` of those, and query again.

Any env with `observe() -> droid.Observation` and `step(action8)` works: `droid.FrankaDroidEnv`
and `stretch4_retarget.Stretch4SimEnv` here, the real robot in `run_stretch4_real.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from examples.vla.molmobot_droid.checkpoint import MolmoBotDroidPolicy, limit_joint_delta
from examples.vla.molmobot_droid.droid import Observation


class RobotEnv(Protocol):
    def observe(self) -> Observation: ...

    def step(self, action8: np.ndarray): ...


@dataclass
class StepInfo:
    step: int
    """Executed actions so far, this one included."""
    query: int
    action: np.ndarray
    """The action executed (after joint-delta limiting)."""
    observation: Observation
    """What the robot looks like after it."""
    step_result: object = None
    """Whatever the env's `step()` returned (Stretch's retargeted joint targets)."""


@dataclass
class RolloutResult:
    steps: int = 0
    queries: int = 0
    stopped_by_callback: bool = False
    inference_seconds: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0


def run_rollout(
    env: RobotEnv,
    policy: MolmoBotDroidPolicy,
    instruction: str,
    execute_horizon: int,
    execute_first_n: int,
    max_steps: int,
    on_step: Callable[[StepInfo], bool | None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RolloutResult:
    """
    Run one episode. `on_step` is called after every executed action and ends the episode by
    returning True (a success check, say); `should_stop` is polled too (Ctrl+C handling).
    """
    result = RolloutResult()
    start = time.perf_counter()
    policy.reset()
    observation = env.observe()
    policy.add_observation(observation.exo_rgb, observation.wrist_rgb)

    while result.steps < max_steps:
        if should_stop is not None and should_stop():
            break
        tic = time.perf_counter()
        chunk = policy.predict_chunk(observation.state8, instruction)
        result.inference_seconds.append(time.perf_counter() - tic)
        result.queries += 1

        for action in chunk[:execute_horizon][:execute_first_n]:
            action = action.copy()
            # As MolmoBot's RealRobotVLAPolicy: limit each step against the state it starts from.
            action[:7] = limit_joint_delta(action[:7], observation.state8[:7])
            step_result = env.step(action)
            observation = env.observe()
            policy.add_observation(observation.exo_rgb, observation.wrist_rgb)
            result.steps += 1
            info = StepInfo(result.steps, result.queries, action, observation, step_result)
            if on_step is not None and on_step(info):
                result.stopped_by_callback = True
                result.wall_seconds = time.perf_counter() - start
                return result
            if result.steps >= max_steps or (should_stop is not None and should_stop()):
                break

    result.wall_seconds = time.perf_counter() - start
    return result


class StopFlag:
    """Ctrl+C ends the running rollout instead of the program; a second one quits."""

    def __init__(self):
        self.stop_requested = False

    def install(self) -> None:
        import signal

        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:
        if self.stop_requested:
            raise KeyboardInterrupt
        print("\nStopping after this step (Ctrl+C again to quit)...")
        self.stop_requested = True

    def __call__(self) -> bool:
        return self.stop_requested

    def clear(self) -> None:
        self.stop_requested = False


def interactive_session(
    env: RobotEnv,
    policy: MolmoBotDroidPolicy,
    default_instruction: str,
    execute_horizon: int,
    execute_first_n: int,
    max_steps: int,
    reset: Callable[[], None],
    on_step: Callable[[StepInfo], bool | None] | None = None,
) -> None:
    """
    Prompt for instructions and run each until `max_steps`, `on_step` returns True, or Ctrl+C.
    `reset` puts the robot back at its start pose ("reset" at the prompt); "quit" exits.
    """
    stop = StopFlag()
    stop.install()
    print(
        "\nType an instruction and press Enter (empty for "
        f"\"{default_instruction}\"), 'reset' to restart the robot, or 'quit'."
    )
    while True:
        try:
            text = input("instruction> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text in ("quit", "exit", "q"):
            return
        if text == "reset":
            reset()
            continue
        instruction = text or default_instruction
        stop.clear()
        result = run_rollout(
            env, policy, instruction, execute_horizon, execute_first_n, max_steps, on_step, should_stop=stop
        )
        inference = np.mean(result.inference_seconds) if result.inference_seconds else 0.0
        print(
            f"{result.steps} steps, {result.queries} queries ({inference:.2f} s each), "
            f"{result.wall_seconds:.0f} s" + (" -- stopped by success check" if result.stopped_by_callback else "")
        )
