#!/usr/bin/env python
"""
Drive the MolmoBot RBY1 policy in MolmoSpaces with instructions you type.

`launch_scripts/run_eval.py` scores a checkpoint headlessly: it replays every
episode in a JSON benchmark against that episode's own recorded instruction and
prints a success rate. This does the opposite -- it loads *one* benchmark episode
(so the scene, object placement, robot pose and cameras are the real evaluation
setup) and then hands you a prompt, so you can tell the policy whatever you like
and watch what it does.

    # RBY1 (Rainbow Robotics RB-Y1) mobile manipulator, door+open policy
    third_party/MolmoBot/MolmoBot/.venv/bin/python \\
        examples/machine_learning/molmobot/rby1_interactive.py \\
        --checkpoint third_party/MolmoBot/MolmoBot/ckpts/molmobot/MolmoBot-RBY1Multitask \\
        --benchmark ~/.cache/molmo-spaces-resources/benchmarks/molmospaces-bench-v2/20260415/procthor-10k/rby1_benchmarks/door_opening_benchmark

    # one-shot, no prompt -- useful for scripting
    ... --instruction "open the door" --steps 200 --once

    # watch it live in a MuJoCo window instead of reading the mp4 afterwards
    ... --viewer

By default this runs headless: MuJoCo renders offscreen through EGL and each
rollout is written to an mp4. `--viewer` opens a passive MuJoCo window so you can
watch the robot move and orbit the camera while the policy drives; it needs a
display, and it still writes the mp4.

Run it with the MolmoBot venv interpreter, not the project one: MolmoBot pins its
own molmospaces commit into `third_party/MolmoBot/MolmoBot/.venv` (Python 3.11),
which is a different revision from the molmospaces this repo installs.
`setup_rby1_sim_eval.sh` creates that venv and fetches the checkpoint.

Prompt commands
---------------
Anything not starting with ':' is an instruction for the policy.

    :go [N]        keep running the current instruction for N more steps
    :steps N       set how many steps an instruction runs for (default 150)
    :objects       what is in this scene, i.e. what you can ask it to pick
    :find TEXT     find episodes whose own goal mentions TEXT, e.g. :find bowl
    :episode N     load a different benchmark episode
    :list [N]      list N episodes around the current one
    :reset         restart the episode (scene is restored, policy state cleared)
    :status        current episode, instruction, step count
    :quit

Choosing what to pick
---------------------
An episode fixes two things: which house is loaded, and where the robot starts.
It does not limit what you can ask for. The house is fully furnished -- a kitchen
episode carries ninety-odd movable objects -- and the episode's own target is
just one of them, so `:objects` lists the rest and any of those names can go in
an instruction. What the episode does constrain is reachability: the robot is
posed in front of its own target, so objects near that spot are the fair asks and
anything across the room needs the policy to drive the base there first.

To start from a scene built around the object you have in mind instead, search
the benchmark for it -- `:find bowl` -- and `:episode N` into a match.

How the instruction reaches the model
-------------------------------------
In simulation the policy reads its goal from `task.get_task_description()`, so
each instruction is installed on the task object the same way MolmoSpaces itself
overrides descriptions (`types.MethodType`). Changing the instruction also resets
the policy: it buffers an action chunk (16 predicted, 8 executed) and, for
door+open, caches a conditioning image and point prompt from the first frame it
sees -- all of which have to be recaptured for the new goal to take effect.

Success reporting
-----------------
`judge_success()` scores the episode's *original* goal, not what you typed. It is
shown because it is a useful signal when you type something close to the recorded
task, and it is meaningless when you type something else.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from pathlib import Path

# MuJoCo picks its renderer when it is imported, which happens deep inside the
# first molmo_spaces import -- long before argparse runs. So the choice is made
# from argv here.
#
# The two variables are not interchangeable. MolmoSpaces renders camera
# observations through its own EGL context (`renderer/opengl_context.py` imports
# `mujoco.egl` and refuses to load unless PYOPENGL_PLATFORM is egl or unset), so
# that one stays egl either way. MUJOCO_GL only picks the backend for MuJoCo's
# own contexts, including the passive viewer's window -- so --viewer sets it to
# glfw and the offscreen camera rendering carries on through EGL beside it.
_WANT_VIEWER = "--viewer" in sys.argv
os.environ.setdefault("MUJOCO_GL", "glfw" if _WANT_VIEWER else "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep MolmoSpaces off the GPU

import numpy as np

# Eval configs, and the policy class each one names in `policy_config.policy_cls`.
# door_plus_open and pick_pnp are the two released RBY1 multitask checkpoints;
# `door` is the door-opening-only config (an alias for MolmoBotRBY1EvalConfig).
EVAL_CONFIGS = {
    "door_plus_open": "MolmoBotRBY1DoorPlusOpenEvalConfig",
    "pick_pnp": "MolmoBotRBY1PickPnPEvalConfig",
    "door": "MolmoBotRBY1DoorEvalConfig",
}


# Fields `SynthVLAPolicy` reads off its policy config that only the Franka config
# (`SynthVLAPolicyConfig`) declares. `None` is the right value for both on RBY1:
# the rescaling that reads `relative_max_joint_delta` is patched out, and
# `states_mode=None` tells `SynthManipMolmoInferenceWrapper` to keep whatever the
# checkpoint was trained with -- which is what the released real-robot RBY1 policy
# does, since it constructs the wrapper with neither argument.
FRANKA_ONLY_POLICY_FIELDS = {"relative_max_joint_delta": None, "states_mode": None}

_PATCHED = False


def patch_released_rby1_policy() -> None:
    """Work around three bugs that stop the released RBY1 policy running in simulation.

    `MolmoBotRBY1MultitaskPolicy` inherits `SynthVLAPolicy`, which was written for
    the single-armed Franka and assumes things the RBY1 configs do not provide:

    1. It reads `policy_config.relative_max_joint_delta` (in `__init__`) and
       `policy_config.states_mode` (in `prepare_model`). Both are declared on
       `SynthVLAPolicyConfig` (Franka) but not on `SynthVLARBY1PolicyConfig`,
       so constructing any RBY1 policy raises `AttributeError`.

    2. `_populate_action_buffer` reads `self.clamp_gripper`, which no `__init__`
       in the RBY1 policy chain ever sets, so the first inference raises
       `AttributeError: ... has no attribute 'clamp_gripper'`.

    3. `SynthVLAPolicy.inference_model` rescales per-step joint deltas through
       `action["arm"]` whenever `action_type == "joint_pos_rel"` -- which is what
       the RBY1 configs set. RBY1 has no `"arm"` move group (its groups are
       `base`, `left_arm`, `left_gripper`, `right_arm`, `right_gripper`, `torso`),
       so the first step would raise `KeyError: 'arm'`.

    These are patched at runtime rather than in the checkout, so
    `third_party/MolmoBot` stays a clean clone. `launch_scripts/run_eval.py`
    builds its policy the same way and hits the same bugs, so it cannot run the
    RBY1 configs as shipped -- `run_eval_rby1.py` beside this file applies this
    same patch and then calls the identical `run_evaluation`.

    Dropping the rescaling is what the released real-robot RBY1 policy already
    does: `olmo/eval/real_robot_molmobot_rby1_door.py`, the standalone policy the
    websocket server serves to a physical RB-Y1, returns the buffered action
    unmodified. This override is that same body.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from olmo.eval.configure_molmo_spaces import (
        MolmoBotRBY1DoorOpeningPolicy,
        SynthVLARBY1PolicyConfig,
    )

    # Pydantic raises AttributeError from __getattr__ for fields a model does not
    # declare, and refuses assignment of undeclared fields -- so the defaults are
    # supplied by intercepting that lookup. Patching the base config class covers
    # every RBY1 config that derives from it, including ones built inside
    # molmo_spaces' own run_evaluation() where there is no instance to reach.
    original_getattr = SynthVLARBY1PolicyConfig.__getattr__

    def __getattr__(self, item):
        if item in FRANKA_ONLY_POLICY_FIELDS:
            return FRANKA_ONLY_POLICY_FIELDS[item]
        return original_getattr(self, item)

    SynthVLARBY1PolicyConfig.__getattr__ = __getattr__

    original_init = MolmoBotRBY1DoorOpeningPolicy.__init__

    def __init__(self, config, task_type):
        original_init(self, config, task_type)
        # `_populate_action_buffer` reads self.clamp_gripper, which nothing in the
        # sim policy chain sets. Read it off the config, where it is declared --
        # door+open leaves it True, pick+pnp sets it False.
        self.clamp_gripper = config.policy_config.clamp_gripper

    def inference_model(self, model_input):
        obs = model_input[0] if isinstance(model_input, list) else model_input
        self.obs_history.append(obs)
        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(model_input)
        action = self.action_buffer[self.buffer_index]
        self.buffer_index += 1
        self.step_count += 1
        return action

    # Both also cover MolmoBotRBY1MultitaskPolicy, which subclasses this.
    MolmoBotRBY1DoorOpeningPolicy.__init__ = __init__
    MolmoBotRBY1DoorOpeningPolicy.inference_model = inference_model


log = logging.getLogger("rby1_interactive")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("Prompt commands")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="RBY1 checkpoint directory")
    p.add_argument("--benchmark", type=Path, required=True,
                   help="benchmark directory holding benchmark.json")
    p.add_argument("--task-type", choices=sorted(EVAL_CONFIGS), default="door_plus_open",
                   help="which released RBY1 policy config to use")
    p.add_argument("--episode", type=int, default=0, help="benchmark episode index to load")
    p.add_argument("--steps", type=int, default=150,
                   help="policy steps run per instruction (10 Hz, so 150 ~ 15s of robot time)")
    p.add_argument("--task-horizon", type=int, default=5000,
                   help="episode step budget; kept high so an interactive session is not cut short")
    p.add_argument("--viewer", action="store_true",
                   help="open a MuJoCo passive viewer and watch the robot live "
                        "(needs a display; switches the renderer from EGL to GLFW)")
    p.add_argument("--video-dir", type=Path, default=Path("rby1_rollouts"),
                   help="where rollout mp4s are written ('none' to disable)")
    p.add_argument("--instruction", type=str, default=None,
                   help="run this instruction immediately on startup")
    p.add_argument("--once", action="store_true",
                   help="with --instruction, run it and exit instead of prompting")
    p.add_argument("--list-episodes", action="store_true",
                   help="print the benchmark's episodes and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="show MolmoSpaces/policy logs")
    return p.parse_args(argv)


def scene_objects(spec) -> list[tuple[str, int]]:
    """Object types present in an episode's scene, most numerous first.

    A benchmark episode does not build a bare scene around its target: the target
    is an object the ProcTHOR house already contains, and `object_poses` places
    every movable object in that house -- ~90 of them for a typical kitchen. Their
    keys are `<synset>_<asset hash>_<instance>`, so the leading segment is the
    word you would actually use in an instruction.
    """
    from collections import Counter

    poses = getattr(spec.scene_modifications, "object_poses", None) or {}
    counts = Counter(name.split("_", 1)[0] for name in poses)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def episode_target(spec) -> str:
    """The synset of the episode's own pickup object, or '' if it has none."""
    name = spec.task.get("pickup_obj_name") or ""
    return name.split("_", 1)[0]


def describe(spec, idx: int) -> str:
    """One-line summary of a benchmark episode."""
    task_cls = spec.task.get("task_cls", "?").rsplit(".", 1)[-1]
    desc = getattr(getattr(spec, "language", None), "task_description", None) or "-"
    return f"[{idx:4d}] house {spec.house_index:<5} {task_cls:<22} {desc}"


class Session:
    """One loaded episode: scene + task + policy, ready to be stepped."""

    def __init__(self, args: argparse.Namespace):
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
        from olmo.eval import configure_molmo_spaces as cms

        self.args = args
        self.episodes = load_all_episodes(args.benchmark)
        if not self.episodes:
            raise SystemExit(f"no episodes found in {args.benchmark}")
        log.info("loaded %d episodes from %s", len(self.episodes), args.benchmark)

        self.config_cls = getattr(cms, EVAL_CONFIGS[args.task_type])
        self.policy = None
        self.sampler = None
        self.task = None
        self.viewer = None
        self.instruction = ""
        self.episode_idx = -1
        self.load_episode(args.episode)

    # ------------------------------------------------------------ episode --
    def load_episode(self, idx: int) -> None:
        """Build the scene and task for a benchmark episode.

        The eval config is rebuilt per episode because JsonEvalTaskSampler mutates
        it in place -- it pins house_inds, camera_config, task_type, scene_dataset
        and data_split to the episode being loaded.
        """
        from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

        if not 0 <= idx < len(self.episodes):
            raise IndexError(f"episode {idx} out of range (0..{len(self.episodes) - 1})")

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.sampler is not None:
            self.sampler.close()
            self.sampler = None
            self.task = None

        spec = self.episodes[idx]
        config = self.config_cls()
        config.policy_config.checkpoint_path = str(self.args.checkpoint)
        config.task_horizon = self.args.task_horizon
        config.seed = spec.seed if spec.seed is not None else idx

        print(f"\nloading {describe(spec, idx)}")
        self.sampler = JsonEvalTaskSampler(config, spec)
        self.task = self.sampler.sample_task(house_index=spec.house_index)
        if self.task is None:
            raise SystemExit(f"task sampler returned no task for episode {idx}")

        # The model is loaded once and carried across episodes -- it is ~6GB and
        # takes far longer to load than a scene does.
        if self.policy is None:
            print(f"loading policy from {self.args.checkpoint} ...")
            self.policy = config.policy_config.policy_cls(config, task_type=config.task_type)
        else:
            self.policy.config = config

        self.task.register_policy(self.policy)
        self.open_viewer()
        self.episode_idx = idx
        self.recorded_task = getattr(getattr(spec, "language", None), "task_description", "") or ""
        self.set_instruction(self.instruction or self.recorded_task)
        self.obs, _info = self.task.reset()  # also calls policy.reset()

    def open_viewer(self) -> None:
        """Attach a passive viewer to the freshly built scene, if --viewer was given.

        The viewer holds a reference to one specific MjModel/MjData pair, so it has
        to be reopened whenever a new scene is compiled -- reusing it across
        episodes would leave it rendering the previous house.
        """
        if not self.args.viewer:
            return
        import mujoco.viewer

        env = self.task.env
        data = env.mj_datas[env.current_batch_index]
        self.viewer = mujoco.viewer.launch_passive(data.model, data)
        self.viewer.opt.sitegroup[0] = False  # hide control sites, as eval does
        self.viewer.sync()

    def sync_viewer(self) -> None:
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close(self) -> None:
        """Tear down in the right order: viewer, then the env that owns the model.

        `launch_passive` runs its window on another thread holding a reference to
        this scene's MjModel/MjData. Letting the interpreter exit without closing
        it first lets the env free the model out from under that thread, which
        ends the process with a GLFW error and a core dump instead of a clean
        exit.
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.sampler is not None:
            self.sampler.close()
            self.sampler = None
            self.task = None

    def reset(self) -> None:
        """Restart the episode. Re-sampling restores object poses and robot pose."""
        spec = self.episodes[self.episode_idx]
        self.task = self.sampler.sample_task(house_index=spec.house_index)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.open_viewer()
        self.task.register_policy(self.policy)
        self.set_instruction(self.instruction)
        self.obs, _info = self.task.reset()  # also calls policy.reset()

    # -------------------------------------------------------- instruction --
    def set_instruction(self, instruction: str) -> None:
        """Install `instruction` as the task's goal and clear cached policy state."""
        self.instruction = instruction

        def get_task_description(_self, _text=instruction) -> str:
            return _text

        self.task.get_task_description = types.MethodType(get_task_description, self.task)
        if self.policy is not None:
            # Drops the action chunk, and the conditioning image/points that
            # door+open captured for the previous goal.
            self.policy.reset()

    # ------------------------------------------------------------ rollout --
    def run(self, steps: int) -> None:
        frames: list[np.ndarray] = []
        print(f'running "{self.instruction}" for {steps} steps ...')
        for i in range(steps):
            frames.append(self._frame(self.obs))
            action = self.policy.get_action(self.obs)
            if action is None:
                print("  policy returned no action, stopping")
                break
            self.obs, _reward, _term, _trunc, infos = self.task.step(action)
            self.sync_viewer()
            if self.task.is_done():
                print(f"  episode ended at step {i + 1} (horizon reached or terminal)")
                break
            if (i + 1) % 25 == 0:
                print(f"  step {i + 1}/{steps}")

        print(f"stopped after {self.task.num_steps_taken()} total steps in this episode")
        try:
            success = bool(self.task.judge_success())
            print(f'  judge_success() for the episode\'s recorded goal '
                  f'("{self.recorded_task}"): {success}')
        except Exception as exc:
            print(f"  judge_success() unavailable: {type(exc).__name__}: {exc}")
        self._write_video(frames)

    def _frame(self, obs) -> np.ndarray:
        """Head camera on top, the two wrist cameras side by side underneath."""
        o = obs[0] if isinstance(obs, list) else obs
        head = np.asarray(o["head_camera"])[..., :3]
        wrists = [np.asarray(o[c])[..., :3] for c in ("wrist_camera_l", "wrist_camera_r")
                  if c in o]
        if not wrists:
            return head
        half = head.shape[1] // 2
        row = np.hstack([_resize(w, half, head.shape[0] // 2) for w in wrists])
        return np.vstack([head, row])

    def _write_video(self, frames: list[np.ndarray]) -> None:
        if not frames or str(self.args.video_dir).lower() == "none":
            return
        import imageio.v2 as imageio

        self.args.video_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in self.instruction)[:50] or "rollout"
        path = self.args.video_dir / f"ep{self.episode_idx:04d}_{slug}.mp4"
        # policy_dt_ms is 100 for the RBY1 configs, i.e. the policy runs at 10 Hz.
        imageio.mimwrite(path, frames, fps=10, macro_block_size=1)
        print(f"  wrote {path} ({len(frames)} frames)")

    def status(self) -> str:
        spec = self.episodes[self.episode_idx]
        return (f"episode {self.episode_idx} (house {spec.house_index}), "
                f"step {self.task.num_steps_taken()}\n"
                f'  your instruction: "{self.instruction}"\n'
                f'  episode\'s recorded goal: "{self.recorded_task}"')


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize, so the video needs no extra dependency."""
    ys = (np.arange(height) * img.shape[0] // height).clip(0, img.shape[0] - 1)
    xs = (np.arange(width) * img.shape[1] // width).clip(0, img.shape[1] - 1)
    return img[ys][:, xs]


def repl(session: Session) -> None:
    steps = session.args.steps
    print("\nType an instruction, or :help for commands. Ctrl-D to quit.\n")
    while True:
        try:
            line = input("molmobot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        if not line.startswith(":"):
            session.set_instruction(line)
            session.run(steps)
            continue

        cmd, _, rest = line[1:].partition(" ")
        rest = rest.strip()
        try:
            if cmd in ("q", "quit", "exit"):
                return
            elif cmd in ("h", "help"):
                print(__doc__.split("Prompt commands\n---------------\n")[1]
                      .split("How the instruction")[0])
            elif cmd == "go":
                session.run(int(rest) if rest else steps)
            elif cmd == "steps":
                steps = int(rest)
                print(f"steps per instruction: {steps}")
            elif cmd == "reset":
                session.reset()
                print("episode reset")
            elif cmd == "episode":
                session.load_episode(int(rest))
            elif cmd == "list":
                n = int(rest) if rest else 10
                lo = max(0, session.episode_idx - n // 2)
                for i in range(lo, min(len(session.episodes), lo + n)):
                    mark = "*" if i == session.episode_idx else " "
                    print(mark + describe(session.episodes[i], i))
            elif cmd == "objects":
                spec = session.episodes[session.episode_idx]
                target = episode_target(spec)
                objects = scene_objects(spec)
                print(f"{sum(n for _, n in objects)} movable objects in this scene "
                      f"-- any of them can be named in an instruction:")
                for name, count in objects:
                    mark = "  <- this episode's target" if name == target else ""
                    print(f"  {name:<24} x{count}{mark}")
                print("The robot starts posed at its own target, so things near it "
                      "are the realistic asks; anything further needs the base to drive.")
            elif cmd == "find":
                if not rest:
                    print("usage: :find <text>   e.g. :find bowl")
                else:
                    hits = [(i, sp) for i, sp in enumerate(session.episodes)
                            if rest.lower() in (getattr(getattr(sp, "language", None),
                                                        "task_description", "") or "").lower()]
                    print(f"{len(hits)} of {len(session.episodes)} episodes match "
                          f"'{rest}'" + (" (first 15):" if len(hits) > 15 else ":"))
                    for i, sp in hits[:15]:
                        print(describe(sp, i))
                    if hits:
                        print(f"load one with :episode {hits[0][0]}")
            elif cmd == "status":
                print(session.status())
            else:
                print(f"unknown command :{cmd} -- try :help")
        except Exception as exc:  # keep the session alive on a bad command
            print(f"error: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log.setLevel(logging.INFO)

    if not args.checkpoint.is_dir():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if not (args.benchmark / "benchmark.json").is_file():
        raise SystemExit(f"no benchmark.json in {args.benchmark}")

    patch_released_rby1_policy()

    if args.list_episodes:
        from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

        for i, spec in enumerate(load_all_episodes(args.benchmark)):
            print(describe(spec, i))
        return 0

    session = Session(args)
    try:
        if args.instruction:
            session.set_instruction(args.instruction)
            session.run(args.steps)
            if args.once:
                return 0
        repl(session)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
