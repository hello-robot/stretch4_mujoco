"""
Helpers for running the MolmoSpaces benchmarks with MolmoBot-DROID, on the Franka DROID
(`droid.FrankaDroidEnv`) or on Stretch 4 (`stretch4_retarget.Stretch4SimEnv`). The runners
are `run_benchmark_franka.py` and `run_benchmark_stretch4.py`.

Episodes are molmospaces' own benchmark JSON (`EpisodeSpec`): the house, object poses, the
robot's base pose and starting arm, and the instruction. Scenes are rebuilt from them with
`custom_scene`, and success uses molmospaces' criteria (see `PickSuccess`), checked through
`env.body_pose()` / `env.body_contacts()` so that it works for either robot.

Every episode writes one video per camera, the scene camera, and a grid of all of them;
the run writes `report.md` and `results.json`. File names start with `run_name()`: the robot
and the flags, joined with underscores.
"""

from __future__ import annotations

import json
import math
import shlex
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from examples.vla.molmobot_droid.checkpoint import HF_REPO, HF_REVISION, MOLMOBOT_COMMIT, POLICY_DT, POLICY_HZ
from examples.vla.molmobot_droid.droid import FRANKA_PEDESTAL_HEIGHT, Observation, RobotPose, quat_to_mat
from examples.vla.molmobot_droid.molmospaces.custom_scene import CustomScene, resolve_scene_xml
from examples.vla.molmobot_droid.rollout import StepInfo, run_rollout

DEFAULT_BENCHMARK = "molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench"


# ---------------------------------------------------------------------------
# Benchmarks and episodes
# ---------------------------------------------------------------------------


def benchmarks_root() -> Path:
    from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR

    return Path(DATA_CACHE_DIR) / "benchmarks"


def resolve_benchmark_dir(benchmark: str) -> Path:
    """
    A benchmark directory (the one holding `benchmark.json`), from a path or a name of the form
    "<suite>/<dataset>/<Benchmark>", e.g. "molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench".
    The suite's newest installed version and the benchmark's newest build are used.
    """
    path = Path(benchmark).expanduser()
    if (path / "benchmark.json").exists():
        return path
    suite, dataset, name = benchmark.split("/")
    suite_dir = benchmarks_root() / suite
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"{suite_dir} not found; install it with molmospaces' resource manager")
    for version in sorted(suite_dir.iterdir(), reverse=True):
        builds = sorted((version / dataset / name).glob("*/benchmark.json"), reverse=True)
        if builds:
            return builds[0].parent
    raise FileNotFoundError(f"No {dataset}/{name} under {suite_dir}")


def list_benchmarks() -> list[str]:
    names = set()
    for found in benchmarks_root().glob("*/*/*/*/*/benchmark.json"):
        suite, _, dataset, name = found.parts[-6:-2]
        names.add(f"{suite}/{dataset}/{name}")
    return sorted(names)


def load_episodes(benchmark: str, episodes: list[int] | None = None, max_episodes: int | None = None):
    """The benchmark's `EpisodeSpec`s, optionally only some indices or the first N."""
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

    benchmark_dir = resolve_benchmark_dir(benchmark)
    all_episodes = load_all_episodes(benchmark_dir)
    indices = episodes if episodes is not None else list(range(len(all_episodes)))
    if max_episodes is not None:
        indices = indices[:max_episodes]
    return benchmark_dir, [(i, all_episodes[i]) for i in indices]


@dataclass
class EpisodeSetup:
    scene: CustomScene
    robot_pose: RobotPose
    link0_height: float
    """`fr3_link0` z: the episode's mocap base z plus molmospaces' 0.58 m pedestal."""
    franka_init_qpos: list[float]
    instruction: str
    task_type: str
    task: dict
    horizon_steps: int


def episode_setup(episode) -> EpisodeSetup:
    """Everything needed to rebuild an episode, from its `EpisodeSpec`."""
    task = episode.task
    base = task["robot_base_pose"]  # x, y, z, qw, qx, qy, qz
    rotation = quat_to_mat(base[3:7])
    object_name = task["pickup_obj_name"]
    object_pose = episode.scene_modifications.object_poses.get(object_name) or task.get("pickup_obj_start_pose")
    scene = CustomScene(
        scene_id=f"{episode.scene_dataset}/{episode.data_split}/{episode.house_index}",
        scene_xml=resolve_scene_xml(f"{episode.scene_dataset}/{episode.data_split}/{episode.house_index}"),
        object_name=object_name,
        object_pos=np.asarray(object_pose[:3], dtype=float),
        object_quat=np.asarray(object_pose[3:7], dtype=float),
        object_poses=dict(episode.scene_modifications.object_poses),
        added_objects=dict(episode.scene_modifications.added_objects),
        removed_objects=list(episode.scene_modifications.removed_objects),
    )
    return EpisodeSetup(
        scene=scene,
        robot_pose=RobotPose(base[0], base[1], math.atan2(rotation[1, 0], rotation[0, 0])),
        link0_height=base[2] + FRANKA_PEDESTAL_HEIGHT,
        franka_init_qpos=list(episode.robot.init_qpos.get("arm", [])),
        instruction=episode.language.task_description,
        task_type=task.get("task_type", "pick"),
        task=task,
        horizon_steps=round(task.get("task_horizon_sec", 20) * POLICY_HZ),
    )


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class BenchmarkEnv(Protocol):
    """What the benchmark needs from a robot env, on top of `rollout.RobotEnv`."""

    robot_root: str
    """Root body name of the robot, as contacts report it."""

    def observe(self) -> Observation: ...

    def step(self, action8: np.ndarray): ...

    def render_scene(self) -> np.ndarray | None: ...

    def body_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]: ...

    def body_contacts(self, name: str) -> list[str]: ...

    def close(self) -> None: ...


class SuccessJudge(Protocol):
    bodies: list[str]
    """Bodies whose poses and contacts it reads (what Stretch's simulator must watch)."""

    def __call__(self, env: BenchmarkEnv) -> tuple[bool, dict[str, float]]: ...


@dataclass
class PickSuccess:
    """
    molmospaces `PickTask.judge_success()`: lifted at least `threshold` above its start and
    touching nothing but the robot.
    """

    object_name: str
    start_z: float
    threshold: float = 0.01

    @property
    def bodies(self) -> list[str]:
        return [self.object_name]

    def __call__(self, env: BenchmarkEnv) -> tuple[bool, dict[str, float]]:
        position, _ = env.body_pose(self.object_name)
        lift = float(position[2] - self.start_z)
        contacts = env.body_contacts(self.object_name)
        touching_robot = env.robot_root in contacts
        only_robot = touching_robot and all(c == env.robot_root for c in contacts)
        return only_robot and lift >= self.threshold, {"lift_m": lift, "held": float(only_robot)}


@dataclass
class PickAndPlaceSuccess:
    """
    molmospaces `PickAndPlaceTask`, simplified to what poses and contacts show: the object
    rests on the receptacle, the robot has let go, and the receptacle moved at most
    `max_receptacle_displacement`.
    """

    object_name: str
    receptacle_name: str
    receptacle_start: np.ndarray
    max_receptacle_displacement: float = 0.15

    @property
    def bodies(self) -> list[str]:
        return [self.object_name, self.receptacle_name]

    def __call__(self, env: BenchmarkEnv) -> tuple[bool, dict[str, float]]:
        contacts = env.body_contacts(self.object_name)
        receptacle_pos, _ = env.body_pose(self.receptacle_name)
        displacement = float(np.linalg.norm(receptacle_pos - self.receptacle_start))
        on_receptacle = self.receptacle_name in contacts
        released = env.robot_root not in contacts
        success = on_receptacle and released and displacement <= self.max_receptacle_displacement
        return success, {"on_receptacle": float(on_receptacle), "receptacle_moved_m": displacement}


def make_judge(setup: EpisodeSetup) -> SuccessJudge:
    task = setup.task
    task_type = setup.task_type
    if task_type == "pick":
        return PickSuccess(
            setup.scene.object_name, float(setup.scene.object_pos[2]), task.get("succ_pos_threshold", 0.01)
        )
    # "next to" is judged on distance to the receptacle rather than resting on it.
    if task_type in ("pick_and_place", "pick_and_place_color") and "place_receptacle_name" in task:
        receptacle = task["place_receptacle_name"]
        start = task.get("place_receptacle_start_pose") or setup.scene.object_poses[receptacle]
        return PickAndPlaceSuccess(
            setup.scene.object_name,
            receptacle,
            np.asarray(start[:3], dtype=float),
            task.get("max_place_receptacle_pos_displacement", 0.15),
        )
    # Opening/closing is judged on an articulation's joint, which Stretch's simulator does
    # not report. Add it here when needed.
    raise NotImplementedError(f"No success check for task type '{task_type}' yet")


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """
    Streams one mp4 per camera, plus a grid of all of them, at the policy rate. Frames are
    written as they come, so an episode never holds its video in memory.
    """

    GRID_TILE_HEIGHT = 360

    def __init__(self, out_dir: Path, prefix: str, fps: float = POLICY_HZ):
        self.out_dir = out_dir
        self.prefix = prefix
        self.fps = fps
        self._writers: dict[str, object] = {}
        self._grid_size: tuple[int, int] | None = None
        out_dir.mkdir(parents=True, exist_ok=True)

    def path(self, camera: str) -> Path:
        return self.out_dir / f"{self.prefix}_{camera}.mp4"

    def _writer(self, camera: str):
        import imageio.v2 as imageio

        if camera not in self._writers:
            self._writers[camera] = imageio.get_writer(
                self.path(camera), fps=self.fps, codec="libx264", quality=7, macro_block_size=8
            )
        return self._writers[camera]

    def add(self, frames: dict[str, np.ndarray]) -> None:
        frames = {name: frame for name, frame in frames.items() if frame is not None}
        for name, frame in frames.items():
            self._writer(name).append_data(_pad_to_block(frame))
        self._writer("grid").append_data(self._grid(frames))

    def _grid(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        height = self.GRID_TILE_HEIGHT
        tiles = []
        for name, frame in frames.items():
            tile = cv2.resize(frame, (max(2, round(frame.shape[1] * height / frame.shape[0])), height))
            cv2.putText(tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        columns = math.ceil(math.sqrt(len(tiles)))
        rows = [tiles[i : i + columns] for i in range(0, len(tiles), columns)]
        width = max(sum(t.shape[1] for t in row) for row in rows)
        grid = np.vstack([_pad_width(np.hstack(row), width) for row in rows])
        # Every frame of a video must be the same size; the first one fixes it.
        if self._grid_size is None:
            self._grid_size = (grid.shape[1] - grid.shape[1] % 8, grid.shape[0] - grid.shape[0] % 8)
        return cv2.resize(grid, self._grid_size)

    def close(self) -> list[Path]:
        paths = []
        for camera, writer in self._writers.items():
            writer.close()
            paths.append(self.path(camera))
        self._writers.clear()
        return paths


def _pad_to_block(frame: np.ndarray, block: int = 8) -> np.ndarray:
    """Pad to a multiple of the encoder's macro block, rather than let it resize."""
    pad_h, pad_w = -frame.shape[0] % block, -frame.shape[1] % block
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0))) if pad_h or pad_w else frame


def _pad_width(image: np.ndarray, width: int) -> np.ndarray:
    return np.pad(image, ((0, 0), (0, width - image.shape[1]), (0, 0)))


def recording_frames(observation: Observation, scene: np.ndarray | None) -> dict[str, np.ndarray]:
    """The policy's two views, any raw camera images, and the scene camera."""
    return {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, **observation.extra_cameras, "scene": scene}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

FLAG_ABBREVIATIONS = {
    "exo_camera": "exo",
    "gripper_camera": "grip",
    "execute-horizon": "eh",
    "execute-horizon-do-only-first-n-steps": "n",
    "head-crop": "crop",
    "slow": "slow",
    "wait-for-arrival": "wait",
    "grasp-offset-mm": "offmm",
    "grasp-offset-deg": "offdeg",
    "use_parallel_gripper": "pg",
    "include_franka": "ghost",
}
"""Short names for run names. `flags` dicts are keyed by the flags' command-line spelling."""


def run_name(robot: str, flags: dict[str, object]) -> str:
    """e.g. "stretch4_exo-left_grip-left_eh-8_n-2_crop-droid_slow-0_wait-1_pg-0_offmm-0,0,0_offdeg-0,0,0"."""
    parts = [robot]
    for key, value in flags.items():
        if isinstance(value, bool):
            value = int(value)
        parts.append(f"{FLAG_ABBREVIATIONS.get(key, key)}-{value}")
    return "_".join(parts).replace("/", "-").replace(" ", "")


@dataclass
class EpisodeResult:
    index: int
    scene_id: str
    object_name: str
    instruction: str
    success: bool = False
    steps: int = 0
    queries: int = 0
    wall_seconds: float = 0.0
    mean_inference_seconds: float = 0.0
    metrics: dict = field(default_factory=dict)
    videos: list[str] = field(default_factory=list)
    error: str | None = None


def run_episode(
    index: int,
    setup: EpisodeSetup,
    env: BenchmarkEnv,
    policy,
    execute_horizon: int,
    execute_first_n: int,
    out_dir: Path,
    prefix: str,
    end_on_success: bool = True,
    on_step: Callable[[StepInfo, BenchmarkEnv], None] | None = None,
    extra_metrics: Callable[[BenchmarkEnv], dict] | None = None,
) -> EpisodeResult:
    """Run one episode on an already built env, recording it, and judge it."""
    result = EpisodeResult(index, setup.scene.scene_id, setup.scene.object_name, setup.instruction)
    judge = make_judge(setup)
    recorder = EpisodeRecorder(out_dir, f"{prefix}_{episode_name(index)}")
    latest: dict[str, object] = {"success": False, "metrics": {}}

    def step_callback(info: StepInfo) -> bool:
        recorder.add(recording_frames(info.observation, env.render_scene()))
        success, metrics = judge(env)
        latest["success"] = latest["success"] or success
        latest["metrics"] = metrics
        if on_step is not None:
            on_step(info, env)
        return end_on_success and success

    try:
        recorder.add(recording_frames(env.observe(), env.render_scene()))
        rollout = run_rollout(
            env, policy, setup.instruction, execute_horizon, execute_first_n, setup.horizon_steps, step_callback
        )
        result.steps = rollout.steps
        result.queries = rollout.queries
        result.wall_seconds = rollout.wall_seconds
        result.mean_inference_seconds = float(np.mean(rollout.inference_seconds)) if rollout.inference_seconds else 0.0
    except Exception as error:  # one bad episode should not end the run
        result.error = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        result.videos = [str(p.name) for p in recorder.close()]

    result.success = bool(latest["success"])
    result.metrics = dict(latest["metrics"])
    if extra_metrics is not None:
        result.metrics.update(extra_metrics(env))
    return result


def write_report(
    out_dir: Path,
    robot: str,
    name: str,
    flags: dict[str, object],
    benchmark_dir: Path,
    results: list[EpisodeResult],
    started: float,
    stopped_at: str | None = None,
    resume_command: str | None = None,
) -> Path:
    """
    `report.md` for people and `results.json` for `compare_benchmarks.py`. `stopped_at` is the
    episode a Ctrl+C interrupted (not in `results`); the report says how to resume from it.
    """
    results = sorted(results, key=lambda r: r.index)
    finished = [r for r in results if r.error is None]
    successes = sum(r.success for r in results)
    rate = successes / len(results) if results else 0.0
    summary = {
        "robot": robot,
        "run_name": name,
        "flags": flags,
        "benchmark": str(benchmark_dir),
        "checkpoint": f"{HF_REPO}@{HF_REVISION[:7]}",
        "molmobot_commit": MOLMOBOT_COMMIT[:7],
        "policy_dt_s": POLICY_DT,
        "episodes": len(results),
        "errors": len(results) - len(finished),
        "successes": successes,
        "success_rate": rate,
        "total_wall_seconds": time.time() - started,
        "stopped_at": stopped_at,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))

    metric_names = sorted({k for r in results for k in r.metrics})
    lines = [
        f"# MolmoBot-DROID benchmark: {robot}",
        "",
        f"**{successes}/{len(results)} succeeded ({rate:.0%})**"
        + (f", {summary['errors']} errored" if summary["errors"] else ""),
        "",
        *(
            [
                f"> **Stopped early** (Ctrl+C) during `{stopped_at}`, which is not counted. "
                f"Resume with `--resume {stopped_at}`:",
                ">",
                f"> `{resume_command}`",
                "",
            ]
            if stopped_at
            else []
        ),
        "| | |",
        "|---|---|",
        f"| Run | `{name}` |",
        f"| Benchmark | `{benchmark_dir}` |",
        f"| Checkpoint | `{summary['checkpoint']}` (MolmoBot code `{summary['molmobot_commit']}`) |",
        f"| Wall time | {summary['total_wall_seconds'] / 60:.1f} min |",
        "",
        "## Flags",
        "",
        "| Flag | Value |",
        "|---|---|",
        *[f"| `--{k}` | `{v}` |" for k, v in flags.items()],
        "",
        "## Episodes",
        "",
        "| # | Scene | Instruction | Success | Steps | Queries | s/query | Wall s | "
        + " | ".join(metric_names)
        + " | Error |",
        "|---|---|---|---|---|---|---|---|" + "---|" * len(metric_names) + "---|",
    ]
    for r in results:
        metrics = " | ".join(_format(r.metrics.get(m, "")) for m in metric_names)
        lines.append(
            f"| {r.index} | {r.scene_id} | {r.instruction} | {'yes' if r.success else 'no'} | {r.steps} | "
            f"{r.queries} | {r.mean_inference_seconds:.2f} | {r.wall_seconds:.0f} | {metrics} | {r.error or ''} |"
        )
    lines += ["", "Videos per episode: `<run>_ep<#>_<camera>.mp4`, plus `_grid.mp4` with every camera.", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def episode_name(index: int) -> str:
    """How an episode is named in output, file names and `--resume`: its index in the benchmark."""
    return f"ep{index:04d}"


def parse_episode_name(name: str) -> int:
    if not (name.startswith("ep") and name[2:].isdigit()):
        raise ValueError(f"Episode names look like ep0003, got '{name}'")
    return int(name[2:])


class BenchmarkRun:
    """
    The bookkeeping both runners share: which episodes are left (`--resume`), the results so
    far (carried over from the run directory's results.json when resuming), the report after
    every episode, and stopping cleanly on Ctrl+C with a report and the command to resume.

        run = BenchmarkRun(out_dir, "franka", name, flags, benchmark_dir, selected, resume)
        with run:
            for index, episode in run.episodes():
                ...
                run.record(result)
    """

    def __init__(self, out_dir: Path, robot: str, name: str, flags: dict, benchmark_dir: Path, selected, resume: str | None):
        self.out_dir, self.robot, self.name, self.flags, self.benchmark_dir = out_dir, robot, name, flags, benchmark_dir
        self.results: list[EpisodeResult] = []
        self.current: str | None = None
        self.started = time.time()
        self.selected = list(selected)
        if resume:
            start = parse_episode_name(resume)
            order = [index for index, _ in self.selected]
            if start not in order:
                raise ValueError(f"{resume} is not among this run's episodes ({episode_name(order[0])}..)")
            done = set(order[: order.index(start)])
            self.selected = self.selected[order.index(start) :]
            previous = out_dir / "results.json"
            if previous.exists():
                summary = json.loads(previous.read_text())
                self.results = [EpisodeResult(**r) for r in summary["results"] if r["index"] in done]
                self.started -= summary.get("total_wall_seconds", 0.0)
            missing = done - {r.index for r in self.results}
            if missing:
                print(f"No earlier results for {sorted(episode_name(i) for i in missing)}; the report leaves them out.")

    def episodes(self):
        for index, episode in self.selected:
            self.current = episode_name(index)
            yield index, episode
        self.current = None

    def record(self, result: EpisodeResult) -> None:
        self.results.append(result)
        self.write()

    def write(self, stopped_at: str | None = None) -> Path:
        return write_report(
            self.out_dir, self.robot, self.name, self.flags, self.benchmark_dir, self.results, self.started,
            stopped_at=stopped_at, resume_command=resume_command(stopped_at) if stopped_at else None,
        )

    def __enter__(self) -> "BenchmarkRun":
        allow_ctrl_c()
        return self

    def __exit__(self, kind, error, traceback_) -> bool:
        if kind is KeyboardInterrupt:
            report = self.write(stopped_at=self.current)
            print(f"\nStopped during {self.current}. Report so far: {report}")
            if self.current:
                print(f"Resume with:\n  {resume_command(self.current)}")
            return True  # handled: the runner returns normally
        self.write()
        return False


def allow_ctrl_c() -> None:
    """
    Make Ctrl+C raise KeyboardInterrupt again. `Stretch4MujocoSimulator.start()` replaces the
    handler with one that only stops the simulator, so call this after starting one.
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)


def resume_command(episode: str) -> str:
    """This command line, with `--resume <episode>` in place of any earlier one."""
    args, skip = [], False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg == "--resume":
            skip = True
            continue
        if arg.startswith("--resume="):
            continue
        args.append(arg)
    main = sys.modules["__main__"]
    module = getattr(getattr(main, "__spec__", None), "name", None)
    program = f"python -m {module}" if module else f"python {sys.argv[0]}"
    return f"{program} {shlex.join(args + ['--resume', episode])}"


def _format(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)
