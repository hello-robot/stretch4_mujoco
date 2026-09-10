"""
Turn Stretch rollouts into a MolmoSpaces JSON benchmark you can evaluate against.

The eight benchmarks in `benchmarks.py` are *released* asset packages: fixed
episode lists that arrive with the MolmoSpaces assets. That works as long as the
task you want to score already has a release. It does not for a task family you
made up locally -- `--task potato`, say -- and no amount of filtering helps: of
the 1000 episodes in MB-Pick, six pick up a potato.

So this builds the benchmark instead, out of the same generation pipeline that
produces the training data:

    # 200 episodes over held-out houses -- NOT the split you fine-tuned on
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task potato --episodes 200 --data-split val --no-export \\
        --output-dir data/stretch_potato_eval

    # freeze them into <assets>/benchmarks/stretch-local/potato/benchmark.json
    python -m examples.machine_learning.molmospaces.build_benchmark \\
        --rollouts data/stretch_potato_eval/rollouts/potato --benchmark potato

    # and score a policy on it exactly like a released benchmark
    python -m examples.machine_learning.molmospaces.run_benchmarks \\
        --benchmark potato --policy molmobot --checkpoint <ckpt>

## Why this is possible at all

Every rollout already carries its own initial conditions. `BaseMujocoTask.reset()`
calls `MlSpacesExpConfig.freeze_task_config()`, which pickles a `SavedEpisode`
-- camera extrinsics resolved to fixed values, the robot's start joint positions,
and a task config holding the base pose, the pickup object, its pose, every
mobile object's pose, and the referral expressions -- and stores it base64-encoded
under `obs_scene["frozen_config"]` in the HDF5. That is the same information an
`EpisodeSpec` holds; this module is the translation, field for field.

Which is also why the benchmark reproduces the episode rather than approximating
it: the poses written here are the ones the rollout actually ran from, not a
re-sample from the same distribution.

## What makes an episode set a benchmark rather than more training data

Two things, neither of which this script can check for you:

1. **A held-out split.** Generate with `--data-split val`. Scoring a policy on
   the houses it was fine-tuned in measures memorisation. The builder warns when
   the rollouts it is given came from `train`, and refuses to overwrite an
   existing benchmark without `--force`, because a benchmark that quietly changes
   under you invalidates every number you have already recorded against it.
2. **Solvable episodes.** Only episodes the expert succeeded at are kept
   (`--include-failures` overrides). An episode the demonstrator could not do is
   not a fair test; it just lowers every score by a constant.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import pickle
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np

from examples.machine_learning.molmospaces.benchmarks import (
    BENCHMARKS,
    LOCALLY_BUILT_BENCHMARK_KEYS,
    default_benchmarks_root,
)
from examples.machine_learning.molmospaces.hdf5_layout import (
    TRAJECTORY_FILE_PATTERN,
    TRAJECTORY_KEY_PATTERN,
)

log = logging.getLogger(__name__)

HOUSE_DIR_PATTERN = re.compile(r"^house_(?P<index>\d+)$")

DEFAULT_TASK_HORIZON_SEC = 20.0
"""
Per-episode step budget, in seconds, written into every episode's task dict.

`eval_main.py` reads `task["task_horizon_sec"]` and converts it to steps with the
eval config's `policy_dt_ms`, and *fails* if the field is missing -- so it is not
optional. 20s is what every episode of MB-Pick and MB-PnP carries, which is the
point: a potato pick and a released pick then get the same budget, and their
scores are comparable.
"""


def _plain(value: Any) -> Any:
    """Strip numpy scalars and `Path`s out of a nested structure, for JSON.

    A frozen task config is full of both -- `np.str_` referral expressions,
    `np.float64` poses, `Path` asset locations -- because it was built from
    simulator state rather than parsed from a file. Pydantic coerces the fields
    it knows the types of, but `EpisodeSpec.task` is a bare `dict`, so anything
    inside it reaches `json.dump` exactly as it is and `np.float64` is not
    serialisable.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def _expert_solved(trajectory: h5py.Group) -> bool:
    """Whether the expert finished the episode holding the object, not just touched it.

    The *last* step's success flag, which is the one that reproduces
    `task.judge_success()` -- the verdict the generation pipeline itself keeps or
    discards an episode on. Measured on a `--keep-failures` potato run: an episode
    that lifted a potato 5cm and then dropped it has `success.any()` True and
    `success[-1]` False, and the pipeline logged it as a failure.

    Deliberately stricter than `lerobot_export._build_episode`, which uses
    `success.any()`. The two want different things from the same flag: a training
    exporter is looking for demonstrated behaviour to imitate, and a brief
    successful lift is still that. A benchmark is asserting the episode is
    *completable*, and an episode whose only demonstration ends in a dropped
    object is not evidence of that.
    """
    success = trajectory["success"][:]
    return bool(len(success)) and bool(success[-1])


def _house_index(h5_path: Path) -> int:
    """The house a trajectory file belongs to, from its `house_<N>/` directory."""
    match = HOUSE_DIR_PATTERN.match(h5_path.parent.name)
    if match is None:
        raise ValueError(
            f"{h5_path} is not under a 'house_<N>' directory, so its house index is "
            "unknown. Point --rollouts at a generation run's task directory "
            "(the one holding house_*/), not at a single house."
        )
    return int(match.group("index"))


@lru_cache(maxsize=None)
def _scene_settings_of_config(config_path: Path) -> tuple[str | None, str | None]:
    """`(scene_dataset, data_split)` out of one `experiment_config_*.pkl`.

    Cached because unpickling one is not cheap: the config holds a
    `Stretch4RobotConfig`, whose construction converts the Stretch URDF to MJCF.
    Every house directory of a run resolves to the same config file, so without
    this a 50-house run would pay for that conversion 50 times.
    """
    try:
        with config_path.open("rb") as handle:
            config = pickle.load(handle)
    except Exception as error:  # noqa: BLE001 - a stale or unreadable pickle is not fatal
        log.warning(f"[benchmark] could not read {config_path}: {error}")
        return None, None
    return getattr(config, "scene_dataset", None), getattr(config, "data_split", None)


def _run_scene_settings(h5_path: Path, search_root: Path) -> tuple[str | None, str | None]:
    """`(scene_dataset, data_split)` for a trajectory file, from its run's config.

    `generate_rollouts()` calls `config.save_config()`, which drops an
    `experiment_config_<timestamp>.pkl` in the run's output directory -- so the
    scene dataset and split are recorded, just not in the HDF5. Walks up from the
    trajectory file so a `--rollouts` pointed at a tree of several runs still
    reads each episode's own settings rather than the first one it finds.

    Returns `(None, None)` if there is no config to read, leaving the caller to
    fall back on `--scene-dataset` / `--data-split`.
    """
    directory = h5_path.parent
    while True:
        configs = sorted(directory.glob("experiment_config_*.pkl"))
        if configs:
            # Latest timestamp: a resumed run writes a new one per invocation.
            return _scene_settings_of_config(configs[-1])
        if directory == search_root or directory == directory.parent:
            return None, None
        directory = directory.parent


def _decode_frozen_config(trajectory: h5py.Group) -> tuple[Any, dict]:
    """`(SavedEpisode, obs_scene)` for one trajectory.

    Raises:
        ValueError: the trajectory has no frozen config, so its initial
            conditions cannot be reconstructed.
    """
    if "obs_scene" not in trajectory:
        raise ValueError("trajectory has no 'obs_scene' group")

    raw = trajectory["obs_scene"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    scene = json.loads(raw)

    blob = scene.get("frozen_config")
    if not blob:
        raise ValueError(
            "obs_scene has no 'frozen_config'. It is written by "
            "MlSpacesExpConfig.freeze_task_config() on task reset, so a trajectory "
            "without one was not produced by the datagen pipeline."
        )
    return pickle.loads(base64.b64decode(blob)), scene


def _camera_specs(camera_config: Any) -> list[dict]:
    """The frozen camera system as benchmark camera specs.

    `freeze_task_config()` has already resolved every randomised camera to a
    fixed one -- a `RobotMountedCameraConfig` with concrete offsets or a
    `FixedExocentricCameraConfig` with a concrete world pose -- which is exactly
    the two shapes the benchmark schema has.

    These are recorded for completeness rather than for use: Stretch episodes go
    through `stretch_episode_override`, which replaces the camera list with a
    fresh `Stretch4CameraSystem` at the episode's resolution. They still have to
    be here and be valid, because `JsonEvalTaskSampler._validate_episode_spec()`
    rejects an episode with an empty `cameras` list.
    """
    specs: list[dict] = []
    for camera in camera_config.cameras:
        if hasattr(camera, "reference_body_names"):
            specs.append(
                {
                    "name": camera.name,
                    "type": "robot_mounted",
                    "reference_body_names": list(camera.reference_body_names),
                    "camera_offset": _plain(list(camera.camera_offset)),
                    "lookat_offset": _plain(list(camera.lookat_offset)),
                    "camera_quaternion": _plain(list(camera.camera_quaternion)),
                    "fov": float(camera.fov),
                    "record_depth": bool(camera.record_depth),
                }
            )
        elif hasattr(camera, "forward"):
            specs.append(
                {
                    "name": camera.name,
                    "type": "exocentric",
                    "pos": _plain(list(camera.pos)),
                    "up": _plain(list(camera.up)),
                    "forward": _plain(list(camera.forward)),
                    "fov": float(camera.fov),
                    "record_depth": bool(camera.record_depth),
                }
            )
        else:
            raise ValueError(
                f"Camera {camera.name!r} is a {type(camera).__name__}, which is neither "
                "robot-mounted nor exocentric and has no benchmark spec equivalent. "
                "freeze_task_config() should have resolved it to one of those."
            )
    return specs


def _task_dict(saved_episode: Any, task_horizon_sec: float) -> dict:
    """The episode's `task` dict: identity, plus every task-spec field it has.

    Driven by `get_task_spec_field_names()` rather than a hand-written field list,
    the same way `JsonEvalTaskSampler` copies the JSON back onto a task config on
    the way in. That is what makes this work for any of the datagen task families
    and not just pick: a pick-and-place episode carries
    `place_receptacle_name`/`place_receptacle_start_pose` on its frozen task
    config, and they come along without this function naming them.
    """
    from molmo_spaces.evaluation.benchmark_schema import get_task_spec_field_names

    task_config = saved_episode.task_config
    task_cls = saved_episode.task_cls_str
    if not task_cls:
        raise ValueError("frozen config has no task_cls_str")

    task: dict[str, Any] = {"task_cls": task_cls}

    for field in sorted(get_task_spec_field_names()):
        if not hasattr(task_config, field):
            continue
        value = getattr(task_config, field)
        if value is None:
            continue
        task[field] = _plain(value)

    if "robot_base_pose" not in task:
        raise ValueError(
            "frozen task config has no robot_base_pose, which every episode spec "
            "needs to place the robot"
        )

    task["task_horizon_sec"] = task_horizon_sec
    return task


def _language_spec(saved_episode: Any, scene: dict) -> dict:
    """The instruction and the referral expressions behind it.

    `task_description` comes from `obs_scene` rather than the frozen task config:
    the config holds the *expressions* a description is built from, and the task
    is what assembles them into the sentence the policy is actually prompted
    with. Using anything else here would score the policy against a prompt it was
    never trained on.
    """
    description = scene.get("task_description")
    if not description:
        raise ValueError(
            "obs_scene has no 'task_description'; JsonEvalTaskSampler rejects an "
            "episode without one"
        )

    task_config = saved_episode.task_config
    return {
        "task_description": str(description),
        "referral_expressions": {
            str(k): str(v) for k, v in (task_config.referral_expressions or {}).items()
        },
        "referral_expressions_priority": _plain(
            task_config.referral_expressions_priority or {}
        ),
    }


def _prune_staged_object_poses(
    object_poses: dict[str, Any], added_objects: dict[str, str]
) -> dict[str, Any]:
    """Drop poses for objects that will not exist in the replayed scene.

    `freeze_task_config()` records a pose for every *mobile* object in the scene,
    which in pick-from-set mode includes the pickupables the sampler added but did
    not use -- `POTATO_PICKUPS_PER_HOUSE` is 10, so nine potatoes are parked on a
    staging platform 25m above the house while one is placed in the room. Only the
    one in the room is in `added_objects`, so only it gets rebuilt at eval time,
    and the other nine poses refer to bodies that are not there.

    Harmless but not free: `JsonEvalTaskSampler.randomize_scene()` logs a warning
    per missing body, which measured out at nine warnings per episode and buried
    the real ones. They are also misleading in a benchmark file, which is supposed
    to describe the scene being evaluated.

    Only siblings are dropped, never scene objects: a key is removed solely when
    it shares a top-level namespace with an `added_objects` key (`pickup/`,
    `place_receptacle/`) yet is not itself in `added_objects`. Scene-native bodies
    are named without slashes (`Irishpotato_<hash>_1_0_2`), so they cannot match.
    """
    namespaces = {name.split("/")[0] for name in added_objects if "/" in name}
    if not namespaces:
        return object_poses

    return {
        name: pose
        for name, pose in object_poses.items()
        if name in added_objects or name.split("/")[0] not in namespaces
    }


def _episode_spec(
    saved_episode: Any,
    scene: dict,
    house_index: int,
    scene_dataset: str,
    data_split: str,
    source: dict,
    task_horizon_sec: float,
) -> dict:
    """One `EpisodeSpec`, validated and dumped to JSON-ready primitives."""
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.utils.task_relevant_objects_and_workspace_utils import (
        get_task_relevant_objects,
    )

    task = _task_dict(saved_episode, task_horizon_sec)
    task_config = saved_episode.task_config
    added_objects = {
        str(name): str(path) for name, path in (task_config.added_objects or {}).items()
    }

    spec = EpisodeSpec(
        source=source,
        house_index=house_index,
        scene_dataset=scene_dataset,
        data_split=data_split,
        seed=None,
        robot={
            # Informational: the robot that runs an episode comes from the eval
            # config, and `stretch_episode_override` overwrites this field anyway.
            "robot_name": "stretch4",
            "init_qpos": _plain(saved_episode.robot_config.init_qpos),
        },
        img_resolution=tuple(saved_episode.camera_config.img_resolution),
        cameras=_camera_specs(saved_episode.camera_config),
        scene_modifications={
            "added_objects": added_objects,
            "object_poses": _prune_staged_object_poses(
                _plain(task_config.object_poses or {}), added_objects
            ),
            "removed_objects": [],
        },
        task=task,
        task_relevant_objects=get_task_relevant_objects(task_config),
        language=_language_spec(saved_episode, scene),
    )
    return spec.model_dump(mode="json")


def collect_episodes(
    rollout_dirs: list[Path],
    task_horizon_sec: float = DEFAULT_TASK_HORIZON_SEC,
    successful_only: bool = True,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read every usable trajectory under `rollout_dirs` as an episode spec.

    Args:
        rollout_dirs: generation-run directories holding `house_*/trajectories*.h5`.
        task_horizon_sec: step budget written into each episode.
        successful_only: skip trajectories the task judged unsuccessful.
        scene_dataset: override the run's recorded scene dataset. Only needed if
            the run's `experiment_config_*.pkl` is missing.
        data_split: override the run's recorded split, same caveat.
        limit: stop after this many episodes.

    Returns:
        Episode spec dicts, in `(house, file, trajectory)` order. Trajectories
        that could not be converted are skipped and logged rather than raised on;
        see the comment at the skip.

    Raises:
        ValueError: no trajectory files under `rollout_dirs`, or one whose house
            index or scene settings could not be determined -- both of which mean
            `--rollouts` is pointed somewhere unexpected, so every episode after
            it would fail the same way.
    """
    trajectory_files: list[tuple[Path, Path]] = []
    for rollout_dir in rollout_dirs:
        rollout_dir = Path(rollout_dir)
        for path in sorted(rollout_dir.rglob("trajectories*.h5")):
            if TRAJECTORY_FILE_PATTERN.match(path.name) is not None:
                trajectory_files.append((rollout_dir, path))

    if not trajectory_files:
        raise ValueError(
            f"No trajectories*.h5 found under {[str(d) for d in rollout_dirs]}. "
            "Point --rollouts at a generation run's task directory, e.g. "
            "data/stretch_potato_eval/rollouts/potato."
        )

    episodes: list[dict] = []
    skipped_unsuccessful = 0
    skipped_unconvertible: list[str] = []

    for search_root, h5_path in trajectory_files:
        house_index = _house_index(h5_path)
        run_dataset, run_split = _run_scene_settings(h5_path, search_root)
        episode_dataset = scene_dataset or run_dataset
        episode_split = data_split or run_split

        if not episode_dataset or not episode_split:
            raise ValueError(
                f"Could not determine the scene dataset and split for {h5_path}: no "
                "experiment_config_*.pkl in the run directory. Pass --scene-dataset "
                "and --data-split explicitly."
            )

        source_date = datetime.date.fromtimestamp(h5_path.stat().st_mtime).isoformat()
        created_date = datetime.date.today().isoformat()

        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in sorted(h5_file.keys()):
                if TRAJECTORY_KEY_PATTERN.match(traj_key) is None:
                    continue
                if limit is not None and len(episodes) >= limit:
                    break

                trajectory = h5_file[traj_key]
                if successful_only and not _expert_solved(trajectory):
                    skipped_unsuccessful += 1
                    continue

                # One malformed trajectory should not sink a build that took
                # hours of simulation to produce, so this is a skip rather than
                # a raise -- but it is counted and reported, because "150
                # episodes from a 200-episode run" is the symptom of a
                # systematic problem and must not pass unnoticed.
                try:
                    saved_episode, scene = _decode_frozen_config(trajectory)
                    episode_length = int(trajectory["success"].shape[0])
                    spec = _episode_spec(
                        saved_episode=saved_episode,
                        scene=scene,
                        house_index=house_index,
                        scene_dataset=episode_dataset,
                        data_split=episode_split,
                        source={
                            "h5_file": str(h5_path.resolve()),
                            "traj_key": traj_key,
                            "episode_length": episode_length,
                            "camera_system_class": type(saved_episode.camera_config).__name__,
                            "source_data_date": source_date,
                            "benchmark_created_date": created_date,
                        },
                        task_horizon_sec=task_horizon_sec,
                    )
                except Exception as error:  # noqa: BLE001 - see comment above
                    skipped_unconvertible.append(f"{h5_path.name}:{traj_key} ({error})")
                    continue

                episodes.append(spec)

        if limit is not None and len(episodes) >= limit:
            break

    if skipped_unsuccessful:
        log.info(
            f"[benchmark] skipped {skipped_unsuccessful} unsuccessful trajectories "
            "(--include-failures keeps them)"
        )
    if skipped_unconvertible:
        log.warning(
            f"[benchmark] skipped {len(skipped_unconvertible)} trajectories that could "
            f"not be converted: {'; '.join(skipped_unconvertible[:5])}"
            + (" ..." if len(skipped_unconvertible) > 5 else "")
        )
    return episodes


def _metadata(episodes: list[dict], rollout_dirs: list[Path], description: str) -> dict:
    """`benchmark_metadata.json`: the same summary MolmoSpaces' releases carry.

    Optional as far as the loader is concerned -- `load_benchmark()` reads it if
    present and shrugs if not -- but it is what makes a built benchmark
    self-describing, and the object-category counts are the check that a potato
    benchmark is in fact all potatoes.
    """
    lengths = [
        episode["source"]["episode_length"]
        for episode in episodes
        if episode.get("source", {}).get("episode_length")
    ]
    categories = Counter(
        str(episode["language"]["referral_expressions"].get("pickup_obj_name", "unknown"))
        for episode in episodes
    )

    metadata = {
        "description": description,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "source_datagen_path": ", ".join(str(Path(d).resolve()) for d in rollout_dirs),
        "num_episodes": len(episodes),
        "num_houses": len({episode["house_index"] for episode in episodes}),
        "task_cls_counts": dict(Counter(episode["task"]["task_cls"] for episode in episodes)),
        "object_category_counts": dict(categories),
        "robot_counts": dict(Counter(episode["robot"]["robot_name"] for episode in episodes)),
        "house_counts": dict(Counter(episode["house_index"] for episode in episodes)),
        "camera_system_class": next(
            (
                episode["source"]["camera_system_class"]
                for episode in episodes
                if episode.get("source")
            ),
            None,
        ),
        "benchmark_created_date": datetime.date.today().isoformat(),
    }
    if lengths:
        metadata["episode_length_stats"] = {
            "min": float(min(lengths)),
            "max": float(max(lengths)),
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
        }
    return metadata


def build_benchmark(
    rollout_dirs: list[Path],
    output_dir: Path,
    description: str = "",
    task_horizon_sec: float = DEFAULT_TASK_HORIZON_SEC,
    successful_only: bool = True,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    limit: int | None = None,
    force: bool = False,
) -> tuple[Path, list[dict]]:
    """Write `benchmark.json` and `benchmark_metadata.json` for a set of rollouts.

    Returns:
        `(benchmark_directory, episodes)`.

    Raises:
        FileExistsError: `output_dir` already holds a benchmark and `force` is
            False. Overwriting one silently would change what every score
            already recorded against it refers to.
        ValueError: no usable trajectories, or one could not be converted.
    """
    output_dir = Path(output_dir)
    benchmark_path = output_dir / "benchmark.json"
    if benchmark_path.exists() and not force:
        raise FileExistsError(
            f"{benchmark_path} already exists. Scores recorded against a benchmark "
            "stop meaning anything if its episodes change underneath them, so pass "
            "--force to replace it deliberately, or --output-dir to build a second one."
        )

    episodes = collect_episodes(
        rollout_dirs=rollout_dirs,
        task_horizon_sec=task_horizon_sec,
        successful_only=successful_only,
        scene_dataset=scene_dataset,
        data_split=data_split,
        limit=limit,
    )
    if not episodes:
        raise ValueError(
            "No episodes to write. Every trajectory was skipped -- if the run has a "
            "low success rate, --include-failures keeps the rest, but an episode the "
            "expert failed is a poor test case."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w") as handle:
        json.dump(episodes, handle, indent=1)
    with (output_dir / "benchmark_metadata.json").open("w") as handle:
        json.dump(_metadata(episodes, rollout_dirs, description), handle, indent=1)

    return output_dir, episodes


@click.command()
@click.option(
    "--rollouts",
    "rollout_dirs",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Generation-run directory holding house_*/trajectories*.h5. Repeatable.",
)
@click.option(
    "--benchmark",
    "benchmark_key",
    type=click.Choice(sorted(LOCALLY_BUILT_BENCHMARK_KEYS)),
    default=None,
    help="Write to the directory this registered benchmark expects, so "
    "`run_benchmarks.py --benchmark <key>` finds it. Mutually exclusive with "
    "--output-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the benchmark here instead of to a registered benchmark's directory.",
)
@click.option(
    "--task-horizon-sec",
    type=float,
    default=DEFAULT_TASK_HORIZON_SEC,
    help="Per-episode step budget in seconds. Defaults to 20, which is what the "
    "released pick benchmarks use.",
)
@click.option(
    "--successful-only/--include-failures",
    default=True,
    help="Keep only episodes the expert solved. On by default: an unsolvable "
    "episode lowers every policy's score without telling them apart.",
)
@click.option(
    "--scene-dataset",
    type=str,
    default=None,
    help="Override the scene dataset recorded in the run's experiment_config pickle.",
)
@click.option(
    "--data-split",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="Override the recorded split. Generate evaluation episodes from 'val'.",
)
@click.option("--limit", type=int, default=None, help="Stop after this many episodes.")
@click.option("--force", is_flag=True, help="Replace an existing benchmark at the destination.")
def main(
    rollout_dirs: tuple[Path, ...],
    benchmark_key: str | None,
    output_dir: Path | None,
    task_horizon_sec: float,
    successful_only: bool,
    scene_dataset: str | None,
    data_split: str | None,
    limit: int | None,
    force: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if (benchmark_key is None) == (output_dir is None):
        raise click.UsageError("Pass exactly one of --benchmark or --output-dir.")

    description = ""
    if benchmark_key is not None:
        benchmark = BENCHMARKS[benchmark_key]
        output_dir = default_benchmarks_root() / benchmark.relative_dir
        description = benchmark.description

    try:
        directory, episodes = build_benchmark(
            rollout_dirs=list(rollout_dirs),
            output_dir=Path(output_dir),
            description=description,
            task_horizon_sec=task_horizon_sec,
            successful_only=successful_only,
            scene_dataset=scene_dataset,
            data_split=data_split,
            limit=limit,
            force=force,
        )
    except (FileExistsError, ValueError) as error:
        # These are all "you asked for something that cannot be done" rather than
        # bugs -- a benchmark already there, no rollouts under the path given, a
        # run with no solved episodes. A traceback buries the explanation.
        raise click.ClickException(str(error)) from error

    splits = Counter(episode["data_split"] for episode in episodes)
    houses = {episode["house_index"] for episode in episodes}
    objects = Counter(
        str(episode["language"]["referral_expressions"].get("pickup_obj_name", "unknown"))
        for episode in episodes
    )

    click.secho(
        f"{len(episodes)} episodes over {len(houses)} houses -> {directory / 'benchmark.json'}",
        fg="green",
    )
    click.echo(f"  splits:  {dict(splits)}")
    click.echo(f"  objects: {dict(objects.most_common(8))}")

    if "train" in splits:
        click.secho(
            f"WARNING: {splits['train']} episodes came from the 'train' split. A policy "
            "fine-tuned on generated data has seen these houses, so a score on them "
            "measures memorisation. Regenerate the evaluation episodes with "
            "--data-split val.",
            fg="yellow",
        )

    if benchmark_key is not None:
        click.echo(
            "\nRun it with:\n"
            "  python -m examples.machine_learning.molmospaces.run_benchmarks "
            f"--benchmark {benchmark_key} --episodes {min(len(episodes), 20)}"
        )


if __name__ == "__main__":
    main()
