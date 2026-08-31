"""
Generate Stretch demonstrations, then export them for fine-tuning.

One command for the whole data half of the pipeline:

    # smallest thing that proves the setup works: 2 episodes, 1 house
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \\
        --task debug

    # the same, watched live in MuJoCo's viewer rather than read off a log
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task debug --output-dir data/stretch_debug --no-export --visualize

    # slow down playback in the viewer (e.g. 2x slower than real-time)
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task debug --output-dir data/stretch_debug --no-export --visualize --slow_rate 2.0

    # a real run: 2000 pick episodes across procthor-objaverse, 8 workers
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task pick --episodes 2000 --num-workers 8 --output-dir data/stretch_pick

    # several families pooled into one training set
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task pick --task pnp --task open --episodes 1000 \
        --output-dir data/stretch_manipulation

    # one object category: pick, with potatoes added to every scene
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --task potato --episodes 2000 --num-workers 8 \
        --output-dir data/stretch_potato --no-export

    # rollouts already on disk; just re-export them
    python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
        --rollouts data/stretch_pick/rollouts --output-dir data/stretch_pick

Two stages, either of which can be run alone (`--no-export`, `--rollouts`):

1. **Generate.** MolmoSpaces' `ParallelRolloutRunner` over one of the Stretch
   datagen configs in `datagen_configs.py`, which writes HDF5 trajectories plus
   side-car MP4s under `<output-dir>/rollouts/<task>/`.
2. **Export.** `lerobot_export.py` turns those into a LeRobot dataset under
   `<output-dir>/lerobot/`, which is what a fine-tuning run consumes.

The generation stage is the expensive one -- it is a full physics rollout with
rendering per episode -- so it keeps its raw output rather than streaming
straight into the export. Re-exporting into a different action space is then
seconds rather than hours, which matters because `--action-space` is the choice
you are most likely to want to change your mind about.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import logging
import os
import pprint
import random
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import click
import mujoco
import numpy as np
from tqdm import tqdm

from examples.machine_learning.molmospaces.finetuning.datagen_configs import (
    DATAGEN_CONFIGS,
    qualified_config_name,
)
from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
    ACTION_SPACES,
    export_lerobot_dataset,
)
from examples.machine_learning.molmospaces.visualize import (
    StretchRerunVisualizer,
    snap_free_camera_to_robot,
)
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.data_generation.pipeline import (
    ParallelRolloutRunner,
    cleanup_context,
    cleanup_episode_resources,
    get_worker_logger,
    log_memory_usage,
    mp_context,
    setup_house_dirs,
    setup_policy,
    setup_viewer,
    worker_stdout_context,
)
from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask
from molmo_spaces.utils.profiler_utils import DatagenProfiler
from molmo_spaces.utils.save_utils import prepare_episode_for_saving, save_trajectories

log = logging.getLogger(__name__)

PROGRESS_POLL_SECONDS = 1.0


class TqdmLoggingHandler(logging.StreamHandler):
    """A stderr log handler that writes through `tqdm.write`.

    `tqdm.write` clears the bar, writes the line, and redraws, so a log record
    lands above the bar rather than on top of it. This only covers the process
    that installs it: with `--num-workers > 1` the rollout workers are separate
    processes with their own handlers on the same terminal, and they will still
    write over the bar -- it just redraws on the next poll.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class DatagenProgressBar:
    """A work-item progress bar fed from the runner's shared counters.

    Rollout progress already exists as the `multiprocessing.Value`s that workers
    bump as they finish houses and episodes, so this only has to read them once
    a second and render them; the workers pay nothing for it.

    It counts *work items* rather than episodes because that is the only number
    here that is both bounded and monotone. A failed episode is retried and a
    house that turns out to be invalid for the task is abandoned, so the episode
    counter can overshoot or undershoot the total that was asked for, while each
    house batch is completed or skipped exactly once. Episodes are shown in the
    postfix, where being approximate costs nothing.

    Polling happens on a thread rather than in the runner's own wait loop so that
    the single-worker in-process path -- which blocks inside the worker function
    itself, and is what `--visualize` uses -- gets the same bar as the pool path.
    """

    def __init__(self, runner: Any, enabled: bool = True) -> None:
        self._runner = runner
        self._total_items = len(runner.work_items)
        self._total_episodes = sum(item[1] for item in runner.work_items)
        self._enabled = enabled and self._total_items > 0
        self._bar: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._start_time = 0.0

    def __enter__(self) -> DatagenProgressBar:
        if self._enabled:
            self._start_time = time.time()
            self._bar = tqdm(
                total=self._total_items,
                desc="rollouts",
                unit="house",
                dynamic_ncols=True,
                file=sys.stderr,
            )
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * PROGRESS_POLL_SECONDS)
            self._thread = None
        if self._bar is not None:
            self.refresh()
            self._bar.close()
            self._bar = None

    def _poll(self) -> None:
        while not self._stop.wait(PROGRESS_POLL_SECONDS):
            self.refresh()

    def refresh(self) -> None:
        bar = self._bar
        if bar is None:
            return

        # Read the counters without taking `counter_lock`, as the periodic wandb
        # logging does: an inconsistent read costs one stale progress line, while
        # contending for the workers' lock from here would put a display detail
        # on the critical path of every worker's bookkeeping.
        runner = self._runner
        done = runner.completed_houses.value + runner.skipped_houses.value
        episodes = runner.total_count.value
        successes = runner.success_count.value

        elapsed = max(time.time() - self._start_time, 1e-9)
        postfix = [f"{episodes}/{self._total_episodes} eps"]
        if episodes:
            postfix.append(f"{successes / episodes:.0%} ok")
        postfix.append(f"{episodes / elapsed * 60:.1f} eps/min")

        bar.n = min(done, self._total_items)
        bar.set_postfix_str(", ".join(postfix), refresh=False)
        bar.refresh()


def trim_memory() -> None:
    """Forces Python GC and glibc heap trimmer to release unused memory back to the OS.

    Only ever a second-order effect: the camera frames that dominate a worker's
    footprint are multi-hundred-kilobyte numpy arrays, which glibc mostly serves
    with `mmap` and returns at `free()` without help. This is here for the churn
    of small objects underneath them, and is only worth calling at a point where
    the big allocations have *already* gone out of scope -- see
    `flush_episode_to_disk`, which is what actually bounds the footprint.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def flush_episode_to_disk(
    worker_logger: Any,
    history: dict,
    sensor_suite: Any,
    save_dir: Path,
    exp_config: Any,
    batch_suffix: str,
    episode_idx: int,
    datagen_profiler: Any = None,
) -> dict | None:
    """Encode one finished episode's videos and drop its camera frames immediately.

    This is the whole memory story of a datagen run. An episode's observation
    history is ~4.3 MiB per step -- five 640x368 RGB streams plus a float32 depth
    stream -- so a 300-step episode is ~1.3 GiB and a 500-step one ~2.1 GiB, and
    *none* of it lands in the HDF5: the frames go out as side-car MP4s.

    The pipeline's own `save_house_trajectories` does this encoding once per
    house, at the end, which means all `samples_per_house` episodes are held as
    raw frames simultaneously -- 4 x 2.1 GiB per worker at the default, times
    `--num-workers`. Calling it per episode instead trades nothing (the encoding
    work is identical and the MP4 filenames come out the same) for a peak of one
    episode's frames rather than a houseful.

    What comes back is the camera-stripped batched tensor dict -- the ~10 MiB of
    poses, joint states and per-camera intrinsics that actually go into the HDF5
    -- so accumulating those across a house costs nothing worth counting.

    Returns:
        The prepared episode, or None if there was nothing to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    if datagen_profiler is not None:
        datagen_profiler.start("save_batch_prep")
    try:
        prepared = prepare_episode_for_saving(
            history,
            sensor_suite,
            fps=exp_config.fps,
            save_dir=save_dir,
            episode_idx=episode_idx,
            save_file_suffix=batch_suffix,
        )
    except Exception as e:
        # A failed encode costs one episode, not the house, and must not be
        # mistaken for a rollout failure by the caller's retry counters.
        worker_logger.error(f"Failed to prepare episode {episode_idx} for saving: {e}")
        traceback.print_exc()
        prepared = None
    finally:
        if datagen_profiler is not None:
            datagen_profiler.end("save_batch_prep")

    return prepared


def save_prepared_trajectories(
    worker_logger: Any,
    prepared_episodes: list[dict],
    save_dir: Path,
    exp_config: Any,
    batch_suffix: str,
    datagen_profiler: Any = None,
    batch_num: int | None = None,
    total_batches: int | None = None,
) -> None:
    """Write already-prepared (camera-stripped) episodes into the house's HDF5.

    The back half of `save_house_trajectories`; the front half -- video encoding
    and frame release -- has already happened per episode in
    `flush_episode_to_disk`.
    """
    if not prepared_episodes:
        worker_logger.warning(f"No trajectory data to save for {save_dir.name}")
        return

    batch_info = f" batch {batch_num}/{total_batches}" if batch_num is not None else ""
    worker_logger.info(
        f"Saving trajectory data for {save_dir.name}{batch_info}: "
        f"{len(prepared_episodes)} episodes"
    )

    try:
        t_start = time.perf_counter()
        if datagen_profiler is not None:
            datagen_profiler.start("save_trajectories")
        save_trajectories(
            prepared_episodes,
            save_dir=save_dir,
            fps=exp_config.fps,
            save_file_suffix=batch_suffix,
            save_mp4s=True,
            logger=worker_logger,
        )
        if datagen_profiler is not None:
            datagen_profiler.end("save_trajectories")
        worker_logger.info(
            f"Successfully saved trajectory data for {save_dir.name} "
            f"in {time.perf_counter() - t_start:.2f}s"
        )
    except Exception as e:
        worker_logger.error(f"Failed to save trajectory data for {save_dir.name}: {e}")
        traceback.print_exc()


def stretch_house_processing_worker(
    worker_id: int,
    exp_config: Any,
    work_items: list[tuple[int, int, int, int]],
    shutdown_event: Any,
    counter_lock: Any,
    house_counter: Any,
    success_count: Any,
    total_count: Any,
    completed_houses: Any,
    skipped_houses: Any,
    max_allowed_sequential_task_sampler_failures: int = 10,
    max_allowed_sequential_rollout_failures: int = 10,
    max_allowed_sequential_irrecoverable_failures: int = 5,
    preloaded_policy: Any = None,
    filter_for_successful_trajectories: bool = False,
    runner_class: Any = None,
    max_items_per_worker: int | None = 10,
) -> None:
    """Worker function that processes a limited number of work items before exiting to recycle memory."""
    worker_logger = get_worker_logger(worker_id)

    if hasattr(exp_config, "datagen_profiler") and exp_config.datagen_profiler:
        datagen_profiler = DatagenProfiler(logger=worker_logger, enabled=True)
    else:
        datagen_profiler = None

    num_sequential_irrecoverable_failures = 0
    task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
    task_sampler.set_datagen_profiler(datagen_profiler)

    items_processed_by_worker = 0
    with worker_stdout_context(worker_logger, worker_id):
        try:
            while True:
                if shutdown_event.is_set():
                    worker_logger.info(
                        f"Worker {worker_id} received shutdown signal, cleaning up..."
                    )
                    break

                with counter_lock:
                    if house_counter.value >= len(work_items):
                        break
                    item_idx = house_counter.value
                    house_counter.value += 1

                current_house_id, batch_samples, batch_num, total_batches = work_items[item_idx]

                worker_logger.info(
                    f"Worker {worker_id} (PID {os.getpid()}) starting house {current_house_id} "
                    f"batch {batch_num}/{total_batches} ({batch_samples} episodes) "
                    f"(item {item_idx + 1}/{len(work_items)})"
                )

                house_success_count, house_total_count, irrecoverable = (
                    runner_class.process_single_house(
                        worker_id,
                        worker_logger,
                        current_house_id,
                        exp_config,
                        batch_samples,
                        shutdown_event,
                        task_sampler,
                        preloaded_policy,
                        max_allowed_sequential_task_sampler_failures,
                        max_allowed_sequential_rollout_failures,
                        filter_for_successful_trajectories=filter_for_successful_trajectories,
                        runner_class=runner_class,
                        batch_num=batch_num,
                        total_batches=total_batches,
                        datagen_profiler=datagen_profiler,
                    )
                )

                with counter_lock:
                    success_count.value += house_success_count
                    total_count.value += house_total_count
                    if house_total_count > 0:
                        completed_houses.value += 1
                    else:
                        skipped_houses.value += 1

                items_processed_by_worker += 1
                trim_memory()
                # Logged after the trim, so a footprint that keeps climbing across
                # work items is a real leak worth chasing, while one that returns
                # to a flat baseline is just the per-episode peak.
                log_memory_usage(
                    worker_logger,
                    prefix=f"Worker {worker_id} after {items_processed_by_worker} "
                    f"work items (house {current_house_id}): ",
                )

                if irrecoverable:
                    num_sequential_irrecoverable_failures += 1
                    if (
                        num_sequential_irrecoverable_failures
                        >= max_allowed_sequential_irrecoverable_failures
                    ):
                        worker_logger.error(
                            f"Worker {worker_id} encountered {num_sequential_irrecoverable_failures} "
                            "sequential irrecoverable failures. Exiting worker."
                        )
                        break
                else:
                    num_sequential_irrecoverable_failures = 0

                # Process recycling check: exit cleanly so kernel frees leaked C/driver memory
                if max_items_per_worker is not None and max_items_per_worker > 0:
                    if items_processed_by_worker >= max_items_per_worker:
                        worker_logger.info(
                            f"Worker {worker_id} (PID {os.getpid()}) completed {items_processed_by_worker} "
                            f"work items (limit: {max_items_per_worker}). Recycling process to free OS/GPU memory."
                        )
                        break

            worker_logger.info(f"Worker {worker_id} finished processing assigned work items")
        finally:
            if datagen_profiler is not None:
                datagen_profiler.log_worker_summary()
            if task_sampler is not None:
                task_sampler.close()
            trim_memory()


class StretchRolloutRunner(ParallelRolloutRunner):
    """ParallelRolloutRunner with free camera snapping, simulation slowdown, Rerun 3D viz, and process recycling."""

    slow_rate: float | None = None
    visualize: bool = False
    rerun_visualizer: StretchRerunVisualizer | None = None
    max_items_per_worker: int = 10
    show_progress: bool = False

    @staticmethod
    def run_single_rollout(
        episode_seed: int,
        task: Any,
        policy: Any,
        profiler: Any = None,
        viewer: Any = None,
        shutdown_event: Any = None,
        datagen_profiler: Any = None,
        end_on_success: bool = False,
    ) -> bool:
        slow_rate = StretchRolloutRunner.slow_rate
        if slow_rate is None:
            env_val = os.environ.get("STRETCH_DATAGEN_SLOW_RATE")
            if env_val:
                try:
                    slow_rate = float(env_val)
                except ValueError:
                    slow_rate = None

        visualize = StretchRolloutRunner.visualize or viewer is not None
        if not visualize and os.environ.get("STRETCH_DATAGEN_VISUALIZE") == "1":
            visualize = True

        rerun_viz = None
        if visualize:
            if StretchRolloutRunner.rerun_visualizer is None:
                StretchRolloutRunner.rerun_visualizer = StretchRerunVisualizer(spawn=True)
            rerun_viz = StretchRolloutRunner.rerun_visualizer
            rerun_viz.start_episode(episode_seed, task, policy=policy)

        if profiler is not None:
            profiler.start("rollout")
        if datagen_profiler is not None:
            datagen_profiler.start("rollout_total")
            datagen_profiler.start("rollout_reset")

        observation, _info = task.reset()

        if datagen_profiler is not None:
            datagen_profiler.end("rollout_reset")

        if viewer is not None:
            snap_free_camera_to_robot(viewer, task)
            viewer.sync()

        if rerun_viz is not None:
            rerun_viz.log_step(0, task, observation, policy=policy)

        try:
            task.env.current_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
            task.env.current_model.opt.sleep_tolerance = 1e-3
        except AttributeError:
            log.debug("Not setting mujoco sleep. Needs version >=mujoco-3.8")

        step_count = 0
        while not task.is_done():
            # Check for shutdown signal
            if shutdown_event is not None and shutdown_event.is_set():
                if datagen_profiler is not None:
                    datagen_profiler.end("rollout_total")
                return False

            if viewer is not None and hasattr(viewer, "is_running") and not viewer.is_running():
                break

            t_step_wall_start = time.perf_counter()
            t_sim_start = (
                task.env.mj_datas[task.env.current_batch_index].time
                if hasattr(task, "env") and hasattr(task.env, "mj_datas") and task.env.mj_datas
                else None
            )

            # Step with policy
            if profiler is not None:
                profiler.start("policy_get_action")
            if datagen_profiler is not None:
                datagen_profiler.start("policy_get_action")
            # An action chunk is a list of actions to be applied open-loop before a
            # new observation is needed.
            action_chunk = policy.get_action_chunk(observation) or [policy.get_action(observation)]
            if profiler is not None:
                profiler.end("policy_get_action")
            if datagen_profiler is not None:
                datagen_profiler.end("policy_get_action")

            # Step the task
            if profiler is not None:
                profiler.start("task_step")
            if datagen_profiler is not None:
                datagen_profiler.start("task_step")
            if action_chunk[0] is None:
                log.info("Policy returned None action, ending episode")
                break
            observation, reward, terminal, truncated, infos = task.step_chunk(
                action_chunk, stop_on_success=end_on_success
            )
            step_count += len(action_chunk)
            if profiler is not None:
                profiler.end("task_step")
            if datagen_profiler is not None:
                datagen_profiler.end("task_step")

            if viewer is not None:
                viewer.sync()

            if rerun_viz is not None:
                rerun_viz.log_step(step_count, task, observation, policy=policy)

            # Add termination if succ
            if end_on_success and "success" in infos[0] and infos[0]["success"]:
                break

            if slow_rate is not None and slow_rate > 0:
                t_sim_end = (
                    task.env.mj_datas[task.env.current_batch_index].time
                    if hasattr(task, "env") and hasattr(task.env, "mj_datas") and task.env.mj_datas
                    else None
                )
                if t_sim_start is not None and t_sim_end is not None and t_sim_end > t_sim_start:
                    sim_dt = t_sim_end - t_sim_start
                else:
                    policy_dt_ms = getattr(getattr(task, "config", None), "policy_dt_ms", 66.0)
                    sim_dt = (policy_dt_ms / 1000.0) * len(action_chunk)

                target_wall_dt = sim_dt * slow_rate
                elapsed_wall = time.perf_counter() - t_step_wall_start
                sleep_time = target_wall_dt - elapsed_wall
                if sleep_time > 0:
                    time.sleep(sleep_time)

        try:
            task.env.current_model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        except AttributeError:
            pass

        # Save profiler summary
        if profiler is not None:
            profiler.end("rollout")
        if datagen_profiler is not None:
            datagen_profiler.end("rollout_total")
            datagen_profiler.record("step_count_indicator", step_count / 1000.0)

        # Check success if method exists
        success = task.judge_success() if hasattr(task, "judge_success") else False
        return success

    @staticmethod
    def process_single_house(
        worker_id: int,
        worker_logger: Any,
        house_id: int,
        exp_config: Any,
        samples_per_house: int,
        shutdown_event: Any,
        task_sampler: Any,
        preloaded_policy: Any = None,
        max_allowed_sequential_task_sampler_failures: int = 10,
        max_allowed_sequential_rollout_failures: int = 10,
        filter_for_successful_trajectories: bool = False,
        runner_class: Any = None,
        batch_num: int | None = None,
        total_batches: int | None = None,
        datagen_profiler: Any = None,
    ) -> tuple[int, int, bool]:
        """Process all episodes for a single house with aggressive memory trimming and cache clearing."""
        house_success_count = 0
        house_total_count = 0
        irrecoverable_failure_in_house = False

        # Setup directories and check for existing output
        house_output_dir, house_debug_dir, batch_suffix, should_skip = setup_house_dirs(
            exp_config, house_id, batch_num, total_batches
        )
        if should_skip:
            worker_logger.info(
                f"SKIPPING HOUSE {house_id} BATCH {batch_num}/{total_batches}: "
                f"Output already exists at {house_output_dir / f'trajectories{batch_suffix}.h5'}"
            )
            return 0, 0, False

        episode_specs, shared_task_sampler = runner_class.load_episodes_for_house(
            exp_config, house_id, batch_suffix, task_sampler, worker_logger
        )

        if not episode_specs:
            worker_logger.warning(f"No episodes to process for house {house_id}")
            return 0, 0, False

        max_attempts = runner_class.get_max_episode_attempts(
            episode_specs, samples_per_house, exp_config
        )

        # Camera-stripped batched tensors, one per kept episode. The raw frames
        # they came from are released as each episode finishes, so this list
        # stays in the tens of megabytes rather than the tens of gigabytes.
        house_prepared_episodes: list[dict] = []
        house_debug_prepared_episodes: list[dict] = []

        num_sequential_task_sampler_failures = 0
        num_sequential_rollout_failures = 0
        viewer = None

        episode_idx = 0
        while episode_idx < max_attempts:
            should_stop = runner_class.should_stop_early(
                len(house_prepared_episodes), samples_per_house, exp_config=exp_config
            )
            if should_stop:
                break

            if shutdown_event.is_set():
                worker_logger.info(f"Worker {worker_id} house {house_id} received shutdown signal")
                irrecoverable_failure_in_house = True
                break

            if num_sequential_task_sampler_failures >= max_allowed_sequential_task_sampler_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} encountered "
                    f"{num_sequential_task_sampler_failures} consecutive task sampling failures."
                )
                irrecoverable_failure_in_house = True
                break

            if num_sequential_rollout_failures >= max_allowed_sequential_rollout_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} rollout failed across "
                    f"{num_sequential_rollout_failures} retries."
                )
                irrecoverable_failure_in_house = True
                break

            episode_spec = runner_class.get_episode_spec_at_index(episode_specs, episode_idx)

            task = None
            policy = None
            episode_task_sampler = None
            success = False
            task_sampling_failed = False
            house_invalid = False

            if datagen_profiler is not None:
                datagen_profiler.start("episode_total")

            episode_config = runner_class.prepare_episode_config(
                exp_config, episode_spec, episode_idx
            )

            with cleanup_context():
                if viewer is not None:
                    viewer.close()
                    viewer = None

                task_sampling_start = time.perf_counter()

                try:
                    episode_task_sampler = runner_class.get_episode_task_sampler(
                        episode_config, episode_spec, shared_task_sampler, datagen_profiler
                    )
                    task = runner_class.sample_task_from_spec(
                        episode_task_sampler, house_id, episode_spec, episode_idx
                    )

                    if task is None:
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}: task sampling returned None"
                        )
                        house_invalid = True
                    else:
                        if datagen_profiler is not None:
                            datagen_profiler.record(
                                "task_sampling", time.perf_counter() - task_sampling_start
                            )
                            task.set_datagen_profiler(datagen_profiler)

                        num_sequential_task_sampler_failures = 0
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}/{max_attempts} "
                            f"collected={len(house_prepared_episodes)}/{samples_per_house}"
                        )
                except HouseInvalidForTask as e:
                    traceback.print_exc()
                    worker_logger.warning(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} HouseInvalidForTask: {e.reason}"
                    )
                    house_invalid = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )
                except Exception as e:
                    traceback.print_exc()
                    worker_logger.error(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} task sampling error: {str(e)}"
                    )
                    num_sequential_task_sampler_failures += 1
                    task_sampling_failed = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )

                if task is not None and not house_invalid and not task_sampling_failed:
                    try:
                        policy = setup_policy(
                            episode_config, task, preloaded_policy, datagen_profiler
                        )
                        viewer = setup_viewer(episode_config, task, policy, viewer)

                        episode_seed = runner_class.get_episode_seed(
                            episode_idx, episode_spec, episode_task_sampler
                        )

                        success = runner_class.run_single_rollout(
                            episode_seed=episode_seed,
                            task=task,
                            policy=policy,
                            profiler=episode_config.profiler,
                            viewer=viewer,
                            shutdown_event=shutdown_event,
                            datagen_profiler=datagen_profiler,
                            end_on_success=exp_config.end_on_success,
                        )

                        num_sequential_rollout_failures = 0

                        object_name = "unknown"
                        if hasattr(task, "config") and hasattr(task.config, "task_config"):
                            if hasattr(task.config.task_config, "pickup_obj_name"):
                                object_name = task.config.task_config.pickup_obj_name

                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} "
                            f"object {object_name} completed with success={success}"
                        )

                        should_save = success or not filter_for_successful_trajectories
                        history = task.get_history()

                        should_save_debug = not should_save and random.random() < 0.01

                        # Encode and release this episode's frames now rather than
                        # at the end of the house. `history` aliases the task's own
                        # observation cache and `prepare_episode_for_saving` empties
                        # it in place, so the frames are gone before the next
                        # episode's scene is loaded.
                        if should_save or should_save_debug:
                            if should_save:
                                target_dir = house_output_dir
                                target_list = house_prepared_episodes
                            else:
                                target_dir = house_debug_dir
                                target_list = house_debug_prepared_episodes
                                worker_logger.info(
                                    f"Saving failed trajectory for debug (seed: {episode_seed})"
                                )

                            prepared = flush_episode_to_disk(
                                worker_logger,
                                history=history,
                                sensor_suite=task.sensor_suite,
                                save_dir=target_dir,
                                exp_config=exp_config,
                                batch_suffix=batch_suffix,
                                episode_idx=len(target_list),
                                datagen_profiler=(
                                    datagen_profiler if should_save else None
                                ),
                            )
                            if prepared is not None:
                                target_list.append(prepared)

                        del history
                        trim_memory()
                        log_memory_usage(
                            worker_logger,
                            prefix=f"Worker {worker_id} house {house_id} "
                            f"after episode {episode_idx}: ",
                        )

                        house_total_count += 1
                        if success:
                            house_success_count += 1
                        else:
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(asset_uid, "rollout failed")

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")
                            datagen_profiler.log_episode_summary(
                                episode_idx=episode_idx,
                                house_id=house_id,
                                success=success,
                            )
                    except Exception as e:
                        worker_logger.error(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} rollout error: {str(e)}"
                        )
                        traceback.print_exc()
                        num_sequential_rollout_failures += 1

                        try:
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(
                                    asset_uid, f"rollout exception: {e}"
                                )
                        except Exception:
                            pass

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")
                else:
                    if datagen_profiler is not None:
                        datagen_profiler.end("episode_total")

                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=episode_task_sampler,
                    preloaded_policy=preloaded_policy,
                    close_task_sampler=runner_class.should_close_episode_task_sampler(),
                )

            if house_invalid:
                irrecoverable_failure_in_house = True
                break

            episode_idx += 1

        if viewer is not None:
            viewer.close()
            viewer = None

        if shutdown_event.is_set():
            worker_logger.info(
                f"Worker {worker_id} house {house_id} shutdown requested, skipping save"
            )
            # The HDF5 is what `setup_house_dirs` resumes off, so not writing it
            # means this house batch gets redone. Videos are now written as each
            # episode finishes rather than alongside the HDF5, so they have to be
            # cleared too -- otherwise the re-run leaves stale MP4s behind
            # whenever it keeps fewer episodes than this attempt did.
            for stale_dir in (house_output_dir, house_debug_dir):
                for stale_mp4 in Path(stale_dir).glob(f"episode_*{batch_suffix}.mp4"):
                    try:
                        stale_mp4.unlink()
                    except OSError as e:
                        worker_logger.warning(f"Could not remove partial video {stale_mp4}: {e}")
            return house_success_count, house_total_count, True

        save_prepared_trajectories(
            worker_logger,
            house_prepared_episodes,
            house_output_dir,
            exp_config,
            batch_suffix,
            datagen_profiler,
            batch_num,
            total_batches,
        )

        save_prepared_trajectories(
            worker_logger,
            house_debug_prepared_episodes,
            house_debug_dir,
            exp_config,
            batch_suffix,
            datagen_profiler=None,
            batch_num=batch_num,
            total_batches=total_batches,
        )

        # Drop the prepared tensors before trimming; the previous version trimmed
        # while they were still in scope, so it could not reclaim them.
        house_prepared_episodes.clear()
        house_debug_prepared_episodes.clear()
        trim_memory()

        worker_logger.info(
            f"Worker {worker_id} completed house {house_id}: "
            f"{house_success_count}/{house_total_count} successful episodes"
        )

        if datagen_profiler is not None:
            datagen_profiler.log_house_summary(
                house_id=house_id,
                success_count=house_success_count,
                total_count=house_total_count,
            )

        return house_success_count, house_total_count, irrecoverable_failure_in_house

    def run(self, preloaded_policy: Any = None) -> tuple[int, int]:
        """Run rollouts using a pool of recycled worker processes to prevent memory leaks."""
        total_expected_episodes = sum(wi[1] for wi in self.work_items)
        self.logger.info(
            f"Starting rollout of {self.total_houses} houses "
            f"split into {len(self.work_items)} work items ({total_expected_episodes} total episodes) "
            f"using {self.config.num_workers} worker processes (recycling every {self.max_items_per_worker} items)"
        )

        self.logger.info("Evaluation configuration:")
        self.logger.info(pprint.pformat(self.config.model_dump()))
        self.config.save_config(output_dir=Path(self.config.output_dir))

        start_time = time.time()

        with DatagenProgressBar(self, enabled=self.show_progress):
            self._run_work_items(preloaded_policy, start_time)

        return self._summarize(start_time)

    def _run_work_items(self, preloaded_policy: Any, start_time: float) -> None:
        """Spawn and babysit worker processes until every work item is claimed."""
        if self.config.num_workers > 1 or (not self.visualize and self.max_items_per_worker):
            target_workers = self.config.num_workers
            active_processes: dict[int, Any] = {}
            next_worker_id = 0

            def spawn_worker(wid: int) -> Any:
                p = mp_context.Process(
                    target=stretch_house_processing_worker,
                    args=(
                        wid,
                        self.config,
                        self.work_items,
                        self.shutdown_event,
                        self.counter_lock,
                        self.house_counter,
                        self.success_count,
                        self.total_count,
                        self.completed_houses,
                        self.skipped_houses,
                        self.max_allowed_sequential_task_sampler_failures,
                        self.max_allowed_sequential_rollout_failures,
                        self.max_allowed_sequential_irrecoverable_failures,
                        preloaded_policy,
                        self.config.filter_for_successful_trajectories,
                        type(self),
                        self.max_items_per_worker,
                    ),
                )
                p.start()
                return p

            initial_count = min(target_workers, len(self.work_items))
            for _ in range(initial_count):
                active_processes[next_worker_id] = spawn_worker(next_worker_id)
                next_worker_id += 1

            last_log_time = start_time
            log_interval = 60

            while active_processes:
                dead_ids = []
                for wid, p in list(active_processes.items()):
                    if not p.is_alive():
                        p.join()
                        p.close()
                        dead_ids.append(wid)

                for wid in dead_ids:
                    del active_processes[wid]
                    if not self.shutdown_event.is_set():
                        with self.counter_lock:
                            has_more_work = self.house_counter.value < len(self.work_items)
                        if has_more_work and len(active_processes) < target_workers:
                            active_processes[next_worker_id] = spawn_worker(next_worker_id)
                            next_worker_id += 1

                current_time = time.time()
                if self.wandb_enabled and (current_time - last_log_time) >= log_interval:
                    try:
                        elapsed_time = current_time - start_time
                        completed = self.completed_houses.value
                        skipped = self.skipped_houses.value
                        success = self.success_count.value
                        total = self.total_count.value
                        active = sum(1 for p in active_processes.values() if p.is_alive())
                        total_work_items = len(self.work_items)
                        success_rate = success / total if total > 0 else 0.0
                        episodes_per_second = total / elapsed_time if elapsed_time > 0 else 0.0
                        completion_percentage = (completed + skipped) / total_work_items * 100

                        import wandb

                        wandb.log(
                            {
                                "elapsed_time_seconds": elapsed_time,
                                "elapsed_time_hours": elapsed_time / 3600,
                                "completed_houses": completed,
                                "skipped_houses": skipped,
                                "success_count": success,
                                "total_count": total,
                                "success_rate": success_rate,
                                "episodes_per_second": episodes_per_second,
                                "active_workers": active,
                                "completion_percentage": completion_percentage,
                            }
                        )
                        self.logger.info(
                            f"Progress: {completed}/{total_work_items} work items completed "
                            f"({completion_percentage:.1f}%), {success}/{total} successful episodes "
                            f"({success_rate * 100:.1f}%), {active} workers active"
                        )
                        last_log_time = current_time
                    except Exception as e:
                        self.logger.warning(f"WandB periodic logging failed: {e}")

                time.sleep(1)

        else:
            # Single-worker in-process mode (used for --visualize interactive viewer)
            stretch_house_processing_worker(
                worker_id=0,
                exp_config=self.config,
                work_items=self.work_items,
                shutdown_event=self.shutdown_event,
                counter_lock=self.counter_lock,
                house_counter=self.house_counter,
                success_count=self.success_count,
                total_count=self.total_count,
                completed_houses=self.completed_houses,
                skipped_houses=self.skipped_houses,
                max_allowed_sequential_task_sampler_failures=self.max_allowed_sequential_task_sampler_failures,
                max_allowed_sequential_rollout_failures=self.max_allowed_sequential_rollout_failures,
                max_allowed_sequential_irrecoverable_failures=self.max_allowed_sequential_irrecoverable_failures,
                preloaded_policy=preloaded_policy,
                filter_for_successful_trajectories=self.config.filter_for_successful_trajectories,
                runner_class=type(self),
                max_items_per_worker=None,
            )

    def _summarize(self, start_time: float) -> tuple[int, int]:
        """Log the run totals, push the final wandb row, and report the counts."""
        success_count_val = self.success_count.value
        total_count_val = self.total_count.value
        completed_houses_val = self.completed_houses.value
        skipped_houses_val = self.skipped_houses.value
        success_rate = success_count_val / total_count_val if total_count_val > 0 else 0.0

        self.logger.info(
            f"Completed {completed_houses_val} work items, skipped {skipped_houses_val} work items"
        )
        self.logger.info(f"Success count: {success_count_val}, Total count: {total_count_val}")
        self.logger.info(f"Success rate: {success_rate * 100:.2f}%")

        if self.wandb_enabled:
            try:
                import wandb

                final_elapsed_time = time.time() - start_time
                wandb.log(
                    {
                        "final_success_count": success_count_val,
                        "final_total_count": total_count_val,
                        "final_success_rate": success_rate,
                        "final_completed_houses": completed_houses_val,
                        "final_skipped_houses": skipped_houses_val,
                        "final_elapsed_time_seconds": final_elapsed_time,
                        "final_elapsed_time_hours": final_elapsed_time / 3600,
                    }
                )
                wandb.finish()
            except Exception as e:
                self.logger.warning(f"WandB final logging failed: {e}")

        return success_count_val, total_count_val


def generate_rollouts(
    task: str,
    output_dir: Path,
    episodes: int | None = None,
    num_workers: int = 1,
    scene_dataset: str | None = None,
    data_split: str | None = None,
    houses: int | None = None,
    seed: int | None = None,
    keep_failures: bool = False,
    visualize: bool = False,
    slow_rate: float | None = None,
    max_items_per_worker: int = 10,
    show_progress: bool = False,
) -> Path:
    """Run the data generation pipeline for one task family.

    Args:
        task: a key of `DATAGEN_CONFIGS`.
        output_dir: where the rollouts go. The pipeline appends its own
            `<ConfigName>/<timestamp>/` beneath this.
        episodes: total episodes to attempt. Spread over `houses` houses.
        num_workers: parallel rollout worker processes.
        scene_dataset: override the config's scene dataset, e.g. `procthor-10k`.
        data_split: `train`, `val` or `test`. Left at the config's default
            (`train`) unless given -- generating fine-tuning data out of `val` is
            how a benchmark score stops meaning anything.
        houses: how many houses to draw from. Defaults to enough that each house
            contributes a handful of episodes rather than hundreds, which is
            what keeps the scene distribution wide.
        seed: task-sampling seed.
        keep_failures: keep failed trajectories in the rollout dataset.
        visualize: watch the rollouts in MuJoCo's passive viewer. Requires
            `num_workers == 1` -- see `main()`.
        slow_rate: slow down simulation by a time factor (e.g. 1.0 for real-time,
            2.0 for 2x slower than real-time).
        max_items_per_worker: number of work items (houses) a worker process handles
            before being recycled to release system and GPU driver memory.
        show_progress: draw a progress bar over the work items on stderr.

    Returns:
        The directory the pipeline actually wrote to.
    """
    module_name, class_name = qualified_config_name(task).split(":")
    importlib.import_module(module_name)
    config = get_config_class(class_name)()

    if scene_dataset is not None:
        config.scene_dataset = scene_dataset
    if data_split is not None:
        config.data_split = data_split
    if seed is not None:
        config.seed = seed
    config.num_workers = num_workers
    config.use_wandb = False
    config.use_passive_viewer = visualize
    config.filter_for_successful_trajectories = not keep_failures

    if episodes is not None:
        _spread_episodes(config, episodes, houses)

    config.output_dir = Path(output_dir) / task
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.save_config()

    StretchRolloutRunner.visualize = visualize
    StretchRolloutRunner.max_items_per_worker = max_items_per_worker
    StretchRolloutRunner.show_progress = show_progress
    if visualize:
        os.environ["STRETCH_DATAGEN_VISUALIZE"] = "1"
    elif "STRETCH_DATAGEN_VISUALIZE" in os.environ:
        del os.environ["STRETCH_DATAGEN_VISUALIZE"]

    if slow_rate is not None:
        StretchRolloutRunner.slow_rate = slow_rate
        os.environ["STRETCH_DATAGEN_SLOW_RATE"] = str(slow_rate)
    elif "STRETCH_DATAGEN_SLOW_RATE" in os.environ:
        del os.environ["STRETCH_DATAGEN_SLOW_RATE"]
        StretchRolloutRunner.slow_rate = None
    else:
        StretchRolloutRunner.slow_rate = None

    log.info(
        f"[datagen] {class_name} | {config.scene_dataset}/{config.data_split} | "
        f"{len(config.task_sampler_config.house_inds)} houses x "
        f"{config.task_sampler_config.samples_per_house} episodes | "
        f"{num_workers} workers (recycling every {max_items_per_worker} items) -> {config.output_dir}"
    )
    successes, total = StretchRolloutRunner(config).run()
    log.info(f"[datagen] {task}: {successes}/{total} episodes succeeded")
    return config.output_dir


def _spread_episodes(config, episodes: int, houses: int | None) -> None:
    """Turn a total episode count into houses x samples-per-house.

    The samplers count in those two numbers rather than in episodes, and how the
    total is split matters: all 2000 episodes in one house is 2000 rollouts of
    one room. Defaults to roughly 4 episodes per house, capped at the 10 the
    sampler will retry a single house for before it gives up on it.
    """
    sampler_config = config.task_sampler_config
    per_house = 4 if houses is None else max(1, episodes // max(houses, 1))
    per_house = min(per_house, 10)
    house_count = houses if houses is not None else max(1, -(-episodes // per_house))

    available = list(sampler_config.house_inds or [])
    if len(available) < house_count:
        # The datagen configs ship a short house list (the first few) for
        # debugging; a real run needs more of the dataset than that.
        available = list(range(house_count))
    sampler_config.house_inds = available[:house_count]
    sampler_config.samples_per_house = per_house
    sampler_config.max_tasks = episodes


@click.command()
@click.option(
    "--task",
    "tasks",
    multiple=True,
    type=click.Choice(sorted(DATAGEN_CONFIGS)),
    default=("pick",),
    help="Task family to generate. Repeatable; several are pooled into one dataset.",
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Episodes to attempt per task. Defaults to the config's own house list.",
)
@click.option("--num-workers", type=int, default=1, help="Parallel rollout workers.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Root for `rollouts/` and `lerobot/`.",
)
@click.option(
    "--rollouts",
    "rollout_dirs",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    help="Skip generation and export these existing rollout directories instead.",
)
@click.option(
    "--action-space",
    type=click.Choice(ACTION_SPACES),
    default="stretch",
    help="Action/state space to export in. Only 'stretch', the native 10-dim "
    "move-group vector. See lerobot_export.py.",
)
@click.option(
    "--scene-dataset",
    type=str,
    default=None,
    help="Override the scene dataset, e.g. procthor-10k for a fast local run.",
)
@click.option(
    "--data-split",
    type=click.Choice(["train", "val", "test"]),
    default=None,
    help="Scene split. Leave unset to use the config's default (train).",
)
@click.option("--houses", type=int, default=None, help="How many houses to draw episodes from.")
@click.option("--seed", type=int, default=None, help="Task-sampling seed.")
@click.option(
    "--keep-failures/--successful-only",
    default=False,
    help="Include episodes the task judged unsuccessful. Off by default: a partial "
    "expert's failures are counter-examples, not demonstrations.",
)
@click.option(
    "--visualize",
    is_flag=True,
    help="Watch each episode live in MuJoCo's passive viewer, from the robot's "
    "chase camera. Forces --num-workers 1.",
)
@click.option(
    "--slow-rate",
    "--slow_rate",
    "slow_rate",
    type=float,
    default=None,
    help="Slow down simulation by a time factor (e.g. 1.0 for real-time, 2.0 for 2x slower than real-time).",
)
@click.option(
    "--max-items-per-worker",
    type=int,
    default=10,
    help="Number of house work items a worker process handles before being recycled to release system/GPU memory.",
)
@click.option(
    "--progress/--no-progress",
    "show_progress",
    default=None,
    help="Show a progress bar for the generation and export stages. Defaults to on "
    "when stderr is a terminal, off when it is a log file.",
)
@click.option("--export/--no-export", "want_export", default=True, help="Run the export stage.")
@click.option(
    "--fps", type=float, default=15.0, help="Frame rate to record in the dataset metadata."
)
def main(
    tasks: tuple[str, ...],
    episodes: int | None,
    num_workers: int,
    output_dir: Path,
    rollout_dirs: tuple[Path, ...],
    action_space: str,
    scene_dataset: str | None,
    data_split: str | None,
    houses: int | None,
    seed: int | None,
    keep_failures: bool,
    visualize: bool,
    slow_rate: float | None,
    max_items_per_worker: int,
    show_progress: bool | None,
    want_export: bool,
    fps: float,
) -> None:
    if show_progress is None:
        show_progress = sys.stderr.isatty()

    # With a bar on screen, this process' own log lines have to go out through
    # `tqdm.write` or they land on top of it. Worker processes keep their own
    # handlers and are not covered; see `TqdmLoggingHandler`.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[TqdmLoggingHandler()] if show_progress else None,
    )

    # MolmoSpaces renders off-screen through EGL; without this a headless run
    # fails inside the camera manager rather than at startup. The passive viewer
    # is unaffected -- it is the C++ `simulate` app, which brings its own GLFW
    # window regardless of what the offscreen camera renderer is using.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    if visualize and num_workers != 1:
        # `ParallelRolloutRunner.run()` only stays in the main process for a
        # single worker; above that the rollouts happen in spawned processes,
        # where a viewer window would be launched per worker if it opened at all.
        click.secho("--visualize forces --num-workers 1.", fg="yellow")
        num_workers = 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rollout_dirs:
        directories = [Path(directory) for directory in rollout_dirs]
        click.secho(f"Exporting {len(directories)} existing rollout directories.", fg="cyan")
    else:
        directories = [
            generate_rollouts(
                task=task,
                output_dir=output_dir / "rollouts",
                episodes=episodes,
                num_workers=num_workers,
                scene_dataset=scene_dataset,
                data_split=data_split,
                houses=houses,
                seed=seed,
                keep_failures=keep_failures,
                visualize=visualize,
                slow_rate=slow_rate,
                max_items_per_worker=max_items_per_worker,
                show_progress=show_progress,
            )
            for task in tasks
        ]

    if not want_export:
        click.secho(f"Rollouts under {output_dir / 'rollouts'}", fg="green")
        return

    dataset_dir = output_dir / "lerobot"
    metadata = export_lerobot_dataset(
        rollout_dirs=directories,
        output_dir=dataset_dir,
        action_space=action_space,
        successful_only=not keep_failures,
        fps=fps,
        show_progress=show_progress,
    )
    click.echo("")
    click.secho(
        f"{metadata.num_episodes} episodes / {metadata.num_frames} frames "
        f"in {action_space} action space -> {dataset_dir}",
        fg="green",
    )
    click.echo(
        "Fine-tune with:\n"
        f"  python -m examples.machine_learning.molmospaces.finetuning.finetune "
        f"--dataset {dataset_dir} --trainer openpi"
    )


if __name__ == "__main__":
    main()

