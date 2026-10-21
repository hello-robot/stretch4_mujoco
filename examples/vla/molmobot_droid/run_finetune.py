"""
Fine-tune MolmoBot-DROID to drive Stretch 4 directly, on the Stretch 4 trajectories
`run_benchmark_follow_franka.py` saves, with MolmoBot's own trainer (`finetune/train.py`).

The fine-tuned policy's state and actions are Stretch 4's joints, the ones retargeting moves
(`finetune/stretch4_joints.py`): base heading, lift, arm, wrist yaw/pitch/roll, an unused slot
and the gripper, in the Franka's 8 slots so the pretrained action head loads. Its exo view is a
head fisheye, DROID-framed (--use_fisheye: left, right, or `tool`, the one on the tool's side each
step, as `run_stretch4_sim.py` and the real robot pick it; with --use_rectified_fisheye, redrawn
as the DROID exo camera instead, as --rectify does), and its wrist view a gripper camera
(--use_gripper), 2 frames 8 steps apart each, as MolmoBot-DROID's are. Everything else about the
model is the checkpoint's.

What is trained. The LLM stays frozen: on 32 GB, its weights and gradients alone would not fit.
- --use_action_head (default): the flow-matching action expert, which turns the LLM's features
  into actions. --no-use_action_head freezes it.
- --use_vision_transformer: the SigLIP2 vision encoder and its connector into the LLM, to adapt
  them to Stretch's cameras. Off by default: it backpropagates through the whole LLM, so each
  step costs several times as much.
The optimizer is 8-bit AdamW (bitsandbytes).

Training runs on one GPU under torchrun, in MolmoBot's own venv (`molmobot/MolmoBot/.venv`,
Python 3.11, git-ignored with the clone). To set it up:

    git clone https://github.com/allenai/MolmoBot.git examples/vla/molmobot_droid/molmobot
    cd examples/vla/molmobot_droid/molmobot/MolmoBot
    uv sync --extra train --python 3.11
    uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchvision \\
        --reinstall-package torchaudio torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \\
        --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv/bin/python bitsandbytes

(The lock's torch is the CUDA 12.6 build, which has no kernels for Blackwell GPUs like the RTX
5090; the cu128 one does.)

Writes to <out>/<run name>/:
- `dataset/`: links to the episodes' run directories, split into train and val;
- `norm_stats.yaml`: Stretch's state and action normalization, computed on the first run;
- `spec.json`: what `finetune/train.py` was given;
- MolmoBot's checkpoints: `step<N>/` every --save-interval steps (sharded, the latest kept,
  for resuming: run the same command again) and the final `step<steps>-unsharded/`, which
  `checkpoint.load_policy()` loads.

Usage:
    python -m examples.vla.molmobot_droid.run_finetune --use_parallel_gripper \\
        --data outputs/molmobot_droid/follow_franka_eh-8_n-8_slow-0_wait-0_pg-1_offmm-4,0,0_offdeg-0,0,0
    python -m examples.vla.molmobot_droid.run_finetune --data outputs/molmobot_droid --use_parallel_gripper \\
        --use_fisheye right --use_gripper right --use_vision_transformer --no-use_action_head
    python -m examples.vla.molmobot_droid.run_finetune --data ... --use_parallel_gripper --use_fisheye tool
    python -m examples.vla.molmobot_droid.run_finetune --data ... --use_parallel_gripper --use_fisheye tool \
        --use_rectified_fisheye
    # Check the whole pipeline in a few minutes: a tiny randomly initialized model, 20 steps.
    python -m examples.vla.molmobot_droid.run_finetune --data ... --use_parallel_gripper --debug --steps 20
    # Any of MolmoBot's TrainConfig options after --, e.g. the action expert's learning rate:
    python -m examples.vla.molmobot_droid.run_finetune --data ... -- --optimizer.action_expert_learning_rate=3e-5
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import click

from examples.vla.molmobot_droid.checkpoint import MOLMOBOT_LOCAL_DIR, download_checkpoint
from examples.vla.molmobot_droid.finetune.dataset import build_dataset, find_episodes
from examples.vla.molmobot_droid.finetune.stretch4_joints import MOVE_GROUPS
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import PARALLEL_GRIPPER_TOOL, STRETCH_GRIPPER_TOOL

MOLMOBOT_DIR = MOLMOBOT_LOCAL_DIR / "MolmoBot"
TORCHRUN = MOLMOBOT_DIR / ".venv" / "bin" / "torchrun"
TRAIN_SCRIPT = Path(__file__).parent / "finetune" / "train.py"
PRESET = "stretch4"
"""The action and camera preset `finetune/train.py` registers (its PRESET)."""

VAL_MAX_EXAMPLES = 256
"""Val examples per evaluation: a few minutes at most, every --val-interval steps."""

TRAINING_ENV = {
    # As MolmoBot's README sets them for training.
    "OLMO_SHARED_FS": "1",
    "OMP_NUM_THREADS": "8",
    "LOG_FILTER_TYPE": "rank0_only",
    "OLMO_NUM_THREADS_ENV_VAR": "8",
    "NCCL_TIMEOUT_MINUTES": "60",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def run_name(fisheye: str, rectified: bool, gripper: str, parallel_gripper: bool, action_head: bool, vit: bool,
             steps: int, debug: bool) -> str:
    return (
        f"stretch4_fish-{fisheye}{'-rect' if rectified else ''}_grip-{gripper}_pg-{int(parallel_gripper)}_ah-{int(action_head)}_vit-{int(vit)}"
        f"_steps-{steps}" + ("_debug" if debug else "")
    )


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--data", multiple=True, required=True, type=click.Path(exists=True, path_type=Path),
              help="A run directory of run_benchmark_follow_franka.py, or a directory holding several: every "
              "trajectory saved under it is used (that matches the gripper and has the cameras). Repeatable.")
@click.option("--use_fisheye", "--use-fisheye", "use_fisheye", type=click.Choice(["left", "right", "tool"]),
              default="left", show_default=True,
              help="The head fisheye, DROID-framed, that is the policy's exo view: tool is the one on the tool's "
              "side each step (fisheye_tool), as run_stretch4_sim.py and the real robot pick it.")
@click.option("--use_rectified_fisheye", "--use-rectified-fisheye", "use_rectified_fisheye", is_flag=True,
              help="The --use_fisheye fisheye redrawn as the DROID exo camera (its _rectified video, as --rectify "
              "draws it) instead of only cropped to its aspect.")
@click.option("--use_gripper", "--use-gripper", "use_gripper", type=click.Choice(["left", "right"]), default="left",
              show_default=True, help="The gripper camera that is the policy's wrist view.")
@click.option("--use_parallel_gripper", "--use-parallel-gripper", "use_parallel_gripper", is_flag=True,
              help="Train on the trajectories recorded with the parallel jaw gripper (PG4) instead of the "
              "Stretch gripper (SG4).")
@click.option("--use_action_head/--no-use_action_head", "--use-action-head/--no-use-action-head", "use_action_head",
              default=True, show_default=True, help="Train the action expert (the action head).")
@click.option("--use_vision_transformer/--no-use_vision_transformer",
              "--use-vision-transformer/--no-use-vision-transformer", "use_vision_transformer", default=False,
              show_default=True, help="Train the vision transformer and its connector into the LLM.")
@click.option("--steps", type=int, default=50_000, show_default=True, help="Optimizer steps.")
@click.option("--global-batch-size", type=int, default=32, show_default=True, help="Examples per optimizer step.")
@click.option("--device-batch-size", type=int, default=8, show_default=True,
              help="Examples per forward pass; the rest of a step's are gradient accumulation.")
@click.option("--val-fraction", type=float, default=0.1, show_default=True,
              help="Of the episodes, how many are held out for validation loss.")
@click.option("--val-interval", type=int, default=1000, show_default=True)
@click.option("--save-interval", type=int, default=2000, show_default=True,
              help="Steps between (sharded) checkpoints, for resuming.")
@click.option("--num-workers", type=int, default=4, show_default=True, help="Data loader workers.")
@click.option("--out", default="outputs/molmobot_droid/finetune", show_default=True, type=click.Path(path_type=Path))
@click.option("--checkpoint", default=None, help="Checkpoint to start from. Default: MolmoBot-DROID from Hugging Face.")
@click.option("--wandb", "use_wandb", is_flag=True, help="Log to Weights & Biases (needs WANDB_PROJECT and WANDB_ENTITY).")
@click.option("--debug", is_flag=True, help="A tiny randomly initialized model instead of the checkpoint's "
              "weights (its config still), batch 1 and no validation: to check the pipeline end to end.")
@click.argument("trainer_options", nargs=-1, type=click.UNPROCESSED)
def main(data, use_fisheye, use_rectified_fisheye, use_gripper, use_parallel_gripper, use_action_head, use_vision_transformer, steps,
         global_batch_size, device_batch_size, val_fraction, val_interval, save_interval, num_workers, out,
         checkpoint, use_wandb, debug, trainer_options):
    """Fine-tune MolmoBot-DROID on Stretch 4 trajectories (see the module docstring). Options after
    the ones here go to MolmoBot's TrainConfig, e.g. --optimizer.action_expert_learning_rate=3e-5."""
    if not (use_action_head or use_vision_transformer):
        raise click.UsageError("Nothing to train: give --use_action_head or --use_vision_transformer (the LLM stays frozen)")
    if not TORCHRUN.exists():
        raise click.ClickException(f"No MolmoBot training venv at {TORCHRUN.parent.parent}: set it up as "
                                   "`python -m examples.vla.molmobot_droid.run_finetune --help`'s module docstring says")

    tool = PARALLEL_GRIPPER_TOOL if use_parallel_gripper else STRETCH_GRIPPER_TOOL
    exo = f"fisheye_{use_fisheye}" + ("_rectified" if use_rectified_fisheye else "")
    cameras = [exo, f"gripper_{use_gripper}"]  # exo then wrist, as MolmoBot-DROID's
    episodes = find_episodes(data)
    usable = [e for e in episodes if e.tool_name == tool and set(cameras) <= set(e.cameras)]
    other_tool = sum(e.tool_name != tool for e in episodes)
    if not usable:
        hint = f" ({other_tool} with the other gripper: {'drop' if use_parallel_gripper else 'add'} --use_parallel_gripper?)" if other_tool else ""
        raise click.ClickException(f"No {tool} trajectories with {cameras} under {', '.join(map(str, data))}{hint}")
    click.secho(
        f"{len(usable)} trajectories, {sum(e.steps for e in usable)} steps"
        + (f" ({len(episodes) - len(usable)} others left out)" if len(usable) < len(episodes) else ""),
        fg="green",
    )

    name = run_name(use_fisheye, use_rectified_fisheye, use_gripper, use_parallel_gripper, use_action_head,
                    use_vision_transformer, steps, debug)
    save_dir = out.resolve() / name
    counts = build_dataset(usable, save_dir / "dataset", val_fraction)
    click.secho(f"Dataset: {save_dir / 'dataset'} ({counts['train']} train, {counts['val']} val)", fg="green")
    checkpoint_dir = Path(checkpoint).resolve() if checkpoint else download_checkpoint()

    validation = (
        ["--no_val"] if counts["val"] == 0
        else ["--val_interval", str(val_interval), "--val_max_examples", str(VAL_MAX_EXAMPLES)]
    )
    args = [
        "--data_paths", str(save_dir / "dataset"),
        "--action_preset", PRESET,
        "--camera_preset", PRESET,
        # --debug makes the global batch the world size: 1.
        "--device_batch_size", str(1 if debug else device_batch_size),
        "--global_batch_size", str(global_batch_size),
        "--stats_path", str(save_dir / "norm_stats.yaml"),
        "--num_workers", str(num_workers),
        "--exp_name", name,
        *validation,
        *(["--debug"] if debug else []),
        # MolmoBot's TrainConfig, as OmegaConf overrides.
        f"--save_folder={save_dir}",
        f"--max_duration={steps}",
        "--ft_llm=False",
        f"--ft_vit={use_vision_transformer}",
        f"--ft_connector={use_vision_transformer}",
        f"--save_interval={save_interval}",
        "--save_final_unsharded_checkpoint=True",
        "--checkpoint_retention_frequency=null",
        *([] if use_wandb else ["--wandb=null"]),
        *trainer_options,
    ]
    spec = {
        "molmobot_dir": str(MOLMOBOT_DIR),
        "checkpoint": str(checkpoint_dir),
        "move_groups": MOVE_GROUPS,
        "cameras": cameras,
        "train_action_head": use_action_head,
        "tool_name": tool,
        "args": args,
    }
    spec_path = save_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    command = [str(TORCHRUN), "--standalone", "--nproc-per-node=1", str(TRAIN_SCRIPT), str(spec_path)]
    click.secho(f"Training {name}: {' '.join(command)}", fg="cyan")
    env = {**TRAINING_ENV, **os.environ}
    completed = subprocess.run(command, env=env, cwd=MOLMOBOT_DIR)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    final = save_dir / f"step{steps}-unsharded"
    click.secho(f"Done: {final}\n  load it with checkpoint.load_policy({str(final)!r})", fg="green")


if __name__ == "__main__":
    main()
