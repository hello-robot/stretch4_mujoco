"""
MolmoBot's own trainer (`launch_scripts/train_molmobot.py`), set up to fine-tune MolmoBot-DROID on
Stretch 4's trajectories. `run_finetune.py` runs it under torchrun in MolmoBot's venv (Python
3.11, its `train` extra), with a spec it writes; it is not meant to be run by hand, and imports
nothing of this repo's.

What it changes, all before `train_molmobot.main()` runs:
- The model is the checkpoint's own config (its `config.yaml`'s `model`), not the one
  `train_molmobot.get_model()` rebuilds from it, so everything MolmoBot-DROID was trained with
  carries over (2 frames 8 steps apart per camera, 4 images, its pooling, the action expert's
  shape, ...). Only the normalization becomes Stretch's, which `train_molmobot` computes from
  the data into the stats file.
- Stretch 4's joints (move groups) and cameras are presets, named PRESET.
- The optimizer is bitsandbytes' 8-bit AdamW (`LocalShardAdamW8bit`) instead of torch's AdamW.
- Without `train_action_head`, the action expert is frozen.
- Every step's metrics the trainer logs, and every validation's, are appended to the spec's
  `metrics` file (`plot_finetune.py` plots them): without W&B they would only reach the console.

Spec (JSON): molmobot_dir, checkpoint, move_groups, cameras, train_action_head, metrics, and args,
`train_molmobot.py`'s command line after the checkpoint.
"""

import json
import os
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
from torch.distributed.tensor import DTensor

PRESET = "stretch4"


class LocalShardAdamW8bit(bnb.optim.AdamW8bit):
    """
    bitsandbytes' 8-bit AdamW for parameters FSDP2 shards (DTensors), which its kernels refuse.
    Each step updates every parameter's local shard in place, through a plain tensor sharing its
    storage, with the gradient's local shard. The state stays keyed by the parameter itself, so
    `param_groups`, `state_dict()` and distributed checkpointing see it as for any optimizer.

    On one GPU, as here, a parameter's local shard is the whole of it, so this is exactly
    AdamW8bit. On several, each rank would quantize its own shard's state blocks.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                local = (p.to_local() if isinstance(p, DTensor) else p).detach()
                local.grad = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                self.state[local] = self.state[p]  # the same dict: what the step writes is p's
                try:
                    if not self.state[local]:
                        self.init_state(group, local, gindex, pindex)
                    self.update_step(group, local, gindex, pindex)
                finally:
                    del self.state[local]
        return loss


def use_8bit_adamw() -> None:
    """Make MolmoBot's optimizer config build `LocalShardAdamW8bit`, with its param groups
    (per-part learning rates, betas, eps, weight decay) as they are."""
    from olmo.train.optim import OptimizerConfig

    def build_optimizer(self, max_grad_norm, max_grad_norm_ratio, model):
        groups = self.get_param_groups(max_grad_norm, max_grad_norm_ratio, model)
        return LocalShardAdamW8bit(
            groups, lr=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay
        )

    OptimizerConfig.build_optimizer = build_optimizer


def freeze_action_expert() -> None:
    """Build models with the action expert's parameters frozen, as `run_trainer` freezes the
    parts `ft_llm`, `ft_vit` and `ft_connector` leave out (it has no flag for this one)."""
    from olmo.models.molmobot.molmobot import MolmoBotConfig

    build_model = MolmoBotConfig.build_model

    def build_frozen(self, *args, **kwargs):
        model = build_model(self, *args, **kwargs)
        for param in model.get_action_expert_parameters():
            param.requires_grad = False
        return model

    MolmoBotConfig.build_model = build_frozen


def record_metrics(path: str) -> None:
    """Append the metrics the trainer logs to `path`, one JSON line per step and per validation:
    {"step", "kind": "train" or "val", "metrics": {name: value}}. Train steps carry the losses
    (`train/...`), learning rates and gradient norms (`optim/...`), throughput and peak GPU memory;
    validations, each evaluator's losses (`<label>/...`). A resumed run appends steps again from its
    checkpoint, and the later lines win."""
    from olmo.torch_util import get_global_rank
    from olmo.train.trainer import Trainer

    log_metrics_to_console = Trainer.log_metrics_to_console
    loss_eval = Trainer.loss_eval

    def write(step: int, kind: str, metrics: dict) -> None:
        values = {}
        for name, value in metrics.items():
            try:
                values[name] = float(value)
            except (TypeError, ValueError):  # e.g. W&B media from inference evaluators
                continue
        with open(path, "a") as file:
            file.write(json.dumps({"step": step, "kind": kind, "metrics": values}) + "\n")

    def log_and_record(self, prefix, metrics):
        if prefix.startswith("[step="):  # the train step's header; evaluators log under their label
            write(self.global_step, "train", metrics)
        log_metrics_to_console(self, prefix, metrics)

    def eval_and_record(self):
        metrics = loss_eval(self)
        if get_global_rank() == 0:
            write(self.global_step, "val", metrics)
        return metrics

    Trainer.log_metrics_to_console = log_and_record
    Trainer.loss_eval = eval_and_record


def checkpoint_model_config(checkpoint: str):
    from olmo.models.model_config import BaseModelConfig

    return BaseModelConfig.load(os.path.join(checkpoint, "config.yaml"), key="model")


def checkpoint_args(checkpoint: str) -> list[str]:
    """`train_molmobot.py`'s flags that would otherwise put its own defaults over the checkpoint's
    settings, set to the checkpoint's."""
    from omegaconf import OmegaConf

    config = OmegaConf.load(os.path.join(checkpoint, "config.yaml"))
    model = config.model
    args = [
        "--seq_len", str(config.data.sequence_length),
        "--n_obs_steps", str(model.n_obs_steps),
        "--obs_step_delta", str(model.obs_step_delta),
        "--action_horizon", str(model.action_horizon),
        "--n_action_steps", str(model.n_action_steps),
        "--action_expert_layer_mode", str(model.action_expert_layer_mode),
    ]
    if model.vision_backbone.use_image_augmentation:
        args.append("--img_aug")
    return args


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    molmobot_dir = spec["molmobot_dir"]
    sys.path[:0] = [molmobot_dir, os.path.join(molmobot_dir, "launch_scripts")]
    import train_molmobot
    from olmo.data import synthmanip_presets as presets

    presets.ACTION_SPECS[PRESET] = dict(spec["move_groups"])
    presets.ACTION_DATASET_KEYS[PRESET] = "joint_pos"
    presets.CAMERA_PRESETS[PRESET] = list(spec["cameras"])
    train_molmobot.get_model = lambda checkpoint, model_type: checkpoint_model_config(checkpoint)
    use_8bit_adamw()
    if not spec["train_action_head"]:
        freeze_action_expert()
    record_metrics(spec["metrics"])

    checkpoint = spec["checkpoint"]
    sys.argv = [train_molmobot.__file__, checkpoint, *checkpoint_args(checkpoint), *spec["args"]]
    train_molmobot.main()


if __name__ == "__main__":
    main()
