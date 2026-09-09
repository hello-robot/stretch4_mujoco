# How long to train, and how to continue a run

Findings from the `stretch_potato` fine-tunes, written down because three of them
contradict what [`README.md`](README.md) currently says and one of them
contradicts what the run's own "best" checkpoint claims.

Everything below is measured, not inferred. The subject run is
`data/stretch_potato/rollouts/molmobot_cleaned_10k/checkpoints/stretch4_potato`
— 10,000 steps, action expert only (`ft_vit`, `ft_llm`, `ft_connector` all
`false` in `step10000/config.yaml:382-384`), finished 2026-09-09.

**No code changes have been made.** Section 7 lists the ones this analysis
implies, for a later decision.

---

## TL;DR

- **"Train until the loss stops falling" is not achievable.** The train loss
  follows a power law with no floor — `L(s) = 57.6 · s^-0.907`, R² 0.966. At
  100,000 steps it is still dropping. Pick the horizon by passes over the data
  and by benchmark success rate; train loss cannot choose it for you.
- **For 331 trajectories, use 30,000 steps, not 50,000.** ~24 passes, ~25 h.
  50,000 buys ~37% lower train loss on data you already fit 7x better than
  held-out, for another 17 hours.
- **`MAX_STEPS` is the learning-rate horizon.** Resuming with a larger one
  re-evaluates a fresh cosine at your current step and jumps the LR up. Pin
  `--scheduler.t_max` to make staged runs continuous.
- **`step<N>_bestfit/` is not trustworthy here.** Its threshold sits ~40x below
  the eval noise, and the validation loss is drawn from unseeded RNG. On the 10k
  run it saved step 4000; step 10000 is clearly better in rollouts. Select by
  `run_benchmarks.py`.

---

## 1. How many steps? (331 trajectories, single task)

### The train loss never flattens

Fitting the post-warmup portion of the run (steps 1500-10000) in log-log space:

```
L(s) = 57.6 · s^-0.907          R² = 0.966
```

A clean power law with **no floor term**. Extrapolated:

| steps | train loss | passes | change vs. previous row |
| --- | --- | --- | --- |
| 10,000 | 0.0136 | 8.2 | — |
| 15,000 | 0.0094 | 12.2 | −31% |
| 20,000 | 0.0072 | 16.3 | −23% |
| 30,000 | 0.0050 | 24.5 | −31% |
| 40,000 | 0.0039 | 32.6 | −23% |
| 50,000 | 0.0032 | 40.8 | −18% |
| 100,000 | 0.0017 | 81.5 | −23% |

At 100,000 steps it is still falling ~23%. So a horizon chosen by "run until it
converges" does not terminate: pick 50,000 and the loss will still be visibly
descending at 50,000.

Two reasons it behaves this way, and both matter:

1. **The schedule manufactures the descent.** Under cosine-to-`alpha_f` the LR
   keeps shrinking to the end of whatever horizon you set, so the curve is
   guaranteed to keep sloping down. The fit above bundles genuine learning
   together with the anneal and cannot separate them — which also means it does
   not directly predict a 50,000-step run's trajectory, where the LR stays high
   far longer.
2. **Falling train loss with flat validation is not progress.** The train/val
   gap at step 10000 is already 7.3x (section 4). More steps widen it.

So the horizon has to come from passes over the data and from benchmark success
rate.

### The budget

296 train trajectories / 39,242 frames at `GLOBAL_BATCH=32`, and 3.01 s/step
measured from the last run (12,050.8 s at step 4000):

| steps | passes | wall clock |
| --- | --- | --- |
| 10,000 | 8.2 | 8.4 h |
| 20,000 | 16.3 | 16.7 h |
| **30,000** | **24.5** | **25.1 h** |
| 40,000 | 32.6 | 33.5 h |
| 50,000 | 40.8 | 41.8 h |

**30,000 is the recommendation.** That is ~24 passes over 296 trajectories of a
single task, already at the outer edge for this data volume and 3x past the
script's own `MAX_EPOCHS_WARN` line (which sits at 12,263 steps —
`10 × 39,242 / 32`). Going to 50,000 costs another 17 hours for ~37% lower loss
on training houses you are already fitting far better than held-out ones.

Pin the schedule so the run is coherent and the retained checkpoints line up
with the decision points:

```bash
MAX_STEPS=30000
# add to the generated script:  EXTRA_ARGS+=(--scheduler.t_max=30000)
```

`save_interval=2000` with `checkpoint_retention_frequency=10000`
(`launch_scripts/train_molmobot.py:597-599`) retains `step10000/`,
`step20000/` and `step30000/`. Benchmark all three. If success rate is still
climbing from 20k to 30k, extending is then a decision backed by evidence.
Set `checkpoint_retention_frequency=5000` if you want finer granularity.

### Two things worth changing on a from-scratch restart

- **`TRAINABLE=vision`.** The ViT was frozen for the whole 10k run. The blocker
  is that the robot never completes the pick, and that is far more likely a
  "the encoder has never seen a Stretch head camera" problem — the premise
  [`README.md`](README.md) opens with — than a "the action expert needs 20,000
  more steps" problem. A frozen DROID encoder cannot adapt to that viewpoint no
  matter how long the expert trains on top of it. Note that more trainable
  capacity on 296 trajectories overfits *sooner*, so hold at 20-30k rather than
  reaching for 50k.
- **A larger `--val-fraction`.** 34 val trajectories is not enough to measure
  anything (section 4).

Changing `TRAINABLE` and the horizon in the same run means not knowing which
helped. If that matters, run 30k action-expert-only first — it is directly
comparable to the 10k already on disk.

---

## 2. `MAX_STEPS` is the learning-rate horizon

`CosWithWarmup.get_lr(initial_lr, step, max_steps)`
(`third_party/MolmoBot/MolmoBot/olmo/train/optim.py:434`) computes the LR as a
**pure function of `(step, max_steps)`**. There is no stored curve and no memory
of a previous run. `t_max` is `None` by default, so `max_steps` is
`cfg.max_duration`, i.e. `MAX_STEPS` (`olmo/train/trainer.py:536-541`).

So changing `MAX_STEPS` re-evaluates a **brand-new cosine at your current step**.
The optimizer moments resume correctly; it is the LR that is discontinuous.

Measured with the real scheduler (action-expert group, peak `1e-4`,
`alpha_f=0.1`):

| situation | LR at step 10000 |
| --- | --- |
| end of a 10k run (`t_max` unset) | `1.00e-05` (fully annealed) |
| resume with `MAX_STEPS=20000`, `t_max` unset | `5.57e-05` — **5.6x jump** |
| resume with `MAX_STEPS=30000`, `t_max` unset | `7.81e-05` — **7.8x jump** |
| `MAX_STEPS=1000000` from scratch | `9.998e-05` — flat, constant-LR at peak |
| `--scheduler.t_max=30000 --max_duration=20000` | `7.805e-05`, then `3.277e-05` at 20000 — **continuous** |

An uncontrolled warm restart. Not fatal — SGDR does this deliberately — but it
spikes the loss and partly undoes an anneal already paid for.

### Setting a huge `MAX_STEPS` and stopping early does not work

At `MAX_STEPS=1e6` the cosine is flat across any range you would actually train:

```
step 200    lr=1.000e-04       step 50000   lr=9.945e-05
step 10000  lr=9.998e-05       step 100000  lr=9.781e-05
```

That is constant-LR-at-peak training that never anneals, so every checkpoint is
a high-LR checkpoint. `save_num_checkpoints_to_keep=1` rotates the intermediate
ones away regardless.

### The fix: pin `scheduler.t_max`

`SchedulerConfig.t_max` (`olmo/train/optim.py:559`) is unset today. When set it
**replaces** `max_steps` inside `get_lr`, and `SchedulerConfig.build()`
propagates it to every per-group `CosWithWarmup`. That decouples the LR horizon
from the stopping point.

Verified: `--scheduler.t_max=30000` merges cleanly through
`merge_with_dotlist` as an `int`, and `stop_at="${max_duration}"` still
interpolates.

Decide the total horizon up front, pass `--scheduler.t_max=<total>` on **every**
stage, and raise `MAX_STEPS` alone to continue. The LR curve is then identical
to one uninterrupted run.

---

## 3. Three ways to continue a run

[`README.md`](README.md) currently describes two and says the third is
impossible. All three work.

| | mechanism | optimizer | step counter | LR |
| --- | --- | --- | --- | --- |
| same `SAVE_FOLDER`, bigger `MAX_STEPS` | `allow_resume` | kept | kept | continuous **iff** `t_max` pinned |
| `RESUME_FROM=` | `--load_path` | kept | kept | same, but into a new folder |
| `CHECKPOINT=` | `initial_model_checkpoint` | reset | reset | one fresh warmup+cosine |

- `allow_resume=True` (`train_molmobot.py:543`) makes `run_trainer.py:180-207`
  scan `save_folder` for `step<N>/` and resume from it **in preference to
  `load_path`**. Same folder plus a bigger `MAX_STEPS` is all a resume needs;
  `RESUME_FROM` is for resuming out of a *different* folder.
- `step<N>/` carries optimizer state only because `finetune.py:811` forces
  `--save_final_optim=True`, overriding MolmoBot's `False`. The README's claim
  that a resume "cannot happen anyway" is out of date.
- `Checkpointer.latest_checkpoint` matches `^step(\d+)$`
  (`checkpointer.py:141-149`), so `step<N>_bestfit/` is never auto-resumed.
- Bestfit checkpoints are saved with `optim=None` on purpose
  (`trainer.py:801-826`), so they are `CHECKPOINT=` only, never `RESUME_FROM=`.
- The step counter carrying over makes `MAX_STEPS` **cumulative**;
  `Trainer.restore_checkpoint` (`trainer.py:665-684`) raises if the restored
  step is already past it.

---

## 4. `step<N>_bestfit/` selects the luckiest eval, not the best model

Confirmed behaviourally: on the 10k run, `step4000_bestfit` was saved as "best",
but in rollouts the step-10000 checkpoint is clearly better. Two compounding
causes.

### 4a. The selection threshold is ~40x below the noise

`Trainer.maybe_save_bestfit_checkpoint` (`olmo/train/trainer.py:736-760`):
`MOLMOBOT_BESTFIT_MIN_DELTA` defaults to `0.005` and is **relative** — 0.5%, per
that method's own docstring — and it saves on *every* improvement rather than
waiting for a plateau.

The observed eval-to-eval bounce is ±20-25%. A threshold that far below the
noise means every downward wiggle clears the bar and rewrites the checkpoint. So
the saved "best" is the **running argmin of a noisy series** — a biased-low
estimator that tracks noise, not quality.

(`MOLMOBOT_BESTFIT_PATIENCE`, default 4, only announces a plateau. It does not
stop training.)

### 4b. The validation loss is not reproducible

`_compute_flow_matching_loss` draws timesteps via `_sample_beta_timesteps`
(`olmo/models/molmobot/molmobot.py:514`) and noise via `torch.randn`
(`:531`, `:536`) with **no `generator=` argument** — unlike the inference path at
`:321`, which does pass one. The dataloader *is* seeded (`seed=691203`,
`shuffle=False`, `max_examples=2000`, `train_molmobot.py:496-522`), so the data
is fixed, but the flow noise and timesteps are redrawn on every evaluation.

Consecutive validation numbers therefore differ partly because the noise
differs, not because the model changed. The 16 `flow_loss_time_*` bins are
indices into randomly-sampled timesteps, so they are not comparable across
evaluations either.

### The validation curve, for the record

```
 500 0.09928   3000 0.07368   5500 0.06785   8000 0.08443
1000 0.10386   3500 0.06176   6000 0.06549   8500 0.10215
1500 0.06963   4000 0.05991*  6500 0.08617   9000 0.07540
2000 0.08296   4500 0.06723   7000 0.07466   9500 0.06178
2500 0.07751   5000 0.07753   7500 0.08516  10000 0.09819
```

No trend after ~step 3500. The marked minimum is a draw, not a verdict.

`CrossEntropyLoss` (~23.36) and `Accuracy` (0.0) are **inert** in this config —
train shows the same ~22.5, because `ft_llm=false` and
`response_logits_only=True`. The flow loss is the only live validation signal.

### The generalization gap is real, but it is not the deployment metric

Per-dimension at step 10000 (train mean over steps>9000 vs val):

| dim | train | val | ratio |
| --- | --- | --- | --- |
| base_x | 0.01015 | 0.15410 | 15.2x |
| base_y | 0.01010 | 0.06918 | 6.8x |
| base_th | 0.00225 | 0.03429 | 15.2x |
| lift | 0.04469 | 0.21060 | 4.7x |
| arm | 0.01865 | 0.09298 | 5.0x |
| wrist_a | 0.01422 | 0.21932 | 15.4x |
| wrist_b | 0.01575 | 0.12295 | 7.8x |
| wrist_c | 0.01121 | 0.07059 | 6.3x |
| grip_l | 0.00403 | 0.00473 | 1.2x |
| grip_r | 0.00400 | 0.00321 | 0.8x |
| **TOTAL** | **0.01351** | **0.09819** | **7.3x** |

Both gripper dimensions match; every motion dimension does not. Generalization
to held-out houses is genuinely weak. But held-out-house flow loss is not the
deployment metric, and it is far too noisy to select checkpoints with.
`run_benchmarks.py` success rate is.

---

## 5. The 10k run was cut off, not converged

Train `action_flow_loss`, mean per 500-step window:

```
    1-  500  0.19875           5001- 5500  0.02919  (+11.5%)
  501- 1000  0.07807 (-60.7%)  5501- 6000  0.02315  (-20.7%)
 1001- 1500  0.06092 (-22.0%)  6001- 6500  0.02193  ( -5.2%)
 1501- 2000  0.05670 ( -6.9%)  6501- 7000  0.02018  ( -8.0%)
 2001- 2500  0.05207 ( -8.2%)  7001- 7500  0.01731  (-14.2%)
 2501- 3000  0.04301 (-17.4%)  7501- 8000  0.01407  (-18.7%)
 3001- 3500  0.03992 ( -7.2%)  8001- 8500  0.01628  (+15.7%)
 3501- 4000  0.03666 ( -8.1%)  8501- 9000  0.01570  ( -3.6%)
 4001- 4500  0.03059 (-16.6%)  9001- 9500  0.01381  (-12.0%)
 4501- 5000  0.02617 (-14.4%)  9501-10000  0.01320  ( -4.4%)
```

Still falling 4.4% in the final window, and falling **while the LR was annealing
to `1e-5`**, which normally flattens a curve. There was headroom left — which is
what section 1 quantifies, and also why the plateau diagnostics in
`training_report.py` should not have been read as "it converged".

---

## 6. Dataset scale

From `valid_trajectory_index.json`:

- train: 119 houses, **296 trajectories, 39,242 frames** (mean 132.6 frames)
- val: 14 houses, **34 trajectories, 4,311 frames** (mean 126.8)
- total: 330 trajectories

At `GLOBAL_BATCH=32`, passes over the training frames are `steps × 32 / 39,242`.
The `MAX_EPOCHS_WARN=10` preflight prompt therefore triggers above **12,263
steps**.

---

## 7. Implied changes, not yet applied

1. **`README.md:158-191`** — the section "Continuing from your own weights, and
   why not a resume" states `save_final_optim=False` and that a resume "cannot
   happen anyway". Both are false now. Rewrite around section 3's table, keeping
   its `7.8e-5` figure (verified) and adding `5.57e-05` for the 20k case.
2. **`README.md`** (step-5 benchmark command, `:73-75`) — it presents
   `step<N>_bestfit` as simply "the best checkpoint". Add section 4's caveat:
   select by benchmark success rate, treat bestfit as a convenience snapshot.
3. **`training_report.py:82-101`** (`LONGER_RUN_ADVICE`) — carries the same stale
   claim, and is *printed at the user* by the plateau diagnostics at `:634` and
   `:679`. It is a module-level string de-duplicated by substring match at
   `:774-776`, so keep it one string; the file runs by path under MolmoBot's
   interpreter and cannot import this repo, so no new imports.
4. **`finetune.py:1557`** — raise the `BESTFIT_MIN_DELTA` default from `0.005`
   (0.5% relative) to something above the noise floor, e.g. `0.05`, and say in
   the comment that the threshold must exceed eval-to-eval noise.
5. **`finetune.py:289`** — optionally add `--scheduler.t_max=<total steps>` to
   `MOLMOBOT_OPTIONAL_FLAGS` so it appears as a commented-out
   `EXTRA_ARGS+=(...)` line beside `--img_aug`, where someone staging a run will
   find it.
6. **Not proposed here:** seeding the flow-loss RNG
   (`molmobot.py:514`, `:531`, `:536`). The checkout is gitignored and
   regenerated, so it needs a `molmobot_repo.py` patch. It is the root cause of
   the unusable validation signal, and worth doing — recorded so the reason the
   curve bounces is not lost.

### Reproducing the numbers

The scheduler table in section 2, from `third_party/MolmoBot/MolmoBot`:

```bash
.venv/bin/python -c "
from olmo.train.optim import SchedulerConfig, SchedulerType
s = SchedulerConfig(name=SchedulerType.multimodal, action_expert_t_warmup=200,
                    alpha_f=0.1, warmup_min_lr=0.0).build()
print(s.get_lr(1e-4, 10000, 20000, group_name='action_expert'))"
```

The loss tables in sections 1, 4 and 5 come from `metrics.jsonl` in the save
folder (`split` is `train` or `eval`; the train metric key is
`train/action_flow_loss`, the eval one `action_flow_loss`). The pass counts come
from `valid_trajectory_index.json`, summed as the preflight in `finetune.py:1203-1210`
does it.
