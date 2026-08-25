# Behaviour cloning a Stretch 4 policy

Fits a small network from scratch to the scripted expert's successful episodes.
The other road — fine-tuning a pretrained VLA — is `../finetuning/`.

```
collect.py   rollouts -> .npz shards
dataset.py   the shard format, and the torch-side reader
train_bc.py  fit the network, write a checkpoint
```

The network itself is `../policies/networks.py`, the checkpoint loader is
`../policies/checkpoint.py`, and `../policies/bc_policy.py` runs it inside a
MolmoSpaces evaluation.

## The pipeline

```bash
# 1. demonstrate -- this is the expensive step, and it lives in ../finetuning/
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task pick --task pnp --episodes 2000 --num-workers 8 \
    --output-dir data/stretch_manip --no-export

# 2. shard what it wrote (--rollouts is repeatable; several runs pool into one)
python -m examples.machine_learning.molmospaces.training.collect \
    --rollouts data/stretch_manip/rollouts --output-dir data/stretch_manip/bc

# 3. fit
python -m examples.machine_learning.molmospaces.training.train_bc \
    --dataset-dir data/stretch_manip/bc --output checkpoints/stretch_manip.pt

# 4. score it on the same benchmarks the expert was scored on
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy bc --checkpoint checkpoints/stretch_manip.pt \
    --benchmark pick --benchmark pnp --episodes 100
```

## Why the demonstrations come from `../finetuning/`

Because a released benchmark's 1000–2000 episodes are the **test set**. Rolling
the expert over them and cloning the result measures memorisation, and there are
only a few thousand of them either way. `../finetuning/datagen_configs.py` drives
MolmoSpaces' data generation pipeline instead — procedurally sampled houses,
objects and robot placements, drawn from the training splits and unbounded — so
both learners here train on the same distribution and are scored on scenes
neither has seen.

Teleop demonstrations work too: `../finetuning/live_recorder.py` writes the same
on-disk format, so `collect.py --rollouts` reads them with no branching.

## Only the successes

Step 2 keeps only the episodes the task judged successful, which is the point of
cloning a partial expert — the minority it completes are the demonstrations, and
the rest are counter-examples. `--keep-failures` overrides it. The raw rollouts
are kept rather than deleted: the videos are the only record of *how* the expert
failed on the episodes that were dropped.

## What the network is

Head and wrist camera images plus seven numbers of proprioception in, a chunk of
eight future actions out. Chunking is what makes an open-loop cloner usable at
15Hz: a single-step policy has to be re-queried every 66ms and stalls wherever
two nearby observations imply opposite motions.

The encoding lives in `../policies/networks.py` rather than here, because three
places have to agree on it exactly — `dataset.py` writes it, `train_bc.py` fits
it, `../policies/bc_policy.py` reads it back — and getting it wrong is silent.
The checkpoint is written self-describing (camera names, chunk size,
normalisation statistics) for the same reason.

`train_bc.py` writes `<checkpoint>_curves.png`, `_history.json` and
`_history.csv` beside the checkpoint, rewritten every epoch, so a run still
going -- or one that died -- can be inspected.
