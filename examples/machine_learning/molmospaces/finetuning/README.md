# Fine-tuning a VLA on Stretch 4

A model trained on a Franka emits Franka joint angles, and nothing in this repo
translates them any more — see *Why a Franka-space model is not simply remapped*
in [`../README.md`](../README.md). The half of the mismatch that killed the
attempt is the half kinematics cannot reach anyway: the model has never seen a
Stretch head camera, and no retarget fixes a viewpoint. So teach it Stretch.

```
datagen_configs.py   Stretch versions of MolmoSpaces' data generation configs
generate_dataset.py  run them, then optionally export -- one command
live_recorder.py     record teleop demonstrations from examples/molmo_environment.py
lerobot_export.py    rollouts -> a LeRobot v2.1 dataset (for openpi / LeRobot)
finetune.py          check the data, prepare it, write the trainer config, launch
```

The trajectory format itself is `../hdf5_layout.py`, one level up because
`../training/` reads it too. Behaviour cloning a small net from scratch is that
other road: [`../training/README.md`](../training/README.md).

## MolmoBot needs no conversion

[MolmoBot](https://github.com/allenai/MolmoBot) trains **directly on MolmoSpaces
trajectories**. `MolmoBot/olmo/data/synthmanip_dataset.py` opens
`{data_path}/{split}/house_*/*.h5` and reads `obs/agent/qpos`,
`actions/joint_pos_rel`, `obs_scene["task_description"]` and
`obs/sensor_data/{camera}` — which is exactly what `generate_dataset.py` writes.

And its action space is configured **by move group**: `--action_move_groups`,
`--camera_names`, `action_spec`. So it learns Stretch's own ten numbers —

| group | dims |
| --- | --- |
| `base` | 3 |
| `lift` | 1 |
| `arm` | 1 |
| `wrist` | 3 |
| `gripper` | 2 (the MJCF's mirrored pair of the one `stretch_gripper` actuator) |

— and `SynthVLAPolicy` hands MolmoSpaces back an action dict keyed by move group,
which is precisely what Stretch's controllers already take. That is why
`--policy molmobot` in `run_benchmarks.py` has no remapper in it.

```bash
# 1. prove the setup: 2 episodes, 1 house, small scene dataset
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task debug --output-dir data/stretch_debug --no-export

# 2. generate for real (--no-export: MolmoBot reads the rollouts as they are)
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task pick --task pnp --episodes 2000 --num-workers 8 \
    --output-dir data/stretch_pick --no-export

# 3. prepare and print the commands
python -m examples.machine_learning.molmospaces.finetuning.finetune \
    --rollouts data/stretch_pick/rollouts/pick --trainer molmobot

# 4. run MolmoBot's two preprocessing steps, then its trainer (in your checkout)
# 5. score the result -- natively, no remapping
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy molmobot --checkpoint <checkpoint> --benchmark pick
```

Step 3 does the two mechanical things that stand between a MolmoSpaces run and a
MolmoBot dataset, both easy to miss:

- **`obs/sensor_data` is empty.** MolmoSpaces strips camera observations before
  batching to keep the HDF5 small
  (`prepare_episode_for_saving(remove_sensors_if_save_dir=True)`), so its saver
  writes that group empty *even though the MP4s are in the same directory*.
  MolmoBot reads the video filename out of it, so every trajectory would look
  image-less. `../hdf5_layout.py`'s `ensure_sensor_data_paths()` fills it in from the
  videos that are already there, under the name the saver itself would have
  used.
- **The houses are flat.** MolmoBot wants `train/` and `val/`;
  `arrange_train_val_split()` symlinks them across, moving houses *whole* so a
  room never lands in both splits.

It also prints MolmoBot's own two preprocessing commands
(`validate_trajectories.py`, `calculate_stats.py`) rather than running them —
they live in its repository and write the manifest and norm-stats YAML its
trainer reads.

## The other trainers, and the LeRobot export

openpi (pi0/pi0.5) and LeRobot want a LeRobot dataset, so they take
`lerobot_export.py`'s output. It writes one action space, Stretch's own:

| | `stretch` |
| --- | --- |
| dimensions | 10 (base step, lift, arm, wrist, gripper) |
| model's action head | reshaped and re-learned |
| pretrained weights | vision and language carry over; the head does not |
| drives the base | yes |
| encoding fidelity | exact — it is what was recorded |

There used to be a `franka` default here, an 8-dimensional encoding that ran the
Franka remapping backwards so a DROID-pretrained checkpoint could keep its action
head. It is gone with the rest of the remapping. What it cost was invisible from
the outside: every recorded pose the virtual arm could not reach was quietly
replaced by the nearest one it could, the encoding was pinned to a *virtual*
Franka mounted on Stretch's mast rather than to the authoring arm, and evaluating
against the wrong frame of the two made the arm reach consistently short with
nothing in the logs to say why. A re-learned action head wants more data. It does
not want a debugging session.

## Where the demonstrations come from

**The scripted expert, procedurally** — `datagen_configs.py` +
`generate_dataset.py`. MolmoSpaces' data generation pipeline samples tasks
procedurally (pick a house, pick an object, place the robot, plan, roll out), so
it is unbounded and drawn from the training splits. The benchmark's own 1000
episodes are the *test set*; training on them measures memorisation. The configs
are also addressable from MolmoSpaces' entry point:

```bash
python -m molmo_spaces.data_generation.main \
    examples.machine_learning.molmospaces.finetuning.datagen_configs:StretchPickDataGenConfig
```

Each is a MolmoSpaces datagen config with three substitutions — Stretch's robot,
cameras and scripted expert — and the task class, sampler and success criteria
left alone, because those are what make the data comparable to the benchmark.
One thing beyond the substitution has to change: **where the robot is placed.**
The samplers put a Franka within 0.7m of the target because that is a Franka's
reach; Stretch's tool cannot come closer than 0.39m to its own base axis or go
past 0.99m. `STRETCH_PLACEMENT` widens those fields to the same 0.55–0.90m band
`../stretch/episode_overrides.py` retargets benchmark episodes into.

(`robot_object_z_offset` is deliberately *not* in that set. The samplers use it
to lift a Franka's base to a workable height, which is meaningless for Stretch —
its base is on the floor and the lift covers the vertical range. Harmlessly so:
`HoloJointsRobotBaseGroup.pose` only reads x, y and yaw, so the sampler's z is
discarded rather than obeyed.)

**You, driving** — `live_recorder.py`, via
`examples/molmo_environment.py --record_dataset`:

```bash
python -m examples.molmo_environment --dataset procthor-10k --house-index 0 \
    --keyboard --record_dataset data/teleop_pick --record-task "pick up the mug"
```

`R` starts an episode, `T` keeps it, `X` discards it. Same on-disk format, so it
feeds the same trainers with no branching downstream. Two decisions worth
knowing:

- **The action for a frame is the next frame's state.** A teleop session has no
  commanded target vector — the operator nudges velocities and the position
  controllers chase what falls out — and for a position-controlled arm the next
  observed state *is* the command, retrospectively. `actions/joint_pos_rel` is
  written from the same pair as a difference. The last frame of an episode has no
  successor and is dropped.
- **The operator delimits episodes.** Recording continuously and slicing later
  fills a dataset with the minutes spent driving between objects, and a
  demonstration of "reach for the mug" that starts thirty seconds before the
  reach teaches a policy to wait.

## Which head camera

`head_camera` is the **centre** camera of Stretch 4's head — MJCF
`camera_center_link`, on `camera_center_optical_link`, 45° vertical FOV, 1.62m
above the floor and 0.086m ahead of the base axis, looking forward and 35° down.

The head also carries `camera_left_link` and `camera_right_link`, the stereo
pair, 7.5cm either side and pitched 47° down. Nothing here uses them. Stretch 4's
head has no pan/tilt joint, so all three are rigid to the base yaw: anything that
turns the base turns the view with it.

`wrist_camera` is `gripper_camera_left_rgb`, on the wrist, looking straight along
the arm.

## Why the fine-tune itself is not in here

The trainers live in the model's own repository — MolmoBot for MolmoBot, openpi
for pi0/pi0.5, LeRobot for ACT/diffusion/SmolVLA — each with its own JAX or
PyTorch stack, distributed launcher and checkpoint format, and none is a
dependency of this repo. Vendoring a copy would rot.

`finetune.py` does the parts that *are* this repo's business: check the data,
prepare it, pool the normalisation statistics where the trainer does not compute
its own, write the trainer config, print (or run) the command, and say how to
score the result. The config is JSON rather than the trainer's native Python
because all three resolve configs from dataclasses whose fields move between
versions — a generated module is a file that stops importing, while JSON of the
same field names stays diffable and reviewable before something spends a day on
it.

## Note on the LeRobot format

The export targets **LeRobot dataset format v2.1** — per-episode parquet under
`data/`, per-camera MP4 under `videos/`, JSON metadata under `meta/` — written
directly with pyarrow. Since `lerobot` is not installed here, the layout is
*targeted*, not validated by the library that defines it. Pass `--validate` to
check it against an installed `lerobot`, and read `meta/stretch_export.json` for
the same shape information in a form that does not depend on anyone's format
version.
