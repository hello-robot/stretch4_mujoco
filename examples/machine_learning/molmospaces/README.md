# Stretch 4 on the MolmoSpaces benchmarks

Runs Stretch 4 against the eight [MolmoSpaces](https://github.com/allenai/molmospaces)
benchmark evaluations, with a scripted expert as the baseline and a
behaviour-cloning pipeline for training a learned policy to beat it.

MolmoSpaces' [evaluation
README](https://github.com/allenai/molmospaces/blob/main/molmo_spaces/evaluation/README.md)
asks an external repo for three things — a policy class, a policy config, and an
eval config extending `JsonBenchmarkEvalConfig`. Stretch needs a fourth, because
no MolmoSpaces benchmark was authored with it: a robot integration, plus a way to
retarget the recorded Franka/RBY1 episodes onto it.

```
benchmarks.py           the eight benchmark evaluations, as a registry
configs.py              eval configs, addressable from eval_main.py
run_benchmarks.py       CLI: run any or all eight, aggregate to CSV
stretch/
  robot_view.py         Stretch as MolmoSpaces move groups
  robot.py              Stretch as a MolmoSpaces Robot; scene attachment
  config.py             robot config + camera system
franka_remapping/
  episode_overrides.py  retarget a Franka/RBY1 episode onto Stretch
  episode_frame.py      carry the authoring arm's frame past the retarget
  franka_arm.py         the Franka Droid arm as pure FK/IK
  pose_solver.py        full 6-DOF tool-pose solving for Stretch
  action_remap.py       Franka joint space <-> Stretch move groups
  vla_policy.py         a Franka-space VLA server, driving Stretch
  vla_client.py         websocket/msgpack transport for that server
finetuning/
  datagen_configs.py    Stretch versions of the MolmoSpaces datagen configs
  generate_dataset.py   generate demonstrations, then optionally export them
  live_recorder.py      record teleop demonstrations from molmo_environment.py
  hdf5_layout.py        the MolmoSpaces trajectory format: writer, repair, splits
  lerobot_export.py     rollouts -> a LeRobot dataset, in either action space
  finetune.py           check the data, prepare it, write the config, launch
policies/
  kinematics.py         reach solving
  scripted.py           the scripted expert (baseline + BC teacher)
  networks.py           the BC network and the action/state encoding
  checkpoint.py         loading and running a checkpoint, robot-stack agnostic
  bc_policy.py          the trained policy, as a MolmoSpaces InferencePolicy
  molmobot_policy.py    a MolmoBot checkpoint driving Stretch natively
training/
  collect.py            roll out the expert, keep what worked
  dataset.py            HDF5 + MP4 rollouts -> training shards
  train_bc.py           fit the network, write a checkpoint
report.py               a finished run -> captioned video, telemetry, summary
telemetry.py            live Rerun streaming and recording
live_policy.py          run a checkpoint in the interactive sim
```

## Setup

```bash
pip install -e ".[molmo]"

# Downloads the benchmark JSONs, scenes, objects and grasp libraries.
# Several GB; only needs doing once.
python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()"
python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager as r; \
           r().install_all_for_source('benchmarks', 'molmospaces-bench-v2'); \
           r().install_all_for_source('benchmarks', 'molmospaces-bench-v1')"
```

Rendering is off-screen through EGL. `run_benchmarks.py` sets `MUJOCO_GL=egl`
itself; if you call `run_evaluation` directly, export it yourself.

## The eight benchmarks

Between `molmospaces-bench-v1` and `-v2` there are around ninety released
benchmark directories, but they cover eight distinct task families — the seven
`TaskSpec` subclasses in MolmoSpaces' `benchmark_schema.py`, with `OpeningTask`
split into its "open" and "close" suites. `benchmarks.py` pins one directory per
family:

| key            | release  | task                                    | episodes |
| -------------- | -------- | --------------------------------------- | -------- |
| `pick`         | MB-Pick  | pick a named object off a surface       | 1000     |
| `pnp`          | MB-PnP   | place it in or on a named receptacle    | 1000     |
| `pnp_next_to`  | MB-PnP-next-to | place it *beside* a named object  | 1000     |
| `pnp_color`    | MB-PnP-color | receptacle told apart only by colour | 1000    |
| `open`         | MB-Open  | open a drawer or cabinet                | 1000     |
| `close`        | MB-Close | close one that starts open              | 1000     |
| `door_opening` | MB-Door  | pull a room door two thirds open        | 2000     |
| `nav_to_obj`   | MS-Nav   | drive to within 1.5m of a named object  | 2000     |

Where both a MolmoSpaces ("MS-", easier, iTHOR/procthor-10k) and a MolmoBot
("MB-", harder, procthor-objaverse) release exists, the harder one is the default
and the other is available as `--alternate ms`. Two of those defaults are not
simply a difficulty preference:

- **`open` / `close`** default to the MolmoBot release because the MolmoSpaces
  iTHOR ones do not load at all in the current asset set:
  `JsonEvalTaskSampler.set_joint_values()` needs a per-joint grasp file for the
  articulated object, and the released `droid` grasp library has none for those
  iTHOR drawers and cabinets, so every episode raises "No joints with grasp file
  found". It is an asset gap rather than anything Stretch-specific — it
  reproduces with MolmoSpaces' own Franka `DummyBenchmarkEvalConfig`. Try
  `--alternate ms` if a later release fills the library in.
- **`door_opening`** cannot be evaluated with Stretch at all, see *Known
  limitations*. It is skipped when sweeping and only runs when named.

```bash
python -m examples.machine_learning.molmospaces.run_benchmarks --list
```

## Running

```bash
# baselines on everything evaluable, a few episodes each
python -m examples.machine_learning.molmospaces.run_benchmarks --episodes 5

# one benchmark, properly, in parallel
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --benchmark pick --episodes 200 --num-workers 8
```

`--policy baseline` (the default) runs the scripted expert on the manipulation
and articulation benchmarks, and MolmoSpaces' own A* planner on navigation.
Results land in `<output>/results.csv` next to the per-episode HDF5 trajectories
and MP4s.

### Baseline, three episodes per benchmark

Enough to show the harness works end to end and to rank the task families;
far too few episodes to quote as a score. Run a few hundred for that.

These were measured *before* the stowed-spawn fix described under
[The episode](#the-episode), so the manipulation rows understate the current
baseline: MB-Pick has since measured 3/8 (37.5%) over eight episodes, against
2/8 on the same episodes beforehand. Re-run the sweep for current numbers.

| benchmark     | episodes | success |
| ------------- | -------- | ------- |
| `pick`        | 3        | 33%     |
| `pnp`         | 3        | 0%      |
| `pnp_next_to` | 3        | 33%     |
| `pnp_color`   | 3        | 0%      |
| `open`        | 3        | 0%      |
| `close`       | 3        | 0%      |
| `nav_to_obj`  | 3        | 100%    |

Navigation is solved because MolmoSpaces' A* planner does the hard part and
Stretch's holonomic base executes its waypoints directly. The manipulation
numbers are the scripted expert's, and they are what the behaviour-cloning
pipeline below is meant to improve on — see *Known limitations* for why the
expert is not a ceiling.

The eval configs are also addressable from MolmoSpaces' entry point directly:

```bash
python molmo_spaces/evaluation/eval_main.py \
    examples.machine_learning.molmospaces.configs:StretchScriptedEvalConfig \
    --benchmark_dir <dir> --no_wandb
```

## Training a policy

```bash
# 1. demonstrate. The expert succeeds on a minority of episodes; those are the data.
python -m examples.machine_learning.molmospaces.training.collect \
    --benchmark pick --benchmark pnp --episodes 300 --output-dir data/stretch_manip

# 2. fit
python -m examples.machine_learning.molmospaces.training.train_bc \
    --dataset-dir data/stretch_manip --output checkpoints/stretch_manip.pt

# 3. score it on the same benchmarks the expert was scored on
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy bc --checkpoint checkpoints/stretch_manip.pt \
    --benchmark pick --benchmark pnp --episodes 100
```

The network takes the head and wrist camera images plus seven numbers of
proprioception, and predicts a chunk of eight future actions. Chunking is what
makes an open-loop cloner usable at 15Hz: a single-step policy has to be
re-queried every 66ms and stalls wherever two nearby observations imply opposite
motions.

Collection draws episodes from the front of the benchmark's own list, so by
default the policy is measured on its ability to imitate rather than to
generalise. For a held-out split, collect on a benchmark's `--alternate` release
and evaluate on the default one.

## How Stretch is fitted to a benchmark written for a Franka

### The base

`examples/molmo_environment.py` keeps Stretch's real free-jointed, three-omniwheel
base and retargets its contact pairs onto the scene floor, because the point
there is to drive the actual robot. That does not work for a benchmark: episodes
freeze the robot at a `task.robot_base_pose`, expect `robot_view.base.pose = ...`
to place it exactly, and drive it with position controllers. So — exactly as
MolmoSpaces does for its own mobile robots — `stretch/robot.py` hangs the robot
off three *virtual* holonomic joints (two slides and a hinge) with position
actuators, and drops the wheel contact pairs. The wheels stay in the model but
never touch anything; the lowest remaining collision geom clears the floor by
about 8cm.

### The move groups

| group     | dof | joints                                              |
| --------- | --- | --------------------------------------------------- |
| `base`    | 3   | virtual holonomic x, y, theta                        |
| `lift`    | 1   | mast                                                 |
| `arm`     | 1   | four telescoping segments, one tendon actuator       |
| `wrist`   | 3   | yaw, pitch, roll                                     |
| `gripper` | 2   | the mirrored finger pair of the one `stretch_gripper` DOF |

The telescoping arm is the awkward one: four MuJoCo joints held equal by
`<equality>` constraints and driven by a single tendon actuator whose range is
their *total* extension. Every MolmoSpaces controller assumes a move group's
`joint_pos` is directly comparable to its actuator control range, so
`StretchTelescopingArmGroup` reports the sum as a one-element position and
distributes a command back as a quarter each. Its Jacobian column is
correspondingly the *mean* of the four segment columns, not their sum.

### The episode

`franka_remapping/episode_overrides.py` is registered as MolmoSpaces'
`robot_override_fn`, which runs per episode with both the episode spec and the
experiment config mutable. It rewrites three things and leaves the scene, the
object poses, the instruction and the success criteria untouched:

- **cameras** — the recorded Franka extrinsics hang off bodies that do not exist
  in a Stretch scene, so they are replaced by Stretch's own head and wrist MJCF
  cameras.
- **`robot.init_qpos`** — keyed by the authoring robot's move groups; replaced
  with Stretch's.
- **`task.robot_base_pose`** — the direction from target to base is preserved,
  since the episode's author validated that approach as reachable and unoccluded.
  Only the distance changes, and only when the target falls outside Stretch's
  0.55–0.90m reach band. Yaw is recomputed to point the base's +x axis — the axis
  the arm extends along — at the target.

Stretch starts each episode **stowed** (arm retracted, lift low, wrist turned
back), and that is load-bearing rather than tidy. A benchmark base pose is a
0.55–0.90m standoff from the target, and an unstowed Stretch already has its tool
0.57m out in front of the base at working height — so the gripper would spawn
exactly where the object is, which in practice means inside the counter, cabinet
or cistern the object sits on, with the robot jammed and unable to move. Measured
over the first eight MB-Pick episodes, an unstowed spawn was interpenetrating the
scene in five of them by up to 19cm; stowed, that is one in eight (two in twelve
over a longer sample). The residual cases are episodes whose authored base pose
is tight for a robot of Stretch's footprint.

### What the kinematics allow

Stretch's manipulator is nearly Cartesian, so reaching decomposes: wrist pitch
and roll choose *how* to grasp and the rest choose *where*. But the two grasp
styles are not equivalent:

- **Horizontal** (the default): the tool sits ahead of the wrist yaw axis, so yaw
  gives real lateral authority. `lift + arm + wrist yaw` is a square,
  well-conditioned system and the base never has to move.
- **Top-down**: the tool hangs directly beneath the yaw axis, the lever arm
  collapses, and yaw stops moving the tool sideways at all. Stretch has no
  wrist-driven lateral freedom reaching straight down — a real one drives its
  base — so top-down solves recruit the holonomic slides, weighted against so
  they only creep when nothing else will do.

Run `--policy scripted_top_down` to compare them.

## Which `--policy` for which model

Retargeting the *episode* makes the benchmark runnable at all. Getting a
particular *model* to drive Stretch is a separate question, and the answer
depends on one thing: what action space the checkpoint was trained on.

| model | action space it emits | how to run it |
| --- | --- | --- |
| `allenai/MolmoBot-DROID` | `franka_joint`: 7 arm joints + gripper | `--policy vla` (remapped) |
| pi0.5 / pi0 DROID, DreamZero | 7 Franka joints + gripper | `--policy vla` (remapped) |
| MolmoBot fine-tuned on Stretch | Stretch's 5 move groups, 10 dims | `--policy molmobot` (native) |
| your BC checkpoint | Stretch's 10-dim encoding | `--policy bc` |
| scripted expert / A\* planner | — | `--policy baseline` |

```bash
# Franka-space model: serve it on the websocket protocol, then remap
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy vla --vla-host localhost --vla-port 8000 --benchmark pick

# MolmoBot fine-tuned on Stretch's own move groups: no remapping at all
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy molmobot --checkpoint /path/to/checkpoint --benchmark pick
```

`--policy molmobot` needs MolmoBot importable
(`git clone https://github.com/allenai/MolmoBot`, then that repo's inner
`MolmoBot/` directory on `PYTHONPATH`); it is not a dependency here.
`policies/molmobot_policy.py` supplies Stretch's move-group spec and delegates to
MolmoBot's own `SynthVLAPolicy`, which already returns an action dict keyed by
move group — exactly what Stretch's controllers take. See `finetuning/README.md`
for how to produce such a checkpoint.

The rest of this section is about the remapped path.

### How the remap works

A joint vector is a robot-specific *parameterisation* of a tool pose, so the
retarget turns it back into the thing it parameterises and works in the world:

```
VLA action (7 joints + gripper)
  -> FrankaArm.forward()          tool pose in the authoring arm's frame
  -> FrankaEpisodeFrame           the same pose in the world
  -> StretchPoseSolver.solve()    lift / arm / wrist / base targets
```

and the observation the model is fed runs the same path backwards through
`FrankaArm.inverse()`, so its proprioception tracks where Stretch's tool actually
is rather than where it was last told to go.

The world is the hinge, and no workspace calibration is fitted by hand. The
episode's authoring Franka was placed so *its* workspace covered the target, and
`episode_frame.py` records that pose on the way past — before
`retarget_base_pose()` overwrites it. Compose a VLA-space tool pose with it and
you get an absolute world pose, and the object is at the same world pose for
either robot.

Solving Stretch's joints for a full 6-DOF pose is unusually clean, because its
wrist turns out to be an exact ZYX Euler triple:

```
R_tool_world = Rz(base_yaw + wrist_yaw) @ Ry(wrist_pitch) @ Rx(wrist_roll)
```

verified to machine precision on the compiled MJCF. So pitch and roll are read
straight off a requested orientation and only the *sum* of base yaw and wrist yaw
is constrained — which leaves one free parameter inside an exactly-matched
orientation, the **yaw split**: turn the base by `+s` and the wrist by `-s` and
the arm swings around the base axis while the tool keeps pointing exactly where
it was asked to. `lift + arm + split` is then a square, well-conditioned system
for position, and unlike the wrist-yaw solve it does *not* go singular reaching
straight down.

### What it measures, and what it costs

Over the pick benchmark's grasp trajectories (Franka joint targets interpolated
from each episode's own `init_qpos` to a top-down grasp at its object, retargeted
step by step):

| quantity                              | median | p90    |
| ------------------------------------- | ------ | ------ |
| grasp-pose position error             | 3.7mm  | 24mm   |
| grasp-pose orientation error          | 0.008 rad | 0.018 rad |
| base rotation used over an episode    | 0.13 rad | 0.16 rad |
| virtual-arm IK error (proprioception) | 0.3mm  | 1.1mm  |

Three things the remap does not fix, all reported per episode in
`RemapTelemetry` rather than hidden:

- **Stretch's minimum reach.** Its tool cannot come closer than ~0.39m to its own
  base axis; a Franka's home posture is tucked in at ~0.23m. Roughly a third of
  the *intermediate* poses of a Franka approach fall in that hole and are
  retargeted to the nearest reachable pose. The grasp pose itself, which is what
  scores, is almost always reachable.
- **The base turns.** Matching an orientation exactly costs the yaw split, which
  swings the head camera. It is small (0.13 rad) but it is not nothing.
- **The cameras are Stretch's.** A DROID-trained model expects two fixed,
  off-robot shoulder views and gets one egocentric view that moves with the
  robot, fed to both inputs. This is the largest single mismatch and the one
  kinematics cannot touch. It is what the next section is for.

Stretch also spawns stowed, a configuration no Franka pose maps to, so the first
~30 steps of an episode drive to the authoring arm's own start pose *without*
querying the model. The model's first observation is then one it could plausibly
have been trained on, and the unstow never enters its action history.

`--policy vla` uses `solver_mode="exact"` (the yaw split). Two alternatives are
implemented and measured worse: `free_azimuth` spends the wrist yaw on reaching
instead and mis-orients the gripper by more than a radian through the approach;
`translating` lets the base drive, which tracks the *approach* better but ends
the grasp worse, because the base wanders off the standoff the episode chose.

## Fine-tuning on Stretch's own data

The other road: stop translating for the model and teach it Stretch. This needs
demonstrations, and the benchmark's own 1000 episodes are the test set — so
`finetuning/` drives MolmoSpaces' *data generation* pipeline instead, which
samples tasks procedurally and is unbounded. `finetuning/README.md` has the
details; the short version is that **MolmoBot needs neither a dataset conversion
nor a remapping.**

It trains straight off MolmoSpaces trajectories, and its action space is
configured by move group — so a Stretch fine-tune learns Stretch's own ten
numbers (`base` 3, `lift` 1, `arm` 1, `wrist` 3, `gripper` 2) and is evaluated
with `--policy molmobot`.

```bash
# 2 episodes, 1 house — check the setup before spending a day on it
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task debug --output-dir data/stretch_debug --no-export

# a real run
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task pick --task pnp --episodes 2000 --num-workers 8 \
    --output-dir data/stretch_pick --no-export

# prepare it and print MolmoBot's commands (preprocessing, then training)
python -m examples.machine_learning.molmospaces.finetuning.finetune \
    --rollouts data/stretch_pick/rollouts/pick --trainer molmobot

# or, for pi0.5, export to LeRobot first and use --trainer openpi
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task pick --episodes 2000 --output-dir data/stretch_pick --action-space franka
python -m examples.machine_learning.molmospaces.finetuning.finetune \
    --dataset data/stretch_pick/lerobot --trainer openpi --base-checkpoint pi05_droid
```

Demonstrations can also come from you rather than the scripted expert:

```bash
python -m examples.molmo_environment --dataset procthor-10k --house-index 0 \
    --keyboard --record_dataset data/teleop_pick --record-task "pick up the mug"
```

`R` starts an episode, `T` keeps it, `X` discards it. It writes the same on-disk
format, so it feeds the same trainers. The action recorded for a frame is the
*next* frame's state — for a position-controlled arm that is the command,
retrospectively — and the operator delimits episodes rather than recording
continuously, because a demonstration that starts thirty seconds before the reach
teaches a policy to wait.

`datagen_configs.py` registers Stretch versions of the MolmoSpaces datagen
configs — same task classes, same success criteria, Stretch's robot, cameras and
scripted expert substituted in. One thing beyond the substitution genuinely has
to change: the samplers place a Franka within 0.7m of the target because that is
a Franka's reach, and Stretch cannot work closer than 0.39m or further than
0.99m. `STRETCH_PLACEMENT` widens those constraints to the same 0.55–0.90m band
the episode retarget uses, so generated and retargeted episodes present the robot
with the same geometry.

They are also addressable from MolmoSpaces' own entry point:

```bash
python -m molmo_spaces.data_generation.main \
    examples.machine_learning.molmospaces.finetuning.datagen_configs:StretchPickDataGenConfig
```

### The action space is the decision

`lerobot_export.py` turns the recorded HDF5 + MP4 rollouts into a LeRobot v2.1
dataset, in one of two spaces:

- **`franka`** (the default) — the same 8-dimensional Franka joint space the
  pretrained model already emits, obtained by running `franka_remapping/`
  *backwards*: Stretch's recorded tool poses become virtual Franka joints. The
  model keeps its action head and its pretrained weights, and is evaluated
  through the same remapper. Because the forward and reverse maps are the same
  code, whatever the retarget cannot express the training data does not contain
  either — the model is never trained to ask for something the robot cannot do.
  On recorded expert rollouts the encoding reproduces the motions to **0.5mm**
  mean; `ExportMetadata.mean_shadow_ik_error_m` reports it per dataset and
  `finetune.py` refuses to launch above 20mm.
- **`stretch`** — Stretch's own 10-dimensional move-group vector, the encoding in
  `policies/networks.py`. Nothing is lost to a retarget and the policy drives the
  base directly, but the action head has to be reshaped and re-learned, so it
  wants far more data.

One trap worth naming, because it is silent: a `franka`-space export encodes
against a *virtual* Franka mounted on Stretch itself (`MAST_MOUNT_FORWARD_M`,
`MAST_MOUNT_HEIGHT_M`, chosen by workspace overlap), while benchmark evaluation
defaults to the *authoring* Franka's recorded frame. A checkpoint fine-tuned on
the former must be evaluated with `frame_source="mast"`, or every action it emits
is offset by the standoff between the two shoulders and the arm reaches
consistently short. `finetune.py` prints the setting the dataset needs.

The fine-tune itself runs in the model's own repository — openpi for pi0/pi0.5,
LeRobot for ACT/diffusion/SmolVLA — because that is where the trainers and their
JAX/PyTorch stacks live, and neither is a dependency here. `finetune.py` does the
parts that are this repo's business: validate the dataset, compute the
normalisation statistics, write the trainer config, and print or run the command.
Serving the result over the same websocket protocol makes it scorable with
`run_benchmarks.py --policy vla`.

## Watching a policy, and keeping proof

Three different questions, three answers.

**Watch an evaluation as it happens.** `--viewer` launches MuJoCo's passive
viewer for each episode:

```bash
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --benchmark pick --episodes 5 --viewer
```

It forces single-worker, since eight workers would open eight windows.
Internally it sets `STRETCH_MOLMOSPACES_VIEWER=1`, which the eval config reads --
`run_evaluation()` builds the config from a class and takes no override, so an
environment variable is the only injection point it leaves open.

The view is a chase camera mounted on the robot's base, because MolmoSpaces'
`setup_viewer` will only aim the viewer at a *fixed MJCF camera*
(`viewer_cam_dict["camera"]`). Left alone it keeps MuJoCo's default framing of
the whole model, and since benchmark houses load in their **ceiling** variant,
that is a sealed building photographed from ~70m away with the robot invisible
inside it. `Stretch4Robot._add_chase_camera()` mounts the camera; the offset was
picked by ray-casting to the robot across six episodes, which showed anything
much beyond a metre behind the robot is inside a wall every time. Roughly one
episode in six still ends up with something between the camera and the robot;
press `[` / `]` in the viewer to cycle to another camera, or Esc for the free
camera. Set `viewer_cam_dict = {"camera": "robot_0/camera_center_link"}` on your
eval config for the robot's own egocentric view instead.

**Export proof of a run.** `report.py` turns MolmoSpaces' raw output — JSON blobs
in HDF5 and unlabelled per-camera MP4s — into artifacts someone else can look at
without this repository:

```bash
# after the fact, on any run directory
python -m examples.machine_learning.molmospaces.report eval_output/stretch4/<run>

# or as part of the run
python -m examples.machine_learning.molmospaces.run_benchmarks --episodes 5 --report
```

Per episode you get a `.mp4` with the cameras tiled side by side and every frame
captioned with the step, the outcome and the language instruction, plus a `.csv`
of per-step joint positions, commanded targets, tool pose and base pose. Per run
you get `summary.md` and `summary.json`.

**Watch training.** `train_bc.py` writes `<checkpoint>_curves.png`,
`_history.csv` and `_history.json` beside the checkpoint, rewritten every epoch
so a run still going — or one that died — can be inspected.

## Running a checkpoint in the live sim

`run_benchmarks.py` scores a checkpoint headlessly on fixed episodes.
`live_policy.py` runs the same checkpoint in the sim you can watch and interfere
with: the MuJoCo viewer, a MolmoSpaces house, and Stretch's *real* omniwheel base
rather than the benchmark's virtual holonomic one.

```bash
# drive manually, then hit SPACE to hand over to the policy
python -m examples.machine_learning.molmospaces.live_policy \
    --checkpoint checkpoints/stretch_manip.pt --dataset procthor-10k --house-index 0

# the bundled room scene, no house download, policy running from the start,
# streaming to Rerun and recording to disk
python -m examples.machine_learning.molmospaces.live_policy \
    --checkpoint checkpoints/stretch_manip.pt --scene default --autostart \
    --rerun --record runs/live_demo
```

The viewer camera follows the robot rather than framing the whole house — a
procthor house is ~30m across and not centred on the origin, so MuJoCo's default
framing leaves the robot a few pixels wide, or off screen. Orbiting and zooming
still work; the camera keeps a fixed world heading as the base turns.
`examples/molmo_environment.py` does the same, and takes `--no-follow-robot` if
you would rather see the whole floorplan.

SPACE toggles the policy; while it is off you keep the full keyboard teleop, so
you can put the robot somewhere interesting and then hand over. `--rerun` opens a
viewer with both camera feeds beside time-series of every joint and every
command; `--record` writes the same telemetry to `telemetry.csv` and the frames
to per-camera MP4s.

### Why one checkpoint runs on both robots

The two robots differ in their base and in nothing else the policy sees.
`StatusStretchJoints` and the MolmoSpaces move groups report the same seven
proprioception numbers in the same units — the simulator's `arm.pos` is the
tendon length, which is the total telescoping extension the move group reports,
and the finger positions are raw URDF joint angles in both. So
`policies/checkpoint.py` owns loading, normalisation, chunking and decoding for
both paths, and neither `bc_policy.py` nor `live_policy.py` does any of it
itself.

The base is the exception, and it is why `networks.py` encodes the base action as
a *relative* step in the base's own frame rather than as a world pose. In the
benchmark that step becomes a holonomic joint target; in the live sim
`apply_action()` divides it by the control period and issues it as a body-frame
velocity to the three omniwheels, clipped so a bad prediction cannot launch the
robot across the house.

Two mismatches remain, and both are visible in the code rather than papered over:
the live sim renders the head camera at 1280×964 and the benchmark at the
episode's resolution, so the 112×112 the network sees is squashed slightly
differently; and the live base tracks a velocity command rather than snapping to
a position target, so it lags the benchmark base by a control step or two.

## Known limitations

- **`door_opening` is RBY1-only upstream.** `DoorOpeningTask` builds its sensor
  suite with `molmo_spaces.env.rby1_sensors.get_rby1_door_opening_sensors`, which
  asserts `"rby1" in robot_config.name` and then adds two `RBY1TCPPoseSensor`s
  for a dual-arm torso Stretch does not have. The task class is chosen by the
  benchmark JSON, so no amount of retargeting on this side reaches it; running it
  needs an upstream change. `benchmarks.py` marks it `supported=False`, the sweep
  skips it, and `--benchmark door_opening` still runs it so the error stays
  reachable.
- **The head does not move.** The SE4 URDF has no head pan or tilt joint, so the
  head camera is a fixed forward-and-down view rigidly tied to base yaw. That is
  a real constraint on navigation and on any vision-driven policy, not something
  this integration chose.
- **The scripted expert is not a planner.** No collision-aware motion planning
  and no grasp-pose search: MolmoSpaces' own solvers do both, but they need
  CuRobo and a Stretch grasp library that does not exist. The expert aims the
  tool at the object's body origin and closes. It succeeds on a minority of
  episodes, which is enough to be a baseline and a teacher but is not a ceiling.
- **Articulation is best-effort.** Open and close grasp the moving leaf body's
  origin, not a detected handle, and drag it along the joint's own arc. Success
  there is well below the pick family.
- **One Stretch per scene.** The move-group lookups use fixed names under a
  single `robot_0/` namespace.
- **The live sim has no task and no score.** `live_policy.py` runs the policy and
  shows you what it does; it does not judge success, because a MolmoSpaces house
  loaded this way carries no task specification. Use `run_benchmarks.py` for
  numbers and the live sim for seeing.
