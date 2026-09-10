# Stretch 4 on the MolmoSpaces benchmarks

Runs Stretch 4 against the eight [MolmoSpaces](https://github.com/allenai/molmospaces)
benchmark evaluations, with a simple ik expert as the baseline and two roads to a
learned policy that beats it: behaviour cloning from scratch ([`training/`](training/README.md))
and fine-tuning a pretrained VLA ([`finetuning/`](finetuning/README.md)).

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
  episode_overrides.py  retarget a Franka/RBY1 episode onto Stretch
hdf5_layout.py          the MolmoSpaces trajectory format: writer, repair, splits
finetuning/             fine-tune a pretrained VLA -- see finetuning/README.md
  datagen_configs.py    Stretch versions of the MolmoSpaces datagen configs
  generate_dataset.py   generate demonstrations, then optionally export them
  live_recorder.py      record teleop demonstrations from molmo_environment.py
  lerobot_export.py     rollouts -> a LeRobot dataset, in either action space
  finetune.py           check the data, prepare it, write the config, launch
policies/
  kinematics.py         reach solving
  simple_ik_policy.py   the simple ik IK expert (baseline + BC teacher)
  networks.py           the BC network and the action/state encoding
  checkpoint.py         loading and running a checkpoint, robot-stack agnostic
  bc_policy.py          the trained policy, as a MolmoSpaces InferencePolicy
  molmobot_policy.py    a MolmoBot checkpoint driving Stretch natively
  molmobot_checkpoint.py what a checkpoint records about how it was trained
training/               behaviour-clone from scratch -- see training/README.md
  collect.py            rollouts -> training shards, keeping what worked
  dataset.py            the shard format, and the torch-side reader
  train_bc.py           fit the network, write a checkpoint
report.py               a finished run -> captioned video, telemetry, summary
visualize.py            what --visualize shows, shared by datagen and eval
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
family, plus one benchmark that is built locally rather than downloaded
(see [below](#a-benchmark-for-a-task-with-no-release)):

| key            | release  | task                                    | episodes |
| -------------- | -------- | --------------------------------------- | -------- |
| `pick`         | MB-Pick  | pick a named object off a surface       | 1000     |
| `potato`       | *local*  | pick up a potato                        | *built*  |
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

`potato` is also skipped when sweeping, for a different reason: it does not exist
until you build it, and a default sweep should not report an ERROR row for that.

```bash
python -m examples.machine_learning.molmospaces.run_benchmarks --list
```

## A benchmark for a task with no release

The eight above are released asset packages — fixed episode lists that arrive
with the MolmoSpaces assets. That covers any task family that *has* a release. It
does not cover a task family invented locally, and filtering does not rescue it:
of the 1000 episodes in MB-Pick, **six** pick up a potato, and MS-Pick has none.

`build_benchmark.py` builds the test set instead, from the same generation
pipeline that produces the training data:

```bash
# 1. generate evaluation episodes -- from a HELD-OUT split
python -m examples.machine_learning.molmospaces.finetuning.generate_dataset \
    --task potato --episodes 200 --data-split val --no-export \
    --output-dir data/stretch_potato_eval

# 2. freeze them into <assets>/benchmarks/stretch-local/potato/benchmark.json
python -m examples.machine_learning.molmospaces.build_benchmark \
    --rollouts data/stretch_potato_eval/rollouts/potato --benchmark potato

# 3. score a policy on it, exactly like a released benchmark
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --benchmark potato --policy molmobot --checkpoint <ckpt>
```

### The arguments

`--rollouts` wants the **task directory** — the one holding `house_*/`, which is
`<output-dir>/rollouts/<task>`. Not the `--output-dir` you handed the generator,
and not a single house. It is repeatable, so several runs can be pooled into one
benchmark.

Exactly one destination, and they are mutually exclusive:

- **`--benchmark <key>`** writes to `<assets>/benchmarks/stretch-local/<key>/`,
  which is where `run_benchmarks.py --benchmark <key>` looks. Only keys
  registered with `locally_built=True` are offered.
- **`--output-dir <path>`** writes anywhere. Use it for a scratch build you do
  not intend to score against.

| flag | when you want it |
| ---- | ---------------- |
| `--force` | Required to replace an existing benchmark. Without it you get a refusal, deliberately — changing a benchmark's episodes invalidates every number already recorded against it. |
| `--limit N` | Cap the size. Handy for cutting a 20-episode smoke benchmark out of a large run. |
| `--include-failures` | Keep episodes the expert did not solve. Off by default; at a high expert success rate it changes nothing. |
| `--task-horizon-sec` | Per-episode step budget. Leave at 20 to stay comparable with the released pick benchmarks. |

Two traps worth naming, because the script will do both without erroring:

- **Pointing `--rollouts` at the training run.** `data/stretch_potato` is
  `--data-split train`; a benchmark built from it scores a fine-tuned policy on
  the houses it trained in. The builder prints a yellow warning when the episodes
  it was handed came from `train`, but it cannot know what you trained on, so
  that warning is the only signal you get. Keep the evaluation run in its own
  `--output-dir`.
- **Expecting `--data-split` to change what gets generated.** On *this* script it
  only overrides the label written into the JSON, for runs whose
  `experiment_config_*.pkl` is missing. The split is chosen at generation time,
  in step 1. Do not use it to relabel train episodes as val.

### Checking a build before you trust it

Score the expert on what you just built:

```bash
python -m examples.machine_learning.molmospaces.run_benchmarks --benchmark potato
```

`--policy baseline` (the default) is the same simple_ik expert that generated the
episodes, and every episode in the benchmark is one it solved — so it should come
back near 100%. Anything much lower means the replay is not reconstructing the
scene the rollout ran in, which is a benchmark bug rather than a policy result.

It is the check that caught the one real bug here, and nothing else would have:
the potato benchmark read 0/3 because the eval path rebuilt the added target
object from the raw asset without the mass correction data generation applies, so
the potato weighed 20kg on replay and no policy could have lifted it. See
[added_pickup_repair.py](added_pickup_repair.py).

This works because **every rollout already carries its own initial conditions.**
`BaseMujocoTask.reset()` calls `MlSpacesExpConfig.freeze_task_config()`, which
pickles camera extrinsics resolved to fixed values, the robot's start joint
positions, the base pose, the pickup object and its pose, every mobile object's
pose, and the referral expressions — and stores it base64-encoded under
`obs_scene["frozen_config"]` in the HDF5. That is the same information an
`EpisodeSpec` holds, so `build_benchmark.py` is a translation rather than a
reconstruction: the poses it writes are the ones the rollout actually ran from.
Verified: a benchmark built from three episodes the expert solved during
generation replays at **3/3**.

Two things make an episode set a benchmark rather than more training data, and
only one of them can be enforced here:

- **A held-out split.** Generate with `--data-split val`. Scoring a policy on the
  houses it was fine-tuned in measures memorisation. The builder cannot know what
  you trained on, so it warns whenever the rollouts it is handed came from
  `train`.
- **Solvable episodes.** Only episodes the expert *finished* are kept, which is
  the last step's success flag rather than `success.any()` — an episode that
  lifted a potato and then dropped it has the latter True and is not evidence the
  episode is completable. `--include-failures` overrides.

Rebuilding refuses to overwrite without `--force`: a benchmark whose episodes
change underneath you invalidates every number already recorded against it. Built
benchmarks live under `benchmarks/stretch-local/` so they can never be mistaken
for `molmospaces-bench-v2`, whose scores are comparable with the MolmoSpaces
leaderboard's, and so the resource manager does not replace them on the next asset
install.

Nothing here is potato-specific except the registry entry — `_task_dict()` copies
whatever task-spec fields the frozen config carries, so the same command builds a
benchmark out of `pnp` or `open` rollouts. Add a `Benchmark(..., locally_built=True)`
entry to give one a `--benchmark` key.

## Running

```bash
# baselines on everything evaluable, a few episodes each
python -m examples.machine_learning.molmospaces.run_benchmarks --episodes 5

# one benchmark, properly, in parallel
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --benchmark pick --episodes 200 --num-workers 8
```

`--policy baseline` (the default) runs the simple ik expert on the manipulation
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
numbers are the simple ik expert's, and they are what the behaviour-cloning
pipeline below is meant to improve on — see *Known limitations* for why the
expert is not a ceiling.

The eval configs are also addressable from MolmoSpaces' entry point directly:

```bash
python molmo_spaces/evaluation/eval_main.py \
    examples.machine_learning.molmospaces.configs:StretchSimpleIKEvalConfig \
    --benchmark_dir <dir> --no_wandb
```

## Training a policy

Demonstrations come from the same place either way: `finetuning/generate_dataset.py`
rolls the simple ik expert over procedurally sampled houses, because a benchmark's
own 1000-2000 episodes are the *test set* and cloning them measures memorisation.
From there the two roads split, and each has its own README.

| | [`training/`](training/README.md) | [`finetuning/`](finetuning/README.md) |
| --- | --- | --- |
| what | behaviour-clone a small net from scratch | fine-tune a pretrained VLA |
| model | `StretchBCNet` (`policies/networks.py`) | MolmoBot, pi0/pi0.5, ACT, SmolVLA |
| data | `.npz` shards via `training/collect.py` | rollouts as-is, or a LeRobot export |
| score with | `run_benchmarks.py --policy bc` | `--policy molmobot` |

`hdf5_layout.py` is the recorded-trajectory format both sides read.

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

`stretch/episode_overrides.py` is registered as MolmoSpaces'
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

Run `--policy simple ik_top_down` to compare them.

## Which `--policy` for which model

Retargeting the *episode* makes the benchmark runnable at all. Getting a
particular *model* to drive Stretch is a separate question, and there is one
answer: the checkpoint has to emit Stretch's own action space. Nothing here
translates another robot's joint vector.

| model | action space it emits | how to run it |
| --- | --- | --- |
| MolmoBot fine-tuned on Stretch | Stretch's 5 move groups, 10 dims | `--policy molmobot` |
| your BC checkpoint | Stretch's 10-dim encoding | `--policy bc` |
| simple ik expert / A\* planner | — | `--policy baseline` |
| `allenai/MolmoBot-DROID`, pi0.5 / pi0 DROID, DreamZero | 7 Franka joints + gripper | not runnable as released — fine-tune it first |

```bash
# a MolmoBot checkpoint fine-tuned on Stretch's own move groups. The checkpoint
# is the step directory -- the one holding config.yaml, not the model_and_optim/
# inside it, which is where MolmoBot's loader looks by itself.
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy molmobot \
    --checkpoint data/stretch_pick/rollouts/molmobot/checkpoints/stretch4_pick/step9500_bestfit \
    --benchmark pick
```

`policies/molmobot_policy.py` supplies Stretch's move-group spec and delegates to
MolmoBot's own `SynthVLAPolicy`, which already returns an action dict keyed by
move group — exactly what Stretch's controllers take. See `finetuning/README.md`
for how to produce such a checkpoint.

**No `PYTHONPATH` to export.** MolmoBot is a clone rather than a dependency, so
`run_benchmarks.py --policy molmobot` puts `third_party/MolmoBot/MolmoBot` on the
import path itself — in `sys.path` for this process and the forked rollout
workers, and in `PYTHONPATH` for the spawned ones
(`finetuning/molmobot_repo.ensure_importable`). A missing checkout is a message
at the command line with the `git clone` in it, not an ImportError once per
worker; `finetune.py` is what clones. What neither can do is install MolmoBot's
*runtime* dependencies: evaluation runs MolmoSpaces and
MolmoBot in one interpreter, so `cached_path`, `transformers` and the rest have to
be in this environment, not in the training venv inside the checkout. The command
says which are missing, before the benchmark loads:

```bash
uv pip install -e ".[molmobot]"   # or the packages it names, one by one
```

**The checkpoint configures the evaluation.** `train_molmobot.py` records its
whole launch line in the `config.yaml` it saves, so the cameras and the action
type come off the checkpoint rather than out of a flag — see
`policies/molmobot_checkpoint.py`. This matters more than it sounds like it
should: serving a VLA a different camera set, or the same set in a different
order, raises nothing anywhere. The policy simply acts on a scene it is not
looking at. Any disagreement with the eval config is logged, and
`configure_from_checkpoint=False` turns the inference off.

**The checkpoint says what state training was in.** Every checkpoint a Stretch
fine-tune writes carries a `training_metrics.json` — step, training loss, best
validation loss and where it happened, learning rates as they had decayed — and
the evaluation logs it as it loads:

```
[molmobot] checkpoint saved at step 9,500/10,000, bestfit, train loss 0.02430,
           best action_flow_loss 0.05910 @ step 9,500, still improving when saved
```

That last clause is the one that matters when a benchmark scores badly: it
separates a policy that is bad from a policy that is early. See
`finetuning/README.md`.

**Relative actions become absolute targets here.** A `joint_pos_rel` checkpoint
emits deltas, and every Stretch move group is commanded absolutely — the gripper
has to be, since a relative finger command cannot hold a grasp. The inversion is
the one MolmoSpaces' own sensor defines, off-by-one and all:
`LastCommandedRelativeJointPosSensor` records
`delta[t] = commanded[t] - qpos[t-1]`, differencing against the qpos it saw on
its *previous* call, so the adapter adds the previous step's measurement rather
than the current one. Over a real pick episode the difference is up to 0.23 rad
of lift and 1.17 rad of wrist per step — enough to reach consistently short with
nothing in the logs to say why.

### Why a Franka-space model is not simply remapped

There used to be a `--policy vla` here, and a `franka_remapping/` package behind
it: it read a DROID-trained model's seven Franka joint targets, ran them through
the Franka's FK to a tool pose, composed that with the authoring arm's recorded
world frame, and solved Stretch's lift, arm, wrist and base for the result. It
worked, in the sense that it produced numbers and scored episodes. It has been
removed, because the numbers it produced were not trustworthy and nothing about
the interface said so:

- **The workspaces do not overlap.** Stretch's tool cannot come closer than
  ~0.39m to its own base axis; a Franka's home posture is tucked in at ~0.23m.
  About a third of the intermediate poses of a Franka approach fell in that hole
  and were silently retargeted to the nearest reachable pose. Grasp-pose error
  measured 3.7mm at the median but 24mm at p90 — the wrong side of a grasp
  tolerance.
- **Matching an orientation cost base rotation.** The exact solve spent a
  base/wrist yaw split to hold the tool where it was asked, which swung the head
  camera by ~0.13 rad every episode.
- **The cameras were never right.** A DROID-trained model expects two fixed,
  off-robot shoulder views and got one egocentric view that moves with the robot,
  fed to both inputs. This was the largest single mismatch and no amount of
  kinematics touches it.
- **It had a silent frame trap.** The same code ran backwards to encode
  fine-tuning data, against a *virtual* Franka on Stretch's mast rather than the
  authoring arm — so evaluating with the wrong `frame_source` made the arm reach
  consistently short with nothing in the logs to say why.

The remaining path is the one the last bullet was already pointing at: stop
translating for the model and teach it Stretch.

## Fine-tuning on Stretch's own data

Teach the model Stretch. Full detail is in
**[`finetuning/README.md`](finetuning/README.md)**; the two things worth knowing
before opening it:

- **MolmoBot needs no conversion at all.** It trains directly on MolmoSpaces
  trajectories and configures its action space by move group, so a Stretch
  fine-tune learns Stretch's own ten numbers and is evaluated with
  `--policy molmobot`.
- **For openpi and LeRobot, `lerobot_export.py` writes Stretch's own
  10-dimensional action space.** A pretrained checkpoint contributes its vision
  and language weights; its action head is re-learned, which wants more data than
  a warm-started head would. That is the whole cost, and it is paid in data
  rather than in a coordinate frame nobody can see.

The fine-tune itself runs in the model's own repository, none of which is a
dependency here. `finetune.py` does the parts that are: check the data, prepare
it, write the trainer config, print or run the command.

## Watching a policy, and keeping proof

Three different questions, three answers.

**Watch an evaluation as it happens.** `--visualize` launches MuJoCo's passive
viewer for each episode:

```bash
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --benchmark pick --episodes 5 --visualize
```

It forces single-worker, since eight workers would open eight windows.
Internally it sets `STRETCH_MOLMOSPACES_VIEWER=1`, which the eval config reads --
`run_evaluation()` builds the config from a class and takes no override, so an
environment variable is the only injection point it leaves open.

The camera is MuJoCo's **free** camera, aimed at the robot at the start of every
episode. Framing it at all is not optional: left alone MuJoCo frames the whole
model, and since benchmark houses load in their **ceiling** variant, that is a
sealed building photographed from ~70m away with the robot invisible inside it.
Free rather than tracking or fixed is what keeps the mouse in charge — a tracking
camera rewrites `lookat` from the robot's position every frame, so it cannot be
panned away and the robot can never leave the centre of the shot. Press `[` / `]`
to cycle to the model's own cameras (`Stretch4Robot` mounts a chase camera on the
base, and there is the robot's egocentric `robot_0/camera_center_link`), Esc to
come back.

`viewer_cam_dict` on the eval config still names the chase camera, since
MolmoSpaces' `setup_viewer` will only aim the viewer at a *fixed MJCF camera* and
that is the best framing available to an `eval_main.py` run, which installs no
hook of ours. `--visualize` overrides it per episode.

Alongside the viewer it streams the episode to Rerun, exactly as
`generate_dataset.py --visualize` does: the scene as meshes, the wrist/tool/object
frames, the target grasp, the waypoint plan with its progress checklist and log,
and whatever camera images the observation carries.

`visualize.py` holds both halves — the camera snap and the Rerun visualizer — so
datagen and evaluation show the same thing. Datagen drives them from its own
rollout runner; evaluation cannot, since `run_evaluation()` instantiates
`JsonEvalRunner` itself, so `install_eval_visualize_hook()` wraps that runner's
`run_single_rollout` and works from the task's `reset`/`step_chunk` rather than
from a second copy of the rollout loop.

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

**Watch training.** `train_bc.py` writes loss curves and history beside the
checkpoint; see [`training/README.md`](training/README.md).

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
`examples/molmo_environment.py` starts from the same problem but solves it with a
free camera aimed at the robot once, which can then be panned anywhere; pass
`--follow-robot` there for the tracking camera this uses.

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
- **The simple ik expert is not a planner.** No collision-aware motion planning
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
