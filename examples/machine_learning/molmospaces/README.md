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
  episode_overrides.py  retarget a Franka/RBY1 episode onto Stretch
policies/
  kinematics.py         reach solving
  scripted.py           the scripted expert (baseline + BC teacher)
  networks.py           the BC network and the action/state encoding
  checkpoint.py         loading and running a checkpoint, robot-stack agnostic
  bc_policy.py          the trained policy, as a MolmoSpaces InferencePolicy
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
| `gripper` | 2   | the two finger joints                                |

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

Run `--policy scripted_top_down` to compare them.

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
