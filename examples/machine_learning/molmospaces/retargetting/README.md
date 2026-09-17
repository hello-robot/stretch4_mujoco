# Retargeting parameter search

Finding the camera and gripper settings that let the released `allenai/MolmoBot-DROID`
checkpoint drive Stretch 4.

`demo_droid_on_stretch.py` raises a question it cannot settle: that checkpoint
reaches Stretch through a retargeting layer *and* a camera it was never trained
on, so when a rollout fails there is no way to tell which of the two is at
fault. This package runs the same four grasps through seven setups that walk
from the Franka and camera the policy knows to the Stretch and fisheye it does
not, one change at a time, and then searches the settings that are actually
free.

```
examples/machine_learning/molmospaces/retargetting/
    params_search.py         the entry point: trials, grid, CMA-ES, report
    setups.py                the seven setups, their eval configs and overrides
    cameras.py               the exo camera: mount, fisheye, rectify, crop
    franka_droid_policy.py   the same checkpoint on a Franka, un-retargeted
    mini_benchmark.py        one kitchen, one robot pose, four objects
    scoring.py               the objective, and the report
```

## Running it

```bash
# the default: sweep every parameter that applies to each setup, in stages.
# 184 trials, 736 rollouts, roughly 9 hours. Prints the plan before it starts.
python -m examples.machine_learning.molmospaces.retargetting.params_search

# the quick comparison table instead: each setup once, at its own defaults.
# 7 trials, ~20 minutes.
python -m ...params_search --search none

# one setup, a grid over dimensions you name
python -m ...params_search --setup stretch_stretchcam \
    --search grid --dim pitch_deg=15:50:4 --dim fovy=50:100:3

# the gripper parameters by CMA-ES
python -m ...params_search --setup stretch_fisheye --search cmaes \
    --dim grasp_offset_m=-0.05:0.15 --dim wrist_tilt_deg=-20:60 \
    --population 6 --generations 5

python -m ...params_search --list-setups   # what the seven are
python -m ...params_search --list-dims     # what can be searched, and what the sweep tries
python -m ...retargetting.diagnose         # measure the retargeting, no policy needed
```

Output lands under `--output-dir` (default `eval_output/retarget_params/`): an
MP4 per rollout, `episodes.csv`, `trials.csv`, `trials.jsonl`, and `report.md`
ranking the trials. **Everything is rewritten after every trial**, so a long
sweep can be stopped at any point and what it has is already on disk.

### What the default sweep does

Per setup, a full grid per stage, carrying the winning parameters into the next
stage:

| stage | dimensions | points | applies to |
|---|---|---:|---|
| camera | `pitch_deg` (4) × `fovy` (4) | 16 | every setup |
| gripper | `grasp_offset_m` (4) × `z_offset_fraction` (2) × `wrist_tilt_deg` (3) | 24 | Stretch setups |

Staged rather than one grid over all five, because the full cross product is 384
points per Stretch setup — days of rollouts, most of them re-measuring a camera
a previous point already showed was bad. Parameters that *interact* are kept in
the same stage: `grasp_offset_m` and `z_offset_fraction` are inseparable (see
the debugging section below), and swept one at a time neither would have looked
like the answer. What staging cannot see is an interaction *across* stages — a
camera that is only good with a particular gripper correction. `--search grid`
over a hand-picked pair is how you check one if you suspect it.

Dimensions declare which robot they apply to, so the gripper stage is skipped on
Franka setups (there is nothing to retarget on the robot the policy was trained
on), and `--dim grasp_offset_m` on a Franka setup is refused rather than running
16 identical trials.

## The mount is fixed; the optics are not

The camera position is not searchable, by design. It is `[0.0788406, -0.075, z]`
in the mount body's frame — where Stretch 4's right head camera physically
mounts — and every setup holds it: on Stretch that body is `base_link`, and on
the Franka it is `fr3_link1` at `z = 1.21024331`, the same point taken relative
to that robot's own base. `fr3_link1` turns with joint 1, so the Franka's view
yaws with the arm the way Stretch's head camera yaws with the base its arm
reaches from. Holding that point still is the premise of the transplant: what
the study varies is the optics around it, so `DIMENSIONS` contains `pitch_deg`
and `fovy` and no translation at all.

Both robots' cameras end up at **the same height in the room** — 1.5432 m above
the floor — which is what makes 2 → 3, 4 → 5 and 6 → 7 comparisons of the robot
rather than of two different viewpoints. That takes different numbers on each,
because their bases are at different heights: Stretch's `base_link` is on the
floor, while a benchmark Franka's `fr3_link0` is 0.75 m up on a pedestal and
`fr3_link1` another 0.333 m above that. So the Franka offset is
1.5432 − 0.75 − 0.333 = **0.460 m** above `fr3_link1`, not the 1.210 m that
would be right if `fr3_link0` stood on the floor. `setups.FRANKA_STRETCHCAM_HEIGHT`
derives it rather than stating it, so the two cannot drift apart.

Mind the sign when searching the tilt: `pitch_deg` is measured **up from
straight down** (0 = floor, 90 = horizon), so pointing further down is a
*smaller* number.

## What can be searched

| dimension | applies to | what it moves |
|---|---|---|
| `pitch_deg` | all | camera tilt, measured up from straight down: smaller looks further *down* (0 = floor, 90 = horizon) |
| `fovy` | all | vertical field of view (71 = DROID, 123 = Stretch's fisheye) |
| `grasp_offset_m` | Stretch | commanded grasp centre along the approach axis |
| `wrist_tilt_deg` | Stretch | extra pitch between the Franka tool frame and Stretch's |
| `z_offset_fraction` | Stretch | how much of the measured lift shortfall to add to targets |

`--list-dims` prints each one's range and the values the sweep tries. The three
Stretch-only dimensions are carried on a Franka setup's parameters but never
read there, so one parameter vector still describes a trial on either robot —
`--dim` refuses to *search* them on a Franka setup, since every trial would be
identical.

Output lands under `--output-dir` (default `eval_output/retarget_params/`): an
MP4 per rollout, `episodes.csv`, `trials.csv`, `trials.jsonl`, and `report.md`
ranking the trials.

One trial is four rollouts of ~300 steps with a VLA in the loop — minutes, not
seconds. That is what makes the default sweep a 9-hour job; `--search none` is
the 20-minute version.

## Debugging the retargeting: what was wrong, and the fix

The first full run put `franka_stretchcam` at 0.965 (3/4 picked) and
`stretch_stretchcam` at 0.776 (1/4) — the *same camera* on the two robots, so
the gap was the retargeting. `diagnose.py` found two geometric faults, both
readable off a standing robot with no policy running.

**1. The grasp centre is past the fingertips.** The retargeting drives Stretch's
`grasp_center_link` to the pose the policy asked for its Robotiq's `grasp_site`
— but those are different points on their grippers. Measured in the base frame:
`grasp_center_link` at x = 0.4667, furthest fingertip geom at x = 0.4516, finger
pads at x = 0.3611. So an object placed "at the grasp centre" is 1.5 cm *beyond*
the fingertips and 10.6 cm beyond the pads. The fingers close behind it and nudge
it. `grasp_offset_m` pushes the commanded centre further along the approach,
which pulls the object deeper into the gripper.

**2. Every grasp is commanded 5.1 cm high.** `measure_tool_height_offset()`
returns +0.103 m here and `z_offset_fraction = 0.5` adds half of it to every
target. Three of the four objects are shorter than that:

| object | height above counter | at +5.1 cm |
|---|---:|---|
| bowl | 4.8 cm | rim still catches |
| potato | 4.0 cm | closes above it |
| salt_shaker | 5.5 cm | grazes the top |
| knife | 2.6 cm | never approaches |

### The search

`grasp_offset_m` × `z_offset_fraction`, four objects per point:

| `grasp_offset_m` | `z_frac=0.0` | `z_frac=0.5` |
|---:|---:|---:|
| 0.00 | 0.776 (1/4) | 0.776 (1/4) |
| 0.03 | 0.792 (1/4) | — |
| 0.06 | 0.945 (3/4) | 0.554 (0/4) |
| **0.09** | **0.980 (3/4)** | 0.818 (1/4) |
| 0.12 | 0.895 (2/4) | 0.871 (1/4) |
| 0.15 | 0.972 (2/4) | 0.915 (3/4) |

The two parameters interact, which is why neither alone had looked like the
answer: at `grasp_offset_m = 0` the height made no difference, because the depth
error was already losing every grasp. Once the depth is right the height becomes
the limiting error — mean score over the four offsets is 0.948 at `z_frac=0`
against 0.789 at 0.5.

At the best point Stretch picks up **the same three objects the Franka picks**,
and fails on the same one:

| object | before (0.0 / 0.5) | after (0.09 / 0.0) | Franka, same camera |
|---|---|---|---|
| bowl | picked | picked | picked |
| potato | 0.5 cm lift | 1.1 cm lift, still failed | failed |
| salt_shaker | 0.4 cm lift | picked | picked |
| knife | never approached (18.7 cm) | picked (2.9 cm, 2.0 cm lift) | picked |

`0.776 (1/4)` → `0.980 (3/4)`, against the Franka's `0.965 (3/4)`. These are now
the defaults for the Stretch setups (`STRETCH_GRASP_OFFSET_M`,
`STRETCH_Z_OFFSET_FRACTION`).

### How much to trust this

Not very much, on its own. Each cell is one rollout per object, and the noise is
a whole object wide: `grasp_offset_m = 0.06 / z_frac = 0` scored 0.860 (2/4) in
one run and 0.945 (3/4) in another, at identical parameters. What survives that
noise is the *shape* — everything at or above 0.06 beats everything at or below
0.03, and `z_frac=0` beats 0.5 at every offset but the last. The peak's exact
location does not; 0.09 is the best single point and agrees with the geometry
(it puts the object at the finger pads), which is why it is the default, but
0.06–0.15 is one plateau as far as this evidence goes.

Confirm on a real benchmark before relying on it:

    run_benchmarks.py --policy molmobot_droid --benchmark pick --episodes 50

## The fisheye, and matching the simulator

Setups 4–7 are meant to show the policy exactly what `Stretch4MujocoSimulator`
shows: a 123-degree fisheye off a camera that is **bolted on sideways**. Four
things have to line up, in this order, and they are the same four the simulator
applies.

1. **Render sideways.** The camera's orientation carries a −90° roll about the
   view axis, because that is how the hardware is mounted. Decomposed
   intrinsic-ZXZ against the compiled MJCF, `camera_right_link`'s optical frame
   is `yaw −90, pitch 43, roll −90`;
   `demo_droid_on_stretch.add_reconstructed_exo_camera` is the same thing with
   the roll dropped, which is right for the *pinhole* setups (2 and 3) and wrong
   for these.
2. **Warp in the sensor frame.** `apply_fisheye_distortion` is handed `fx, fy,
   cx, cy` calibrated on the real 1920×1200 sensor, with the optical centre
   ~20 px off centre. That calibration only means anything in the sensor's own
   frame, so the render has to be sideways before the warp — hence the order.
   The render is 640×400, the sensor's 1.6 aspect, or the calibration cannot be
   projected onto the frame at all.
3. **Turn it upright.** `quarter_turns=-1` undoes the mount, exactly as
   `StatusStretchCamera.get_camera_data` does on the robot. The output is
   400×640 portrait.
4. **Crop, optionally.** Off by default, so a fisheye setup delivers the frame
   the hardware delivers. `--exo-crop 640x360` trades field of view for the
   landscape shape the checkpoint was trained on.

Steps 1 and 3 are a pair: the quarter turn exists to undo the roll. With the
roll missing it instead lays an already-upright frame on its side, and the warp
in step 2 is applied along axes the lens does not have — which is what this
package did before the mount roll was put back.

### Checked against the simulator

`stretch_fisheye`'s camera was rendered beside the MJCF camera the simulator
uses (`robot_0/camera_right_link` at its own `fovy` and render size, warped and
turned by `StretchCameras.cam_nav_rgb_se4_right`'s own callback), in the same
scene from the same robot pose:

| this package's exo camera | mean abs diff | pixels >10 apart |
|---|---:|---:|
| at the setup's mount (`STRETCH_STRETCHCAM_HEIGHT`) | 17.5 / 255 | 33.7 % |
| at the measured optical pose (`STRETCH_HEAD_CAMERA_OPTICAL_POSE`) | **0.03 / 255** | **0.03 %** |

The second row is the pipeline being exactly right — 0.03/255 is resampling
noise on edges. The first row is the same pipeline at a mount point 1.45 cm
forward and 6.9 cm below the real one, so all that separates it from the
simulator is parallax.

That 7 cm is the reconstruction `demo_droid_on_stretch.py` uses, and every setup
here shares it deliberately: it is what keeps the difference between the pinhole
rows and the fisheye rows down to the lens. Swap
`STRETCH_HEAD_CAMERA_OPTICAL_POSE` in (it is recorded in `setups.py` for exactly
this) if you would rather have `stretch_fisheye` be the simulator's camera to
the pixel and accept a 7 cm offset between it and `stretch_stretchcam`.

## How success and failure are decided

Nothing in this package judges a grasp. The verdict comes from MolmoSpaces'
`PickTask`, unmodified, which is what makes a score here comparable with a
`run_benchmarks.py` number.

### The predicate

At every step `PickTask.get_info()` computes two things about the target object
(`molmo_spaces/tasks/pick_task.py`, around line 135):

```python
lift_height = object.position[2] - task_config.pickup_obj_start_pose[2]

# every current contact where exactly one side is the target object
only_robot_collision = robot_collision and not non_robot_collision

success = only_robot_collision and lift_height >= succ_pos_threshold
```

So an episode succeeds when **both** hold:

1. **The object is off everything but the robot.** Contacts are compared by
   *root body*, so the whole robot tree counts, not just the fingers — a bowl
   balanced on a forearm passes. Resting on the counter fails, because the
   counter is a non-robot contact. Touching nothing at all also fails: at least
   one robot contact is required, so an object knocked into the air does not
   count.
2. **It has risen at least `succ_pos_threshold` above where it started** — 1 cm,
   set by `mini_benchmark.SUCCESS_LIFT_M`.

The height is measured against `pickup_obj_start_pose` **as recorded in the
benchmark JSON**, not against wherever the object actually is at reset. Those
agree here to well under a millimetre because `mini_benchmark.settled_pose()`
drops each object on the counter and records the pose it comes to rest in — if
it instead authored a guessed pose the object would settle downwards at reset
and quietly raise the bar.

That threshold is also why `added_pickup_repair` matters. A THOR prefab attached
straight from `install_uid()` weighs tens of kilograms, and nothing lifts it
1 cm; the repair is applied both when the benchmark is built and when an episode
is loaded.

### When the verdict is taken

`run_single_rollout` (`molmo_spaces/data_generation/pipeline.py:701`) loops until
`task.is_done()`, then returns `task.judge_success()` — which re-evaluates the
predicate **at the final state**. Two settings shape that:

| setting | value here | effect |
|---|---|---|
| `end_on_success` | `True` | the loop breaks the moment the predicate first holds |
| `terminate_upon_success` | `False` (inherited) | success does not itself make the task terminal |
| `task_horizon` | 20 s at 15 Hz ≈ 300 steps | `--episode-steps` overrides it |

Because `end_on_success` is on, the rollout stops at the first success and the
final-state re-check agrees with it — so in practice the verdict is "did it ever
lift the object clear", not "was it holding it when time ran out". Turn that off
and an episode that lifts an object and then drops it would score as a failure.
The horizon comes from the benchmark's `task_horizon_sec` (20) converted through
`policy_dt_ms` (66); running out of it is an ordinary failure.

### Where the verdict ends up

One boolean reaches four places, and they cannot disagree:

| where | how it reads |
|---|---|
| the MP4 filename | `..._success.mp4` / `..._failure.mp4` / `..._incomplete.mp4` |
| `episodes.csv` | the `success` and `completed` columns |
| `report.md` | `picked up` / `failed` / `crashed`, and **picked** in the summary table |
| `trials.csv` | `successes` and `success_rate`, counted over completed episodes only |

`GraspProbe` takes it from the rollout's own return value rather than
recomputing it, so the score in this package and the `success_count` MolmoSpaces
reports are the same number. Everything else the probe records — approach,
contact, lift — is extra detail about a rollout whose success was decided
elsewhere.

## The objective, and why it is not the success rate

Success rate is the number that matters and the wrong thing to optimise. Four
episodes give five possible success rates, most settings score zero, and a
search that can only see zeros has nothing to climb: a camera that at least
brings the gripper to the object would be indistinguishable from one pointed at
the ceiling.

So every episode also gets a continuous score in `[0, 1]`, built from what a
*failed* rollout still tells you:

| term | what it measures | weight |
|---|---|---:|
| approach | fraction of the initial gripper-to-object gap that was closed | 0.45 |
| contact | whether the robot ever touched the object | 0.25 |
| lift | fraction of the 1cm success threshold the object came off the counter | 0.30 |

A success scores 1.0 outright, so the ranking never prefers a near-miss to a
grasp. The three terms are the order a grasp goes wrong in, so a setting that
fails later scores higher than one that fails earlier and the search has a
gradient long before anything succeeds. The trial's score is the mean over the
four objects, and an errored trial scores 0 rather than being dropped — a
setting that crashes the renderer is a bad setting, and the search should learn
to stay away from it rather than propose it again.

### Crashed episodes

A rollout that raises is not scored at all — it is excluded from the trial mean,
not counted as a failure. The distinction matters: counting a crash as a zero
would teach the search that whatever parameters were in flight when the GPU
filled up are bad ones. A trial whose episodes *all* crashed shows `n/a` and is
not ranked.

You can spot these without reading a log. The MP4 recorder names the file after
the outcome, and a rollout that raised never reports one, so its video is
`..._incomplete.mp4` — the `report.md` cell says `crashed`, and the run logs a
warning naming the `running_log.log` with the traceback.

By far the most common cause is `torch.OutOfMemoryError` from the policy because
something else is holding the card. The DROID checkpoint wants ~11 GiB plus
activations, and each step allocates about 744 MiB, so a stale notebook kernel
is enough to tip it over. Check `nvidia-smi` before a long run, and consider
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

None of this is recorded by the evaluation pipeline (its H5 has `success` and
`rewards`, and neither says how close a failure came), so `scoring.GraspProbe`
watches the rollout itself through the same observer hook the MP4 recorder uses.
See `scoring.py`.

## The CMA-ES implementation

`params_search.SimpleCMAES` is about sixty lines of textbook
(μ/μ_w, λ)-CMA-ES — Hansen's tutorial algorithm with the standard parameter
defaults — written out rather than taken from the `cma` package, which is not in
this repository's environment and would be a new dependency for one script.

### What the algorithm does

CMA-ES keeps a multivariate normal over the search space and moves it towards
wherever the good points were. Each generation it samples λ points, keeps the
best μ, and updates three things: **where** the distribution is centred, **how
far** it reaches, and **which directions** it reaches furthest in. That third
part is what makes it worth the code over random search or a grid: it learns the
shape of the landscape, so a valley running diagonally through (pitch, fovy) is
followed along its floor rather than crossed.

It suits this problem because the objective is expensive (four VLA rollouts per
evaluation), noisy (contact-rich physics, a stochastic policy), and has no
usable gradient. CMA-ES is derivative-free, uses only the *ranking* of the
scores rather than their values, and is the standard choice at this budget.

### The state

| field | what it is |
|---|---|
| `mean` | centre of the sampling distribution: the current best guess |
| `sigma` | overall step size, scaling everything the distribution does |
| `covariance` | the shape: which directions are promising, and how correlated |
| `path_sigma` | evolution path used to decide whether `sigma` is too big or small |
| `path_c` | evolution path used for the rank-one covariance update |
| `weights` | the μ recombination weights, `log(μ + 0.5) - log(i)`, normalised |

### One generation

**`ask()`** — eigendecompose the covariance into `B · diag(d)`, and for each of
the λ points draw `z ~ N(0, I)`, map it through that basis to `y = B·(d ⊙ z)`,
and sample `x = mean + sigma · y`. So `sigma` sets the scale and the covariance
sets the shape and orientation.

**`tell(scores)`** — sort by score descending (this maximises; CMA-ES is
conventionally written to minimise) and take the best μ:

1. **Recombination.** The new `mean` is the weighted average of the best μ
   points. The weights are log-decreasing, so the best point counts for more
   than the μ-th.

2. **The step-size path.** `path_sigma` accumulates the mean's displacement,
   whitened by `C^(-1/2)` so successive steps are comparable. If consecutive
   steps keep pointing the same way, the path grows long and `sigma` is raised —
   the search is making steady progress and should move faster. If they cancel
   out, the path stays short and `sigma` shrinks — the search is circling an
   optimum and should refine. The final line is exactly that comparison, against
   `chi_n`, the expected length of a random walk of the same number of steps.

3. **The covariance path and update.** `path_c` accumulates the same
   displacement unwhitened, and `h_sigma` switches it off when `path_sigma` has
   grown implausibly long — which happens right after a large `sigma` increase,
   where the step is an artefact of the rescaling rather than of the landscape.
   The covariance is then a blend of three things: what it already was, a
   **rank-one** term `path_c · path_cᵀ` that stretches it along the direction the
   mean has been travelling, and a **rank-μ** term summing the selected points'
   own outer products, which captures the local shape from this generation
   alone. The rank-one term learns a long-run direction from few samples; the
   rank-μ term learns the shape fast when λ is large. Small populations lean on
   the first.

### The choices this implementation makes

**No restarts, no IPOP.** The budget here is tens of evaluations, not thousands.
The machinery that earns its keep over long runs — detecting stagnation,
restarting with a doubled population — would never come into play.

**Bounds by clipping.** `--dim name=lo:hi` is a box, and `ask()` clips what it
proposes into it. Clipping biases the search towards a boundary it is pressed
against, which is the known weakness of the approach; it is acceptable here
because every bound is a physical limit (a camera cannot have a negative field
of view) and a search that wants to sit on one is telling you something. The
*unclipped* samples are what `tell()` updates from, so the distribution is not
also distorted by the projection.

**Starting point and initial step size.** The mean starts at the setup's own
defaults, clipped into the box, so generation zero is a neighbourhood of the
configuration you would otherwise have run by hand. `sigma0` is a quarter of the
mean axis range: wide enough to leave that neighbourhood on the first
generation, narrow enough that most of a small population lands inside the box
rather than on its faces.

**Population.** `--population 6`, `--generations 5` by default, so 30 trials =
120 rollouts. That is small for CMA-ES — `4 + 3·ln(n)` is the usual λ, which is
about 6 for two dimensions and 8 for four — so search two or three dimensions at
a time, not eight.

**Noise is not handled explicitly.** Each point is evaluated once, on four fixed
episodes with a fixed seed, and there is no re-evaluation or averaging over
repeats. The four-object mean is the only variance reduction. Treat a
single-generation improvement as suggestive and the trend across generations as
the result.

### Reading the output

Every point CMA-ES evaluates is an ordinary trial: a row in `trials.csv`, a
section in `report.md`, four MP4s. The log line at the end of each generation
reports the best score of that generation and the distribution's new `mean` and
`sigma` — a `sigma` that keeps shrinking means it has found something and is
refining; one that keeps growing means the landscape is flat and the ranking is
noise.

`report.md` ranks every trial by score regardless of which generation produced
it, so the top row is the best setting found, not the last one tried.

## What this cannot tell you

One house, one robot pose, four objects, one episode each. A setting that wins
here has beaten the others *on this kitchen*. The point is to rank settings
against each other cheaply; confirm the winner with
`run_benchmarks.py --policy molmobot_droid` over a real benchmark.
