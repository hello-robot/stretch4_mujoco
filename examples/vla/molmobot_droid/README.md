# MolmoBot-DROID on Stretch 4

These scripts run [MolmoBot-DROID](https://huggingface.co/allenai/MolmoBot-DROID), a
vision-language-action (VLA) policy trained on a Franka arm, on Stretch 4. Stretch 4 runs it in
simulation or on the real robot, by **retargeting**:

- The policy keeps believing it is driving a Franka.
- Each Franka action is converted to the pose of the Franka's tool center point (TCP).
- [`stretch4_kinematics`](https://github.com/hello-robot/stretch4_kinematics) inverse
  kinematics (IK) then finds the Stretch 4 joints that put Stretch's tool at that pose.
- Stretch's measured joints are mapped back to Franka joints for the policy's state input.

Stretch 4 is always built and simulated by this repository's `stretch4_mujoco` library.
Scenes, objects and benchmarks come straight from
[MolmoSpaces](https://github.com/allenai/molmospaces).

## The model

| | |
|---|---|
| Checkpoint | `allenai/MolmoBot-DROID`, Apache-2.0. Sim-only training, no real robot data. Called "MolmoBot (F=2)" in the [paper](https://arxiv.org/abs/2603.16861). |
| Network | Molmo2-4B: SigLIP2 so400m vision encoder and a Qwen3-style 36-layer LLM, plus a flow-matching action expert (10 integration steps). |
| Robot | Franka FR3 with a Robotiq 2F-85 gripper, on a pedestal. Model: molmospaces `franka_droid`. |
| Images | Exo camera, then wrist camera. Each shows the current frame and the one 8 control steps earlier, so 4 images per query. |
| Image size | 640x368, the size molmospaces renders the training cameras at (`FrankaDroidCameraSystem`). The MolmoBot demo notebook uses 640x360 and the benchmark JSON lists 624x352; the model resizes anyway. |
| State | 8 values: the 7 FR3 joint angles plus the Robotiq `left_driver_joint` angle (0.003 open to 0.824 closed). |
| Action | A chunk of 16 actions, each 8 values. The first 7 are absolute FR3 joint positions in radians. The last is the gripper command, 0–255, thresholded at 128 to open (0) or closed (255). |
| Rate | 15 Hz (66 ms per step). |
| Safety | Each executed step is limited to 0.2 rad per joint, measured from the current state (DROID's IK limit). |
| Training cameras | The `franka_one_random_then_wrist` preset: the exo camera is a randomized ZED 2 stand-in placed around the workspace (50–90° FOV), not just the fixed DROID shoulder camera. |
| Training height | The Franka is placed relative to the object, not the floor. Its mocap base sits at object z − 0.75 m, plus noise from U(−0.30, 0.25), under a 0.58 m pedestal. So `fr3_link0` is about 0.17 m below the object. |
| Tasks | `pick` (the focus here), `pick_and_place`, `pick_and_place_next_to`, `pick_and_place_color`, `open`, `close`. See `checkpoint.TASKS` for the instruction templates. |
| Memory | About 10 GB of VRAM in bf16, peaking near 11 GB during inference. About 0.2–0.3 s per query on an RTX 5090. |

Measured here:

- Loading takes about 12 s once the weights are cached.
- `close()` brings VRAM back to a few MB.
- If memory fragments, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Loading and releasing it

The Hugging Face repo has only the weights: `model.pt` (19 GB) and an OLMo `config.yaml`. The
network is defined in MolmoBot's `olmo` package. MolmoBot requires Python 3.11 and this repo
uses 3.12+, so it is not pip-installed:

- `checkpoint.ensure_molmobot_code()` clones `allenai/MolmoBot` at a pinned commit into
  `~/.cache/molmobot/` and imports it by path.
- Set `MOLMOBOT_PATH` to use your own checkout instead.
- Its runtime dependencies are in this repo's `molmobot-droid` extra.

```python
from examples.vla.molmobot_droid.checkpoint import load_policy, build_instruction

with load_policy() as policy:          # frees the GPU on exit
    policy.reset()                       # per episode
    policy.add_observation(exo, wrist)   # every executed step
    chunk = policy.predict_chunk(state8, build_instruction("pick", "red mug"))
```

`load_policy()` keeps a single cached model, so a second call reuses it. Loading a different
checkpoint frees the first one, and `unload_policy()` frees it explicitly. The observation
history is capped at the 9 frames a query can reach.

## Setup

```bash
uv sync --inexact --extra molmo --extra molmobot-droid
```

On Linux the extra pulls torch from the CUDA 12.8 index. Blackwell GPUs (RTX 50xx) need it.

The first run downloads:

- the checkpoint, into the Hugging Face cache;
- the MolmoBot code;
- each house's assets, into the MolmoSpaces cache.

## Retargeting (`franka_retarget/stretch4_retarget.py`)

```
Franka q (7) ──FK──> grasp_site pose ──fr3_link0 in world──> world
   ──TCP_ALIGN · grasp offset──> where Stretch's grasp_center_link should be
   ──into Stretch's current base_footprint──> stretch4_kinematics IK
   ──> base rotation, lift, arm, wrist yaw/pitch/roll   (+ gripper open/close)
```

- **Where the virtual Franka stands.** `fr3_link0` sits above Stretch's starting
  `base_footprint`, facing the same way. Its height is the training height for the target
  object.
  - The head cameras transplanted onto the Franka (`--exo_camera left|right|center`) sit at the
    same place relative to that floor point.
  - Note that `base_link` in Stretch's model is 0.028 m above `base_footprint`. The footprint is
    the reference.
- **Degrees of freedom.** IK solves 6 DOF for a 6-D pose: base rotation, lift, arm, and wrist
  yaw, pitch and roll. Base translation is not used.
- **Out of reach.**
  - When Stretch cannot reach a target, the closest pose within 5 cm / 20° is used and counted
    as `ik_clamped`.
  - Beyond that it holds still and the step counts as an `ik_failure`.
  - The common case is the Franka's home pose over a tall counter. Stretch's lift tops out
    about 0.12 m short of a downward-pointing Franka TCP there.
- **Back to the Franka.** Stretch's tool pose, with the grasp offset taken back out, goes
  through damped least squares IK on the Franka. It is seeded with the previous solution so
  the redundant elbow doesn't flip.
- **TCP frames.**
  - Franka `grasp_site`: approaches along +z, fingers on y.
  - Stretch `grasp_center_link`: approaches along +x, fingers on y.
  - `TCP_ALIGN` maps one onto the other, which also puts both wrist cameras on the same side.
- **Arm offset.**
  - stretch4_kinematics puts the tool 40.75 mm further out than the URDF for the same arm
    extension.
  - Cause: stretch4_urdf's `merge_arm()` sums `joint.origin[3, :3]` (the homogeneous row)
    where it means `joint.origin[:3, 3]`.
  - `measure_arm_offset()` measures this against the URDF and corrects for it, so it stays
    right once that is fixed.
- **Wrist roll range.**
  - The robot (stretch4_body `range_deg: [-65, 245]`) and the simulator (`FLIP_WRIST_ROLL_RANGE`)
    roll from −65° to 245°. The URDF, and so stretch4_kinematics, has that mirrored.
  - The retargeter overrides the limit in its own copy of the kinematics model.
  - IK is multi-start (current joints, then a few canonical wrist poses), so it finds e.g.
    roll +180° rather than stalling at −65°.
- **Wrist orientation.**
  - IK tries several seeds. Of the exact solutions, it prefers:
    1. an upright gripper (|roll| ≤ 90°), since the wrist servo can't hold an upside-down
       gripper pitched down;
    2. wrist joints at least 0.15 rad inside their limits;
    3. the least joint change.
  - **Flipped grasps.** When Stretch's wrist can't reach the Franka's gripper orientation (its
    roll can't pass −65°), it turns the gripper half a turn about the approach axis
    (`TCP_FLIP`) instead.
    - That is the same grasp for two symmetric fingers.
    - The reverse mapping takes the flip back out.
    - The gripper image is turned 180° so the policy still sees its wrist view upright.
- **Parallel gripper.**
  - IK always runs on stretch4_kinematics' Stretch-gripper model. Its PG4 support is only on an
    unpublished branch.
  - For PG4, the target is shifted by the PG4's TCP relative to the Stretch gripper's. That
    transform comes from the two URDFs: 68.1 mm along the approach axis. Everything before the
    wrist roll is identical for both tools.
- **Where Stretch starts.**
  - Stretch spawns with its lift 90% up (1.08 m). The other joints are at the model's home pose.
  - Both its initial joint position and the model's `home` keyframe, which
    `Stretch4MujocoSimulator.start()` homes to, are set to that lift height.
  - It then moves to the Franka's start pose before the policy runs.
  - At the model's lift of 0, the gripper would often start under a table and homing would
    drive it up into it.

### Flags (Stretch 4 scripts)

| Flag | Default | |
|---|---|---|
| `--slow` | off | Every joint at 20% of its default speed. |
| `--wait-for-arrival / --no-wait-for-arrival` | on | Wait for each action to arrive (`wait_command()`) before the next. Never less than one 66 ms step. |
| `--head-crop droid\|none` | `droid` | Center-crop head images to the 640:368 aspect, then resize to 640x368. |
| `--execute-horizon` | 8 | How many of the 16 predicted actions to keep. |
| `--execute-horizon-do-only-first-n-steps` | 2 | Execute this many of those, then query again. |
| `--use_parallel_gripper` | off | Stretch 4 with the parallel jaw gripper (PG4) instead of the Stretch gripper (SG4). |
| `--grasp-offset-mm x,y,z` | `-9,0,0` with SG4; `4,0,0` with PG4 | Stretch's tool relative to the Franka TCP, in the TCP frame (x = approach, y = between the fingers). The defaults line each tool's fingers up with the Robotiq's. The policy never sees it. |
| `--grasp-offset-deg roll,yaw,pitch` | `0,0,0` | The same, rotations about x, z and y. |
| `--exo_camera droid\|left\|right\|center` | `left` | The policy's exo view. `droid` is the DROID shoulder camera where the virtual Franka's would be (sim only). |
| `--gripper_camera left\|right` | `left` | Which gripper camera stands in for the wrist camera. |
| `--include_franka` | off | Overlay the Franka being driven: see-through, no collisions. Moving in Rerun and the scene camera. In MuJoCo's viewer it is static at home, in geom group 5 (press `5` to show it). It is in group 5 so that none of Stretch's cameras, and so not the policy, can see it. |

## Scripts

All are run as modules from the repo root, e.g. `python -m examples.vla.molmobot_droid.run_franka`.

| Script | |
|---|---|
| `run_franka.py` | Franka DROID in a MolmoSpaces house (`--scene-id procthor-10k/val/0 --object-type mug`). Type instructions; watch in the passive viewer and Rerun. |
| `run_stretch4_sim.py` | The same with Stretch 4, plus all the flags above and `--include_franka`. |
| `run_stretch4_real.py` | The real robot (`--robot_ip`), run continuously: type an instruction + Enter and it works on it until told otherwise. Typing another instruction switches to it; Enter or Space on an empty line stops the robot; `home` raises the lift and returns to the start pose; `quit` or Ctrl+C exits. `--max-steps` (default 0, unlimited) caps an instruction. Needs `send_gripper_and_head_images_with_joint_states.py -r` from stretch4_compliant_gripper running on the robot, with the camera sides matching the flags. `--object-height` sets the virtual Franka's height. |
| `run_benchmark_franka.py` | A MolmoSpaces benchmark on the Franka (default `FrankaPickDroidMiniBench`). `--list` shows what is installed. |
| `run_benchmark_stretch4.py` | The same benchmark on Stretch 4. Its footprint goes where each episode puts the Franka's base. Furniture Stretch would spawn inside is removed and listed (`furniture_removed` in the report). The Franka's pedestal there is a mocap body, which static chairs never push on. |
| `compare_benchmarks.py` | `--franka <run> --stretch4 <run>`: side-by-side videos and `comparison_report.md`. |

A benchmark run writes to `outputs/molmobot_droid/<run name>/`. The run name is the robot
plus the flags, joined with underscores, e.g.
`stretch4_exo-left_grip-left_eh-8_n-2_crop-droid_slow-0_wait-1_pg-0_offmm--9,0,0_offdeg-0,0,0_ghost-0`.
The directory contains:

- `<run name>_ep<#>_<camera>.mp4` for each camera: `exo` and `wrist` (what the policy saw),
  `head_raw` and `gripper_raw` (Stretch), and `scene`;
- `<run name>_ep<#>_grid.mp4`, all of them tiled;
- `report.md` and `results.json`.

Episodes are named by their index in the benchmark, e.g. `ep0003`. That name is printed for
each episode and used in the video file names.

**Ctrl+C** stops a benchmark run cleanly:

- The simulator is closed, and the interrupted episode is not counted.
- `report.md` and `results.json` are written for the episodes finished so far. The report
  notes where the run stopped.
- The exact command to continue is printed, with `--resume ep0003`.

**Resuming** reruns from that episode onward, into the same run directory. Results from before
it are kept, so the final report covers the whole run.

Success uses molmospaces' criteria:

- **Pick:** the object is lifted at least 1 cm and touches only the robot.
- **Pick-and-place:** the object rests on the receptacle, the robot has let go, and the
  receptacle moved at most 15 cm.
- **Open/close and next-to:** not implemented yet.

On Stretch 4, success reads object poses and contacts through `sim.watch_bodies()`,
`pull_body_poses()` and `pull_body_contacts()`.

## Tests

```bash
python -m pytest tests/test_molmobot_droid.py tests/test_body_states.py
python -m pytest examples/vla/molmobot_droid/franka_retarget/test_retargetting.py -s
```

`test_retargetting.py` opens Rerun; set `RERUN_SAVE=<file.rrd>` to record instead.

- It overlays the Franka on Stretch 4, for both grippers, using each robot's own MuJoCo model.
- It commands the Franka through 20 poses and has Stretch follow each one.
- At every pose it asserts the tools match (within 3 mm and 1°) and that mapping Stretch back
  recovers the Franka's TCP.

## Files

| File | |
|---|---|
| `checkpoint.py` | Download, load and release the model; task instructions. |
| `droid.py` | Franka spawning (molmospaces' own `FrankaRobot`), exo cameras, the ghost, Franka FK/IK, the in-process Franka env. |
| `molmospaces/custom_scene.py` | Any house plus target object, for either robot; picking a robot pose; the client-side `SceneMirror` that renders cameras Stretch's simulator doesn't have. |
| `molmospaces/benchmark.py` | Benchmark episodes, success checks, recording and reports. |
| `franka_retarget/stretch4_retarget.py` | Retargeting, its flags, and the Stretch 4 sim env. |
| `franka_retarget/test_retargetting.py` | The 20-pose retargeting test, shown in Rerun. Its `conftest.py` stops the repo root, which is a package named `stretch4_mujoco`, from shadowing the library under pytest. |
| `rollout.py` | The query/execute loop and the interactive prompt. |
| `rerun_scene.py` | The Rerun view: the whole house in 3D from the simulated model, cameras, tool target vs. actual. |
