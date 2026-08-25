# Franka → Stretch 4 remapping

Everything here exists because the MolmoSpaces benchmarks were made with a
different robot. Two independent layers:

| layer | files | needed by |
| ----- | ----- | --------- |
| **episode** | `episode_overrides.py`, `episode_frame.py` | every Stretch evaluation, including the scripted expert |
| **action** | `franka_arm.py`, `pose_solver.py`, `action_remap.py`, `vla_policy.py`, `vla_client.py` | only a policy that emits Franka joint angles |

The episode layer is registered unconditionally by `../configs.py`. The action
layer is only touched by `--policy vla`.

## The episode layer

`stretch_episode_override()` is MolmoSpaces' `robot_override_fn`, called once per
episode with the episode spec and the experiment config both mutable. It rewrites
the cameras, the start configuration and the base pose, and leaves the scene, the
object poses, the instruction and the success criteria alone — so the task being
scored is still the benchmark's. `../README.md` covers the details.

It also **records** what it is about to overwrite. The base pose and the start
configuration are what makes the episode runnable *and* the frame a Franka-space
policy's numbers live in, so this is the last moment they exist.
`episode_frame.py` holds them from there to `policy.reset()`.

## The action layer

### Why it goes through the tool pose

A joint vector is a robot-specific *parameterisation* of a tool pose. Stretch has
no seven-joint arm, so the retarget undoes the parameterisation and works in the
quantity that means the same thing on both robots:

```
VLA action (7 joints + gripper)
  -> FrankaArm.forward()          tool pose in the authoring arm's frame
  -> FrankaEpisodeFrame           the same pose in the world
  -> StretchPoseSolver.solve()    lift / arm / wrist / base targets
```

and inbound, the same path backwards through `FrankaArm.inverse()`. Both
directions use the *same* `franka_droid/model.xml` MolmoSpaces authored the
episodes with, so the link lengths, the Robotiq mounting and the
`gripper/grasp_site` tool point are the benchmark's own rather than a re-derived
DH table.

The world is the hinge, and there is no hand-fitted workspace calibration
anywhere in the benchmark path. The episode's author placed the Franka's shoulder
where *its* workspace covered the target; compose a VLA-space tool pose with that
recorded pose and you get an absolute world pose, and the object is at the same
world pose for either robot.

### Why Stretch's wrist makes the pose solve exact

Measured on the compiled MJCF and asserted in
`tests/test_franka_remapping.py`, to machine precision:

```
R_tool_world = Rz(base_yaw + wrist_yaw) @ Ry(wrist_pitch) @ Rx(wrist_roll)
```

Stretch's three wrist joints are a textbook ZYX Euler triple. So a requested
orientation is *read off* rather than solved for: pitch and roll are forced, and
only the sum of base yaw and wrist yaw is constrained. That leaves one free
parameter inside an exactly-matched orientation — the **yaw split**, `base_yaw +=
s, wrist_yaw -= s`, which swings the arm around the base axis while the tool keeps
pointing exactly where it was asked to. `lift + arm + split` is then square and
well-conditioned for position, and unlike a wrist-yaw solve it does not go
singular reaching straight down: its lever arm is base-axis-to-wrist-axis rather
than wrist-axis-to-tool.

Three sources of representation freedom get searched before the solve
(`fit_wrist`), because a ZYX triple is not unique and Stretch's limits are
lopsided (yaw and pitch run −1.135…4.276 rad, roll runs −4.276…1.135):

- the canonical decomposition;
- the mirror branch `(A + π, π − p, r + π)` — the same orientation with the
  gripper folded back over the arm, which is legal on this wrist and costs about
  0.3m of forward reach, so it is scored down outside `NATURAL_PITCH_RANGE`;
- near-vertical approaches, where the decomposition degenerates and azimuth
  trades against roll one-for-one. That is the *common* case — a top-down grasp —
  and it is a gift rather than a hazard: the azimuth can go wherever the wrist
  can reach and be paid for in roll, which for a symmetric two-finger gripper
  pointing straight down is very nearly free.

### Measured fidelity

Franka joint targets interpolated from each pick episode's own `init_qpos` to a
top-down grasp at its object, retargeted step by step:

| quantity | median | p90 |
| -------- | ------ | --- |
| grasp-pose position error | 3.7mm | 24mm |
| grasp-pose orientation error | 0.008 rad | 0.018 rad |
| base rotation used over an episode | 0.13 rad | 0.16 rad |
| virtual-arm IK error (proprioception) | 0.3mm | 1.1mm |

`solver_mode` picks between three DOF sets. `exact` (the yaw split) is the
default and the best measured. `free_azimuth` spends the wrist yaw on reaching
instead and mis-orients the gripper by more than a radian through the approach.
`translating` lets the base drive, which tracks the approach better but ends the
grasp worse, because the base wanders off the standoff the episode chose.

One counter-intuitive knob: `max_base_rotation` is a **guard, not a tuning
parameter**. A binding cap distorts the descent direction rather than merely
truncating it — capping at 0.5 rad takes the median grasp-pose error from 5mm to
257mm — while the rotation a good solve actually wants is 0.13 rad. Raise it
rather than lower it.

### What it does not fix

All three are reported per episode in `RemapTelemetry`, not hidden:

- **Minimum reach.** Stretch's tool cannot come closer than ~0.39m to its base
  axis; a Franka's home posture is at ~0.23m. About a third of the intermediate
  poses of an approach land in that hole and get the nearest reachable pose. The
  grasp pose, which is what scores, almost always does not.
- **The base turns.** Exact orientation costs the yaw split, which swings the head
  camera.
- **The cameras.** Stretch's head camera is fed to both of the model's exocentric
  inputs. A DROID-trained model expects two fixed, off-robot shoulder views and
  gets one egocentric view that moves with the robot, twice. Kinematics cannot
  touch this; `../finetuning/` is the fix.

### The handover

Stretch spawns stowed — necessarily, or its gripper materialises inside whatever
the object is sitting on. No Franka joint vector maps to that configuration, so
the first `handover_steps` (30, two seconds at 15Hz) drive to the authoring arm's
own start pose *without* querying the model, ending early on arrival. The model's
first observation is then one it could plausibly have been trained on, and the
unstow never enters its action history.

## Frames, and the one trap

`frame_source` decides which frame a VLA's numbers are read in:

- `episode` (default) — the authoring arm's recorded pose. Right for a
  **pretrained** Franka checkpoint.
- `mast` — a virtual Franka bolted to Stretch's own base. Right for a checkpoint
  **fine-tuned** on `../finetuning/lerobot_export.py --action-space franka`,
  because that is the frame the export encodes in.

Getting it wrong is silent: the two frames differ by the standoff between the two
shoulders, so every action is offset by it and the arm reaches consistently short
or long. `../finetuning/finetune.py` prints the setting a given dataset needs.

## Running it

```bash
# serve a checkpoint on the openpi-style websocket protocol, then:
python -m examples.machine_learning.molmospaces.run_benchmarks \
    --policy vla --vla-host localhost --vla-port 8000 --benchmark pick
```

Concretely, this is the path for `allenai/MolmoBot-DROID` (served with MolmoBot's
`launch_scripts/serve_molmo.py --action-type joint_pos`), for pi0.5/pi0 DROID
checkpoints, and for DreamZero — anything trained on MolmoBot's `franka_joint`
action preset (`arm(7), gripper(1)`) or DROID's equivalent.

It is **not** the path for a MolmoBot checkpoint fine-tuned on Stretch's own move
groups. That one emits Stretch's ten numbers directly and wants
`--policy molmobot`; see `../policies/molmobot_policy.py`. `include_endpoint` in
`remote_config` is the switch for MolmoBot's server, which routes without an
`endpoint` field (openpi-style) unlike DreamZero's.

Everything about *how* the remap behaves is a field on `FrankaVLAPolicyConfig`:
`action_space` (`joint_position` / `joint_velocity`), `gripper_mapping`
(`normalized` / `aperture`), `frame_source`, `solver_mode`, `chunk_size`,
`grasping_type`, `handover_steps`.
