"""
Running Franka-authored MolmoSpaces work on Stretch 4.

Everything in this package exists because the benchmark was made with a
different robot. Two layers, and they are independent:

**The episode layer** -- `episode_overrides.py`, `episode_frame.py`. A benchmark
episode freezes its authoring robot into the JSON: a start configuration keyed
by that robot's move groups, cameras mounted on its links, a base pose chosen
for its workspace. The override rewrites those three things for Stretch and
records what it rewrote. This layer is needed by *every* Stretch evaluation,
including the scripted expert, which is why `configs.py` registers it
unconditionally.

**The action layer** -- `franka_arm.py`, `pose_solver.py`, `action_remap.py`,
`vla_policy.py`, `vla_client.py`. A policy trained on the Franka emits seven
joint angles. Stretch has no seven-joint arm, so the numbers are turned back
into the tool pose they parameterise, moved into the world through the frame the
episode layer recorded, and solved for Stretch's own joints. This layer is only
needed if you are actually running such a policy.

If instead you want a policy that speaks Stretch natively, `../finetuning/`
generates Stretch demonstrations and exports them for fine-tuning -- the same
destination by the other road.
"""

from examples.machine_learning.molmospaces.franka_remapping.action_remap import (
    FrankaActionRemapper,
    RemapTelemetry,
)
from examples.machine_learning.molmospaces.franka_remapping.episode_frame import (
    FrankaEpisodeFrame,
)
from examples.machine_learning.molmospaces.franka_remapping.episode_overrides import (
    retarget_base_pose,
    stretch_episode_override,
    stretch_home_init_qpos,
)
from examples.machine_learning.molmospaces.franka_remapping.franka_arm import FrankaArm
from examples.machine_learning.molmospaces.franka_remapping.pose_solver import (
    StretchPoseSolver,
    fit_wrist,
)

__all__ = [
    "FrankaActionRemapper",
    "FrankaArm",
    "FrankaEpisodeFrame",
    "RemapTelemetry",
    "StretchPoseSolver",
    "fit_wrist",
    "retarget_base_pose",
    "stretch_episode_override",
    "stretch_home_init_qpos",
]
