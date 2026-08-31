"""
Run a trained policy inside a live, interactive Stretch simulation.

`run_benchmarks.py` scores a checkpoint headlessly against fixed episodes. This
runs the same checkpoint in the sim you can actually watch and interfere with:
the MuJoCo viewer from `Stretch4MujocoSimulator`, in a MolmoSpaces house built by
`examples/molmo_environment.py`, with the real omniwheel base rather than the
benchmark's virtual holonomic one.

    # a MolmoSpaces house, drive manually, hit SPACE to hand over to the policy
    python -m examples.machine_learning.molmospaces.live_policy \\
        --checkpoint checkpoints/stretch_manip.pt --dataset procthor-10k --house-index 0

    # the plain room scene, no house download, policy running from the start
    python -m examples.machine_learning.molmospaces.live_policy \\
        --checkpoint checkpoints/stretch_manip.pt --scene default --autostart

SPACE toggles the policy on and off; while it is off you keep the keyboard
teleop, so you can drive the robot somewhere interesting and then hand over.
`--rerun` streams both cameras and the full telemetry to a Rerun viewer, and
`--record` writes the same telemetry to CSV and the camera frames to MP4.

Why the same checkpoint works on a different robot
--------------------------------------------------
The benchmark robot and this one differ in their base -- virtual holonomic
joints there, three omniwheels here -- and in nothing else that the policy sees.
The seven proprioception numbers come out of `StatusStretchJoints` in the same
units the MolmoSpaces move groups report (`arm.pos` is the tendon length, i.e.
total telescoping extension; the finger positions are raw URDF joint angles),
and `networks.py` deliberately encodes the base action as a *relative* step in
the base's own frame. That step becomes a `move_to` on the arm joints and a
holonomic velocity command on the base, see `apply_action()`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import click
import numpy as np

from examples.machine_learning.molmospaces.policies.checkpoint import TrainedPolicy
from examples.machine_learning.molmospaces.policies.networks import STATE_GROUPS
from examples.machine_learning.molmospaces.stretch.config import (
    HEAD_CAMERA,
    HEAD_CAMERA_LEFT,
    HEAD_CAMERA_LEFT_MJCF_NAME,
    HEAD_CAMERA_MJCF_NAME,
    HEAD_CAMERA_RIGHT,
    HEAD_CAMERA_RIGHT_MJCF_NAME,
    WRIST_CAMERA_LEFT,
    WRIST_CAMERA_RIGHT,
    WRIST_LEFT_CAMERA_MJCF_NAME,
    WRIST_RIGHT_CAMERA_MJCF_NAME,
)
from examples.molmo_environment import STRETCH_ROOT_BODY
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

log = logging.getLogger(__name__)


def _camera_for_mjcf_name(mjcf_name: str) -> StretchCameras:
    """The `StretchCameras` member that renders a given MJCF camera.

    Derived rather than hand-written, because the two stacks name the same
    physical camera differently and getting the pairing wrong is invisible:
    feeding a policy trained on `gripper_camera_left_rgb` the *right* gripper
    camera produces plausible-looking images from 2cm away and a policy that
    quietly does worse.

    Several members can share an MJCF camera -- the head centre camera has a
    full-resolution and a low-resolution member -- so prefer the one Stretch 4
    exposes outwardly. `Stretch4MujocoSimulator` swaps in the low-resolution
    variant internally and reports frames back under the outward name.
    """
    matches = [
        camera
        for camera in StretchCameras.all_stretch4()
        if camera.camera_name_in_mjcf == mjcf_name
    ]
    if not matches:
        raise ValueError(
            f"No StretchCameras member renders MJCF camera {mjcf_name!r}; "
            f"available: {sorted(c.camera_name_in_mjcf for c in StretchCameras.all_stretch4())}"
        )
    return matches[0]


# The policy's camera names, mapped to the simulator cameras that render the very
# same MJCF cameras `Stretch4CameraSystem` used during training.
CAMERA_FOR_TRAINED_NAME: dict[str, StretchCameras] = {
    HEAD_CAMERA: _camera_for_mjcf_name(HEAD_CAMERA_MJCF_NAME),
    WRIST_CAMERA_LEFT: _camera_for_mjcf_name(WRIST_LEFT_CAMERA_MJCF_NAME),
    WRIST_CAMERA_RIGHT: _camera_for_mjcf_name(WRIST_RIGHT_CAMERA_MJCF_NAME),
    HEAD_CAMERA_LEFT: _camera_for_mjcf_name(HEAD_CAMERA_LEFT_MJCF_NAME),
    HEAD_CAMERA_RIGHT: _camera_for_mjcf_name(HEAD_CAMERA_RIGHT_MJCF_NAME),
}

# Move groups the policy commands as absolute joint positions, and the simulator
# actuator each maps onto. The base is handled separately, as a velocity.
ACTUATOR_FOR_JOINT_TARGET: dict[tuple[str, int], Actuators] = {
    ("lift", 0): Actuators.lift,
    ("arm", 0): Actuators.arm,
    ("wrist", 0): Actuators.wrist_yaw,
    ("wrist", 1): Actuators.wrist_pitch,
    ("wrist", 2): Actuators.wrist_roll,
    ("gripper", 0): Actuators.gripper_right_finger,
    ("gripper", 1): Actuators.gripper_left_finger,
}

# Caps on what a single policy step is allowed to ask the base to do, in m/s and
# rad/s. A behaviour-cloned base command is a small displacement; dividing a
# spurious large one by the control period would otherwise launch the robot
# across the house before the next step could correct it.
MAX_BASE_SPEED_MPS = 0.6
MAX_BASE_TURN_RADPS = 1.2


def read_state(status) -> np.ndarray:
    """The policy's 7-vector of proprioception, from a `StatusStretchJoints`.

    Deliberately built from `STATE_GROUPS` rather than hard-coded, so it cannot
    drift out of step with what `networks.encode_state` produces on the
    benchmark side.
    """
    per_group = {
        "lift": [status.lift.pos],
        "arm": [status.arm.pos],
        "wrist": [status.wrist_yaw.pos, status.wrist_pitch.pos, status.wrist_roll.pos],
        "gripper": [status.gripper_right_finger.pos, status.gripper_left_finger.pos],
    }
    return np.concatenate(
        [np.asarray(per_group[group][:width], dtype=np.float32) for group, width in STATE_GROUPS]
    )


def apply_action(
    sim: Stretch4MujocoSimulator,
    targets: dict[str, np.ndarray],
    base_xytheta: np.ndarray,
    control_period_s: float,
    joint_limits: dict[Actuators, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Send one decoded action to the simulator. Returns what was commanded.

    The arm joints take absolute targets directly, clipped to `joint_limits` --
    which is `Stretch4MujocoSimulator.pull_joint_limits()`, read off the MJCF's
    own joint ranges by the simulator process. A network's output is only bounded
    by what it was trained on, so nothing else stops a target sailing past the
    end of a joint's travel, and an actuator commanded past its limit holds a
    permanent position error against a joint that cannot move any further. An
    actuator the simulator reports no limit for is passed through untouched
    rather than clipped against a number invented here.

    The base cannot take an absolute target: this robot steers three omniwheels,
    so `TrainedPolicy` hands back an absolute world pose that has to be turned
    into the velocity that would close that gap over one control period, rotated
    into the base's own frame.
    """
    limits = joint_limits or {}
    commanded: dict[str, float] = {}
    for (group, index), actuator in ACTUATOR_FOR_JOINT_TARGET.items():
        value = float(np.asarray(targets[group]).reshape(-1)[index])
        limit = limits.get(actuator)
        if limit is not None:
            clipped = float(np.clip(value, limit[0], limit[1]))
            if clipped != value:
                log.debug(
                    f"[policy] {actuator.name} target {value:+.3f} clipped to {clipped:+.3f} "
                    f"by the model's limits {limit}"
                )
            value = clipped
        sim._move_to(actuator, value)
        commanded[actuator.name] = value

    goal = np.asarray(targets["base"], dtype=float)
    delta_world = goal[:2] - np.asarray(base_xytheta[:2], dtype=float)
    yaw = float(base_xytheta[2])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    forward = (cos_yaw * delta_world[0] + sin_yaw * delta_world[1]) / control_period_s
    left = (-sin_yaw * delta_world[0] + cos_yaw * delta_world[1]) / control_period_s
    turn = float(np.arctan2(np.sin(goal[2] - yaw), np.cos(goal[2] - yaw))) / control_period_s

    forward = float(np.clip(forward, -MAX_BASE_SPEED_MPS, MAX_BASE_SPEED_MPS))
    left = float(np.clip(left, -MAX_BASE_SPEED_MPS, MAX_BASE_SPEED_MPS))
    turn = float(np.clip(turn, -MAX_BASE_TURN_RADPS, MAX_BASE_TURN_RADPS))

    sim.base.set_velocity(forward, left, turn)
    commanded.update({"base_forward": forward, "base_left": left, "base_turn": turn})
    return commanded


class PolicyRunner:
    """Drives the simulator from a checkpoint, on its own thread, toggleable.

    Runs at the checkpoint's control rate rather than as fast as it can: the
    policy was cloned from an expert commanding one action per 66ms step, so its
    action *scale* is tied to that period, and the base velocity conversion in
    `apply_action` divides by it explicitly.
    """

    def __init__(
        self,
        sim: Stretch4MujocoSimulator,
        policy: TrainedPolicy,
        control_hz: float = 15.0,
        telemetry_sink=None,
    ) -> None:
        self.sim = sim
        self.policy = policy
        self.control_period_s = 1.0 / control_hz
        self.telemetry_sink = telemetry_sink

        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.step_count = 0
        # Pulled once: these come from the compiled model, which does not change
        # while the simulator is up, and each pull is an IPC round trip.
        self._joint_limits: dict[Actuators, tuple[float, float]] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def toggle(self) -> bool:
        if self._enabled.is_set():
            self._enabled.clear()
            # Hand back a stationary base rather than leaving the last velocity
            # command running -- the base holds a velocity until told otherwise.
            self.sim.base.set_velocity(0.0, 0.0, 0.0)
            click.secho("[policy] paused", fg="yellow")
        else:
            self.policy.reset()
            self._enabled.set()
            click.secho("[policy] running", fg="green")
        return self._enabled.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._enabled.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set() and self.sim.is_running():
            if not self._enabled.wait(timeout=0.1):
                continue
            started = time.time()
            try:
                self._step()
            except Exception as error:  # noqa: BLE001 - a bad step must not kill the sim
                log.exception(f"[policy] step failed, pausing: {error}")
                self._enabled.clear()
                self.sim.base.set_velocity(0.0, 0.0, 0.0)
            time.sleep(max(0.0, self.control_period_s - (time.time() - started)))

    def _step(self) -> None:
        status = self.sim.pull_status()
        frames = self.sim.pull_camera_data().get_all()
        images = {}
        for trained_name, camera in CAMERA_FOR_TRAINED_NAME.items():
            pixels = frames.get(camera)
            if pixels is None:
                return  # cameras not warmed up yet; try again next step
            images[trained_name] = pixels

        state = read_state(status)
        base_xytheta = np.array([status.base.x, status.base.y, status.base.theta])
        targets = self.policy.act(images, state, base_xytheta)
        if self._joint_limits is None:
            self._joint_limits = self.sim.pull_joint_limits()
        commanded = apply_action(
            self.sim, targets, base_xytheta, self.control_period_s, self._joint_limits
        )

        self.step_count += 1
        if self.telemetry_sink is not None:
            self.telemetry_sink.record(
                step=self.step_count,
                sim_time=status.time,
                state=state,
                base_xytheta=base_xytheta,
                commanded=commanded,
                images=images,
            )


def build_scene_model(scene: str | None, dataset: str, split: str, house_index: int, variant: str):
    """The compiled model to run in, or None for `Stretch4MujocoSimulator`'s default.

    Reuses `examples/molmo_environment.py` wholesale rather than reimplementing
    scene assembly -- that module already knows how to attach Stretch's real
    omniwheel base into a MolmoSpaces house and retarget its contact pairs.
    """
    if scene == "default":
        return None

    from examples.molmo_environment import build_model, resolve_molmospaces_scene

    scene_xml_path = scene or resolve_molmospaces_scene(dataset, split, house_index, variant)
    return build_model(
        scene_xml_path=scene_xml_path,
        x=None,
        y=None,
        z=0.0,
        quat=[1.0, 0.0, 0.0, 0.0],
    )


@click.command()
@click.option(
    "--checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Checkpoint from training/train_bc.py.",
)
@click.option(
    "--scene",
    type=str,
    default=None,
    help="Scene XML path, or 'default' for the bundled room scene (no download). "
    "Omit to load a MolmoSpaces house from --dataset.",
)
@click.option("--dataset", type=str, default="procthor-10k", help="MolmoSpaces house dataset.")
@click.option("--split", type=str, default="val", help="Dataset split.")
@click.option("--house-index", type=int, default=0, help="House index within the split.")
@click.option(
    "--variant", type=str, default="base", help="House variant; 'base' omits the ceiling."
)
@click.option("--control-hz", type=float, default=15.0, help="Policy control rate.")
@click.option(
    "--execute-chunk-steps",
    type=int,
    default=None,
    help="Steps of each predicted chunk to execute before re-querying. Defaults to the "
    "whole chunk, as in evaluation.",
)
@click.option("--autostart", is_flag=True, help="Start with the policy already running.")
@click.option("--rerun", is_flag=True, help="Stream cameras and telemetry to a Rerun viewer.")
@click.option(
    "--record",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write telemetry.csv and per-camera MP4s to.",
)
@click.option("--headless", is_flag=True, help="Run without the MuJoCo viewer.")
def main(
    checkpoint: Path,
    scene: str | None,
    dataset: str,
    split: str,
    house_index: int,
    variant: str,
    control_hz: float,
    execute_chunk_steps: int | None,
    autostart: bool,
    rerun: bool,
    record: Path | None,
    headless: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from examples.machine_learning.molmospaces.telemetry import LiveTelemetry
    from stretch4_mujoco.sim_teleop import KeyboardTeleop, print_keyboard_help

    policy = TrainedPolicy.load(checkpoint, execute_chunk_steps=execute_chunk_steps)
    model = build_scene_model(scene, dataset, split, house_index, variant)

    cameras = list(CAMERA_FOR_TRAINED_NAME.values())
    sim = Stretch4MujocoSimulator(model=model, cameras_to_use=cameras, camera_hz=control_hz * 2)
    # Follow the robot: a policy that drives across a house is impossible to
    # watch under Mujoco's default whole-scene framing.
    sim.start(headless=headless, viewer_track_body=STRETCH_ROOT_BODY)

    telemetry = LiveTelemetry(
        camera_names=list(CAMERA_FOR_TRAINED_NAME),
        use_rerun=rerun,
        output_dir=record,
    )
    runner = PolicyRunner(sim, policy, control_hz=control_hz, telemetry_sink=telemetry)

    teleop = KeyboardTeleop(sim)
    print_keyboard_help()
    click.secho("SPACE toggles the trained policy on and off.", fg="cyan", bold=True)
    _install_space_toggle(teleop, runner)
    teleop.start()
    runner.start()
    if autostart:
        runner.toggle()

    try:
        while sim.is_running():
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        teleop.stop()
        telemetry.close()
        sim.stop()
        click.secho(f"Policy ran for {runner.step_count} steps.", fg="green")


def _install_space_toggle(teleop, runner: PolicyRunner) -> None:
    """Make SPACE toggle the policy, leaving every other teleop key alone.

    Hooks `on_press`/`on_release` rather than `keyboard_control`, for two
    reasons. `KeyboardTeleop` only dispatches `KeyCode` keys to
    `keyboard_control`, and SPACE arrives as `Key.space`; and it re-dispatches
    held keys at 30Hz, which for a toggle would flip the policy on and off
    thirty times a second. `held` gives the edge detection pynput does not,
    since it repeats `on_press` while a key is down.

    Sharing `KeyboardTeleop`'s listener rather than starting a second one is
    deliberate: two global pynput listeners on the same keys race each other.
    Both wrappers must be installed before `teleop.start()`, which is when the
    listener binds to these attributes.
    """
    from pynput import keyboard

    original_on_press, original_on_release = teleop.on_press, teleop.on_release
    held = False

    def on_press(key):
        nonlocal held
        if key == keyboard.Key.space:
            if not held:
                held = True
                runner.toggle()
        return original_on_press(key)

    def on_release(key):
        nonlocal held
        if key == keyboard.Key.space:
            held = False
        return original_on_release(key)

    teleop.on_press = on_press
    teleop.on_release = on_release


if __name__ == "__main__":
    main()
