"""
MolmoBot-DROID driving a Franka DROID in any MolmoSpaces house while Stretch 4 follows it blind
(`franka_retarget/follow_franka.py`): the policy sees only the Franka's cameras, Stretch 4 is
invisible to them, and only Stretch 4 touches anything. For seeing how well Stretch 4 does what
the Franka does: whether it picks the object up (molmospaces' pick criterion, judged on
Stretch), how far its tool is from where the Franka's action put it, how often it cannot follow.

Type an instruction and press Enter, and it runs until Stretch has picked the object up (unless
--no-stop-on-success) or --max-steps, as `run_stretch4_sim.py` runs (see
`franka_retarget/console.py`'s KEYS_HELP): Enter or Space stops it, `home` or `reset` puts both
robots back at the start pose (the objects stay where they are), `quit` exits.

The start pose can be changed between instructions with `run_stretch4_sim.py`'s commands
(`StretchStartPoseEditor`), but on the Franka, which Stretch follows: `jog` moves its joints and
pedestal with the keys, `pose` prints them, `set q1 ... q7` (rad) puts both robots there,
`height <m>` sets the pedestal and `offset x y z [roll yaw pitch]` the grasp offset (mm, deg).
`stats` prints how closely Stretch has followed since the start pose. `object` says what the
target is and what can replace it; `object <name> [index]` puts that in its place (a THOR
category or prefab, or tennis_ball; see `custom_scene.find_asset()`): the scene is rebuilt and
the simulator restarted, with both robots back where they were.

Watch it in stretch4_mujoco's viewer (the Franka is in geom group 5: press 5 to see it, static at
home) and in Rerun: the policy's views (the Franka's exo and wrist cameras), Stretch's head
fisheyes and gripper cameras as a policy would be shown them, the scene camera with the Franka
see-through over Stretch, and the house in 3D with the Franka's tool target and Stretch's tool.
`fisheye_tool` is the fisheye on the tool's side, switching as `run_stretch4_sim.py` switches
its exo view. With --rectify, the fisheyes are shown redrawn as the DROID exo camera would see
them (`FisheyeRectifier`) instead of only cropped to its aspect; the policy still sees the Franka's.

Usage:
    python -m examples.vla.molmobot_droid.run_follow_franka --scene-id procthor-10k/val/0 --object-type mug
    python -m examples.vla.molmobot_droid.run_follow_franka --use_parallel_gripper --no-wait-for-arrival
    python -m examples.vla.molmobot_droid.run_follow_franka --custom_franka_start_pose
    python -m examples.vla.molmobot_droid.run_follow_franka --rectify
"""

from __future__ import annotations

import click
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, build_instruction, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.console import KEYS_HELP, Console, continuous_session
from examples.vla.molmobot_droid.franka_retarget.follow_franka import STRETCH_CAMERAS, TOOL_FISHEYE, FollowFrankaEnv
from examples.vla.molmobot_droid.franka_retarget.start_pose_editor import StretchStartPoseEditor
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    CUSTOM_START_QPOS,
    custom_start_link0_height,
    custom_start_option,
    franka_yaw_for_start,
    params_from_kwargs,
    rectify_option,
    retarget_options,
    spawn_stretch4,
)
from examples.vla.molmobot_droid.molmospaces.benchmark import PickSuccess, allow_ctrl_c
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    STRETCH_ROOT_BODY,
    SceneMirror,
    SwapTarget,
    object_command,
    run_with_swaps,
    load_custom_scene,
    load_custom_scene_stretch4,
    scene_camera_pose,
    virtual_franka_link0_height,
)
from examples.vla.molmobot_droid.rollout import StepInfo
from examples.vla.molmobot_droid.run_franka import resolve_pose, scene_options
from examples.vla.molmobot_droid.run_stretch4_sim import go_to_carried_pose, with_nearby_free_bodies

RERUN_CAMERAS = ["exo", "wrist", "fisheye_left", "fisheye_right", TOOL_FISHEYE, "gripper_left", "gripper_right", "scene"]
"""The policy's (the Franka's) views, Stretch's as a policy would be shown them (TOOL_FISHEYE: the
fisheye on the tool's side, as `run_stretch4_sim.py` picks its exo view), the scene camera."""


def spawn_follow_sim(stretch_scene, params, judged_bodies: list[str]):
    """
    Stretch 4's simulator for following the Franka (not started): rendering every camera that is
    recorded, publishing the poses of the bodies success is judged on and of the free objects
    near the robot, which the Franka's views are rendered with (`SceneMirror` moves only those).
    """
    sim = spawn_stretch4(stretch_scene, params, cameras=list(STRETCH_CAMERAS.values()))
    sim.watch_bodies(with_nearby_free_bodies(stretch_scene, stretch_scene.watched_bodies + judged_bodies))
    return sim


class FollowRerunLogger:
    """Logs a `FollowFrankaEnv` step: the mirror scene, the cameras, the Franka's tool target vs. Stretch's tool."""

    def __init__(self, env: FollowFrankaEnv):
        self.env = env
        stretch_scene = env.stretch_scene
        moving = [STRETCH_ROOT_BODY, stretch_scene.scene.object_name, stretch_scene.franka.base_name]
        self.scene = rerun_scene.RerunScene(env.mirror.model, env.mirror.data, moving)

    def log(self, step: int, observation, metrics: dict[str, float] | None = None) -> None:
        """Step `step`, as of `observation`: Stretch's tool from the state it read, not now."""
        env = self.env
        sim_time, footprint, joints = env.observed_state
        rerun_scene.set_step(step, sim_time)
        self.scene.log(env.mirror.data)
        images = {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, "scene": env.render_scene()}
        images.update({name: observation.extra_cameras[name] for name in RERUN_CAMERAS if name in observation.extra_cameras})
        rerun_scene.log_cameras(images)
        actual = env.retargeter.stretch_tool_world(footprint, joints)
        rerun_scene.log_tool_poses(env.retargeter.last_target_tool_world, actual)
        if metrics:
            rerun_scene.log_metrics(metrics)


def stats_text(env: FollowFrankaEnv) -> str:
    stats = env.stats()
    tool = (
        f"Stretch's tool {stats['tool_error_mean_m'] * 1000:.0f} mm from the Franka's on average, "
        f"{stats['tool_error_max_m'] * 1000:.0f} mm at most; "
        if "tool_error_mean_m" in stats
        else ""
    )
    return (
        f"Since the start pose: {tool}{stats['ik_clamped']:.0f} actions clamped "
        f"({stats['ik_twisted']:.0f} twisted), {stats['ik_failures']:.0f} Stretch could not follow."
    )


@click.command()
@scene_options
@retarget_options(camera_options=False)
@rectify_option
@custom_start_option
@click.option("--stop-on-success/--no-stop-on-success", default=True, show_default=True,
              help="End an instruction once Stretch 4 has picked the object up.")
def main(scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, checkpoint, rerun,
         custom_franka_start_pose, stop_on_success, **kwargs):
    params = params_from_kwargs(kwargs)
    scene = load_custom_scene(scene_id, object_type, object_index, in_place_of=in_place_of)
    pose = resolve_pose(scene, robot_pose)
    console = None

    def say(message: str, color: str | None = None) -> None:
        if console is not None:
            console.event(message, color)
        else:
            click.secho(message, fg=color)

    def run_scene(scene, carry: dict) -> None:
        """Both robots in `scene` until `quit`, or until `object` swaps the target (SwapTarget,
        its `carry` the Franka's joints and pedestal, which the next scene starts from)."""
        nonlocal console
        click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
        link0_height = virtual_franka_link0_height(scene)
        start_q7 = np.array(FRANKA_HOME_QPOS)
        if custom_franka_start_pose:
            link0_height = custom_start_link0_height(params, scene.floor_z)
            start_q7 = np.array(CUSTOM_START_QPOS)
            click.secho(f"Franka fr3_link0 at z={link0_height:.3f} (molmospaces would put it at "
                        f"{virtual_franka_link0_height(scene):.3f})", fg="green")
        if "pedestal" in carry:
            link0_height = scene.floor_z + carry["pedestal"]
        # Stretch spawns turned the way the start pose would turn its base, so it does not have to.
        franka_yaw = franka_yaw_for_start(params, link0_height, start_q7, scene.floor_z)
        stretch_scene = load_custom_scene_stretch4(
            scene, pose, include_franka=True, tool_name=params.tool_name, link0_height=link0_height,
            stretch_yaw=-franka_yaw,
        )
        if stretch_scene.removed_bodies:
            say(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", "yellow")
        if stretch_scene.backed_off:
            say(f"Moved {stretch_scene.backed_off:.2f} m back from the target so Stretch 4 clears what it "
                f"rests on: --robot-pose {stretch_scene.robot_pose}", "yellow")

        sim = spawn_follow_sim(stretch_scene, params, [scene.object_name])
        env = None
        try:
            sim.start(viewer_look_at_body=STRETCH_ROOT_BODY)
            allow_ctrl_c()  # start() takes Ctrl+C over to stop only the simulator, leaving this waiting on the console
            env = FollowFrankaEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
            env.start_q7 = start_q7
            env.on_exo_side_change = lambda side, facing, tool_side: say(
                f"◐ {TOOL_FISHEYE} from the {side} head camera (tool facing {facing}, on the {tool_side})"
            )
            logger = None
            step_count = 0
            if rerun:
                eye = scene_camera_pose(stretch_scene.robot_pose, scene.object_pos)
                if carry:  # a swapped-in target: a recording of its own, with its own scene geometry
                    rerun_scene.new_recording(f"{scene.scene_id}: {scene.object_category}", eye=eye)
                else:
                    rerun_scene.init_rerun(
                        "MolmoBot-DROID Stretch 4 following the Franka", RERUN_CAMERAS,
                        main_cameras=["exo", "wrist", "gripper_left"], eye=eye,
                    )
                rerun_scene.set_step(0)
                env.observe()  # poses the mirror
                logger = FollowRerunLogger(env)

            def log_pose() -> None:
                """A Rerun step outside a rollout; rollout steps and these share the `step` timeline."""
                nonlocal step_count
                if logger is not None:
                    step_count += 1
                    logger.log(step_count, env.observe())

            def go_to_start() -> None:
                try:
                    env.reset()
                finally:
                    log_pose()

            say("Moving Stretch 4 to the Franka's start pose...", "yellow")
            go_to_start()
            go_to_carried_pose(env, carry, say)
            log_pose()
            policy = load_policy(checkpoint)
            judge: PickSuccess | None = None
            succeeded = False

            def on_step(info: StepInfo) -> bool:
                nonlocal step_count, judge, succeeded
                if not sim.is_running():
                    raise SystemExit("The simulator stopped")
                if info.step == 1:  # a new instruction: lifted from where the object is now
                    judge = PickSuccess(scene.object_name, float(env.body_pose(scene.object_name)[0][2]))
                    succeeded = False
                success, metrics = judge(env)
                if success and not succeeded:
                    say(f"✓ Stretch 4 picked up the {scene.object_category} at step {info.step} "
                        f"(lifted {metrics['lift_m'] * 100:.1f} cm). {stats_text(env)}", "green")
                succeeded = succeeded or success
                step_count += 1
                if logger is not None:
                    error = env.tool_error_m()
                    logger.log(step_count, info.observation, {
                        "gripper_command": info.action[7], "query": info.query, "lift_m": metrics["lift_m"],
                        "ik_failures": env.retargeter.ik_failures, "ik_clamped": env.retargeter.ik_clamped,
                        **({"tool_error_m": error} if error is not None else {}),
                    })
                return stop_on_success and success

            def extra_commands(text: str) -> bool:
                if text.strip() == "stats":
                    console.event(stats_text(env))
                    return True
                return object_command(text, scene, console.event)

            if console is None:
                click.echo(
                    f"\n{KEYS_HELP}\n"
                    "  'stats' + Enter                 how closely Stretch has followed since the start pose\n"
                    "  'object [name [index]]' + Enter swap another object in for the target\n"
                )
                console = Console()
            say(f"The target's instruction: {build_instruction('pick', scene.object_category)}", "green")
            editor = StretchStartPoseEditor(
                env, go_to=env.move_to_franka_pose, set_pedestal_height=stretch_scene.set_franka_pedestal_height,
                keys=console.keys, say=console.event, on_move=log_pose, extra_commands=extra_commands,
            )
            try:
                continuous_session(env, policy, params, max_steps, on_step, console, editor, reset=go_to_start)
            except SwapTarget as swap:
                swap.carry.update(q7=editor.franka_pose(), pedestal=stretch_scene.franka.pedestal_height)
                raise
        finally:
            if env is not None:
                env.close()
            sim.stop()

    try:
        run_with_swaps(scene, run_scene, say)
    except KeyboardInterrupt:
        pass
    finally:
        if console is not None:
            console.close()
        unload_policy()

if __name__ == "__main__":
    main()
