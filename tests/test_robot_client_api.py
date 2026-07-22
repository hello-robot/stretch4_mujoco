import time
import click
import numpy as np
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


def wait_for_joint(status_fn, target_val, timeout=10.0, tolerance=0.03):
    start_real_time = time.time()
    while time.time() - start_real_time < timeout:
        current_val = status_fn()
        if np.isclose(current_val, target_val, atol=tolerance):
            return True
        time.sleep(0.05)
    return False


def test_subsystems():
    click.secho("Initializing Stretch4MujocoSimulator...", fg="cyan")
    sim = Stretch4MujocoSimulator()

    click.secho("Starting simulator in headless mode...", fg="cyan")
    sim.start(headless=True)

    try:
        # Check that subsystems exist and match expected types/structures
        click.secho("Verifying existence of subsystems...", fg="cyan")
        assert hasattr(sim, "base")
        assert hasattr(sim, "omnibase")
        assert hasattr(sim, "arm")
        assert hasattr(sim, "lift")
        assert hasattr(sim, "end_of_arm")
        # Check that old API methods are removed from public interface
        assert not hasattr(sim, "move_to")
        assert not hasattr(sim, "move_by")
        assert not hasattr(sim, "set_base_velocity")

        assert hasattr(sim.end_of_arm, "wrist_yaw")
        assert hasattr(sim.end_of_arm, "wrist_pitch")
        assert hasattr(sim.end_of_arm, "wrist_roll")
        assert hasattr(sim.end_of_arm, "stretch_gripper")
        assert hasattr(sim.end_of_arm, "parallel_gripper")

        if hasattr(sim, "head"):
            assert hasattr(sim.head, "head_pan")
            assert hasattr(sim.head, "head_tilt")

        click.secho("Subsystems verified successfully!", fg="green")

        # Test status retrieval
        click.secho("Verifying status retrieval...", fg="cyan")
        base_status = sim.base.status
        arm_status = sim.arm.status
        lift_status = sim.lift.status
        wrist_yaw_status = sim.end_of_arm.wrist_yaw.status

        click.secho(f"Base Status: {base_status}", fg="blue")
        click.secho(f"Arm Status: {arm_status}", fg="blue")
        click.secho(f"Lift Status: {lift_status}", fg="blue")
        click.secho(f"Wrist Yaw Status: {wrist_yaw_status}", fg="blue")

        if hasattr(sim, "head"):
            head_pan_status = sim.head.head_pan.status
            click.secho(f"Head Pan Status: {head_pan_status}", fg="blue")


        # Test movement of lift (gravity loaded)
        click.secho("Testing lift.move_by (gravity-loaded joint)...", fg="cyan")
        start_lift = sim.lift.status.pos
        sim.lift.move_by(0.1)
        # Give simulation some time to run and apply force
        time.sleep(2.0)
        end_lift = sim.lift.status.pos
        click.secho(f"Lift moved from {start_lift:.4f} to {end_lift:.4f}", fg="green")
        # Lift must show positive displacement under the commanded upward relative movement
        assert end_lift > start_lift + 0.003, f"Lift did not move up. Ended at {end_lift:.4f}"

        # Test movement of arm (gravity loaded)
        click.secho("Testing arm.move_to...", fg="cyan")
        start_arm = sim.arm.status.pos
        sim.arm.move_to(0.2)
        click.secho("Calling wait_command() to wait for arm motion completion...", fg="cyan")
        wait_success = sim.wait_command(timeout=5.0)
        assert wait_success, "wait_command() timed out or failed"
        end_arm = sim.arm.status.pos
        click.secho(f"Arm moved from {start_arm:.4f} to {end_arm:.4f}", fg="green")
        assert end_arm > start_arm + 0.01, f"Arm did not extend. Ended at {end_arm:.4f}"

        # Test wrist yaw move_by (unloaded joint)
        click.secho("Testing end_of_arm.wrist_yaw.move_by (unloaded joint)...", fg="cyan")
        start_yaw = sim.end_of_arm.wrist_yaw.status.pos
        sim.end_of_arm.wrist_yaw.move_by(0.2)
        reached = wait_for_joint(lambda: sim.end_of_arm.wrist_yaw.status.pos, start_yaw + 0.2, tolerance=0.04)
        end_yaw = sim.end_of_arm.wrist_yaw.status.pos
        click.secho(f"Wrist Yaw moved from {start_yaw:.4f} to {end_yaw:.4f}", fg="green")
        assert reached or end_yaw > start_yaw + 0.1, f"Wrist yaw did not rotate significantly (ended at {end_yaw:.4f})"

        # Test base translate_by
        click.secho("Testing base.translate_by...", fg="cyan")
        start_base_x = sim.base.status.x
        sim.base.translate_by(0.2)
        reached = wait_for_joint(lambda: sim.base.status.x, start_base_x + 0.2, tolerance=0.04)
        end_base_x = sim.base.status.x
        click.secho(f"Base X translated from {start_base_x:.4f} to {end_base_x:.4f}", fg="green")
        assert reached, f"Base X did not translate to target {start_base_x + 0.2:.4f} (ended at {end_base_x:.4f})"

        # Test base rotate_by
        click.secho("Testing base.rotate_by...", fg="cyan")
        start_base_theta = sim.base.status.theta
        sim.base.rotate_by(0.1)
        reached = wait_for_joint(lambda: sim.base.status.theta, start_base_theta + 0.1, tolerance=0.04)
        end_base_theta = sim.base.status.theta
        click.secho(f"Base theta rotated from {start_base_theta:.4f} to {end_base_theta:.4f}", fg="green")
        assert reached, f"Base theta did not rotate to target {start_base_theta + 0.1:.4f} (ended at {end_base_theta:.4f})"

        # Test base set_velocity
        click.secho("Testing base.set_velocity...", fg="cyan")
        sim.base.set_velocity(0.1, 0.0, 0.0)
        time.sleep(0.5)
        sim.base.set_velocity(0.0, 0.0, 0.0)

        click.secho("All subsystem API tests passed successfully!", fg="green", bold=True)

    finally:
        click.secho("Stopping simulator...", fg="cyan")
        sim.stop()


if __name__ == "__main__":
    test_subsystems()
