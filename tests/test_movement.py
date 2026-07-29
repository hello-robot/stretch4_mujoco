import time
import click
import numpy as np
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


def wait_for_sim_time(sim, duration_s):
    t_start = sim.pull_status().time
    while sim.pull_status().time - t_start < duration_s:
        time.sleep(0.01)


def test_joint_and_base_movement():
    click.secho("Starting Stretch4MujocoSimulator for movement verification...", fg="cyan")
    sim = Stretch4MujocoSimulator()
    sim.start(headless=True)

    time.sleep(1.0)

    try:
        # ==========================================
        # 1. Base Tests
        # ==========================================
        click.secho("\n--- Testing Base Movement ---", fg="yellow")

        # 1a. Base translate_by
        start_x = sim.base.status.x
        sim.base.translate_by(0.20)
        time.sleep(0.3)
        assert sim.wait_command(timeout=5.0, position_tolerance=0.001)
        end_x = sim.base.status.x
        disp_x = end_x - start_x
        click.secho(f"Base translate_by(0.20): start={start_x:.4f}, end={end_x:.4f}, disp={disp_x:.4f}m", fg="green")
        assert np.isclose(disp_x, 0.20, atol=0.04), f"Base translate_by failed: disp={disp_x:.4f}m"

        # 1b. Base rotate_by
        start_theta = sim.base.status.theta
        rotate_by = np.radians(90)
        sim.base.rotate_by(rotate_by)
        assert sim.wait_command(timeout=5.0, position_tolerance=0.001)
        end_theta = sim.base.status.theta
        disp_theta = end_theta - start_theta
        click.secho(f"Base rotate_by({rotate_by}): start={start_theta:.4f}, end={end_theta:.4f}, disp={disp_theta:.4f}rad", fg="green")
        assert np.isclose(disp_theta, rotate_by, atol=0.08), f"Base rotate_by failed: disp={disp_theta:.4f}rad"

        # 1c. Base set_velocity
        start_x = sim.base.status.x
        start_y = sim.base.status.y
        sim.base.set_velocity(vx_m=0.10, vy_m=0.0, w_r=0.0)
        wait_for_sim_time(sim, 2.0)
        sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
        time.sleep(0.5)
        end_x = sim.base.status.x
        end_y = sim.base.status.y
        disp_dist = np.hypot(end_x - start_x, end_y - start_y)
        click.secho(f"Base set_velocity(0.10 m/s for 2s): disp={disp_dist:.4f}m (Expected ~0.20m)", fg="green")
        assert np.isclose(disp_dist, 0.20, atol=0.05), f"Base set_velocity failed: disp={disp_dist:.4f}m"


        # ==========================================
        # 2. Arm Tests
        # ==========================================
        click.secho("\n--- Testing Arm Movement ---", fg="yellow")

        # 2a. Arm move_by
        start_arm = sim.arm.status.pos
        sim.arm.move_by(0.10)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_arm = sim.arm.status.pos
        disp_arm = end_arm - start_arm
        click.secho(f"Arm move_by(0.10): start={start_arm:.4f}, end={end_arm:.4f}, disp={disp_arm:.4f}m", fg="green")
        assert np.isclose(disp_arm, 0.10, atol=0.03), f"Arm move_by failed: disp={disp_arm:.4f}m"

        # 2b. Arm set_velocity
        start_arm = sim.arm.status.pos
        v_arm = 0.08
        sim.arm.set_velocity(v_arm)
        wait_for_sim_time(sim, 2.0)
        sim.arm.set_velocity(0.0)
        time.sleep(0.5)
        end_arm = sim.arm.status.pos
        disp_arm = end_arm - start_arm
        click.secho(f"Arm set_velocity(0.08 m/s for 2s): start={start_arm:.4f}, end={end_arm:.4f}, disp={disp_arm:.4f}m (Expected ~0.16m)", fg="green")
        assert np.isclose(disp_arm, 0.16, atol=0.03), f"Arm set_velocity failed: disp={disp_arm:.4f}m"


        # ==========================================
        # 3. Lift Tests
        # ==========================================
        click.secho("\n--- Testing Lift Movement ---", fg="yellow")

        # 3a. Lift move_by
        start_lift = sim.lift.status.pos
        sim.lift.move_by(0.10)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_lift = sim.lift.status.pos
        disp_lift = end_lift - start_lift
        click.secho(f"Lift move_by(0.10): start={start_lift:.4f}, end={end_lift:.4f}, disp={disp_lift:.4f}m", fg="green")
        assert np.isclose(disp_lift, 0.10, atol=0.03), f"Lift move_by failed: disp={disp_lift:.4f}m"

        # 3b. Lift set_velocity
        start_lift = sim.lift.status.pos
        v_lift = 0.08
        sim.lift.set_velocity(v_lift)
        wait_for_sim_time(sim, 1.5)
        sim.lift.set_velocity(0.0)
        time.sleep(0.5)
        end_lift = sim.lift.status.pos
        disp_lift = end_lift - start_lift
        click.secho(f"Lift set_velocity(0.08 m/s for 1.5s): start={start_lift:.4f}, end={end_lift:.4f}, disp={disp_lift:.4f}m (Expected ~0.12m)", fg="green")
        assert np.isclose(disp_lift, 0.12, atol=0.03), f"Lift set_velocity failed: disp={disp_lift:.4f}m"


        # ==========================================
        # 4. Wrist Yaw Tests
        # ==========================================
        click.secho("\n--- Testing Wrist Yaw Movement ---", fg="yellow")

        # 4a. Wrist Yaw move_by
        start_yaw = sim.end_of_arm.wrist_yaw.status.pos
        sim.end_of_arm.wrist_yaw.move_by(0.20)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_yaw = sim.end_of_arm.wrist_yaw.status.pos
        disp_yaw = end_yaw - start_yaw
        click.secho(f"Wrist Yaw move_by(0.20): start={start_yaw:.4f}, end={end_yaw:.4f}, disp={disp_yaw:.4f}rad", fg="green")
        assert np.isclose(disp_yaw, 0.20, atol=0.04), f"Wrist Yaw move_by failed: disp={disp_yaw:.4f}rad"

        # 4b. Wrist Yaw set_velocity
        start_yaw = sim.end_of_arm.wrist_yaw.status.pos
        v_yaw = -0.20
        sim.end_of_arm.wrist_yaw.set_velocity(v_yaw)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.wrist_yaw.set_velocity(0.0)
        time.sleep(0.5)
        end_yaw = sim.end_of_arm.wrist_yaw.status.pos
        disp_yaw = end_yaw - start_yaw
        click.secho(f"Wrist Yaw set_velocity(-0.20 rad/s for 1.0s): start={start_yaw:.4f}, end={end_yaw:.4f}, disp={disp_yaw:.4f}rad (Expected ~-0.20rad)", fg="green")
        assert np.isclose(disp_yaw, -0.20, atol=0.04), f"Wrist Yaw set_velocity failed: disp={disp_yaw:.4f}rad"


        # ==========================================
        # 5. Wrist Pitch Tests
        # ==========================================
        click.secho("\n--- Testing Wrist Pitch Movement ---", fg="yellow")

        # 5a. Wrist Pitch move_by
        start_pitch = sim.end_of_arm.wrist_pitch.status.pos
        sim.end_of_arm.wrist_pitch.move_by(0.20)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_pitch = sim.end_of_arm.wrist_pitch.status.pos
        disp_pitch = end_pitch - start_pitch
        click.secho(f"Wrist Pitch move_by(0.20): start={start_pitch:.4f}, end={end_pitch:.4f}, disp={disp_pitch:.4f}rad", fg="green")
        assert np.isclose(disp_pitch, 0.20, atol=0.04), f"Wrist Pitch move_by failed: disp={disp_pitch:.4f}rad"

        # 5b. Wrist Pitch set_velocity
        start_pitch = sim.end_of_arm.wrist_pitch.status.pos
        v_pitch = -0.20
        sim.end_of_arm.wrist_pitch.set_velocity(v_pitch)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.wrist_pitch.set_velocity(0.0)
        time.sleep(0.5)
        end_pitch = sim.end_of_arm.wrist_pitch.status.pos
        disp_pitch = end_pitch - start_pitch
        click.secho(f"Wrist Pitch set_velocity(-0.20 rad/s for 1.0s): start={start_pitch:.4f}, end={end_pitch:.4f}, disp={disp_pitch:.4f}rad (Expected ~-0.20rad)", fg="green")
        assert np.isclose(disp_pitch, -0.20, atol=0.04), f"Wrist Pitch set_velocity failed: disp={disp_pitch:.4f}rad"


        # ==========================================
        # 6. Wrist Roll Tests
        # ==========================================
        click.secho("\n--- Testing Wrist Roll Movement ---", fg="yellow")

        # 6a. Wrist Roll move_by
        start_roll = sim.end_of_arm.wrist_roll.status.pos
        sim.end_of_arm.wrist_roll.move_by(0.20)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_roll = sim.end_of_arm.wrist_roll.status.pos
        disp_roll = end_roll - start_roll
        click.secho(f"Wrist Roll move_by(0.20): start={start_roll:.4f}, end={end_roll:.4f}, disp={disp_roll:.4f}rad", fg="green")
        assert np.isclose(disp_roll, 0.20, atol=0.04), f"Wrist Roll move_by failed: disp={disp_roll:.4f}rad"

        # 6b. Wrist Roll set_velocity
        start_roll = sim.end_of_arm.wrist_roll.status.pos
        v_roll = 0.20
        sim.end_of_arm.wrist_roll.set_velocity(v_roll)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.wrist_roll.set_velocity(0.0)
        time.sleep(0.5)
        end_roll = sim.end_of_arm.wrist_roll.status.pos
        disp_roll = end_roll - start_roll
        click.secho(f"Wrist Roll set_velocity(0.20 rad/s for 1.0s): start={start_roll:.4f}, end={end_roll:.4f}, disp={disp_roll:.4f}rad (Expected ~0.20rad)", fg="green")
        assert np.isclose(disp_roll, 0.20, atol=0.04), f"Wrist Roll set_velocity failed: disp={disp_roll:.4f}rad"


        # ==========================================
        # 7. Gripper Tests
        # ==========================================
        click.secho("\n--- Testing Gripper Movement ---", fg="yellow")

        # 7a. Gripper move_by (open)
        start_grip = sim.end_of_arm.stretch_gripper.status.pos
        sim.end_of_arm.stretch_gripper.move_by(0.10)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_grip = sim.end_of_arm.stretch_gripper.status.pos
        disp_grip = end_grip - start_grip
        click.secho(f"Gripper move_by(0.10): start={start_grip:.4f}, end={end_grip:.4f}, disp={disp_grip:.4f}", fg="green")
        assert np.isclose(disp_grip, 0.10, atol=0.03), f"Gripper move_by failed: disp={disp_grip:.4f}"

        # 7b. Gripper set_velocity (close)
        start_grip = sim.end_of_arm.stretch_gripper.status.pos
        sim.end_of_arm.stretch_gripper.set_velocity(-0.10)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.stretch_gripper.set_velocity(0.0)
        time.sleep(0.5)
        end_grip = sim.end_of_arm.stretch_gripper.status.pos
        disp_grip = end_grip - start_grip
        click.secho(f"Gripper set_velocity(-0.10 for 1.0s): start={start_grip:.4f}, end={end_grip:.4f}, disp={disp_grip:.4f}", fg="green")
        assert np.isclose(disp_grip, -0.10, atol=0.03), f"Gripper set_velocity failed: disp={disp_grip:.4f}"


        click.secho("\nALL MOVEMENT TESTS PASSED SUCCESSFULLY!", fg="green", bold=True)

    finally:
        sim.stop()


if __name__ == "__main__":
    test_joint_and_base_movement()
