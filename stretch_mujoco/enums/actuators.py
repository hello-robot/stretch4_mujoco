from enum import Enum
from functools import cache

from stretch_mujoco.datamodels.status_stretch_joints import StatusStretchJoints


class Actuators(Enum):
    """
    An enum for the joints defined in the URDF.
    """

    arm = 0
    gripper = 1
    head_pan = 2
    head_tilt = 3
    lift = 4
    wrist_pitch = 5
    wrist_roll = 6
    wrist_yaw = 7

    base_rotate = 8
    base_translate = 9
    base_translate_y = 10
    left_wheel_vel = 11
    right_wheel_vel = 12
    back_wheel_vel = 13

    gripper_right_finger = 14
    gripper_left_finger = 15



    def get_joint_names_in_mjcf(self) -> list[str]:
        """
        An actuator may have multiple joints in the MJCF. Return their names here. Useful for querying positions from Mujoco.

        Opposite mapping to get_actuator_by_joint_names_in_mjcf()
        """
        if self == Actuators.left_wheel_vel:
            return ["joint_left_wheel", "left_wheel_joint"]
        if self == Actuators.right_wheel_vel:
            return ["joint_right_wheel", "right_wheel_joint"]
        if self == Actuators.back_wheel_vel:
            return ["joint_back_wheel", "back_wheel_joint"]
        if self == Actuators.lift:
            return ["joint_lift", "lift_joint"]
        if self == Actuators.arm:
            return ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3",
                    "arm_l4_joint", "arm_l3_joint", "arm_l2_joint", "arm_l1_joint"]
        if self == Actuators.wrist_yaw:
            return ["joint_wrist_yaw", "wrist_yaw_joint"]
        if self == Actuators.wrist_pitch:
            return ["joint_wrist_pitch", "wrist_pitch_joint"]
        if self == Actuators.wrist_roll:
            return ["joint_wrist_roll", "wrist_roll_joint"]
        if self == Actuators.gripper:
            return ["joint_gripper_slide", "gripper_slide_joint"]
        if self == Actuators.gripper_left_finger:
            return ["joint_gripper_finger_left", "gripper_finger_left_joint"]
        if self == Actuators.gripper_right_finger:
            return ["joint_gripper_finger_right", "gripper_finger_right_joint"]
        if self == Actuators.head_pan:
            return ["joint_head_pan", "head_pan_joint"]
        if self == Actuators.head_tilt:
            return ["joint_head_tilt", "head_tilt_joint"]

        raise NotImplementedError(f"Joint names for {self} are not defined.")

    @staticmethod
    @cache
    def get_actuator_by_joint_names_in_mjcf(joint_name: str) -> "Actuators":
        """
        Joint names defined in the mjcf, return their Actuator here.

        Opposite mapping to get_joint_names_in_mjcf()
        """
        if joint_name in ("joint_left_wheel", "left_wheel_joint"):
            return Actuators.left_wheel_vel
        if joint_name in ("joint_right_wheel", "right_wheel_joint"):
            return Actuators.right_wheel_vel
        if joint_name in ("joint_back_wheel", "back_wheel_joint"):
            return Actuators.back_wheel_vel
        if joint_name == 'translate_mobile_base' or joint_name == 'position':
            return Actuators.base_translate
        if joint_name == 'rotate_mobile_base':
            return Actuators.base_rotate

        if joint_name in ("joint_lift", "lift_joint"):
            return Actuators.lift
        if "joint_arm" in joint_name or "arm_l" in joint_name:
            return Actuators.arm
        if joint_name in ("joint_wrist_yaw", "wrist_yaw_joint"):
            return Actuators.wrist_yaw
        if joint_name in ("joint_wrist_pitch", "wrist_pitch_joint"):
            return Actuators.wrist_pitch
        if joint_name in ("joint_wrist_roll", "wrist_roll_joint"):
            return Actuators.wrist_roll
        if joint_name in ("joint_gripper_slide", "gripper_slide_joint", "gripper_aperture"):
            return Actuators.gripper
        if "joint_gripper_finger_left" in joint_name or "gripper_finger_left" in joint_name:
            return Actuators.gripper_left_finger
        if "joint_gripper_finger_right" in joint_name or "gripper_finger_right" in joint_name:
            return Actuators.gripper_right_finger
        if joint_name in ("joint_head_pan", "head_pan_joint"):
            return Actuators.head_pan
        if joint_name in ("joint_head_tilt", "head_tilt_joint"):
            return Actuators.head_tilt

        raise NotImplementedError(f"Actuator for {joint_name} is not defined.")



    def _get_status_attribute(self, is_position: bool, status: StatusStretchJoints) -> float:
        attribute_name = "pos" if is_position else "vel"
        if self == Actuators.arm:
            return getattr(status.arm, attribute_name)
        if self == Actuators.gripper:
            return getattr(status.gripper, attribute_name)
        if self == Actuators.head_pan:
            return getattr(status.head_pan, attribute_name)
        if self == Actuators.head_tilt:
            return getattr(status.head_tilt, attribute_name)
        if self == Actuators.lift:
            return getattr(status.lift, attribute_name)
        if self == Actuators.wrist_pitch:
            return getattr(status.wrist_pitch, attribute_name)
        if self == Actuators.wrist_roll:
            return getattr(status.wrist_roll, attribute_name)
        if self == Actuators.wrist_yaw:
            return getattr(status.wrist_yaw, attribute_name)
        if self == Actuators.gripper_left_finger:
            return getattr(status.gripper_left_finger, attribute_name)
        if self == Actuators.gripper_right_finger:
            return getattr(status.gripper_right_finger, attribute_name)

        raise NotImplementedError(
            f"Get {'Position' if is_position else 'Velocity'} for {self.name} is not implemented."
        )

    def _get_base_status_attribute(
        self, is_position: bool, status: StatusStretchJoints
    ) -> tuple[float, float, float]:
        x = "x" if is_position else "x_vel"
        y = "y" if is_position else "y_vel"
        theta = "theta" if is_position else "theta_vel"
        if self in [Actuators.base_rotate, Actuators.base_translate, Actuators.base_translate_y]:
            return (
                getattr(status.base, x),
                getattr(status.base, y),
                getattr(status.base, theta),
            )

        raise NotImplementedError(
            f"Get {'Position' if is_position else 'Velocity'}  for {self.name} is not implemented."
        )

    def get_position(self, status: StatusStretchJoints) -> float:
        if self in [Actuators.base_rotate, Actuators.base_translate, Actuators.base_translate_y]:
            raise Exception(f"Please use `get_position_relative()` for {self.name}")
        return self._get_status_attribute(True, status)

    def get_position_relative(self, status: StatusStretchJoints) -> tuple[float, float, float]:
        if self not in [Actuators.base_rotate, Actuators.base_translate, Actuators.base_translate_y]:
            raise Exception(f"Please use `get_position()` for {self.name}")
        return self._get_base_status_attribute(True, status)

    def get_velocity(self, status: StatusStretchJoints) -> float:
        if self in [Actuators.base_rotate, Actuators.base_translate, Actuators.base_translate_y]:
            raise Exception(f"Please use `get_velocity_relative()` for {self.name}")
        return self._get_status_attribute(False, status)

    def get_velocity_relative(self, status: StatusStretchJoints) -> tuple[float, float, float]:
        if self not in [Actuators.base_rotate, Actuators.base_translate,Actuators.base_translate_y]:
            raise Exception(f"Please use `get_velocity()` for {self.name}")
        return self._get_base_status_attribute(False, status)
