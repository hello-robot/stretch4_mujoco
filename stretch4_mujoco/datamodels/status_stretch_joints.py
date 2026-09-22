import copy
from dataclasses import asdict, dataclass, field
from stretch4_mujoco.utils import dataclass_from_dict

@dataclass
class PositionVelocity:
    pos: float
    vel: float
    effort: float = 0.0

    @staticmethod
    def default():
        return PositionVelocity(0, 0, 0.0)

@dataclass
class BaseStatus:
    x:float
    y:float
    theta:float
    x_vel:float
    theta_vel:float
    active_translate_x: bool = False
    active_translate_y: bool = False
    active_rotate: bool = False

    @staticmethod
    def default():
        return BaseStatus(0, 0, 0, 0, 0, False, False, False)

@dataclass
class StatusStretchJoints:
    time: float
    fps:float
    sim_to_real_time_ratio_msg: str
    base:BaseStatus
    lift: PositionVelocity
    arm: PositionVelocity
    head_pan: PositionVelocity
    head_tilt: PositionVelocity
    wrist_yaw: PositionVelocity
    wrist_pitch: PositionVelocity
    wrist_roll: PositionVelocity
    gripper: PositionVelocity
    gripper_left_finger: PositionVelocity
    gripper_right_finger: PositionVelocity
    is_self_colliding: bool = False
    actuators_in_motion: list[str] = field(default_factory=list)
    """MJCF actuator names whose motion profile is still ramping.

    A rate-limited joint accelerates from rest, so for the first tick of a move
    its position has barely changed -- which is indistinguishable, to a caller
    watching for position stability, from having finished. This is the server
    saying "there is still a commanded move in flight", and is what
    `wait_command()` and `wait_while_is_moving()` check alongside position.
    Empty for actuators with no recorded limits, which have no profile.
    """

    def __getitem__(self, name:str):
        """For backward compatibility: allows access with the square brackets []"""
        return getattr(self, name)

    def to_dict(self):
        return asdict(self)

    def copy(self):
        return StatusStretchJoints.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data:dict)-> "StatusStretchJoints":
        return dataclass_from_dict(StatusStretchJoints, dict_data) #type: ignore


    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusStretchJoints(
            0,
            0,
            "",
            BaseStatus.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            PositionVelocity.default(),
            False,
        )
