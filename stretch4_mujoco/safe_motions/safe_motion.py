from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer


class SafeMotion:
    """Base class for all safe motions.

    The sim counterpart of `stretch4_body.behavior.safe_motions.SafeMotion`. One
    difference in where it sits in the loop: on the robot a safe motion runs
    *before* `push_command()` and overrides the subsystem state that is about to
    be pushed, because the motor mode it sets then latches in firmware. The sim
    has no firmware to latch anything, so a safe motion runs *after*
    `push_command()` and overrides `mjdata.ctrl` directly -- it still gets the
    last word before the physics step, and re-asserting itself every cycle is
    what stands in for the latch.

    Subclasses implement `step()`, returning whether they are overriding motion.
    """

    def __init__(self, name: str, mujoco_server: "MujocoServer"):
        self.name = name
        self.mujoco_server = mujoco_server
        self.params: dict[str, Any] = mujoco_server.robot_settings.get(name, {})
        self.status: dict[str, Any] = {}

    @property
    def dt(self) -> float:
        """One control interval, the period a safe motion is stepped at."""
        return 1.0 / self.mujoco_server.control_rate_hz

    def step(self) -> bool:
        """Run one control cycle.

        Returns:
            Whether this safe motion is currently overriding commanded motion.
        """
        raise NotImplementedError
