import importlib
from typing import TYPE_CHECKING

import click

from stretch4_mujoco.safe_motions.motion_overrides import MotionOverrides
from stretch4_mujoco.safe_motions.safe_motion import SafeMotion

if TYPE_CHECKING:
    from stretch4_mujoco.mujoco_server import MujocoServer


class SafeMotionManager:
    """Runs the enabled safe motions once per control cycle.

    Mirrors `stretch4_body.behavior.safe_motions.SafeMotionManager`, including
    reading which plug-ins to load from the robot settings, so the sim and the
    robot enable the same set. A model whose settings list no controllers -- the
    Stretch 3 model -- gets an empty manager that costs nothing to step.
    """

    def __init__(self, mujoco_server: "MujocoServer"):
        self.mujoco_server = mujoco_server
        self.controllers: dict[str, SafeMotion] = {}
        self.status: dict[str, list[str]] = {"safe_motions_triggered": []}
        # Shared, so two safe motions holding the same joint do not release it
        # out from under each other. See `MotionOverrides`.
        self.overrides = MotionOverrides(mujoco_server)

        settings = mujoco_server.robot_settings
        for name in settings.get("safe_motion_manager", {}).get("controllers", []):
            params = settings.get(name)
            if not params or not params.get("enabled", 1):
                continue
            module = importlib.import_module(params["py_module_name"])
            controller = getattr(module, params["py_class_name"])(
                mujoco_server, self.overrides
            )
            self.controllers[name] = controller
            click.secho(f"Started SafeMotion {name}", fg="green")

    def step(self) -> None:
        self.status["safe_motions_triggered"] = [
            name
            for name, controller in self.controllers.items()
            if controller.step()
        ]
