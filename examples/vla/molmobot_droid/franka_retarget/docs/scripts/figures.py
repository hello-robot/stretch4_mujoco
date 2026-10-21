"""
What the figure scripts here share: the scene they are rendered in, where they are saved, and how
their panels are labelled. Each script writes the PNG of its own name to franka_retarget/docs/.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from examples.vla.molmobot_droid.droid import RobotPose
from examples.vla.molmobot_droid.molmospaces.custom_scene import CustomScene, load_custom_scene, suggest_robot_pose

DOCS = Path(__file__).resolve().parents[1]

SCENE_ID = "procthor-10k/val/0"
OBJECT_TYPE, IN_PLACE_OF = "tennis_ball", "toaster"
"""A green tennis ball on a kitchen counter, a yellow canister and a white kettle beside it: colours
to tell the views apart by."""

PANEL_SCALE = 0.75
"""Panels are saved at this fraction of the 640x368 the policy sees."""


def load_scene() -> tuple[CustomScene, RobotPose]:
    """The scene, and the robot pose `run_franka.py` / `run_stretch4_sim.py` would pick in it."""
    scene = load_custom_scene(SCENE_ID, OBJECT_TYPE, in_place_of=IN_PLACE_OF)
    return scene, suggest_robot_pose(scene)


def label(image: np.ndarray, text: str) -> np.ndarray:
    """`image` (RGB) at PANEL_SCALE, under a white band with `text` in it."""
    image = cv2.resize(image, None, fx=PANEL_SCALE, fy=PANEL_SCALE, interpolation=cv2.INTER_AREA)
    band = np.full((30, image.shape[1], 3), 255, np.uint8)
    cv2.putText(band, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([band, image])


def save(rows: list[list[np.ndarray]], name: str) -> Path:
    """Rows of labelled panels as docs/<name>.png."""
    path = DOCS / f"{name}.png"
    cv2.imwrite(str(path), np.vstack([np.hstack(row) for row in rows])[..., ::-1])
    print(f"Wrote {path}")
    return path
