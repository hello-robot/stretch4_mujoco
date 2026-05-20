import os
import sys

# Enable GPU acceleration by default on Linux
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=true")

from .stretch_mujoco_simulator import StretchMujocoSimulator
from .stretch4_mujoco_simulator import Stretch4MujocoSimulator
from .utils import models_path
