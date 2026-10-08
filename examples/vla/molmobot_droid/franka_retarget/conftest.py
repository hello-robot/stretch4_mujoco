"""
The repo root has an `__init__.py` and the library's name, so pytest imports the modules under
examples/ as `stretch4_mujoco.examples...`, and in doing so registers the repo root itself as
`stretch4_mujoco`, shadowing the library. Load the library in its place before the tests
import it.
"""

import importlib.util
import sys
from pathlib import Path

_shadow = sys.modules.get("stretch4_mujoco")
_library_dir = Path(__file__).resolve().parents[4] / "stretch4_mujoco"
if _shadow is not None and Path(_shadow.__file__).parent != _library_dir:
    _spec = importlib.util.spec_from_file_location(
        "stretch4_mujoco", _library_dir / "__init__.py", submodule_search_locations=[str(_library_dir)]
    )
    _library = importlib.util.module_from_spec(_spec)
    sys.modules["stretch4_mujoco"] = _library
    _spec.loader.exec_module(_library)
    # The examples package stays reachable under the name pytest gave it.
    _library.examples = sys.modules.get("stretch4_mujoco.examples")
