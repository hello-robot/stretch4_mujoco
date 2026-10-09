"""
The repo root has an `__init__.py` and the library's name, so pytest imports the modules under
examples/ as `stretch4_mujoco.examples...`: it puts the repo's parent directory on sys.path and
registers the repo root itself as `stretch4_mujoco`, shadowing the library. On that path, sibling
checkouts (~/repos/stretch4_urdf, stretch4_kinematics, ...) shadow their installed packages too,
as empty namespace packages. Import the real ones while that directory is off sys.path, so they
are cached before pytest puts it back for each test module.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[4]
_parent = _repo.parent

_saved_path = list(sys.path)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _parent]
for _sibling in sorted(p.name for p in _parent.iterdir() if p.is_dir() and p.name.isidentifier()):
    _module = sys.modules.get(_sibling)
    if _module is not None and getattr(_module, "__file__", None) is None:
        del sys.modules[_sibling]  # a namespace package made from the sibling checkout
    if _sibling != _repo.name and importlib.util.find_spec(_sibling) is not None:
        try:
            importlib.import_module(_sibling)
        except Exception:  # not ours to fix; the tests that need it will say so
            sys.modules.pop(_sibling, None)

_shadow = sys.modules.get("stretch4_mujoco")
_library_dir = _repo / "stretch4_mujoco"
if _shadow is not None and Path(_shadow.__file__).parent != _library_dir:
    _spec = importlib.util.spec_from_file_location(
        "stretch4_mujoco", _library_dir / "__init__.py", submodule_search_locations=[str(_library_dir)]
    )
    _library = importlib.util.module_from_spec(_spec)
    sys.modules["stretch4_mujoco"] = _library
    _spec.loader.exec_module(_library)
    # The examples package stays reachable under the name pytest gave it.
    _library.examples = sys.modules.get("stretch4_mujoco.examples")

sys.path[:] = _saved_path
