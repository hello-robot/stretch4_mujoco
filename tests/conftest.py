"""Shared pytest configuration for the stretch4_mujoco tests."""

import functools
import os
import subprocess
import sys

import pytest

from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

# Importing stretch4_mujoco first picks the same MUJOCO_GL backend the tests use.
_GL_PROBE = "import stretch4_mujoco, mujoco; mujoco.GLContext(64, 64).free()"


@functools.cache
def _gl_unavailable_reason() -> str | None:
    """
    Check whether an OpenGL context can be created on this machine.

    Probes in a separate process, so a failed or leftover context cannot affect
    the test process. The result is cached for the session.

    Returns:
        None if a context was created, otherwise the probe's last error line.
    """
    result = subprocess.run(
        [sys.executable, "-c", _GL_PROBE], capture_output=True, text=True, timeout=60
    )
    if result.returncode == 0:
        return None
    lines = result.stderr.strip().splitlines()
    return lines[-1] if lines else f"exit code {result.returncode}"


@pytest.fixture(scope="session", autouse=True)
def generated_robot_model() -> None:
    """
    Generate the Stretch 4 robot MJCF once, before any test runs.

    The default scene includes the generated MJCF, which is gitignored, so a
    fresh checkout has none until a simulator is constructed. Tests that load
    the scene directly would otherwise depend on an earlier test, which may be
    skipped, having generated it. Overwrites any previously generated model.

    Returns:
        None.
    """
    Stretch4MujocoSimulator.get_robot_xml_path()


def pytest_configure(config: pytest.Config) -> None:
    """
    Register the markers the test modules use.

    Args:
        config: The pytest config.

    Returns:
        None.
    """
    config.addinivalue_line(
        "markers",
        "requires_gl: needs an OpenGL context; skipped on CI machines without one",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Skip `requires_gl` tests on CI when no OpenGL context can be created.

    Hosted macOS runners have no GPU, and MuJoCo's CGL backend only accepts an
    accelerated pixel format. Outside CI the tests always run, so a broken GL
    setup on a developer machine still fails loudly.

    Args:
        config: The pytest config.
        items: The collected tests, modified in place.

    Returns:
        None.
    """
    if not os.environ.get("CI"):
        return
    gl_items = [item for item in items if item.get_closest_marker("requires_gl")]
    if not gl_items:
        return
    reason = _gl_unavailable_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=f"No OpenGL context on this CI runner: {reason}")
    for item in gl_items:
        item.add_marker(skip)
