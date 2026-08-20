"""R12-801: SimulatorIdentity generation and research context guard.

build_simulator_identity() introspects the running Python environment, host
platform, and (optionally) the compiled MuJoCo model to produce an exact,
versioned provenance record that uniquely identifies the physics world.

assert_research_context() refuses to allow a learning/search session to start
when physical hardware could be addressed, or when the identity is degenerate
(missing hashes that would allow wrong-world experience to be reused).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import platform
import subprocess
import sys
from typing import Optional

from .models import SimulatorIdentity

_SCENE_SCHEMA_VERSION = "1.0"


def _scene_compiler_version() -> str:
    try:
        import sys as _sys
        _native = pathlib.Path(__file__).resolve().parents[4] / "native_mujoco"
        if str(_native) not in _sys.path:
            _sys.path.insert(0, str(_native))
        from scene_compiler import COMPILER_VERSION  # type: ignore[import]
        return str(COMPILER_VERSION)
    except (ImportError, AttributeError):
        return "unknown"


def _repo_root() -> Optional[pathlib.Path]:
    here = pathlib.Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists():
            return p
    return None


def _run_git(*args: str) -> str:
    root = _repo_root()
    try:
        r = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(root) if root else None,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _sha256_path(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_simulator_identity(
    *,
    model_path: str = "",
    scene_path: str = "",
    scene_revision: str = "",
    backend_name: str = "native_mujoco",
    compiled_xml: Optional[str] = None,
    calibration_profile_id: str = "default",
    sensor_effect_profile_id: str = "default",
    physics_profile_id: str = "default",
    protocol_version: int = 1,
) -> SimulatorIdentity:
    """Build a SimulatorIdentity from the current environment.

    Hashes model and scene source files on disk.  If compiled_xml is provided
    (the MJCF string produced by scene_compiler before mujoco.MjModel loading),
    it is hashed as compiled_model_sha256 — this captures compiler behaviour
    without storing the non-portable binary model.
    """
    try:
        import mujoco as _mj
        mujoco_version = getattr(_mj, "__version__", "unknown")
    except ImportError:
        mujoco_version = "not_installed"

    git_sha = _run_git("rev-parse", "HEAD")
    dirty_output = _run_git("status", "--porcelain")
    working_tree_dirty = bool(dirty_output)

    return SimulatorIdentity(
        repository_git_sha=git_sha,
        working_tree_dirty=working_tree_dirty,
        model_source_path=model_path,
        model_sha256=_sha256_path(model_path),
        compiled_model_sha256=_sha256_text(compiled_xml) if compiled_xml else "",
        scene_source_path=scene_path,
        scene_sha256=_sha256_path(scene_path),
        scene_revision=scene_revision,
        scene_schema_version=_SCENE_SCHEMA_VERSION,
        scene_compiler_version=_scene_compiler_version(),
        physics_profile_id=physics_profile_id,
        protocol_version=protocol_version,
        mujoco_version=mujoco_version,
        python_version=sys.version,
        host_os=platform.system(),
        host_arch=platform.machine(),
        backend_name=backend_name,
        calibration_profile_id=calibration_profile_id,
        sensor_effect_profile_id=sensor_effect_profile_id,
    )


# ---------------------------------------------------------------------------
# Research context guard
# ---------------------------------------------------------------------------

class ResearchContextError(RuntimeError):
    """Raised when the environment is unsafe for simulator-only research."""


def assert_research_context(identity: Optional[SimulatorIdentity] = None) -> None:
    """Raise ResearchContextError if this environment must not run research episodes.

    Rules enforced (EPIC-8 Section 5.1):
    1. Physical motion must not be enabled (REACHY_ENABLE_MOTION != "true").
       Learning/search episodes must run in simulation only.
    2. If an identity is provided, it must have a non-empty model_sha256 so
       that identity-based filtering is meaningful.
    3. If an identity is provided with a scene_source_path, scene_sha256 must
       also be set — a missing hash with a known path is almost certainly a
       bug that would let wrong-scene experience pass compatibility checks.
    """
    if os.environ.get("REACHY_ENABLE_MOTION", "false").lower() == "true":
        raise ResearchContextError(
            "Physical motion is enabled (REACHY_ENABLE_MOTION=true). "
            "Learning/search episodes must run in simulation only. "
            "Unset REACHY_ENABLE_MOTION before starting a study."
        )

    if identity is None:
        return

    if not identity.model_sha256:
        raise ResearchContextError(
            "SimulatorIdentity.model_sha256 is empty. "
            "Build the identity from a real model file before starting a study; "
            "an empty hash would allow any model's experience to be reused."
        )

    if identity.scene_source_path and not identity.scene_sha256:
        raise ResearchContextError(
            f"SimulatorIdentity has scene_source_path='{identity.scene_source_path}' "
            "but scene_sha256 is empty. "
            "Hash the scene file when building the identity so compatibility "
            "checks can distinguish different scene configurations."
        )
