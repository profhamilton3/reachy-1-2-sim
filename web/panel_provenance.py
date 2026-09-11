"""One answer to "which world is this?", shared by everything that asks (#90).

Three things had grown a second copy, and each copy is a way for the panel to
stop matching its own records with no error to show for it:

  * `_robot_model()` — the MJCF whose hash IS `model_sha256`.  Two copies that
    resolved differently would have the recorder and the library computing
    different identities, and nothing would ever match.
  * `_observed_board()` — the obstacle set.  `check_reuse` requires exact set
    equality between the board one side recorded and the board the other side
    asks about, so two implementations must agree byte for byte in behaviour
    or retrieval silently stops working.
  * the identity cache itself.

THE CACHE KEY IS THE WHOLE IDENTITY'S INPUTS, NOT JUST THE SCENE'S
------------------------------------------------------------------
The first version keyed on the scene file alone, which was right when the
identity was mostly the scene.  It is not any more: `build_simulator_identity`
also hashes the robot model and shells out to git for the commit and whether
the tree is dirty.  The panel is a long-lived container process, so a cache
keyed on the scene alone will answer `working_tree_dirty=False` for the rest of
the process's life after a single edit — and `ReusePolicy.require_clean_tree`,
the guard whose entire job is refusing exactly that, would pass on provenance
from hours ago.

So the key carries the model file's stat as well, and the git half — which no
stat can watch — is bounded by time instead.  `IDENTITY_TTL_S` is the longest
the panel may go on believing a git answer.  It is not free (two subprocesses),
which is why it is seconds rather than milliseconds, and it is not forever,
which is the whole point.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
import time
from typing import Any, List, Optional, Tuple

#: How long a git-derived identity may be believed, in seconds.  A plan made
#: inside this window after an edit can still claim a clean tree; a plan made
#: outside it cannot.  Bounded staleness, stated rather than unbounded and
#: silent.
IDENTITY_TTL_S = 30.0


def ensure_paths() -> None:
    """Put `src` on the path: a checkout, or /opt in the image."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "..", "src"), "/opt/src"):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def robot_model() -> str:
    """The robot MJCF the simulator compiles, or "" if it is not here.

    `model_sha256` means this file, and without it the identity is degenerate —
    the gate refuses to answer about a world it cannot tell apart from any
    other, and `query_compatible_trials` raises rather than matching
    everything.  It lives in the repo, so the panel can hash it even though it
    never loads it: the panel talks to the simulator over the SDK bridge and
    has no model of its own.
    """
    for candidate in (repo_root() / "native_mujoco" / "model" / "reachy_1_2.xml",
                      pathlib.Path("/opt/native_mujoco/model/reachy_1_2.xml")):
        if candidate.is_file():
            return str(candidate)
    return ""


def relative(path: str) -> str:
    """A repo-relative path, or the bare filename if it is outside the repo.

    Absolute paths from a developer's machine are provenance nobody else can
    use and a small leak of where this ran.  The scene file's identity is its
    hash, which `build_simulator_identity` records; the path is a label.
    """
    if not path:
        return ""
    try:
        return str(pathlib.Path(path).resolve().relative_to(repo_root()))
    except (ValueError, OSError):
        return os.path.basename(path)


def observed_board(scene) -> Optional[List[str]]:
    """Which objects were on the board, or None if that was not observed.

    None is a real answer and the conservative one.  `on_board is True` and not
    merely truthy, because it is None until a snapshot has been applied and
    unknown is not absent — the same test `SceneView.on_board_tagged` makes for
    the same reason.
    """
    if scene is None or getattr(scene, "error", ""):
        return None
    objects = getattr(scene, "objects", None) or {}
    if any(o.on_board is None for o in objects.values()):
        return None
    return sorted(oid for oid, o in objects.items() if o.on_board is True)


def _stat_key(path: str) -> Tuple[str, int, int]:
    try:
        st = os.stat(path) if path else None
    except OSError:
        st = None
    return (path, st.st_mtime_ns if st else 0, st.st_size if st else 0)


class IdentityCache:
    """`build_simulator_identity`, cached on everything it actually reads.

    One instance per caller.  `for_scene()` returns the identity with the
    requested scene revision stamped on it — the revision is the cheap half and
    the half that moves, and rebuilding two git subprocesses for one string
    would be paying a lot for it.
    """

    def __init__(self, scene_file: str = "", *, backend_name: str = "sdk_bridge",
                 label_paths: bool = False, ttl_s: float = IDENTITY_TTL_S) -> None:
        self._scene_file = scene_file
        self._backend_name = backend_name
        self._label_paths = label_paths
        self._ttl_s = ttl_s
        self._identity: Any = None
        self._key: Optional[Tuple[Any, ...]] = None
        self._built_at = 0.0

    def for_scene(self, scene_revision: str = ""):
        ensure_paths()
        from reachy_ai.experience.identity import build_simulator_identity

        model = robot_model()
        absolute = os.path.abspath(self._scene_file) if self._scene_file else ""
        key = (_stat_key(absolute), _stat_key(model))
        now = time.monotonic()
        expired = (now - self._built_at) > self._ttl_s
        if self._identity is None or self._key != key or expired:
            identity = build_simulator_identity(
                model_path=model, scene_path=absolute,
                backend_name=self._backend_name)
            if self._label_paths:
                identity = dataclasses.replace(
                    identity,
                    model_source_path=relative(model),
                    scene_source_path=relative(self._scene_file))
            self._identity = identity
            self._key = key
            self._built_at = now
        return dataclasses.replace(self._identity, scene_revision=scene_revision)
