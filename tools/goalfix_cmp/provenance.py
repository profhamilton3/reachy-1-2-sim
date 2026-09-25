"""Provenance / build-parity gates (plan §3.2-§3.3).

Everything goes through an injected runner, ``runner(argv) -> (rc, stdout,
stderr)`` -- the production CLI supplies a subprocess runner; tests supply
canned output, so no test here ever calls docker.

The pure check functions below (``check_build_parity``,
``check_image_manifest_diff``, ``check_cycle``, ``validate_arm_map``) take
already-PARSED structured input, not raw command text -- ``parse_*``
helpers do that parsing (from ``docker image inspect``/``pip freeze``/
``dpkg -W``/``sha256sum`` output) so the CLI layer is a thin, separately
testable seam and the gate LOGIC itself has a clean, direct-to-construct
input shape for fixtures.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Runner = Callable[[Sequence[str]], Tuple[int, str, str]]

#: The plan's own rep order (§3, ABBA BAAB ABBA), 12 reps, 6 A and 6 B.
ARM_MAP_ORDER: Tuple[str, ...] = tuple("ABBABAABABBA")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

#: The three files a reviewed tooling merge is allowed to change inside the
#: image (plan §10); the manifest-diff gate passes only on exactly this set.
EXPECTED_DIFF_FILES: Tuple[str, ...] = (
    "/opt/fake_reachy_server.py", "/opt/mujoco_remote_backend.py", "/opt/reset_watcher.py")

#: The COPY line (Dockerfile) at or after which the two images' layers may
#: first diverge -- everything up to and including the pip-install layer
#: must be identical (plan §3.2).
_DIVERGENCE_MARKER = "fake_reachy_server.py"


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Parsing (the CLI's own seam -- never exercised by an offline test)
# ---------------------------------------------------------------------------

def parse_pip_freeze(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, _, version = line.partition("==")
            out[name.strip().lower()] = version.strip()
    return out


def parse_dpkg(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and not line.startswith("Desired="):
            name, version = parts[0], parts[1]
            out[name] = version
    return out


def parse_sha256sum_manifest(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, path = line.split(None, 1)
        out[path.lstrip("*")] = digest.lower()
    return out


# ---------------------------------------------------------------------------
# check_build_parity (plan §3.2)
# ---------------------------------------------------------------------------

@dataclass
class ImageBuildInfo:
    base_digest: str
    layers: Tuple[str, ...]          # ordered layer identifiers, base first
    pip_freeze: Dict[str, str]
    dpkg: Dict[str, str]
    python_version: str


def check_build_parity(a: ImageBuildInfo, b: ImageBuildInfo) -> GateResult:
    if a.base_digest != b.base_digest:
        return GateResult(False, f"base digest differs: {a.base_digest} != {b.base_digest}")

    first_diff = None
    for i, (la, lb) in enumerate(zip(a.layers, b.layers)):
        if la != lb:
            first_diff = i
            break
    else:
        if len(a.layers) != len(b.layers):
            first_diff = min(len(a.layers), len(b.layers))

    marker_idx_a = next((i for i, l in enumerate(a.layers) if _DIVERGENCE_MARKER in l), None)
    if first_diff is not None:
        if marker_idx_a is None or first_diff < marker_idx_a:
            return GateResult(
                False, f"layers diverge at index {first_diff}, before the "
                f"{_DIVERGENCE_MARKER} COPY layer (index {marker_idx_a})")

    if a.pip_freeze != b.pip_freeze:
        diff = {k for k in set(a.pip_freeze) | set(b.pip_freeze)
                if a.pip_freeze.get(k) != b.pip_freeze.get(k)}
        return GateResult(False, f"pip freeze differs: {sorted(diff)}")
    if a.dpkg != b.dpkg:
        diff = {k for k in set(a.dpkg) | set(b.dpkg) if a.dpkg.get(k) != b.dpkg.get(k)}
        return GateResult(False, f"dpkg differs: {sorted(diff)}")
    if a.python_version != b.python_version:
        return GateResult(False, f"python version differs: {a.python_version} != {b.python_version}")
    return GateResult(True)


# ---------------------------------------------------------------------------
# check_image_manifest_diff (plan §3.2)
# ---------------------------------------------------------------------------

def check_image_manifest_diff(
    man_a: Dict[str, str], man_b: Dict[str, str], expected: Dict[str, str],
) -> GateResult:
    """``man_a``/``man_b``: {path: sha256}, from ``sha256sum`` manifests of
    each image's filesystem. ``expected``: {path: sha256}, computed by the
    caller from ``git show <sha>:<file>`` (a read-only git call) for each of
    ``EXPECTED_DIFF_FILES``."""
    all_paths = set(man_a) | set(man_b)
    if not all_paths:
        return GateResult(False, "empty or unreadable manifest")
    differing = {p for p in all_paths if man_a.get(p) != man_b.get(p)}
    if differing != set(EXPECTED_DIFF_FILES):
        extra = differing - set(EXPECTED_DIFF_FILES)
        missing = set(EXPECTED_DIFF_FILES) - differing
        return GateResult(
            False, f"differing set != expected: extra={sorted(extra)} missing={sorted(missing)}")
    for path in EXPECTED_DIFF_FILES:
        exp = expected.get(path)
        if exp is None or not _HEX64_RE.match(exp.lower()):
            return GateResult(False, f"no valid expected hash for {path}")
        actual_b = man_b.get(path)
        if actual_b is None or actual_b.lower() != exp.lower():
            return GateResult(False, f"{path}: B's hash {actual_b} != expected {exp}")
    return GateResult(True)


# ---------------------------------------------------------------------------
# arm_map validation (plan §3.3)
# ---------------------------------------------------------------------------

@dataclass
class ArmMapEntry:
    rep: int                 # 1..12
    arm: str                 # "A" | "B"
    image_tag: str
    opt_hashes: Dict[str, str]   # the three /opt/*.py sha256 hashes


def validate_arm_map(entries: Sequence[ArmMapEntry]) -> GateResult:
    if len(entries) != 12:
        return GateResult(False, f"expected 12 reps, got {len(entries)}")
    by_rep = {e.rep: e for e in entries}
    if sorted(by_rep) != list(range(1, 13)):
        return GateResult(False, f"rep numbers must be 1..12 exactly, got {sorted(by_rep)}")
    arms = [by_rep[i].arm for i in range(1, 13)]
    if tuple(arms) != ARM_MAP_ORDER:
        return GateResult(False, f"arm order {''.join(arms)} != required {''.join(ARM_MAP_ORDER)}")
    if arms.count("A") != 6 or arms.count("B") != 6:
        return GateResult(False, "must be exactly 6 A and 6 B reps")
    tags = [e.image_tag for e in entries]
    if len(set(tags)) != len(tags):
        return GateResult(False, "image tags must be distinct across all 12 reps")
    for e in entries:
        for path, h in e.opt_hashes.items():
            if not _HEX64_RE.match(h.lower()):
                return GateResult(False, f"rep {e.rep} {path}: not a 64-char lowercase hex hash")
    return GateResult(True)


# ---------------------------------------------------------------------------
# check_cycle (plan §3.3)
# ---------------------------------------------------------------------------

@dataclass
class ObservedCycle:
    running_image_id: str
    opt_hashes: Dict[str, str]
    supervisor_start_times: Dict[str, float]   # process name -> epoch seconds
    recreate_timestamp: float
    host_git_sha: str
    host_tree_dirty: bool


def check_cycle(
    rep: int, arm_map: Dict[int, ArmMapEntry], observed: ObservedCycle,
) -> GateResult:
    entry = arm_map.get(rep)
    if entry is None:
        return GateResult(False, f"rep {rep} not in arm_map")
    if observed.running_image_id != entry.image_tag:
        return GateResult(
            False, f"running image {observed.running_image_id!r} != "
            f"arm_map[{rep}] {entry.image_tag!r} (version mismatch)")
    if observed.opt_hashes != entry.opt_hashes:
        diff = {k for k in set(observed.opt_hashes) | set(entry.opt_hashes)
                if observed.opt_hashes.get(k) != entry.opt_hashes.get(k)}
        return GateResult(False, f"/opt hash mismatch: {sorted(diff)}")
    for proc, start_ts in observed.supervisor_start_times.items():
        if start_ts < observed.recreate_timestamp:
            return GateResult(False, f"{proc} started before the recreate (stale container)")
    if observed.host_tree_dirty:
        return GateResult(False, "host git tree is dirty")
    return GateResult(True)


def write_versions_doc(host_sha: str, host_dirty: bool, bridge_sha: str) -> Dict[str, object]:
    """The ``versions_<cycle>.json`` payload -- both the host/native SHA and
    the bridge SHA, named explicitly (plan §3.3)."""
    return {"host_native_sha": host_sha, "host_tree_dirty": host_dirty, "bridge_sha": bridge_sha}
