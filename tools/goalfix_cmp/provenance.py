"""Provenance / build-parity gates (plan §3.2-§3.3; assignment T6).

Everything goes through an injected runner, ``runner(argv) -> (rc, stdout,
stderr)`` -- the production CLI supplies a subprocess runner
(``default_runner``); tests supply canned output, so no test here ever
calls docker.

The pure check functions below (``check_build_parity``,
``check_image_manifest_diff``, ``check_cycle``, ``validate_arm_map``) take
already-PARSED structured input, not raw command text -- ``parse_*``
helpers do that parsing (from ``docker image inspect``/``docker
history``/``pip freeze``/``dpkg -W``/``sha256sum`` output) so the CLI layer
is a thin, separately testable seam and the gate LOGIC itself has a clean,
direct-to-construct input shape for fixtures.

T6 (review §3.6/M4) rewrites every one of these against the readiness
review's demonstrated failures:
* ``ArmMapEntry`` gained ``image_id``/``bridge_sha``; ``validate_arm_map``
  now requires every A entry (and every B entry) to share one tag/id/sha/
  hash-set, with A and B differing, instead of rejecting the plan's real
  2-tag map for "non-distinct" per-rep tags.
* ``check_cycle`` compares the running image ID (not a tag), requires the
  host SHA to equal the tooling merge commit M with a clean tree, and
  requires non-empty hashes and a non-empty required-supervisor-program
  list -- all silently accepted as passing before.
* ``check_image_manifest_diff`` now verifies A's three hashes too, not
  only B's.
* ``check_build_parity`` takes the layer-divergence index from
  ``docker history --no-trunc`` CREATED-BY lines, never by searching a
  file name inside an opaque layer digest (which "fails always" on real
  digests, per the review).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Runner = Callable[[Sequence[str]], Tuple[int, str, str]]

#: The plan's own rep order (§3, ABBA BAAB ABBA), 12 reps, 6 A and 6 B.
ARM_MAP_ORDER: Tuple[str, ...] = tuple("ABBABAABABBA")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

#: The three files a reviewed tooling merge is allowed to change inside the
#: image (plan §10); the manifest-diff gate passes only on exactly this set,
#: and every arm_map entry's opt_hashes must carry exactly these keys.
EXPECTED_DIFF_FILES: Tuple[str, ...] = (
    "/opt/fake_reachy_server.py", "/opt/mujoco_remote_backend.py", "/opt/reset_watcher.py")

#: The COPY-step text (docker history's CREATED-BY column, not a layer
#: digest) at or after which the two images' layers may first diverge --
#: everything up to and including the pip-install layer must be identical
#: (plan §3.2).
_DIVERGENCE_MARKER = "fake_reachy_server.py"


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


class ProvenanceCliError(RuntimeError):
    """A runner call the CLI needed did not produce usable output (rc != 0,
    empty stdout, or truncated/malformed structured output). Callers map
    this to rc=3 (evidence incomplete), never a pass."""


# ---------------------------------------------------------------------------
# Parsing (the CLI's own seam -- never exercised against real docker by a
# test; tests parse canned TEXT, which is not a docker call)
# ---------------------------------------------------------------------------

def parse_pip_freeze(text: str) -> Dict[str, str]:
    """Every non-comment, non-blank line, including ``@`` (direct
    reference) and ``-e`` (editable) forms, which a plain ``name==version``
    split cannot represent -- those compare by their whole line instead."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e") or "@" in line or "==" not in line:
            out[line] = line
            continue
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


def parse_docker_history(text: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``docker history --no-trunc --format '{{.ID}}\\t{{.CreatedBy}}'
    <image>`` output, oldest-first (docker prints newest-first) -- (layer
    IDs, created-by text), same order, same length. A malformed line (not
    exactly one tab) raises ``ProvenanceCliError``."""
    rows: List[Tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ProvenanceCliError(f"docker history: malformed line {line!r}")
        rows.append((parts[0].strip(), parts[1].strip()))
    rows.reverse()  # oldest (base) first, matching ImageBuildInfo.layers order
    if not rows:
        return (), ()
    ids, created_by = zip(*rows)
    return ids, created_by


# ---------------------------------------------------------------------------
# check_build_parity (plan §3.2; T6)
# ---------------------------------------------------------------------------

@dataclass
class ImageBuildInfo:
    base_digest: str
    layers: Tuple[str, ...]          # ordered layer digests, base first
    created_by: Tuple[str, ...]      # docker history's CREATED-BY text, same order/length
    pip_freeze: Dict[str, str]
    dpkg: Dict[str, str]
    python_version: str


def check_build_parity(a: ImageBuildInfo, b: ImageBuildInfo) -> GateResult:
    if len(a.layers) != len(a.created_by) or len(b.layers) != len(b.created_by):
        return GateResult(False, "layers and created_by must be the same length")
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

    # T6: the divergence marker is searched in the CREATED-BY text (the
    # COPY command docker recorded), never in a layer digest -- a digest
    # never contains a file name, which is why this "failed always" on
    # real layer digests before.
    marker_idx_a = next(
        (i for i, cb in enumerate(a.created_by) if _DIVERGENCE_MARKER in cb), None)
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
# check_image_manifest_diff (plan §3.2; T6: verify BOTH A's and B's hashes)
# ---------------------------------------------------------------------------

def check_image_manifest_diff(
    man_a: Dict[str, str], man_b: Dict[str, str],
    expected_a: Dict[str, str], expected_b: Dict[str, str],
) -> GateResult:
    """``man_a``/``man_b``: {path: sha256}, from ``sha256sum`` manifests of
    each image's filesystem. ``expected_a``/``expected_b``: {path: sha256},
    computed by the caller from ``git show <sha>:<file>`` (a read-only git
    call) for each of ``EXPECTED_DIFF_FILES``, A's SHA and B's SHA
    respectively -- T6 checks A's hashes too; the old code never did."""
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
        exp_a, exp_b = expected_a.get(path), expected_b.get(path)
        if exp_a is None or not _HEX64_RE.match(exp_a.lower()):
            return GateResult(False, f"no valid expected A hash for {path}")
        if exp_b is None or not _HEX64_RE.match(exp_b.lower()):
            return GateResult(False, f"no valid expected B hash for {path}")
        actual_a, actual_b = man_a.get(path), man_b.get(path)
        if actual_a is None or actual_a.lower() != exp_a.lower():
            return GateResult(False, f"{path}: A's hash {actual_a} != expected {exp_a}")
        if actual_b is None or actual_b.lower() != exp_b.lower():
            return GateResult(False, f"{path}: B's hash {actual_b} != expected {exp_b}")
    return GateResult(True)


# ---------------------------------------------------------------------------
# arm_map validation (plan §3.3; T6)
# ---------------------------------------------------------------------------

@dataclass
class ArmMapEntry:
    rep: int                 # 1..12
    arm: str                 # "A" | "B"
    image_tag: str
    image_id: str
    bridge_sha: str
    opt_hashes: Dict[str, str]   # the three /opt/*.py sha256 hashes, all three keys required


def validate_arm_map(
    entries: Sequence[ArmMapEntry], *, expected_bridge_sha: Optional[Dict[str, str]] = None,
) -> GateResult:
    """T6: every A entry shares ONE tag/id/sha/hash-set; every B entry
    shares ANOTHER; A and B differ. ``expected_bridge_sha`` (optional --
    e.g. ``{"A": "8c0dad2...", "B": "67730a1..."}``, passed in by the
    caller, never hard-coded here) additionally pins each arm's bridge SHA
    to a specific value."""
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

    for e in entries:
        if set(e.opt_hashes) != set(EXPECTED_DIFF_FILES):
            return GateResult(
                False, f"rep {e.rep}: opt_hashes must have exactly the three required keys, "
                f"got {sorted(e.opt_hashes)}")
        for path, h in e.opt_hashes.items():
            if not _HEX64_RE.match(h.lower()):
                return GateResult(False, f"rep {e.rep} {path}: not a 64-char lowercase hex hash")

    identity = {}
    for arm_label in ("A", "B"):
        group = [e for e in entries if e.arm == arm_label]
        first = group[0]
        first_key = (first.image_tag, first.image_id, first.bridge_sha,
                     tuple(sorted(first.opt_hashes.items())))
        for e in group[1:]:
            key = (e.image_tag, e.image_id, e.bridge_sha, tuple(sorted(e.opt_hashes.items())))
            if key != first_key:
                return GateResult(
                    False, f"arm {arm_label}: rep {e.rep} does not share arm {arm_label}'s "
                    f"tag/id/sha/hash-set (rep {first.rep})")
        identity[arm_label] = first_key
        if expected_bridge_sha and arm_label in expected_bridge_sha:
            if first.bridge_sha != expected_bridge_sha[arm_label]:
                return GateResult(
                    False, f"arm {arm_label}: bridge_sha {first.bridge_sha!r} != "
                    f"expected {expected_bridge_sha[arm_label]!r}")

    if identity["A"] == identity["B"]:
        return GateResult(False, "A and B must not share the same tag/id/sha/hash-set")
    return GateResult(True)


# ---------------------------------------------------------------------------
# check_cycle (plan §3.3; T6)
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
    rep: int, arm_map: Dict[int, ArmMapEntry], observed: ObservedCycle, *,
    expected_host_sha: str, required_supervisor_programs: Sequence[str],
) -> GateResult:
    """T6: compares the running IMAGE ID (never a tag, and never against
    ``entry.image_tag`` as the old code did), requires non-empty hashes and
    a non-empty ``required_supervisor_programs`` (an empty list is rc 3,
    per the assignment -- it can never confirm the container recreated),
    and requires the HOST SHA to equal ``expected_host_sha`` (M, the
    tooling merge commit, passed on the command line) with a clean tree --
    none of which the old code read at all."""
    entry = arm_map.get(rep)
    if entry is None:
        return GateResult(False, f"rep {rep} not in arm_map")
    if observed.running_image_id != entry.image_id:
        return GateResult(
            False, f"running image ID {observed.running_image_id!r} != "
            f"arm_map[{rep}].image_id {entry.image_id!r} (version mismatch)")

    if not entry.opt_hashes or any(not v for v in entry.opt_hashes.values()):
        return GateResult(False, f"rep {rep}: arm_map opt_hashes are empty")
    if not observed.opt_hashes or any(not v for v in observed.opt_hashes.values()):
        return GateResult(False, "observed /opt hashes are empty")
    if observed.opt_hashes != entry.opt_hashes:
        diff = {k for k in set(observed.opt_hashes) | set(entry.opt_hashes)
                if observed.opt_hashes.get(k) != entry.opt_hashes.get(k)}
        return GateResult(False, f"/opt hash mismatch: {sorted(diff)}")

    if not required_supervisor_programs:
        return GateResult(
            False, "required_supervisor_programs is empty -- cannot confirm the "
            "container actually recreated")
    missing = [p for p in required_supervisor_programs
               if p not in observed.supervisor_start_times]
    if missing:
        return GateResult(False, f"no supervisor start time for {sorted(missing)}")
    stale = [p for p in required_supervisor_programs
             if observed.supervisor_start_times[p] <= observed.recreate_timestamp]
    if stale:
        return GateResult(False, f"{sorted(stale)} started before/at the recreate (stale container)")

    if observed.host_tree_dirty:
        return GateResult(False, "host git tree is dirty")
    if observed.host_git_sha != expected_host_sha:
        return GateResult(
            False, f"host SHA {observed.host_git_sha!r} != expected {expected_host_sha!r}")
    return GateResult(True)


def write_versions_doc(
    *, host_native_kernel_sha: str, host_tree_dirty: bool, bridge_arm: str, bridge_sha: str,
    running_image_id: str, opt_hashes: Dict[str, str],
    supervisor_start_times: Dict[str, float],
) -> Dict[str, object]:
    """The ``versions_<cycle>.json`` payload -- every plan §3.3 field, named
    explicitly. T6: the old version omitted the image ID, the three
    hashes, ``bridge_arm`` and the supervisor start times entirely."""
    return {
        "host_native_kernel_sha": host_native_kernel_sha,
        "host_tree_dirty": host_tree_dirty,
        "bridge_arm": bridge_arm,
        "bridge_sha": bridge_sha,
        "running_image_id": running_image_id,
        "opt_hashes": dict(opt_hashes),
        "supervisor_start_times": dict(supervisor_start_times),
    }


# ---------------------------------------------------------------------------
# Production runner + CLI (never exercised against real docker by a test)
# ---------------------------------------------------------------------------

def default_runner(argv: Sequence[str]) -> Tuple[int, str, str]:
    import subprocess
    proc = subprocess.run(list(argv), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _require(rc: int, out: str, what: str) -> str:
    if rc != 0:
        raise ProvenanceCliError(f"{what} exited {rc}")
    if not out.strip():
        raise ProvenanceCliError(f"{what} produced empty stdout")
    return out


def inspect_image(runner: Runner, tag: str) -> ImageBuildInfo:
    """Every runner call ``check_build_parity`` needs for one image tag, via
    a throwaway container for pip/dpkg/python (plan §3.2)."""
    rc, out, err = runner(["docker", "image", "inspect", tag])
    out = _require(rc, out, f"docker image inspect {tag}")
    try:
        data = json.loads(out)[0]
        base_digest = data["RootFS"]["Layers"][0]
        layers = tuple(data["RootFS"]["Layers"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise ProvenanceCliError(f"docker image inspect {tag}: malformed JSON: {exc}") from exc

    rc, out, err = runner(
        ["docker", "history", "--no-trunc", "--format", "{{.ID}}\t{{.CreatedBy}}", tag])
    _, created_by = parse_docker_history(_require(rc, out, f"docker history {tag}"))

    rc, out, err = runner(["docker", "run", "--rm", tag, "pip", "freeze"])
    pip_freeze = parse_pip_freeze(_require(rc, out, f"pip freeze ({tag})"))

    rc, out, err = runner(["docker", "run", "--rm", tag, "dpkg-query", "-W"])
    dpkg = parse_dpkg(_require(rc, out, f"dpkg-query -W ({tag})"))

    rc, out, err = runner(["docker", "run", "--rm", tag, "python3", "--version"])
    python_version = _require(rc, out, f"python3 --version ({tag})").strip()

    return ImageBuildInfo(base_digest, layers, created_by, pip_freeze, dpkg, python_version)


def compute_expected_hash(runner: Runner, sha: str, path: str) -> str:
    """``git show <sha>:<file> | sha256sum``, computed by this CLI through
    the runner (a read-only git call) -- never by hand."""
    rc, out, err = runner(["git", "show", f"{sha}:{path.lstrip('/')}"])
    if rc != 0 or not out:
        raise ProvenanceCliError(f"git show {sha}:{path} failed (rc={rc}): {err}")
    return hashlib.sha256(out.encode()).hexdigest()


def read_image_manifest(runner: Runner, container_name: str) -> Dict[str, str]:
    """``sha256sum`` over ``/opt`` (excluding ``__pycache__``),
    ``/etc/supervisor/conf.d`` and ``/entrypoint.sh``, read with
    ``docker compose exec`` (plan §3.2)."""
    rc, out, err = runner([
        "docker", "compose", "exec", "-T", container_name, "sh", "-c",
        "find /opt /etc/supervisor/conf.d /entrypoint.sh -type f "
        "-not -path '*/__pycache__/*' -exec sha256sum {} +",
    ])
    return parse_sha256sum_manifest(_require(rc, out, "sha256sum manifest"))


def _cli(argv: Optional[Sequence[str]] = None, runner: Optional[Runner] = None) -> int:
    import argparse

    from tools.goalfix_cmp._io import RC_INCONCLUSIVE, RC_OK, RC_STOP, write_result

    runner = runner or default_runner
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-parity")
    bp.add_argument("--image-a", required=True)
    bp.add_argument("--image-b", required=True)
    bp.add_argument("--out", required=True)

    mn = sub.add_parser("manifest")
    mn.add_argument("--container-a", required=True)
    mn.add_argument("--container-b", required=True)
    mn.add_argument("--sha-a", required=True)
    mn.add_argument("--sha-b", required=True)
    mn.add_argument("--out", required=True)

    am = sub.add_parser("arm-map")
    am.add_argument("--arm-map", required=True, help="JSON file: list of ArmMapEntry fields")
    am.add_argument("--expected-bridge-sha-a")
    am.add_argument("--expected-bridge-sha-b")
    am.add_argument("--out", required=True)

    cy = sub.add_parser("cycle")
    cy.add_argument("--rep", type=int, required=True)
    cy.add_argument("--arm-map", required=True)
    cy.add_argument("--container", required=True)
    cy.add_argument("--recreate-timestamp", type=float, required=True)
    cy.add_argument("--required-supervisor-programs", nargs="+", required=True)
    cy.add_argument("--expected-host-sha", required=True)
    cy.add_argument("--out", required=True)
    cy.add_argument("--out-versions", required=True)

    args = p.parse_args(argv)

    try:
        if args.cmd == "build-parity":
            a = inspect_image(runner, args.image_a)
            b = inspect_image(runner, args.image_b)
            r = check_build_parity(a, b)
        elif args.cmd == "manifest":
            man_a = read_image_manifest(runner, args.container_a)
            man_b = read_image_manifest(runner, args.container_b)
            expected_a = {f: compute_expected_hash(runner, args.sha_a, f)
                          for f in EXPECTED_DIFF_FILES}
            expected_b = {f: compute_expected_hash(runner, args.sha_b, f)
                          for f in EXPECTED_DIFF_FILES}
            r = check_image_manifest_diff(man_a, man_b, expected_a, expected_b)
        elif args.cmd == "arm-map":
            raw = json.loads(open(args.arm_map).read())
            entries = [ArmMapEntry(**e) for e in raw]
            expected_bridge_sha = {}
            if args.expected_bridge_sha_a:
                expected_bridge_sha["A"] = args.expected_bridge_sha_a
            if args.expected_bridge_sha_b:
                expected_bridge_sha["B"] = args.expected_bridge_sha_b
            r = validate_arm_map(entries, expected_bridge_sha=expected_bridge_sha or None)
        elif args.cmd == "cycle":
            raw = json.loads(open(args.arm_map).read())
            arm_map = {e["rep"]: ArmMapEntry(**e) for e in raw}
            entry = arm_map.get(args.rep)
            manifest = read_image_manifest(runner, args.container)
            rc_id, out_id, _ = runner(["docker", "inspect", "-f", "{{.Image}}", args.container])
            running_image_id = _require(rc_id, out_id, "docker inspect (image id)").strip()
            rc_sv, out_sv, _ = runner(["docker", "compose", "exec", "-T", args.container,
                                        "supervisorctl", "status"])
            supervisor_start_times = _parse_supervisorctl_status(
                _require(rc_sv, out_sv, "supervisorctl status"))
            rc_sha, out_sha, _ = runner(["git", "rev-parse", "HEAD"])
            host_git_sha = _require(rc_sha, out_sha, "git rev-parse HEAD").strip()
            rc_dirty, out_dirty, _ = runner(["git", "status", "--porcelain"])
            host_tree_dirty = bool(_require0(rc_dirty, out_dirty))
            observed = ObservedCycle(
                running_image_id=running_image_id,
                opt_hashes={f: manifest.get(f) for f in EXPECTED_DIFF_FILES},
                supervisor_start_times=supervisor_start_times,
                recreate_timestamp=args.recreate_timestamp,
                host_git_sha=host_git_sha, host_tree_dirty=host_tree_dirty)
            r = check_cycle(args.rep, arm_map, observed,
                             expected_host_sha=args.expected_host_sha,
                             required_supervisor_programs=args.required_supervisor_programs)
            if entry is not None:
                write_result(args.out_versions, RC_OK, write_versions_doc(
                    host_native_kernel_sha=host_git_sha, host_tree_dirty=host_tree_dirty,
                    bridge_arm=entry.arm, bridge_sha=entry.bridge_sha,
                    running_image_id=running_image_id, opt_hashes=observed.opt_hashes,
                    supervisor_start_times=supervisor_start_times))
        else:
            raise ProvenanceCliError(f"unknown subcommand {args.cmd!r}")
    except (ProvenanceCliError, OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    rc = RC_OK if r.ok else RC_STOP
    return write_result(args.out, rc, {"ok": r.ok, "reason": r.reason})


def _require0(rc: int, out: str) -> str:
    if rc != 0:
        raise ProvenanceCliError(f"git status --porcelain exited {rc}")
    return out


def _parse_uptime_to_seconds(text: str) -> float:
    """``supervisorctl status``'s own uptime column: ``H:MM:SS`` or
    ``N day(s), H:MM:SS``."""
    text = text.strip()
    days = 0
    if "day" in text:
        day_part, _, text = text.partition(",")
        days = int(day_part.strip().split()[0])
        text = text.strip()
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def _parse_supervisorctl_status(text: str, *, now: Optional[float] = None) -> Dict[str, float]:
    """``supervisorctl status`` reports an UPTIME, not an absolute start
    time; converted here as ``now - uptime`` (``now`` defaults to wall time
    at the moment of this call). This is second-granularity and taken at
    CLI-invocation time rather than at the instant the status was actually
    captured -- a real, bounded skew, not a guessed value; noted in the
    handoff as a precision limit, not treated as unknown. A line with no
    parseable uptime is skipped, never given a fabricated time."""
    import time as _time
    now = _time.time() if now is None else now
    out: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.split(None, 1)[0]
        m = re.search(r"uptime\s+(.+)$", line)
        if not m:
            continue
        try:
            out[name] = now - _parse_uptime_to_seconds(m.group(1))
        except (ValueError, IndexError):
            continue
    return out


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
