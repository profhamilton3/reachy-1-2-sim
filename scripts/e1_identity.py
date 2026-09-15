"""E1 readiness (assignment 2026-09-14, work item 2): verify that the
simulator backend `scripts/measure_route_clearance.py` is about to record
from is ACTUALLY the native MuJoCo bridge on this machine -- never trusted
from the client's own `REACHY_SIM_BACKEND` (or any other env var, or from
`--record-root` defaulting to `REACHY_SIM_RECORD`, which is a path HINT
only). `verify_simulator_identity` is called before `record_joint_log`; on
any failure the recorder refuses to run. `safety.gate_check` and every
other production safety path are untouched -- this module is new, standalone,
and read by the recorder alone.

CONTRACT DETAIL RESOLVED BEFORE IMPLEMENTATION (owner, 2026-09-14)
-------------------------------------------------------------------
"Loopback is a restriction, not proof of simulator identity. Establish the
SDK endpoint's connection to the verified simulator bridge; fail closed if
that cannot be established. Joint agreement alone is insufficient."

What this module can and cannot prove
--------------------------------------
`read_sdk_joints` talks to the fake gRPC server (`fake_reachy_server.py`,
:50051 by default) over the SAME protobuf services the real robot's SDK
answers -- by design, so `reachy_sdk` (the real v1 package, unmodified)
works against either. Read `fake_reachy_server.py`'s own registrations:
Joint, Sensor, Fan, HeadKinematics, ArmKinematics, Camera -- five services,
all mirroring the physical robot's API, none of them simulator-aware. There
is no sixth, simulator-only RPC to ask "which run directory are you
bridging to" -- adding one would mean diverging from the real robot's
protocol this fake server exists to stand in for, which is out of scope
here (and would still leave the REAL robot, which this module must also
refuse, unable to answer it). So THE ONLY SIGNAL REACHABLE THROUGH
`host:port` IS JOINT VALUES.

Joint-value agreement with a snapshot of `states.jsonl` proves correlation,
never identity: no motion is authorised anywhere in E1 readiness work (nor
in the pilot's own parked recording, which is deliberately motionless), so
a decoupled bridge parked at a similar pose could satisfy a SINGLE snapshot
comparison by coincidence. This module cannot close that gap by talking to
the gRPC endpoint harder -- there is nothing more to ask it. It closes as
much of the gap as the available signals allow, and REFUSES rather than
guesses past the rest:

  1. Loopback-only host -- excludes the robot regardless of env. Necessary,
     not sufficient (this alone was the old, insufficient design).
  2. Exactly ONE live run directory under `record_root`. Two
     simultaneously-fresh `run_*` directories is refused as ambiguous
     rather than silently picking the newest: "which server is `host:port`
     actually talking to" has no answer when more than one candidate is
     alive at once, and picking one anyway would be the same mistake as
     trusting an env var.
  3. INTERLEAVED, not sequential, round-by-round correlation: each of
     `min_reads` rounds (>= `min_read_interval_s` apart) re-reads BOTH the
     one live run directory's latest sample AND the SDK, at that same
     moment, and requires (a) the run directory's sample is fresh (age <=
     `max_round_age_s`, tighter than the initial liveness bound) and its
     `sim_step` has advanced since the previous round, and (b) every one of
     the 8 `rig_routes.R_JOINTS` agrees with the SDK's reading to within
     `max_joint_deg` -- AT THAT ROUND, never against a cached snapshot from
     an earlier one. A stale run directory or an unrelated bridge would
     have to coincidentally satisfy this every round, not just once.
  4. Scene identity: `manifest.scene_sha256 == sha256(scene_path)`, plus
     the sha of every file in the `extends` chain (the manifest only
     hashes the leaf).
  5. Bridge backend: `/status` (issue #40) reports `backend ==
     "mujoco-remote"` and `frames_stale is False`. Unreachable is a
     failure, not a downgrade -- it is the one existing detection of the
     container bridge's claimed backend, and even that is itself sourced
     from a sidecar file the bridge last wrote (`web/camera_server.py`'s
     `_detect_backend`), not a live process check -- named here so nobody
     mistakes it for stronger evidence than it is.

None of this is a cryptographic binding of the TCP session at `host:port`
to the process that owns the run directory. That would need either a
change to the real robot's SDK protocol (out of scope: this module talks to
the SAME protocol the physical robot answers) or OS-level socket/process
introspection (not attempted here -- named as a residual limitation in the
handoff and ADR-0003, alongside "avoid redesigning the general safety
system"). "Fail closed" here means every one of the checks above must
hold, every round -- not that identity is proven beyond what the SDK
protocol actually carries.

Two clocks, deliberately not conflated
---------------------------------------
`now_ns` (default `time.monotonic_ns`) is used for every freshness/ordering
comparison against `states.jsonl` samples' own `wall_time_ns` -- which,
despite its name, is ALSO `time.monotonic_ns()` (see
`measure_route_clearance.py`'s "Sample wall-clock" docstring section for
the full evidence trail). This module and the native server it is checking
are both host-native processes (this module runs on the host, same as
`native_mujoco/server.py`; never inside the Docker container), so this is a
same-clock comparison, not a cross-process guess.

`wall_clock_ns` (default `time.time_ns`) is used ONLY for
`checked_at_wall_ns` -- an audit timestamp for a human reading the sidecar
later. It is never compared against anything: mixing it with `now_ns`
readings would be exactly the wall-clock/monotonic-clock confusion this
module's sibling docstring warns about.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import pathlib
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from reachy_ai.motion import rig_routes as R  # noqa: E402

#: The 8 joints identity is checked over -- the same set
#: `measure_route_clearance.py` records (ARM7 + r_gripper).
R_JOINT_NAMES: Tuple[str, ...] = R.R_JOINTS


@dataclass(frozen=True)
class SimulatorIdentityCheck:
    ok: bool
    reasons: Tuple[str, ...]
    run_dir: str
    manifest: dict
    scene_chain_sha256: Dict[str, str]
    joint_agreement_deg: Dict[str, float]
    status: dict
    checked_at_wall_ns: int

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "run_dir": self.run_dir,
            "manifest": self.manifest,
            "scene_chain_sha256": self.scene_chain_sha256,
            "joint_agreement_deg": self.joint_agreement_deg,
            "status": self.status,
            "checked_at_wall_ns": self.checked_at_wall_ns,
        }


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _sha256_file(path: pathlib.Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _extends_chain(path: pathlib.Path) -> List[pathlib.Path]:
    """`path`, then every ancestor named by `extends:`, child-to-root, by
    reading each file's own `extends:` key directly (this module only needs
    the file list to hash -- the full merge semantics live in
    `native_mujoco/scene_io.py` and are not reimplemented here)."""
    chain: List[pathlib.Path] = []
    seen = set()
    current = path.resolve()
    while current not in seen:
        seen.add(current)
        chain.append(current)
        try:
            doc = yaml.safe_load(current.read_text())
        except OSError:
            break
        if not isinstance(doc, dict):
            break
        parent_ref = doc.get("extends")
        if not parent_ref:
            break
        current = (current.parent / str(parent_ref)).resolve()
    return chain


def _read_last_state(states_path: pathlib.Path) -> Optional[dict]:
    """The last well-formed JSON line of a states.jsonl -- tolerant of a
    torn last line from a concurrent writer (line-buffered, but a reader
    can still catch a partial flush)."""
    try:
        data = states_path.read_bytes()
    except OSError:
        return None
    for raw in reversed(data.splitlines()):
        raw = raw.strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None


def _live_run_dirs(
    record_root: pathlib.Path, now_ns_value: int, max_state_age_s: float,
) -> List[pathlib.Path]:
    """`run_*` directories under `record_root` with a `states.jsonl` whose
    last sample is fresh -- liveness alone, independent of whether
    `manifest.json` exists (a missing manifest is reported as its own,
    distinct reason once a single live candidate is chosen, not folded into
    "no live directory")."""
    if not record_root.is_dir():
        return []
    live = []
    for candidate in sorted(record_root.glob("run_*")):
        states_path = candidate / "states.jsonl"
        if not states_path.is_file():
            continue
        last = _read_last_state(states_path)
        if last is None:
            continue
        wall_time_ns = last.get("wall_time_ns")
        if not isinstance(wall_time_ns, (int, float)):
            continue
        age_s = abs(now_ns_value - wall_time_ns) / 1e9
        if age_s <= max_state_age_s:
            live.append(candidate)
    return live


def _default_http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def verify_simulator_identity(
    *,
    host: str,
    port: int,
    scene_path: str,
    record_root: str,
    status_url: str = "http://localhost:8080/status",
    read_sdk_joints: Callable[[], Dict[str, float]],
    now_ns: Callable[[], int] = time.monotonic_ns,
    wall_clock_ns: Callable[[], int] = time.time_ns,
    http_get: Callable[[str], dict] = _default_http_get,
    sleep: Callable[[float], None] = time.sleep,
    max_joint_deg: float = 1.0,
    max_state_age_s: float = 1.0,
    max_round_age_s: float = 0.2,
    min_reads: int = 3,
    min_read_interval_s: float = 0.1,
) -> SimulatorIdentityCheck:
    """See the module docstring for what each check does and does not
    prove. `read_sdk_joints` must return `{sdk_name: degrees}` for at least
    `R_JOINT_NAMES`, read from an SDK object already connected to
    `host:port` -- this function never opens that connection itself.
    """
    reasons: List[str] = []

    if not _is_loopback(host):
        return SimulatorIdentityCheck(
            ok=False,
            reasons=(f"recorder is for the simulator bridge on this "
                     f"machine; a physical robot host is refused "
                     f"(host={host!r} is not loopback)",),
            run_dir="", manifest={}, scene_chain_sha256={},
            joint_agreement_deg={}, status={},
            checked_at_wall_ns=wall_clock_ns())

    scene_p = pathlib.Path(scene_path).resolve()
    chain = _extends_chain(scene_p)
    scene_chain_sha256: Dict[str, str] = {}
    for p in chain:
        sha = _sha256_file(p)
        scene_chain_sha256[str(p)] = sha
        if sha is None:
            reasons.append(f"could not hash {p} (part of {scene_p}'s "
                           "extends chain)")

    root_p = pathlib.Path(record_root)
    run_dir = ""
    manifest: dict = {}
    joint_agreement: Dict[str, float] = {}

    live = _live_run_dirs(root_p, now_ns(), max_state_age_s)
    if len(live) == 0:
        reasons.append(
            f"no live physics run directory under {record_root!r} "
            f"(expected a run_* whose states.jsonl has a sample within "
            f"{max_state_age_s}s of now) -- is the native server running "
            "with --record set to this path?")
    elif len(live) > 1:
        reasons.append(
            f"{len(live)} run directories under {record_root!r} all look "
            f"live at once ({[str(p) for p in live]}) -- refusing rather "
            "than guessing which one host:port is actually bridging to; "
            "stop every simulator instance but the one you mean to use")
    else:
        run_dir_path = live[0]
        run_dir = str(run_dir_path)
        manifest_path = run_dir_path / "manifest.json"
        if not manifest_path.is_file():
            reasons.append(f"{run_dir}/manifest.json is missing")
        else:
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                reasons.append(f"{run_dir}/manifest.json unreadable: {exc}")

        prev_sim_step: Optional[int] = None
        for round_i in range(min_reads):
            state = _read_last_state(run_dir_path / "states.jsonl")
            if state is None:
                reasons.append(f"round {round_i}: states.jsonl has no "
                               "readable sample")
                break
            wall_time_ns = state.get("wall_time_ns")
            if not isinstance(wall_time_ns, (int, float)):
                reasons.append(f"round {round_i}: latest sample has no "
                               "numeric wall_time_ns")
                break
            age_s = abs(now_ns() - wall_time_ns) / 1e9
            if age_s > max_round_age_s:
                reasons.append(
                    f"round {round_i}: latest sample is {age_s:.3f}s old "
                    f"(> {max_round_age_s}s) -- the physics looks paused, "
                    "or this run directory is going stale")
                break
            sim_step = state.get("sim_step")
            if prev_sim_step is not None and not (
                    isinstance(sim_step, (int, float))
                    and sim_step > prev_sim_step):
                reasons.append(
                    f"round {round_i}: sim_step did not advance "
                    f"({prev_sim_step} -> {sim_step}) -- physics is paused "
                    "or this is not the live run directory")
                break
            prev_sim_step = sim_step

            sdk_joints = read_sdk_joints()
            server_joints = {j.get("name"): j for j in state.get("joints", [])}
            round_bad: List[str] = []
            for name in R_JOINT_NAMES:
                entry = server_joints.get(name)
                if entry is None:
                    round_bad.append(f"{name}: not in this state sample")
                    continue
                server_deg = math.degrees(entry.get("position_rad", 0.0))
                sdk_deg = sdk_joints.get(name)
                if sdk_deg is None:
                    round_bad.append(f"{name}: SDK did not report it")
                    continue
                diff = abs(float(sdk_deg) - server_deg)
                joint_agreement[name] = max(joint_agreement.get(name, 0.0), diff)
                if diff > max_joint_deg:
                    round_bad.append(
                        f"{name}: |sdk {sdk_deg:.2f} - server "
                        f"{server_deg:.2f}| = {diff:.2f} deg > "
                        f"{max_joint_deg} deg")
            if round_bad:
                reasons.append(
                    f"round {round_i} joint disagreement: " +
                    "; ".join(round_bad))
                break

            if round_i + 1 < min_reads:
                sleep(min_read_interval_s)

        if manifest:
            manifest_sha = manifest.get("scene_sha256")
            manifest_scene_path = manifest.get("scene_path")
            leaf_sha = scene_chain_sha256.get(str(scene_p))
            if manifest_scene_path and pathlib.Path(
                    manifest_scene_path).resolve() != scene_p:
                reasons.append(
                    f"manifest.scene_path ({manifest_scene_path!r}) is not "
                    f"{scene_p} -- the running server was started with a "
                    "different --scene")
            if manifest_sha != leaf_sha:
                reasons.append(
                    f"scene sha mismatch: manifest.scene_sha256="
                    f"{manifest_sha!r}, {scene_p} hashes to {leaf_sha!r}")

    status: dict = {}
    try:
        status = http_get(status_url)
    except Exception as exc:  # noqa: BLE001 -- any transport failure refuses
        reasons.append(f"{status_url} unreachable: {exc}")
    else:
        backend = status.get("backend")
        if backend != "mujoco-remote":
            reasons.append(
                f"{status_url} reports backend={backend!r}, not "
                "'mujoco-remote'")
        if status.get("frames_stale", True) is not False:
            reasons.append(
                f"{status_url} reports frames_stale="
                f"{status.get('frames_stale')!r}")

    return SimulatorIdentityCheck(
        ok=not reasons, reasons=tuple(reasons), run_dir=run_dir,
        manifest=manifest, scene_chain_sha256=scene_chain_sha256,
        joint_agreement_deg=joint_agreement, status=status,
        checked_at_wall_ns=wall_clock_ns())
