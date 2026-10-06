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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import yaml

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "native_mujoco"))

from reachy_ai.motion import rig_routes as R  # noqa: E402
from joint_map import by_mjcf_index as _joint_by_mjcf_index  # noqa: E402

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


#: Read at most this many trailing bytes per attempt when hunting for the
#: last state line (F1, PR #118 review) -- a >=20 Hz poll against a
#: multi-hundred-MB states.jsonl must not cost a whole-file read on every
#: call. Grown geometrically within a single call when the window turns
#: out to hold no complete line (a state record wider than the window, or
#: nothing but a torn final write), so a single oversized record still
#: terminates instead of falling back to reading everything unconditionally.
_TAIL_READ_BYTES = 64 * 1024


def _read_last_state(
    states_path: pathlib.Path, tail_bytes: int = _TAIL_READ_BYTES,
) -> Optional[dict]:
    """The last well-formed JSON line of a states.jsonl, read from the TAIL
    of the file rather than the whole thing. Tolerant of a torn last line
    from a concurrent writer (line-buffered, but a reader can still catch a
    partial flush): the fragment nearest our seek point is dropped, since a
    seek mid-file can only ever land inside a line, never at a boundary the
    writer chose, and the true last line -- torn or not -- is always the
    one furthest from that seek point.

    `tail_bytes` is clamped to at least 1 (H1, issue #119): `tail_bytes=0`
    would otherwise pin `window` at 0 forever (`min(size, 0*2) == 0` and
    `0 >= size` is false for any non-empty file), and `tail_bytes<0` would
    reach `f.read()` with a negative count and raise `ValueError` instead
    of the `OSError` this function otherwise fails closed on.
    """
    tail_bytes = max(1, tail_bytes)
    try:
        size = states_path.stat().st_size
    except OSError:
        return None
    window = min(tail_bytes, size) if size else 0
    try:
        with open(states_path, "rb") as f:
            while True:
                f.seek(size - window)
                data = f.read(window)
                lines = data.split(b"\n")
                if size - window > 0:
                    lines = lines[1:]
                for raw in reversed(lines):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                if window >= size:
                    return None
                window = min(size, window * 2)
    except OSError:
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
    in_container: Callable[[], bool] = lambda: pathlib.Path(
        "/.dockerenv").exists(),
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

    if in_container():
        reasons.append(
            "running inside a container; the recorder shares the host "
            "monotonic clock with the native server and must run from a "
            "host shell")

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

        contacts_tracked = manifest.get("contacts_tracked")
        if contacts_tracked is not True:
            reasons.append(
                f"manifest.contacts_tracked is {contacts_tracked!r}; the "
                "server is not tracking arm-object contacts -- refusing "
                "to record")

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


@dataclass(frozen=True)
class ComplianceCheck:
    """Bounded pre-motion check (issue #116): did the physics actually reach
    the compliance state the operator just commanded, before the route
    primitive runs. See `require_compliance` for what it reads and why."""
    ok: bool
    reasons: Tuple[str, ...]
    per_joint: Dict[str, dict]
    waited_s: float
    last_state_age_s: float
    #: Which "newly applied command" evidence passed when a `cmd_baseline`
    #: was given: "advanced" (same-sequence cmd_seq advance) or
    #: "bridge_restart" (see `require_compliance`). None on the legacy
    #: `min_cmd_seq` path and when neither was given.
    cmd_evidence: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "per_joint": self.per_joint,
            "waited_s": self.waited_s,
            "last_state_age_s": self.last_state_age_s,
            "cmd_evidence": self.cmd_evidence,
        }


@dataclass(frozen=True)
class CmdBaseline:
    """Pre-`turn_on` position in the run directory's streams (see
    `capture_cmd_baseline`): the baseline state's `cmd_seq` and `seq`, and
    the byte offsets just past the last complete line of `states.jsonl`
    (that line is the baseline state) and of `commands.jsonl`."""
    cmd_seq: int
    state_seq: int
    states_offset: int
    commands_offset: int

    def as_dict(self) -> dict:
        return {
            "cmd_seq": self.cmd_seq,
            "state_seq": self.state_seq,
            "states_offset": self.states_offset,
            "commands_offset": self.commands_offset,
        }


def _is_plain_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _tail_complete_line(
    path: pathlib.Path, *, need_line: bool, tail_bytes: int = _TAIL_READ_BYTES,
) -> Tuple[int, Optional[bytes]]:
    """(byte offset just past the last complete newline-terminated line of
    `path`, that line's bytes without the newline or None if `need_line` is
    False). Offset 0 and None for a file with no complete line. Bounded tail
    reads only (window grows geometrically, as in `_read_last_state`).
    Raises OSError."""
    tail_bytes = max(1, tail_bytes)
    size = path.stat().st_size
    if size == 0:
        return 0, None
    window = min(tail_bytes, size)
    with open(path, "rb") as f:
        while True:
            start = size - window
            f.seek(start)
            data = f.read(window)
            end_idx = data.rfind(b"\n")
            if end_idx < 0:
                if window >= size:
                    return 0, None
                window = min(size, window * 2)
                continue
            offset = start + end_idx + 1
            if not need_line:
                return offset, None
            prev = data.rfind(b"\n", 0, end_idx)
            if prev >= 0:
                return offset, data[prev + 1:end_idx]
            if start == 0:
                return offset, data[:end_idx]
            window = min(size, window * 2)


def capture_cmd_baseline(
    run_dir, *, tail_bytes: int = _TAIL_READ_BYTES,
) -> Tuple[Optional[CmdBaseline], str]:
    """Capture the pre-`turn_on` baseline for `require_compliance(...,
    cmd_baseline=...)`. Returns `(baseline, "")`, or `(None, reason)` --
    fail closed -- when the streams are missing/unreadable, hold no complete
    state line, the last complete state line is not JSON, its `cmd_seq` is
    not a non-negative non-bool int (absent, null, bool, str, float -- even
    an integral float -- and non-finite all refuse), or its `seq` is not a
    non-bool int.

    Order: `commands.jsonl`'s offset is fixed FIRST, then `states.jsonl`'s
    (whose last complete line is the baseline state), so every command row
    written after the baseline state is also after the commands offset.
    Bounded tail reads only; never reads a whole states file.
    """
    run_dir_path = pathlib.Path(run_dir)
    try:
        commands_offset, _ = _tail_complete_line(
            run_dir_path / "commands.jsonl", need_line=False,
            tail_bytes=tail_bytes)
    except OSError as exc:
        return None, f"commands.jsonl unreadable: {exc}"
    try:
        states_offset, raw = _tail_complete_line(
            run_dir_path / "states.jsonl", need_line=True,
            tail_bytes=tail_bytes)
    except OSError as exc:
        return None, f"states.jsonl unreadable: {exc}"
    if raw is None:
        return None, "states.jsonl has no complete state line"
    try:
        state = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"last complete states.jsonl line is not JSON: {exc}"
    if not isinstance(state, dict):
        return None, "last complete states.jsonl line is not a JSON object"
    if "cmd_seq" not in state:
        return None, "baseline state has no cmd_seq (missing_key)"
    cmd_seq = state["cmd_seq"]
    if not _is_plain_int(cmd_seq) or cmd_seq < 0:
        return None, f"baseline state cmd_seq is not a non-negative int: {cmd_seq!r}"
    state_seq = state.get("seq")
    if not _is_plain_int(state_seq):
        return None, f"baseline state seq is not an int: {state_seq!r}"
    return CmdBaseline(
        cmd_seq=cmd_seq, state_seq=state_seq,
        states_offset=states_offset, commands_offset=commands_offset), ""


class _AppendedLines:
    """Incrementally reads the COMPLETE lines of a JSONL file appended after
    a byte offset. An unterminated trailing line is ignored until its
    newline arrives. `error` becomes sticky on the first complete line that
    does not parse to a JSON object (or on an unreadable file)."""

    def __init__(self, path: pathlib.Path, offset: int, label: str):
        self.path, self.pos, self.label = path, offset, label
        self.rows: List[dict] = []
        self.error: Optional[str] = None

    def poll(self) -> None:
        if self.error:
            return
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except OSError as exc:
            self.error = f"{self.label} unreadable: {exc}"
            return
        end = data.rfind(b"\n")
        if end < 0:
            return
        for raw in data[:end].split(b"\n"):
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.error = (f"{self.label}: a complete line after the "
                              f"baseline is not JSON ({exc})")
                return
            if not isinstance(row, dict):
                self.error = (f"{self.label}: a complete line after the "
                              "baseline is not a JSON object")
                return
            self.rows.append(row)
        self.pos += end + 1


def _bridge_restart_reasons(
    baseline: CmdBaseline, x_state: dict,
    commands: _AppendedLines, states: _AppendedLines,
) -> List[str]:
    """Empty when the bridge-restart evidence (N2, see `require_compliance`)
    holds for last state `x_state`; otherwise the failed parts, each
    prefixed "bridge-restart path: "."""
    pre = "bridge-restart path: "
    commands.poll()
    states.poll()
    if commands.error:
        return [pre + commands.error]
    if states.error:
        return [pre + states.error]

    # (a) rows after the baseline are exactly one new sequence 1..m
    seqs: List[int] = []
    for row in commands.rows:
        if row.get("type") == "reset":
            return [pre + "(a) a reset row was appended after the baseline"]
        if row.get("type") == "joint_command":
            seq = row.get("seq")
            if not _is_plain_int(seq):
                return [pre + f"(a) a joint_command row has a non-int seq ({seq!r})"]
            seqs.append(seq)
    if not seqs:
        return [pre + "(a) no joint_command row was appended after the baseline"]
    m = len(seqs)
    if seqs != list(range(1, m + 1)):
        return [pre + f"(a) appended joint_command seqs {seqs[:12]} are not "
                      "exactly 1..m of one new sequence"]

    # (b) native proof of an applied change after the baseline
    x_seq = x_state.get("seq")
    if not _is_plain_int(x_seq) or x_seq <= baseline.state_seq:
        return [pre + f"(b) last state seq {x_seq!r} is not past the baseline "
                      f"state seq {baseline.state_seq}"]
    considered = []
    for row in states.rows:
        seq = row.get("seq")
        if _is_plain_int(seq) and baseline.state_seq < seq <= x_seq:
            considered.append(row)
    first = None
    for i, row in enumerate(considered):
        if row.get("cmd_seq") != baseline.cmd_seq or not _is_plain_int(row.get("cmd_seq")):
            first = i
            break
    if first is None:
        return [pre + "(b) no appended state shows an applied change of cmd_seq "
                      f"away from the baseline ({baseline.cmd_seq}); a received "
                      "command row alone is not evidence that it was applied"]
    prev = None
    for row in considered[first:]:
        v = row.get("cmd_seq")
        if not _is_plain_int(v) or not (1 <= v <= m):
            return [pre + f"(b) state seq {row.get('seq')} has cmd_seq {v!r} "
                          f"outside [1, {m}]"]
        if prev is not None and v < prev:
            return [pre + f"(b) cmd_seq decreased ({prev} -> {v}) within the new sequence"]
        prev = v
    last = considered[-1]
    if last.get("seq") != x_seq or last.get("cmd_seq") != x_state.get("cmd_seq"):
        return [pre + f"(b) the last state (seq {x_seq}) was not found among "
                      "the states appended after the baseline"]

    # (c)
    if not (1 <= x_state["cmd_seq"] <= m):
        return [pre + f"(c) cmd_seq={x_state['cmd_seq']} is outside [1, {m}]"]
    return []


def _commands_with_compliance_tail(
    commands_path: pathlib.Path, n: int = 5,
) -> List[dict]:
    """The last `n` commands.jsonl entries that carried a non-null
    `compliant` entry for at least one joint -- oldest first. Tolerant of a
    torn last line, same reasoning as `_read_last_state`."""
    try:
        raw_lines = commands_path.read_text().splitlines()
    except OSError:
        return []
    out: List[dict] = []
    for raw in reversed(raw_lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        compliant = entry.get("compliant")
        if isinstance(compliant, list) and any(v is not None for v in compliant):
            out.append(entry)
            if len(out) >= n:
                break
    out.reverse()
    return out


def _describe_sent_compliance(entry: dict) -> str:
    names = []
    for i, v in enumerate(entry.get("compliant") or []):
        if v is None:
            continue
        joint = _joint_by_mjcf_index(i)
        name = joint.sdk_name if joint is not None else f"idx{i}"
        names.append(f"{name}={v}")
    return f"seq {entry.get('seq')}: " + ", ".join(names)


def require_compliance(
    run_dir: str,
    joints: Sequence[str],
    *,
    compliant: bool = False,
    timeout_s: float = 3.0,
    max_state_age_s: float = 0.5,
    min_cmd_seq: Optional[int] = None,
    cmd_baseline: Optional[CmdBaseline] = None,
    read_last_state: Callable[[pathlib.Path], Optional[dict]] = _read_last_state,
    now_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> ComplianceCheck:
    """Poll the run directory's state stream (never the SDK client's cached
    `joint.compliant`, which is written by the client and read back only at
    connect -- see the module docstring) until every named joint reports
    `compliant == <expected>` on a FRESH sample, or `timeout_s` elapses.

    Never sends anything, never retries motion, never touches tolerances.

    Timing/sequence contract (F2, PR #118 review)
    -----------------------------------------------
    `max_state_age_s` alone proves the sample is CURRENT, not that it
    postdates any particular call -- compliance that already held before
    `turn_on` (e.g. it persisted through a reset) can satisfy a fresh
    sample by coincidence. `min_cmd_seq` closes that gap: pass
    `state["cmd_seq"]` (see `native_mujoco.protocol.State.cmd_seq`, the
    last-*applied* command sequence) read from the run directory
    IMMEDIATELY BEFORE the `turn_on`/`turn_off` call this check is gating.
    A sample is only accepted once its own `cmd_seq` has advanced STRICTLY
    PAST that baseline -- i.e. the server has applied at least one command
    submitted after the baseline was captured.

    A `cmd_seq` newer than the baseline is NECESSARY but not on its own
    SUFFICIENT proof that it was specifically the gated call's command that
    landed -- any command from any sender advances the same counter, so a
    concurrent unrelated command would also satisfy it. It is the
    combination of all three checks -- `min_cmd_seq` (something new was
    applied), freshness (the sample is current, not a stale disk read),
    and the per-joint `compliant` match (the actual effect is present) --
    that ties the evidence to the current attempt; none of the three is
    load-bearing alone. `min_cmd_seq=None` (the default) skips this check
    entirely, keeping every pre-F2 caller's contract unchanged and relying
    on the freshness window alone, as before.

    Bridge-restart caveat (H4, issue #119): `cmd_seq` is the bridge's own
    counter (`mujoco_remote_backend.py`'s `_cmd_seq`), which resets to 0
    when the container restarts, while the native server's `_cmd_seq`
    baseline keeps its old, higher value. After a container restart,
    `require_compliance(min_cmd_seq=<old high>)` fails closed until the
    new counter overtakes that stale baseline -- read that as a bridge
    restart, not a physics fault.

    `cmd_baseline` (assignment 2026-10-05; a004, 2026-10-02)
    ---------------------------------------------------------
    `min_cmd_seq` compares numbers from ONE bridge process, so it cannot
    cross a bridge restart (the H4 caveat above): a004's r1 setup `turn_on`
    after a container recreate moved the native `cmd_seq` 2 -> 1 and was
    rejected although the sample was fresh and every joint stiff. Pass a
    `CmdBaseline` from `capture_cmd_baseline` (taken immediately before the
    call) INSTEAD of `min_cmd_seq` (passing both raises ValueError); the
    legacy `min_cmd_seq` behaviour above is unchanged. Four questions, each
    answered by its own evidence: sequence identity (command-row `seq` and
    state `cmd_seq`, comparable only within one bridge), state freshness
    (`wall_time_ns` age, unchanged), the compliance effect (every requested
    joint's `compliant` equals the request, unchanged), and "a command was
    newly applied", which is accepted by either:

      N1 "advanced": last state X's `cmd_seq` (a non-bool int) is greater
         than the baseline's -- the same-sequence case, as before; or
      N2 "bridge_restart": X.cmd_seq <= baseline.cmd_seq AND, reading only
         complete lines appended after the baseline offsets,
         (a) the appended joint_command `seq` values are exactly 1..m (no
             gap, no repeat, no old value; any appended `reset` row, a
             non-int seq, or a complete unparseable line rejects);
         (b) X.seq > baseline.state_seq and, among appended states up to X,
             the first state F with cmd_seq != baseline.cmd_seq exists (the
             native server changes cmd_seq only when it APPLIES a command,
             so this -- not a received row -- is the proof of an apply),
             and every state from F through X has a non-bool int cmd_seq in
             [1, m], non-decreasing;
         (c) X.cmd_seq is in [1, m].
    N2 failures are reported with the prefix "bridge-restart path: ".
    An old-bridge command in flight carries a seq above the baseline, so it
    can never yield X.cmd_seq <= baseline; restart evidence from before the
    baseline is excluded by the offsets.

    Limits (fail closed, deliberately not papered over): if the new
    sequence's final applied value equals the old baseline and no
    intermediate change was sampled (old 1 -> new 1), N2 rejects. This is
    still not causal proof that the gated call's own command landed -- the
    concurrent-sender caveat above applies unchanged -- and N2 is not
    evidence about any restart other than the one whose rows follow the
    baseline. `ComplianceCheck.cmd_evidence` records which path passed.
    """
    if min_cmd_seq is not None and cmd_baseline is not None:
        raise ValueError(
            "pass either min_cmd_seq (legacy) or cmd_baseline, not both")
    appended_commands = appended_states = None
    if cmd_baseline is not None:
        appended_commands = _AppendedLines(
            pathlib.Path(run_dir) / "commands.jsonl",
            cmd_baseline.commands_offset, "commands.jsonl")
        appended_states = _AppendedLines(
            pathlib.Path(run_dir) / "states.jsonl",
            cmd_baseline.states_offset, "states.jsonl")
    run_dir_path = pathlib.Path(run_dir)
    states_path = run_dir_path / "states.jsonl"
    commands_path = run_dir_path / "commands.jsonl"
    poll_period_s = 1.0 / 20.0  # >= 20 Hz, per the assignment

    start_ns = now_ns()
    while True:
        state = read_last_state(states_path)
        waited_s = (now_ns() - start_ns) / 1e9
        per_joint: Dict[str, dict] = {}
        bad: List[str] = []
        fresh = False
        age_s = float("inf")
        cmd_evidence: Optional[str] = None

        if state is None:
            bad.append("states.jsonl has no readable sample yet")
        else:
            wall_time_ns = state.get("wall_time_ns")
            if isinstance(wall_time_ns, (int, float)):
                age_s = abs(now_ns() - wall_time_ns) / 1e9
            fresh = age_s <= max_state_age_s

            if min_cmd_seq is not None:
                state_cmd_seq = state.get("cmd_seq")
                if not (isinstance(state_cmd_seq, (int, float))
                        and not isinstance(state_cmd_seq, bool)
                        and math.isfinite(state_cmd_seq)):
                    bad.append(
                        f"state has no numeric cmd_seq to compare against "
                        f"min_cmd_seq={min_cmd_seq} -- refusing to treat it "
                        "as evidence of the current attempt")
                elif state_cmd_seq <= min_cmd_seq:
                    bad.append(
                        f"cmd_seq={state_cmd_seq} has not advanced past the "
                        f"pre-call baseline (min_cmd_seq={min_cmd_seq}) -- "
                        "this sample may predate the current turn_on/"
                        "turn_off attempt")

            if cmd_baseline is not None:
                x_cmd_seq = state.get("cmd_seq")
                if not _is_plain_int(x_cmd_seq):
                    bad.append(
                        f"state has no int cmd_seq ({x_cmd_seq!r}) to compare "
                        "against cmd_baseline -- refusing to treat it as "
                        "evidence of the current attempt")
                elif x_cmd_seq > cmd_baseline.cmd_seq:
                    cmd_evidence = "advanced"
                else:
                    restart_bad = _bridge_restart_reasons(
                        cmd_baseline, state, appended_commands, appended_states)
                    if restart_bad:
                        bad.append(
                            f"cmd_seq={x_cmd_seq} has not advanced past the "
                            f"pre-call baseline (cmd_baseline.cmd_seq="
                            f"{cmd_baseline.cmd_seq}) -- this sample may "
                            "predate the current turn_on/turn_off attempt")
                        bad.extend(restart_bad)
                    else:
                        cmd_evidence = "bridge_restart"

            server_joints = {j.get("name"): j for j in state.get("joints", [])}
            for name in joints:
                entry = server_joints.get(name)
                if entry is None:
                    bad.append(f"{name}: not in the last state sample")
                    continue
                raw_compliant = entry.get("compliant")
                if not isinstance(raw_compliant, bool):
                    # F3, PR #118 review: an absent/null/non-bool
                    # `compliant` must never be coerced to False ("stiff").
                    # A state stream that does not report compliance at all
                    # must not be able to pass a compliant=False gate by
                    # accident -- fail closed for every requested joint.
                    per_joint[name] = {
                        "compliant": None,
                        "effort": entry.get("effort"),
                        "seq": state.get("seq"),
                        "sim_step": state.get("sim_step"),
                    }
                    bad.append(
                        f"{name}: compliant field is missing or not a bool "
                        f"({raw_compliant!r}) -- refusing to treat unknown "
                        "compliance as stiff")
                    continue
                entry_compliant = raw_compliant
                per_joint[name] = {
                    "compliant": entry_compliant,
                    "effort": entry.get("effort"),
                    "seq": state.get("seq"),
                    "sim_step": state.get("sim_step"),
                }
                if entry_compliant != compliant:
                    bad.append(
                        f"{name}: compliant={entry_compliant} (want "
                        f"{compliant}), effort={entry.get('effort')}")

        if fresh and not bad:
            return ComplianceCheck(
                ok=True, reasons=(), per_joint=per_joint,
                waited_s=waited_s, last_state_age_s=age_s,
                cmd_evidence=cmd_evidence)

        if waited_s >= timeout_s:
            reasons = list(bad)
            if not fresh:
                reasons.append(
                    f"state stream is stale (last sample {age_s:.3f}s old, "
                    f"> {max_state_age_s}s max) -- refusing to pass on old "
                    "data")
            sent = _commands_with_compliance_tail(commands_path, n=5)
            if sent:
                reasons.append(
                    "commands sent with a compliant flag (most recent "
                    "last): " + " | ".join(
                        _describe_sent_compliance(e) for e in sent))
            return ComplianceCheck(
                ok=False, reasons=tuple(reasons), per_joint=per_joint,
                waited_s=waited_s, last_state_age_s=age_s)

        sleep(poll_period_s)
