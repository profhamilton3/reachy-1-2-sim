"""Stage A vertical slice (assignment
``outputs/assignment-2026-09-25-sonnet-pr144-slice-then-h1-h10.md``, Stage A):
4 real-bridge cycles, reps 1-4, order A B B A (``provenance.ARM_MAP_ORDER``'s
own first four characters), laid out exactly as plan §6 describes a real
E1 session, so the SHIPPED cycle/summary CLIs at e54be0e can be run against
it in non-validation mode.

Builds on ``real_bridge.py`` (the Stage 1 V3 harness: ``reachy_sdk`` ->
loopback gRPC -> ``FakeJointService`` -> ``KinematicBridge(MujocoRemoteBackend)``
-> ``NativeStub``) -- reuses its ``build_session``/``BridgeGen``/``Observer``/
row-builder functions UNCHANGED, and adds only what Stage A itself needs:

* a 4-cycle loop (vs. ``run_two_cycles``'s 2), labelled A/B/B/A;
* synthetic echo injection for the A cycles (see ``inject_synthetic_echoes``);
* a per-cycle bridge-log capture (a ``logging.FileHandler`` on the
  ``mujoco_remote_backend`` logger, attached/detached per generation);
* REAL ``scripts/e1_stage1/reset_verify.py snapshot``/``verify`` subprocess
  calls around each reset, against a live-flushed mirror of the run
  directory (see ``_flush_partial_run_dir``) -- not a synthesised transcript;
* the plan §6 evidence layout: ``<ev>/e1_server_runs/<run>/{commands,states}.jsonl``,
  a ROOT ``SHA256SUMS`` keyed ``./e1_server_runs/<run>/...``, and
  linker-shaped sidecars with a real ``scene`` block for
  ``scenes/e1_boards/B4_pool_box_1_r2c3.yaml``.

DOCUMENTED PLACEHOLDERS AND KNOWN GAPS (excluded from every conclusion; see
the Stage A handoff for the full discussion):

* ``NativeStub`` never runs ``fake_reachy_server.py``'s own process
  bootstrap, so ``reset_watcher`` (spawned only by that bootstrap,
  ``fake_reachy_server.py:674``) never runs in this harness. Every
  watcher-sourced tripwire line is therefore structurally absent here, not
  evidence that no such event occurred.
* **Fixed (coordinator guidance, 2026-09-25):** ``NativeStub`` is now
  constructed via ``build_stage_a_session`` with the REAL native reset
  keyframe -- ``native_mujoco/model/reachy_1_2.xml:465-469``
  (``<keyframe><key name="home" qpos="0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0
  0 0.45 0  0 0" .../></keyframe>``, in MJCF/``JOINT_TABLE`` ``mjcf_index``
  order, radians) -- NOT ``R.HOME`` (a ``rig_routes`` SDK convenience pose
  used by Stage 1's own ``real_bridge.build_session``, whose
  ``r_gripper=-45deg`` is a "ready" pose, not the native reset pose). This
  uses ``NativeStub``'s own documented ``pose=`` constructor kwarg
  unchanged -- ``NativeStub`` itself is not modified. Under the real
  keyframe (``r_gripper=0``, ``r_wrist_roll=0``, every gross joint 0), every
  cycle's post-reset pose genuinely classifies as ``"stiff-zero"`` under
  ``scripts/e1_stage1/plan.classify_start_variant`` -- the start-variant
  gate now PASSES for real on this harness, rather than failing on a
  fixture pose-fidelity gap.
* ``NativeStub``'s constructor takes one ``compliant: bool`` for ALL 21
  joints -- there is no per-joint compliance option to source the plan's
  "antennas stiff by default" (plan §4) from. Checked: neither
  ``NativeStub.__init__`` nor any other public method sets per-joint
  compliance at construction/reset time (only ``move_by_hand``, position
  only). ``mujoco_remote_backend.py:780``'s own ``compliant=uid not in
  (33, 34)`` antenna default exists only in ``build_snapshot``'s "joint the
  server hasn't reported yet" fallback branch, which this harness never
  reaches (``NativeStub`` reports all 21 joints every tick). So there is
  genuinely no documented option to source this from -- kept as a
  documented harness-fidelity gap, per the coordinator's fallback
  instruction, not routed around: the compliance gate is EXPECTED to keep
  failing on ``['l_antenna', 'r_antenna']`` on every cycle.
* the harness's own reset call (``BridgeGen.reset``) goes through
  ``MujocoRemoteBackend.request_reset()`` directly; there is no separate
  ``reset.sh``/``reset_watcher`` generation/ack protocol to reproduce, so
  this module MINTS its own gen/ack label per cycle (``str(rep)``, equal to
  the expected post-reset epoch index) and passes it to
  ``reset_verify.py verify`` as both ``gen`` and ``ack`` -- there is no
  separate ack-mismatch failure mode available to observe on this harness.
* ``make_fixtures.write_evidence`` (the shared JSONL writer other tests use)
  writes JSON with default (SPACED) separators. ``reset_verify.py``'s own
  reset-line matcher (``_reset_lines``) looks for the literal, UNSPACED
  substring ``"type":"reset"`` (matching the real recorder's
  ``json.dumps(..., separators=(",", ":"))``). This module therefore writes
  its own JSONL with compact separators (``_dump_compact``), matching the
  real recorder -- NOT a ``tools/`` or ``make_fixtures.py`` change, purely
  additive here.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
for _p in (_REPO, _REPO / "src", _REPO / "native_mujoco", _REPO / "tests" / "integration",
           _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import real_bridge as rb  # noqa: E402
from native_stub import IDX  # noqa: E402
import make_fixtures as mf  # noqa: E402

from reachy_ai.motion import rig_routes as R  # noqa: E402
from tools.goalfix_cmp import provenance as pv  # noqa: E402

ARM8 = R.ARM7 + ("r_gripper",)
ARM_IDX = [IDX[n] for n in ARM8]

#: provenance.ARM_MAP_ORDER's own first four characters (assignment A1).
ARM_LABELS: Tuple[str, ...] = pv.ARM_MAP_ORDER[:4]
assert ARM_LABELS == ("A", "B", "B", "A")

#: Historical bridge SHAs from the plan (b4-goalfix-comparison-plan-2026-09-24.md
#: §1's table) -- LABELS only. This harness runs the SAME e54be0e checkout's
#: mujoco_remote_backend.py/fake_reachy_server.py for every cycle; no image
#: swap happens. See the module docstring and the handoff for the "arm is a
#: label, not a different binary" statement the assignment requires.
BRIDGE_SHA = {"A": "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8",
              "B": "67730a1ecf646544825fb60c00129cf46de6d307"}

RUN_NAME = "run_stage_a_slice"
RESET_VERIFY_PY = _REPO / "scripts" / "e1_stage1" / "reset_verify.py"
SCENE_PATH = _REPO / "scenes" / "e1_boards" / "B4_pool_box_1_r2c3.yaml"


# ---------------------------------------------------------------------------
# Compact-JSON JSONL writer (see module docstring: reset_verify.py's
# reset-line matcher needs the real recorder's unspaced separators).
# ---------------------------------------------------------------------------

def _dump_compact(row: dict) -> str:
    return json.dumps(row, separators=(",", ":"))


def _write_jsonl_compact(path: Path, rows: Sequence[dict]) -> None:
    text = "\n".join(_dump_compact(r) for r in rows)
    if rows:
        text += "\n"
    path.write_text(text)


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-cycle bridge log capture (mujoco_remote_backend's own logger; see
# module docstring re: reset_watcher never running in this harness).
# ---------------------------------------------------------------------------

class BridgeLogCapture:
    def __init__(self, path: Path):
        self.path = path
        self._logger = logging.getLogger("mujoco_remote_backend")
        self._handler = logging.FileHandler(str(path), mode="w")
        self._handler.setLevel(logging.WARNING)
        self._handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def __enter__(self) -> "BridgeLogCapture":
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc) -> None:
        self._handler.flush()
        self._logger.removeHandler(self._handler)
        self._handler.close()
        if not self.path.exists():
            self.path.write_text("")


# ---------------------------------------------------------------------------
# reset_verify.py -- run for real, as a subprocess (assignment: "actually
# running scripts/e1_stage1/reset_verify.py against the fixture run dir").
# ---------------------------------------------------------------------------

def run_reset_verify(argv: Sequence[str]) -> Tuple[int, str, str]:
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "src"), str(_REPO / "native_mujoco"), str(_REPO / "scripts"),
         env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [sys.executable, str(RESET_VERIFY_PY)] + list(argv),
        cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=30)
    return proc.returncode, proc.stdout, proc.stderr


def _flush_partial_run_dir(session: "rb.V3Session", run_dir: Path,
                            reset_events_committed: Sequence[tuple]) -> None:
    """Writes ``run_dir/{commands,states}.jsonl`` (compact JSON) from
    everything observed/committed SO FAR -- the live mirror
    ``reset_verify.py`` needs to exist at each point it is invoked. Mirrors
    ``real_bridge.emit_evidence``'s own merge algorithm exactly, factored so
    it can be called mid-session, not just once at the end."""
    state_rows = [rb._state_row(m) for m in session.observer.snapshot()]
    with session.stub.lock:
        raw_commands = list(session.stub.commands)

    reset_events_sorted = sorted(reset_events_committed, key=lambda e: e[0])
    closing_steps = rb._closing_epoch_last_steps(state_rows)
    if len(closing_steps) < len(reset_events_sorted):
        raise AssertionError(
            f"stage_a_slice: {len(reset_events_sorted)} committed reset() calls but only "
            f"{len(closing_steps)} sim_step drops observed so far -- a reset did not land, "
            "or the observer missed a state")
    closing_steps = closing_steps[:len(reset_events_sorted)]

    events: List[tuple] = [
        (t, "reset", (seed, closing_steps[k]))
        for k, (t, seed, _live) in enumerate(reset_events_sorted)
    ]
    events += [(e["t"], "command", e) for e in raw_commands]
    events.sort(key=lambda e: e[0])

    command_rows: List[dict] = []
    for _, kind, payload in events:
        if kind == "reset":
            seed, step = payload
            command_rows.append(rb._reset_row(seed, step, time.time() - session._session_start))
        else:
            command_rows.append(rb._command_row(payload))

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_compact(run_dir / "states.jsonl", state_rows)
    _write_jsonl_compact(run_dir / "commands.jsonl", command_rows)


# ---------------------------------------------------------------------------
# Synthetic echo injection for A cycles (assignment A1: "A cycles carry
# synthetic echo injection, as in the existing A-like fixtures. ... Document
# exactly what is injected.") Mutates session.stub.commands IN PLACE, the
# same technique FlightSim._emit_command(forced=...) uses (make_fixtures.py):
# a right-arm joint's target this tick is overwritten with a float32-exact
# copy of a REAL earlier state's reported position for that joint, so the
# resulting command is bit-exact against a genuine past sample -- exactly
# the shape verify_goal_feedback/echo.py's exact-match test looks for.
# ---------------------------------------------------------------------------

@dataclass
class InjectedEcho:
    stub_command_index: int
    joint_name: str
    joint_index: int
    source_state_wall_time_ns: int
    value_rad: float


ECHO_MARGIN_TICKS = 8          # keep clear of goto edges (near-end exemption)
ECHO_STRIDE = 5                 # inject at most every Nth eligible index
ECHO_LOOKBACK_S = (0.10, 0.40)  # source state must be this far in the past
ECHO_MIN_GAP_S = 0.02           # never the immediately preceding sample


def inject_synthetic_echoes(
    session: "rb.V3Session", first_command_index: int, last_command_index: int,
    *, seed: int, max_injections: int = 24,
) -> List[InjectedEcho]:
    """Injects into ``session.stub.commands[first_command_index:last_command_index]``
    (the setup leg's own raw stub-log index range, from
    ``LegBounds.move_calls``). Skips the first/last ``ECHO_MARGIN_TICKS`` of
    that range (mirrors ``FlightSim``'s own near-edge exclusion -- see its
    docstring), tries one right-arm joint per eligible index (round-robin
    over ``ARM8``), and only commits a candidate that is NOT already equal
    to the previous command's own target for that joint (would be
    indistinguishable from an ordinary carry, per ``FlightSim.fly``'s own
    same-value skip)."""
    rng = random.Random(seed)
    injected: List[InjectedEcho] = []
    with session.stub.lock:
        commands = session.stub.commands
        states = session.stub.states
    if not states:
        return injected

    lo = first_command_index + ECHO_MARGIN_TICKS
    hi = last_command_index - ECHO_MARGIN_TICKS
    if hi <= lo:
        return injected

    joint_cycle = 0
    for i in range(lo, hi, ECHO_STRIDE):
        if len(injected) >= max_injections:
            break
        cmd_t = commands[i]["t"]
        jname = ARM8[joint_cycle % len(ARM8)]
        joint_cycle += 1
        jidx = IDX[jname]

        window = [s for s in states
                  if ECHO_MIN_GAP_S <= (cmd_t - s["t"]) <= ECHO_LOOKBACK_S[1]
                  and (cmd_t - s["t"]) >= ECHO_LOOKBACK_S[0]]
        if not window:
            continue
        src = rng.choice(window)
        candidate = float(np.float32(src["pos"][jidx]))
        prev_val = commands[i - 1]["target"][jidx] if i > 0 else None
        if prev_val is not None and candidate == prev_val:
            continue  # would look like an ordinary carry -- skip (documented)
        commands[i]["target"][jidx] = candidate
        injected.append(InjectedEcho(i, jname, jidx, int(src["t"] * 1e9), candidate))
    return injected


# ---------------------------------------------------------------------------
# Real native reset keyframe (coordinator guidance, 2026-09-25): source the
# initial/reset pose NativeStub is seeded with from native_mujoco's own
# keyframe, not R.HOME (see module docstring). NativeStub is not modified;
# only its own documented `pose=` constructor kwarg is used.
# ---------------------------------------------------------------------------

#: native_mujoco/model/reachy_1_2.xml:465-469, <keyframe><key name="home"
#: qpos="..."/></keyframe> -- 21 values, MJCF/JOINT_TABLE mjcf_index order,
#: radians. Cited verbatim, never hand-derived.
NATIVE_HOME_QPOS: Tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.45, 0.0, 0.0, 0.0,
)


def _native_home_pose_rad() -> Dict[str, float]:
    from joint_map import JOINT_TABLE
    by_idx = {e.mjcf_index: e.sdk_name for e in JOINT_TABLE}
    return {by_idx[i]: NATIVE_HOME_QPOS[i] for i in range(21)}


def build_stage_a_session() -> "rb.V3Session":
    """``real_bridge.build_session``'s own body, with the REAL native reset
    keyframe (see module docstring) in place of ``R.HOME`` -- everything
    else (``NativeStub``/``Observer`` construction, the ready-wait) is
    unchanged from Stage 1's own function, which is not modified."""
    session_start = time.monotonic()
    stub = rb.NativeStub(pose=_native_home_pose_rad(), bias=None, compliant=True).start()
    observer = rb.Observer(stub, session_start).start()
    rb._wait(lambda: len(observer.snapshot()) > 3, what="observer receiving states")
    return rb.V3Session(stub, observer, _session_start=session_start)


# ---------------------------------------------------------------------------
# The 4-cycle Stage A scenario itself
# ---------------------------------------------------------------------------

@dataclass
class StageACycle:
    name: str
    rep: int
    arm: str
    bounds: "rb.CycleBounds"
    bridge_log_path: Path
    reset_gen: int
    reset_record_path: Path
    reset_snapshot_stdout: str
    reset_verify_stdout: str
    reset_verify_rc: int
    injected_echoes: List[InjectedEcho] = field(default_factory=list)
    turn_on_state_msg: Optional[dict] = None   # raw observer msg, right after this cycle's own turn_on
    recreate_timestamp: float = 0.0


@dataclass
class StageASession:
    session: "rb.V3Session"
    cycles: List[StageACycle]
    run_dir: Path
    control_dir: Path


def run_stage_a_session(
    ev_dir: Path, control_dir: Path, *, arm_labels: Sequence[str] = ARM_LABELS,
    cycle_names: Optional[Sequence[str]] = None, echo_seed: int = 20260925,
) -> StageASession:
    """Assignment A1: 4 cycles, reps 1-4, order A B B A. Each cycle: a
    container-recreate equivalent (new backend/bridge/gRPC server against
    the SAME NativeStub), a reset through the bridge's own reset path (with
    a REAL reset_verify.py snapshot/verify pair around it), a fresh client +
    turn_on + fly_route(PLACE_ROUTE), then a fresh client + turn_on +
    fly_route(LIFT_TO_PRESENT). Clients are closed before each recreate
    (ruling A4, already implemented by ``BridgeGen.close``)."""
    cycle_names = list(cycle_names or [f"S2-B4-c-r{i}" for i in range(1, len(arm_labels) + 1)])
    run_dir = ev_dir / "e1_server_runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    session = build_stage_a_session()
    gen0 = session.new_gen()
    c0 = gen0.client()
    c0.turn_on("r_arm")
    time.sleep(0.3)

    cycles: List[StageACycle] = []

    for rep, (name, arm) in enumerate(zip(cycle_names, arm_labels), start=1):
        bridge_log_path = control_dir / f"bridge_log_{name}.txt"
        with BridgeLogCapture(bridge_log_path):
            session.gens[-1].close()
            recreate_timestamp = time.time()
            gen = session.new_gen()

            pre_reset_sim_step = session.observer.last_sim_step()
            reset_gen = rep  # == the epoch this reset starts (Stage 0 is epoch 0)
            reset_t = time.monotonic()

            _flush_partial_run_dir(session, run_dir, session.reset_events)
            snap_rc, snap_out, snap_err = run_reset_verify(["snapshot", str(run_dir)])
            if snap_rc != 0:
                raise AssertionError(f"reset_verify.py snapshot failed rc={snap_rc}: {snap_out}{snap_err}")

            gen.reset()
            session.reset_events.append((reset_t, reset_gen, pre_reset_sim_step))
            time.sleep(0.15)

            _flush_partial_run_dir(session, run_dir, session.reset_events)
            gen_label = str(reset_gen)
            verify_rc, verify_out, verify_err = run_reset_verify(
                ["verify", str(run_dir), gen_label, gen_label, snap_out.strip()])
            reset_record_path = control_dir / f"reset_{gen_label}.txt"
            # "capturing their stdout exactly as reset.sh would" -- reset.sh
            # echoes verify's own stdout line into the record; a failure's
            # stdout is empty, so the reason (its stderr) is appended,
            # documented, never invented as a synthetic STOP: line.
            record_text = verify_out
            if verify_rc != 0:
                record_text += f"# reset_verify.py exited {verify_rc}: {verify_err.strip()}\n"
            reset_record_path.write_text(record_text)

            setup_first_seq = session.observer.last_seq()
            c_setup = gen.client()
            c_setup.turn_on("r_arm")
            time.sleep(0.3)
            turn_on_state_msg = session.observer.snapshot()[-1]
            setup_moves: List["rb.MoveCall"] = []
            flown_setup = rb._fly_or_record(session, c_setup.r_arm, R.PLACE_ROUTE,
                                             f"{name}-setup", setup_moves)
            setup_last_seq = session.observer.last_seq()

            injected: List[InjectedEcho] = []
            if arm == "A" and setup_moves:
                # Target the plan §7.1 AFFECTED SEGMENT itself (the last
                # HOVER-assigned setpoint through the HOVER->REST_SHUT goto
                # and its hold), not the whole leg -- so the injected
                # echoes actually land where genuine_echo_count/echo._subclassify
                # look, exercising that check meaningfully rather than by
                # chance. Falls back to the whole leg if HOVER/REST_SHUT
                # were not both flown (documented; would itself be a
                # finding -- see the handoff).
                if "HOVER" in flown_setup and "REST_SHUT" in flown_setup:
                    h = flown_setup.index("HOVER")
                    r = flown_setup.index("REST_SHUT")
                    first_ci = setup_moves[h].first_command_index
                    last_ci = setup_moves[r].last_command_index
                else:
                    first_ci = setup_moves[0].first_command_index
                    last_ci = setup_moves[-1].last_command_index
                injected = inject_synthetic_echoes(
                    session, first_ci, last_ci, seed=echo_seed + rep)

            time.sleep(0.3)
            flight_first_seq = session.observer.last_seq()
            c_flight = gen.client()
            c_flight.turn_on("r_arm")
            time.sleep(0.3)
            flight_moves: List["rb.MoveCall"] = []
            flown_flight = rb._fly_or_record(session, c_flight.r_arm, R.LIFT_TO_PRESENT,
                                              f"{name}-flight", flight_moves)
            flight_last_seq = session.observer.last_seq()

            bounds = rb.CycleBounds(
                setup=rb.LegBounds(setup_first_seq, setup_last_seq, flown_setup, setup_moves),
                flight=rb.LegBounds(flight_first_seq, flight_last_seq, flown_flight, flight_moves),
                reset_sim_step=pre_reset_sim_step, reset_seed=reset_gen)
            session.cycles.append(bounds)

            cycles.append(StageACycle(
                name=name, rep=rep, arm=arm, bounds=bounds, bridge_log_path=bridge_log_path,
                reset_gen=reset_gen, reset_record_path=reset_record_path,
                reset_snapshot_stdout=snap_out, reset_verify_stdout=verify_out,
                reset_verify_rc=verify_rc, injected_echoes=injected,
                turn_on_state_msg=turn_on_state_msg, recreate_timestamp=recreate_timestamp))

    return StageASession(session=session, cycles=cycles, run_dir=run_dir, control_dir=control_dir)


# ---------------------------------------------------------------------------
# Evidence emission (plan §6 layout, assignment A2)
# ---------------------------------------------------------------------------

@dataclass
class StageAEvidence:
    ev_dir: Path
    run_dir: Path
    control_dir: Path
    sha256sums_root: Path
    derived_sha256sums: Path   # H6 workaround, see module docstring / handoff
    native_log_path: Path
    arm_map_path: Path
    sidecars: Dict[str, Dict[str, Path]]   # cycle -> {"setup": path, "flight": path}
    manifests: Dict[str, Path]
    versions: Dict[str, Path]
    prep: Dict[str, Path]
    start_variant: Dict[str, Path]


def _scene_block() -> dict:
    import e1_identity  # noqa: E402
    chain = e1_identity._extends_chain(SCENE_PATH)
    chain_sha256 = {str(p): e1_identity._sha256_file(p) for p in chain}
    return {"path": str(SCENE_PATH.resolve()), "chain_sha256": chain_sha256,
            "board_object_ids": ["pool_box_1"]}


def _write_sidecar(control_dir: Path, name: str, kind: str, route_name: str, ts: str,
                    first_seq: int, last_seq: int, run_dir: Path,
                    scene_block: Optional[dict] = None) -> Path:
    doc = {
        "alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}],
        "log": f"route_clearance_{route_name}_{ts}.log",
        "server_run_dir": str(run_dir.resolve()),
    }
    if scene_block is not None:
        doc["scene"] = scene_block
    path = control_dir / f"route_clearance_{route_name}_{ts}.link.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def _git(*args: str) -> Tuple[int, str, str]:
    proc = subprocess.run(["git", *args], cwd=str(_REPO), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def emit_stage_a_evidence(stage: StageASession, ev_dir: Path) -> StageAEvidence:
    """Writes the plan §6 layout: ``<ev>/e1_server_runs/<run>/{commands,states}.jsonl``,
    a ROOT ``SHA256SUMS`` keyed ``./e1_server_runs/<run>/...``, linker-shaped
    sidecars (with a real B4 scene block on the setup sidecar), and every
    ``control/`` input the shipped ``cycle``/``summary`` CLIs read outside
    ``--validation-mode``: ``cycle_<id>.json`` manifests, ``versions_<cycle>.json``,
    ``prep_<cycle>.json`` (compliance), ``start_variant_<cycle>.json``, a
    12-entry ``arm_map.json``, and an EMPTY placeholder native log (NativeStub
    emits no native-process lines at all -- documented, never fabricated)."""
    session = stage.session
    run_dir = stage.run_dir
    control_dir = stage.control_dir

    # Final, full write of the run directory (compact JSON; see module
    # docstring). This subsumes every partial flush already written --
    # `run_stage_a_session` appended each reset directly onto
    # `session.reset_events` (the same field `real_bridge.run_two_cycles`
    # populates), so the merge here is exactly `_flush_partial_run_dir`'s
    # own algorithm, over the now-complete event set.
    with session.stub.lock:
        raw_commands = list(session.stub.commands)
    state_rows = [rb._state_row(m) for m in session.observer.snapshot()]
    assert len(session.reset_events) == len(stage.cycles), (
        len(session.reset_events), len(stage.cycles))
    reset_events_sorted = sorted(session.reset_events, key=lambda e: e[0])
    closing_steps = rb._closing_epoch_last_steps(state_rows)
    assert len(closing_steps) == len(reset_events_sorted), (
        len(closing_steps), len(reset_events_sorted))

    events: List[tuple] = [
        (t, "reset", (seed, closing_steps[k]))
        for k, (t, seed, _live) in enumerate(reset_events_sorted)
    ]
    events += [(e["t"], "command", e) for e in raw_commands]
    events.sort(key=lambda e: e[0])
    command_rows: List[dict] = []
    for _, kind, payload in events:
        if kind == "reset":
            seed, step = payload
            command_rows.append(rb._reset_row(seed, step, time.time() - session._session_start))
        else:
            command_rows.append(rb._command_row(payload))

    states_path = run_dir / "states.jsonl"
    commands_path = run_dir / "commands.jsonl"
    _write_jsonl_compact(states_path, state_rows)
    _write_jsonl_compact(commands_path, command_rows)

    # Root SHA256SUMS, keyed ./e1_server_runs/<run>/... (assignment A2 / H6).
    rel_states = f"./e1_server_runs/{RUN_NAME}/states.jsonl"
    rel_commands = f"./e1_server_runs/{RUN_NAME}/commands.jsonl"
    sha_root = ev_dir / "SHA256SUMS"
    sha_root.write_text(
        f"{sha256_of(states_path)}  {rel_states}\n"
        f"{sha256_of(commands_path)}  {rel_commands}\n")

    # H6 workaround (documented obstacle, see handoff): a DERIVED digest
    # file inside the run directory itself, prefix-stripped, so the CLI can
    # be run with --ev-dir == the run directory (needed for the sidecar
    # server_run_dir check) while still verifying real bytes -- mirrors the
    # coordinator's own V1 approach (v1-rerun-b4-s1.md, "derived-SHA256SUMS").
    derived = run_dir / "derived-SHA256SUMS"
    derived.write_text(
        f"{sha256_of(states_path)}  states.jsonl\n"
        f"{sha256_of(commands_path)}  commands.jsonl\n")

    native_log_path = control_dir / "native_session.log"
    native_log_path.write_text("")  # NativeStub emits no native-process lines (documented placeholder)

    sidecars: Dict[str, Dict[str, Path]] = {}
    manifests: Dict[str, Path] = {}
    versions: Dict[str, Path] = {}
    prep: Dict[str, Path] = {}
    start_variant: Dict[str, Path] = {}

    scene_block = _scene_block()

    for cyc in stage.cycles:
        ts = f"{RUN_NAME}-{cyc.rep:02d}"
        setup_sidecar = _write_sidecar(
            control_dir, cyc.name, "setup", "PLACE_ROUTE", ts,
            cyc.bounds.setup.first_seq, cyc.bounds.setup.last_seq, run_dir,
            scene_block=scene_block)
        flight_sidecar = _write_sidecar(
            control_dir, cyc.name, "flight", "LIFT_TO_PRESENT", ts,
            cyc.bounds.flight.first_seq, cyc.bounds.flight.last_seq, run_dir)
        sidecars[cyc.name] = {"setup": setup_sidecar, "flight": flight_sidecar}

        manifest = {
            "rep": cyc.rep, "cycle": cyc.name,
            "setup_sidecar": setup_sidecar.name, "flight_sidecar": flight_sidecar.name,
            "reset_gen": cyc.reset_gen, "reset_record": cyc.reset_record_path.name,
            "bridge_log": cyc.bridge_log_path.name,
        }
        manifest_path = control_dir / f"cycle_{cyc.name}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        manifests[cyc.name] = manifest_path

        versions[cyc.name] = _write_versions(control_dir, cyc)
        prep[cyc.name] = _write_prep(control_dir, cyc)
        start_variant[cyc.name] = _write_start_variant(control_dir, cyc)

    arm_map_path = control_dir / "arm_map.json"
    arm_map_path.write_text(json.dumps(_build_arm_map(), indent=2))

    return StageAEvidence(
        ev_dir=ev_dir, run_dir=run_dir, control_dir=control_dir,
        sha256sums_root=sha_root, derived_sha256sums=derived,
        native_log_path=native_log_path, arm_map_path=arm_map_path,
        sidecars=sidecars, manifests=manifests, versions=versions,
        prep=prep, start_variant=start_variant)


# ---------------------------------------------------------------------------
# arm_map.json / versions_<cycle>.json / prep_<cycle>.json / start_variant_<cycle>.json
# ---------------------------------------------------------------------------

_OPT_FILES = {
    "/opt/fake_reachy_server.py": _REPO / "fake_reachy_server.py",
    "/opt/mujoco_remote_backend.py": _REPO / "mujoco_remote_backend.py",
    "/opt/reset_watcher.py": _REPO / "reset_watcher.py",
}


def _real_opt_hashes() -> Dict[str, str]:
    """Real sha256 of the ACTUAL checked-out files this harness imports.
    Both arms run this SAME checkout (no docker image swap in Stage A), so
    these are identical for A and B -- only the arm_map's bridge_sha/tag/id
    LABELS differ (see the module docstring)."""
    return {opt_path: sha256_of(real_path) for opt_path, real_path in _OPT_FILES.items()}


def _image_identity(arm: str) -> Tuple[str, str]:
    """Placeholder image tag/id (documented): no docker image is built in
    Stage A (forbidden). These exist only so validate_arm_map's identity
    tuple has real, distinct, well-formed strings to compare -- never
    checked against an actual `docker inspect`."""
    tag = f"cmp-{arm}-{BRIDGE_SHA[arm][:8]}"
    image_id = f"sha256:{('0' * 58)}{arm.lower()}{'0' * 5}"
    return tag, image_id


def _build_arm_map() -> List[dict]:
    opt_hashes = _real_opt_hashes()
    entries = []
    for rep in range(1, 13):
        arm = pv.ARM_MAP_ORDER[rep - 1]
        tag, image_id = _image_identity(arm)
        entries.append({
            "rep": rep, "arm": arm, "image_tag": tag, "image_id": image_id,
            "bridge_sha": BRIDGE_SHA[arm], "opt_hashes": opt_hashes,
        })
    return entries


REQUIRED_SUPERVISOR_PROGRAMS = ["reachy-sdk-server"]


def _write_versions(control_dir: Path, cyc: StageACycle) -> Path:
    rc, out, err = _git("rev-parse", "HEAD")
    host_sha = out.strip() if rc == 0 else ""
    rc2, out2, _ = _git("status", "--porcelain")
    host_dirty = bool(out2.strip()) if rc2 == 0 else True
    tag, image_id = _image_identity(cyc.arm)
    doc = {
        "host_native_kernel_sha": host_sha, "host_tree_dirty": host_dirty,
        "bridge_arm": cyc.arm, "bridge_sha": BRIDGE_SHA[cyc.arm],
        "running_image_id": image_id, "opt_hashes": _real_opt_hashes(),
        "supervisor_start_times": {
            REQUIRED_SUPERVISOR_PROGRAMS[0]: cyc.recreate_timestamp + 1.0},
        "recreate_timestamp": cyc.recreate_timestamp,
    }
    path = control_dir / f"versions_{cyc.name}.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


#: plan §4: r_arm (8 joints) stiff, both antennas stiff by default, every
#: other joint compliant by default. NativeStub itself does not reproduce
#: the antenna default (see module docstring) -- this is the REAL plan
#: definition, not tuned to match the harness, so the antenna entries are
#: EXPECTED to mismatch "actual" on every cycle (a documented finding, not
#: a workaround target).
def _expected_compliant21() -> List[bool]:
    out = [True] * 21
    for name in ARM8:
        out[IDX[name]] = False
    out[IDX["l_antenna"]] = False
    out[IDX["r_antenna"]] = False
    return out


def _write_prep(control_dir: Path, cyc: StageACycle) -> Path:
    msg = cyc.turn_on_state_msg
    actual = [False] * 21
    if msg is not None:
        by_name = {j["name"]: j["compliant"] for j in msg["joints"]}
        from make_fixtures import JOINT_ORDER
        actual = [bool(by_name.get(n, True)) for n in JOINT_ORDER]
    doc = {"cycle": cyc.name, "expected_compliant21": _expected_compliant21(),
           "actual_compliant21": actual}
    path = control_dir / f"prep_{cyc.name}.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def _write_start_variant(control_dir: Path, cyc: StageACycle) -> Path:
    """Real classification of the REAL pose observed at this cycle's own
    turn_on. Seeded from the real native keyframe (see module docstring:
    ``build_stage_a_session``), so this is EXPECTED to classify as
    "stiff-zero" on every cycle -- a real pass, not a tuned one."""
    import sys as _sys
    sys_path_added = str(_REPO / "scripts") not in _sys.path
    if sys_path_added:
        _sys.path.insert(0, str(_REPO / "scripts"))
    from e1_stage1 import plan as _plan

    msg = cyc.turn_on_state_msg
    pose_deg = {}
    if msg is not None:
        by_name = {j["name"]: j["position_rad"] for j in msg["joints"]}
        for name in ARM8:
            pose_deg[name] = float(np.degrees(by_name.get(name, 0.0)))
    declared = _plan.classify_start_variant(pose_deg)
    doc = {"cycle": cyc.name, "pose": pose_deg, "start_variant": declared}
    path = control_dir / f"start_variant_{cyc.name}.json"
    path.write_text(json.dumps(doc, indent=2))
    return path
