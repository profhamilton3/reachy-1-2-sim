"""V3 real-bridge fixture (Stage 1 assignment, 2026-09-25, §2): the ONLY
in-repo module that drives

    reachy_sdk 0.7.0 -> loopback gRPC -> fake_reachy_server.FakeJointService
      -> KinematicBridge(MujocoRemoteBackend) -> tests/integration/native_stub.NativeStub

and turns what it observes into native-recorder-schema ``commands.jsonl`` /
``states.jsonl`` plus linker-style ``<cycle>-{setup,flight}.link.json``
sidecars, so the shipped ``cycle`` CLI can be run against a real bridge
session instead of a synthetic ``FlightSim``.

Reuses the CONSTRUCTION PATTERN (not the fixture function) of
``tests/integration/test_goal_reporting_sdk_path.py:92-150`` (``Rig``,
``make_rig``). Every product module (``fake_reachy_server``,
``mujoco_remote_backend``, ``kinematic_backend``, ``native_stub``) is
imported, never modified, never subclassed to change behaviour.

No Docker, no native MuJoCo server, no hardware, no motion. Loopback
sockets only. Nothing here reads recorded evidence.

DOCUMENTED PLACEHOLDERS (native_stub.py's toy plant cannot supply these --
excluded from every F1-F7 assertion, never invented as data):

  * ``wall_time_ns`` on every state row is the OBSERVER's own receipt-time
    ``time.monotonic_ns()``, not a native write-time stamp -- a genuine
    measurement, but of different origin than the real recorder's field of
    the same name (which stamps at the moment native pushes the state, one
    hop earlier than a websocket client's own recv()). Locates windows only
    (per the ``States`` docstring); never used for an exact-timing
    assertion here.
  * ``scene_revision``: ``""`` (``NativeStub`` never sets one).
  * ``paused``: always ``False`` (``NativeStub`` has no pause handler --
    this harness cannot produce a paused state; it is not evidence that a
    real session never pauses).
  * ``objects`` / ``grippers`` / ``force_sensors`` / ``interactive`` /
    ``warnings`` / ``contacts``: ``[]`` (``NativeStub``'s ``state`` message
    carries none of these fields).
  * A reset row's ``wall_time_s`` is wall-clock elapsed since THIS SESSION's
    own start, not the real recorder's ``_started_at``-relative figure
    (which this harness has no access to). ``evidence.py`` never reads this
    field; it is written only so the row matches the recorder's own key set.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
for _p in (_REPO, _REPO / "src", _REPO / "native_mujoco", _REPO / "tests" / "integration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_fixtures as mf  # noqa: E402  (write_evidence: exact recorder schema, shared helper)

# Product modules -- imported, never modified.
import grpc  # noqa: E402
from reachy_sdk import ReachySDK  # noqa: E402
from reachy_sdk_api import fan_pb2_grpc, joint_pb2_grpc, sensor_pb2_grpc  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import mujoco_remote_backend as mrb  # noqa: E402
from fake_reachy_server import FakeFanService, FakeJointService, FakeSensorService  # noqa: E402
from mujoco_remote_backend import KinematicBridge, MujocoRemoteBackend  # noqa: E402
from native_stub import NativeStub  # noqa: E402

from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.tasks import rig_motion  # noqa: E402
from tools.goalfix_cmp._units import pose8_rad  # noqa: E402


# ---------------------------------------------------------------------------
# Observer: a raw websocket client on the stub's own connection list --
# hello -> hello_ack, then every `state` message, per assignment §2.2
# ("an observer websocket client ... does hello -> hello_ack, and records
# every `state` message").
# ---------------------------------------------------------------------------

class Observer:
    def __init__(self, stub: NativeStub, session_start_monotonic: float):
        self.stub = stub
        self._session_start = session_start_monotonic
        self.states: List[dict] = []   # raw wire dicts + "_observed_wall_time_ns"
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_evt: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> "Observer":
        self._thread = threading.Thread(target=self._run, daemon=True, name="v3-observer")
        self._thread.start()
        assert self._ready.wait(5), "V3 observer did not connect/hello_ack in time"
        return self

    def stop(self) -> None:
        if self._loop is not None and self._stop_evt is not None:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        if self._thread is not None:
            self._thread.join(5)

    def _run(self) -> None:
        import websockets

        loop = asyncio.new_event_loop()
        self._loop = loop

        async def main():
            self._stop_evt = asyncio.Event()
            async with websockets.connect(self.stub.url()) as ws:
                await ws.send(json.dumps(
                    {"type": "hello", "protocol_version": 1, "client_id": "v3-observer"}))
                ack = json.loads(await ws.recv())
                assert ack.get("type") == "hello_ack", ack
                self._ready.set()
                while not self._stop_evt.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
                    msg = json.loads(raw)
                    if msg.get("type") == "state":
                        msg["_observed_wall_time_ns"] = time.monotonic_ns()
                        with self._lock:
                            self.states.append(msg)

        loop.run_until_complete(main())
        loop.close()

    def snapshot(self) -> List[dict]:
        with self._lock:
            return list(self.states)

    def last_seq(self) -> Optional[int]:
        with self._lock:
            return self.states[-1]["seq"] if self.states else None

    def last_sim_step(self) -> Optional[int]:
        with self._lock:
            return self.states[-1]["sim_step"] if self.states else None


def _wait(pred, timeout=5.0, what="condition"):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


# ---------------------------------------------------------------------------
# A bridge "generation": one MujocoRemoteBackend + KinematicBridge + gRPC
# server against the SAME NativeStub -- the container-recreate equivalent
# (assignment §2.3 step 2). Stopping one and building a fresh one is the
# recreate; the NativeStub (native's stand-in) and Observer are untouched.
# ---------------------------------------------------------------------------

class BridgeGen:
    def __init__(self, stub: NativeStub):
        self.stub = stub
        self.backend = MujocoRemoteBackend(url=stub.url())
        self.backend.start()
        _wait(lambda: self.backend.connection_state == mrb.ConnectionState.READY,
              what="new bridge generation connected")
        self.server = grpc.server(ThreadPoolExecutor(max_workers=16))
        joint_backend = KinematicBridge(self.backend)
        joint_pb2_grpc.add_JointServiceServicer_to_server(FakeJointService(joint_backend), self.server)
        sensor_pb2_grpc.add_SensorServiceServicer_to_server(FakeSensorService(self.backend), self.server)
        fan_pb2_grpc.add_FanControllerServiceServicer_to_server(FakeFanService(), self.server)
        self.port = self.server.add_insecure_port("127.0.0.1:0")
        self.server.start()
        self.clients: List[ReachySDK] = []

    def client(self) -> ReachySDK:
        c = ReachySDK(host="127.0.0.1", sdk_port=self.port, camera_port=self.port,
                      restart_port=self.port)
        time.sleep(0.2)
        self.clients.append(c)
        return c

    def reset(self, timeout: float = 5.0) -> None:
        evt = self.backend.request_reset()
        assert evt.wait(timeout), "no reset_ack from the bridge's own reset path"
        _wait(lambda: self.backend._last_target is not None, what="post-reset baseline")

    def close(self) -> None:
        # A4 (coordinator ruling, 2026-09-25 stage-2a rulings addendum):
        # close every client's own gRPC channel (the SDK's own object,
        # `ReachySDK._grpc_channel` -- standard grpc API, not an SDK
        # patch) BEFORE tearing down the server/backend it talks to.
        # Left open, each client's background `_sync_loop` thread spins
        # a tight BlockingIOError reconnect loop against the now-dead
        # port once the server stops -- CPU contention inside the
        # harness that the coordinator's independent re-run traced the
        # C2 tau-spread failures to (stalls, not bridge behaviour).
        for c in self.clients:
            try:
                c._grpc_channel.close()
            except Exception:
                pass
        self.server.stop(0)
        self.backend.stop()


# ---------------------------------------------------------------------------
# Row conversion (native-recorder schema; native_mujoco/protocol.py +
# native_mujoco/recorder.py) -- assignment §2.2 "Field layout".
# ---------------------------------------------------------------------------

def _state_row(msg: dict) -> dict:
    joints = [
        {"name": j["name"], "uid": j.get("uid", 0), "position_rad": j["position_rad"],
         "velocity_rad_s": j.get("velocity_rad_s", 0.0), "effort": j.get("effort", 0.0),
         "compliant": j["compliant"]}
        for j in msg["joints"]
    ]
    return {
        "type": "state", "seq": msg["seq"], "sim_step": msg["sim_step"],
        "sim_time_s": msg["sim_time_s"],
        "wall_time_ns": msg["_observed_wall_time_ns"],           # PLACEHOLDER, see module docstring
        "cmd_seq": msg["cmd_seq"],
        "scene_revision": "",                                     # PLACEHOLDER
        "paused": False,                                          # PLACEHOLDER
        "joints": joints,
        "objects": [], "grippers": [], "force_sensors": [],       # PLACEHOLDERs
        "interactive": [], "warnings": [], "contacts": [],        # PLACEHOLDERs
    }


def _command_row(entry: dict) -> dict:
    return {
        "type": "joint_command", "seq": entry["seq"], "target_rad": list(entry["target"]),
        "mask": None, "compliant": entry.get("compliant"),
        "speed_limit_rad_s": None, "torque_limit_percent": None,
    }


def _reset_row(seed: Optional[int], sim_step: int, wall_time_s: float) -> dict:
    return {"type": "reset", "seed": seed, "sim_step": sim_step, "wall_time_s": wall_time_s}  # wall_time_s: PLACEHOLDER


def write_sidecar(control_dir: Path, cycle: str, kind: str, first_seq: int, last_seq: int,
                   *, log_name: str, server_run_dir: str = "v3-real-bridge") -> Path:
    """As far as the harness allows, in the shape ``scripts/link_e1_flight.py``
    writes (assignment §2.2 "Sidecars"): the ``alignment`` list of
    ``server_seq``s that ``evidence.leg_from_sidecar`` reads, plus ``log``
    (route name) and ``server_run_dir`` -- fields CB3 says a real cycle
    manifest must be checked against. ``scene``/``chain_sha256`` are left
    out entirely (documented as absent, never a fabricated placeholder --
    this harness has no board scene)."""
    path = control_dir / f"{cycle}-{kind}.link.json"
    path.write_text(json.dumps({
        "alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}],
        "log": log_name,
        "server_run_dir": server_run_dir,
    }))
    return path


# ---------------------------------------------------------------------------
# Scenario orchestration (assignment §2.3)
# ---------------------------------------------------------------------------

@dataclass
class MoveCall:
    pose: Dict[str, float]
    seconds: float
    first_command_index: int   # index into NativeStub.commands (stub-log order)
    last_command_index: int    # exclusive


@dataclass
class LegBounds:
    first_seq: int
    last_seq: int
    flown: List[str] = field(default_factory=list)
    #: One entry per ``move()`` call ``fly_route`` actually issued for this
    #: leg (the initial goto AND every re-stream pass, in order) --
    #: threaded through ``fly_route``'s own ``move=`` parameter (an
    #: existing, unmodified extension point), never by re-implementing
    #: `fly_route`'s retry loop. F1/F2/F3 group these by identical
    #: ``pose`` (a retry repeats the SAME target) to recover per-waypoint
    #: pass counts.
    move_calls: List[MoveCall] = field(default_factory=list)


@dataclass
class CycleBounds:
    setup: LegBounds
    flight: LegBounds
    reset_sim_step: int
    reset_seed: Optional[int]


@dataclass
class V3Session:
    stub: NativeStub
    observer: Observer
    gens: List[BridgeGen] = field(default_factory=list)
    cycles: List[CycleBounds] = field(default_factory=list)
    findings: Dict[str, object] = field(default_factory=dict)  # RouteError etc.
    #: (monotonic_time, seed, sim_step_in_force) at each bridge-triggered
    #: reset this session ran -- `time.monotonic()`, the SAME clock
    #: `NativeStub.commands[i]["t"]` uses, so the two streams can be
    #: merged in true issue order (assignment §2.2's "the sim_step in
    #: force when the reset is applied"; this harness's own equivalent of
    #: the recorder writing the reset row inline as it happens).
    reset_events: List[tuple] = field(default_factory=list)
    _session_start: float = 0.0

    def new_gen(self) -> BridgeGen:
        g = BridgeGen(self.stub)
        self.gens.append(g)
        return g

    def close(self) -> None:
        for g in self.gens:
            g.close()
        self.observer.stop()
        self.stub.stop()


def build_session() -> V3Session:
    session_start = time.monotonic()
    # NativeStub's own `pose` kwarg is in RADIANS (it stands in for the
    # native process, which never sees a degree value) -- R.HOME is an SDK
    # DEGREE pose (assignment invariant 4), so it must go through the same
    # route_rad/pose8_rad boundary as everything else, never assigned
    # unconverted. Passing degree values here directly is exactly the
    # M1/T1 units bug this harness's own evidence must not reproduce.
    stub = NativeStub(pose=pose8_rad(R.HOME), bias=None, compliant=True).start()
    observer = Observer(stub, session_start).start()
    _wait(lambda: len(observer.snapshot()) > 3, what="observer receiving states")
    return V3Session(stub, observer, _session_start=session_start)


def _make_logging_move(session: V3Session, move_calls: List[MoveCall]):
    """``fly_route``'s own ``move=`` extension point (unmodified signature,
    default ``None`` -> ``sdk_move``) -- wraps the REAL ``rig_motion.
    sdk_move`` unchanged, and records each call's command-log bracket so
    F1/F2/F3 can identify every re-stream pass fly_route actually issued,
    without re-implementing its retry loop."""
    def move(arm, pose, seconds):
        with session.stub.lock:
            a = len(session.stub.commands)
        rig_motion.sdk_move(arm, pose, seconds)
        time.sleep(0.12)   # let the bridge's 20 ms sends land (same margin
        # test_goal_reporting_sdk_path.py's own Recorder uses for the
        # identical reason -- goto() can return slightly before its last
        # queued command has reached the stub through the async bridge).
        with session.stub.lock:
            b = len(session.stub.commands)
        move_calls.append(MoveCall(dict(pose), seconds, a, b))
    return move


def _fly_or_record(session: V3Session, arm, route, key: str,
                    move_calls: List[MoveCall]) -> List[str]:
    """Calls ``rig_motion.fly_route`` with its PRODUCT DEFAULTS
    (settle_s/retries/settle_pass_s untouched, per assignment §2.3 step 4) --
    never overridden here. A ``RouteError`` (toy-plant sag/bias) is recorded
    as a finding, never masked."""
    try:
        return rig_motion.fly_route(arm, route, move=_make_logging_move(session, move_calls))
    except rig_motion.RouteError as exc:
        session.findings.setdefault("route_errors", []).append({"key": key, "error": str(exc)})
        return []


def run_two_cycles(session: V3Session) -> V3Session:
    """assignment §2.3: Stage 0 turn_on, then for each of 2 cycles --
    container-recreate equivalent, reset through the bridge's reset path, a
    fresh client + turn_on + fly_route(PLACE_ROUTE) with default settle_s/
    retries/settle_pass_s, then another fresh client + turn_on +
    fly_route(LIFT_TO_PRESENT)."""
    # Stage 0: turn_on on the FIRST bridge generation (pre-recreate).
    gen0 = session.new_gen()
    c0 = gen0.client()
    c0.turn_on("r_arm")
    time.sleep(0.3)

    for cyc_i in (1, 2):
        # Container-recreate equivalent: stop the previous generation, bring
        # up a new MujocoRemoteBackend/KinematicBridge/gRPC server against
        # the SAME NativeStub -- bridge seq restarts, native cmd_seq carries.
        session.gens[-1].close()
        gen = session.new_gen()

        pre_reset_sim_step = session.observer.last_sim_step()
        reset_seed = 100 + cyc_i
        reset_t = time.monotonic()
        gen.reset()
        session.reset_events.append((reset_t, reset_seed, pre_reset_sim_step))
        time.sleep(0.1)

        setup_first_seq = session.observer.last_seq()
        c_setup = gen.client()
        c_setup.turn_on("r_arm")
        time.sleep(0.3)
        setup_moves: List[MoveCall] = []
        flown_setup = _fly_or_record(session, c_setup.r_arm, R.PLACE_ROUTE, f"cycle{cyc_i}-setup", setup_moves)
        setup_last_seq = session.observer.last_seq()

        time.sleep(0.3)   # a settle before the flight leg's own turn_on (plan §5 P8)
        flight_first_seq = session.observer.last_seq()
        c_flight = gen.client()
        c_flight.turn_on("r_arm")
        time.sleep(0.3)
        flight_moves: List[MoveCall] = []
        flown_flight = _fly_or_record(session, c_flight.r_arm, R.LIFT_TO_PRESENT, f"cycle{cyc_i}-flight", flight_moves)
        flight_last_seq = session.observer.last_seq()

        session.cycles.append(CycleBounds(
            setup=LegBounds(setup_first_seq, setup_last_seq, flown_setup, setup_moves),
            flight=LegBounds(flight_first_seq, flight_last_seq, flown_flight, flight_moves),
            reset_sim_step=pre_reset_sim_step, reset_seed=reset_seed,
        ))
    return session


# ---------------------------------------------------------------------------
# Evidence emission (native-recorder schema; tmp_path only -- never
# committed, per assignment §2.2 "The generated files are not committed").
# ---------------------------------------------------------------------------

def _closing_epoch_last_steps(state_rows: Sequence[dict]) -> List[int]:
    """Per reset boundary, in occurrence order, the last ``sim_step`` of
    the epoch it closes -- exactly what
    ``evidence.check_epoch_counts`` calls ``last_step_of_closing_epoch``
    (a ``sim_step`` drop), computed here the SAME way so the reset row
    this adapter writes is guaranteed to agree with that check, rather
    than guessed at from a live sample that can race the plant's own next
    50 Hz tick between the sample and the reset actually landing."""
    out = []
    for i in range(1, len(state_rows)):
        if state_rows[i]["sim_step"] < state_rows[i - 1]["sim_step"]:
            out.append(state_rows[i - 1]["sim_step"])
    return out


def emit_evidence(session: V3Session, out_dir: Path, control_dir: Path,
                   cycle_names: Sequence[str]):
    """Writes ``states.jsonl``/``commands.jsonl``/``SHA256SUMS`` (via
    ``make_fixtures.write_evidence``, the exact-schema shared helper) plus
    one ``<cycle>-{setup,flight}.link.json`` sidecar pair per cycle in
    ``session.cycles`` (assignment §2.2/§2.3). Returns the (states_path,
    commands_path, sha_path) tuple."""
    state_rows = [_state_row(m) for m in session.observer.snapshot()]
    with session.stub.lock:
        raw_commands = list(session.stub.commands)

    # NativeStub logs commands in one flat list with no reset markers of
    # its own (unlike the real recorder, which writes the reset row inline
    # as server.py applies it -- native_mujoco/recorder.py's
    # `record_reset` call site). This harness recovers the SAME ordering
    # from wall-clock time: `session.reset_events` was stamped with
    # `time.monotonic()` at each `reset()` call, the identical clock
    # `NativeStub.commands[i]["t"]` uses (both `time.monotonic()`), so
    # merging the two streams by "t" reproduces true issue order exactly
    # -- this is a fixture-construction detail (how THIS adapter builds
    # the merged stream), not an invented recorder field. The recorded
    # `sim_step` itself comes from `_closing_epoch_last_steps`, not the
    # live sample taken when `reset()` was called (see that function).
    reset_events_sorted = sorted(session.reset_events, key=lambda e: e[0])
    closing_steps = _closing_epoch_last_steps(state_rows)
    if len(closing_steps) != len(reset_events_sorted):
        raise AssertionError(
            f"V3 harness: {len(reset_events_sorted)} reset() calls but "
            f"{len(closing_steps)} sim_step drops observed -- a reset did not "
            "land, or the observer missed a state; record this as a finding, "
            "do not paper over it with a guessed sim_step")

    events: List[tuple] = [
        (t, "reset", (seed, closing_steps[k]))
        for k, (t, seed, _live_sample) in enumerate(reset_events_sorted)
    ]
    events += [(e["t"], "command", e) for e in raw_commands]
    events.sort(key=lambda e: e[0])

    command_rows: List[dict] = []
    for _, kind, payload in events:
        if kind == "reset":
            seed, step = payload
            command_rows.append(_reset_row(seed, step, time.time() - session._session_start))
        else:
            command_rows.append(_command_row(payload))

    mf.write_evidence(out_dir, state_rows, command_rows)

    control_dir.mkdir(parents=True, exist_ok=True)
    for cyc_name, cb in zip(cycle_names, session.cycles):
        write_sidecar(control_dir, cyc_name, "setup", cb.setup.first_seq, cb.setup.last_seq,
                      log_name="route_clearance_PLACE_ROUTE_v3")
        write_sidecar(control_dir, cyc_name, "flight", cb.flight.first_seq, cb.flight.last_seq,
                      log_name="route_clearance_LIFT_TO_PRESENT_v3")
    return out_dir / "states.jsonl", out_dir / "commands.jsonl", out_dir / "SHA256SUMS"
