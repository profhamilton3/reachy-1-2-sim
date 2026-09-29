"""Child-process entry points of the U-only process-separated C2 diagnostic.

PREPARATION ONLY: nothing here has been run.  Each role is a fresh interpreter started by the
coordinator (``c2_process_split.py coordinate``) with explicit argv/env and stdout/stderr sent
to files.  Roles:

* ``N``  -- NativeStub (real keyframe pose, via ``stage_a_slice.build_stage_a_session``) +
  ``Observer`` (evidence writer) + a small request server.  Lives for the whole run.
* ``B``  -- ONE bridge generation: ``real_bridge.BridgeGen`` (``MujocoRemoteBackend`` +
  ``KinematicBridge`` + ``FakeJointService``/sensor/fan servicers + gRPC server on
  ``127.0.0.1:0``), unmodified, plus the per-cycle bridge-log capture.
* ``C``  -- the ``reachy_sdk`` 0.7.0 clients of one cycle (setup + flight) and the route calls
  (``rig_motion.fly_route`` through ``real_bridge._fly_or_record``, unmodified).

No diagnostic capture: nothing wraps, patches, subclasses or monkeypatches any ``reachy_sdk``,
``grpc``, ``fake_reachy_server``, ``mujoco_remote_backend``, ``kinematic_backend`` or
``native_stub`` code.  The only wrapper is the slice's own logging ``move=`` wrapper
(``real_bridge._make_logging_move``), reached through the documented proxy below.

Importing this module imports only the standard library and ``c2ps_common`` (heavy modules are
imported inside the role functions), so the request-side proxies can be unit tested.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import c2ps_common as cc  # noqa: E402


# ---------------------------------------------------------------------------
# Documented duck-typed proxies passed to UNMODIFIED fixture functions
# ---------------------------------------------------------------------------

class StubUrl:
    """What ``real_bridge.BridgeGen(stub)`` needs of the stub: ``stub.url()``.  BridgeGen stores
    ``stub`` and never touches it again, so in process B this stands in for the NativeStub that
    lives in process N."""

    def __init__(self, port: int):
        self._port = int(port)

    def url(self) -> str:
        return f"ws://127.0.0.1:{self._port}"


class RemoteCommands:
    """``len(stub.commands)`` performs the request to N (``count_commands``): this is the read
    ``real_bridge._make_logging_move`` makes before and after each ``sdk_move``, so that the
    counts are N's own and are never compared across processes."""

    def __init__(self, n_port: int, timeout: float = cc.N_REQUEST_TIMEOUT_S):
        self._port, self._timeout = n_port, timeout

    def __len__(self) -> int:
        return int(cc.rpc(self._port, "count_commands", timeout=self._timeout))


class RemoteStub:
    def __init__(self, n_port: int):
        # `with session.stub.lock:` in the fixture: N takes its own lock inside the request.
        self.lock = contextlib.nullcontext()
        self.commands = RemoteCommands(n_port)


class RemoteSession:
    """What ``real_bridge._make_logging_move``/``_fly_or_record`` need of a ``V3Session``:
    ``.stub.lock``, ``len(.stub.commands)`` and ``.findings``."""

    def __init__(self, n_port: int):
        self.stub = RemoteStub(n_port)
        self.findings: Dict[str, object] = {}


class AppendLogCapture:
    """Same logger, level and line format as ``stage_a_slice.BridgeLogCapture``, but opened in
    APPEND mode: cycle k's log is written by two processes in turn (B_{k-1} while it stops, then
    B_k); the coordinator truncates the file once at the start of the cycle, which is what the
    slice's mode "w" open does."""

    def __init__(self, path):
        self.path = Path(path)
        self._logger = logging.getLogger("mujoco_remote_backend")
        self._handler = logging.FileHandler(str(self.path), mode="a")
        self._handler.setLevel(logging.WARNING)
        self._handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def __enter__(self) -> "AppendLogCapture":
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
# Common bootstrap
# ---------------------------------------------------------------------------

class _Proc:
    def __init__(self, role: str, args):
        self.role, self.name = role, args.name
        self.clone = args.clone
        self.proc_dir = Path(args.proc_dir)
        self.start_utc = cc.utc_now()
        self.exit_evt = threading.Event()
        self.records: Dict[str, Any] = {}

    def provenance(self, point: str) -> dict:
        rec = cc.collect_provenance(self.role, self.name, self.clone, self.start_utc, point=point)
        self.records[point] = rec
        cc.write_json_atomic(self.proc_dir / f"{self.name}.provenance.json", {
            "role": self.role, "name": self.name, "records": self.records})
        return rec

    def ready(self, server: cc.ControlServer, ok: bool, problems: List[str], **extra: Any) -> None:
        cc.write_json_atomic(self.proc_dir / f"{self.name}.ready.json", {
            "ok": ok, "port": server.port, "pid": os.getpid(), "problems": problems, **extra})

    def fail(self, error: str) -> None:
        cc.write_json_atomic(self.proc_dir / f"{self.name}.ready.json", {
            "ok": False, "error": error, "pid": os.getpid(), "port": 0})

    def finish(self) -> None:
        """Wait for the exit request, record the final provenance point, and leave without
        running non-daemon SDK/gRPC thread joins."""
        self.exit_evt.wait()
        rec = self.provenance("before_exit")
        cc.write_json_atomic(self.proc_dir / f"{self.name}.exit.json", {
            "name": self.name, "pid": os.getpid(), "exit_utc": cc.utc_now(), "ok": rec["ok"]})
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


# ---------------------------------------------------------------------------
# N: NativeStub + Observer + request server
# ---------------------------------------------------------------------------

def make_n_handlers(session, sas, rb, ev_dir: Path, control_dir: Path, exit_evt: threading.Event):
    """The request handlers of process N over an already-built ``session`` (a
    ``real_bridge.V3Session``).  ``sas``/``rb`` are the ``stage_a_slice``/``real_bridge`` modules
    (injected so the handlers can be unit tested on an in-process fake stub/observer); every
    slice function used here is called UNMODIFIED.  Returns ``(handlers, post_reply)``."""
    run_dir = Path(ev_dir) / "e1_server_runs" / sas.RUN_NAME
    st: Dict[str, Any] = {"pending": None, "turn_on": {}, "recreate": {}, "injected": {}, "cycles": []}
    lock = threading.Lock()

    def count_commands() -> int:
        with session.stub.lock:
            return len(session.stub.commands)

    def last_seq() -> Optional[int]:
        return session.observer.last_seq()

    def last_sim_step() -> Optional[int]:
        return session.observer.last_sim_step()

    def turn_on_state(key: str) -> dict:
        msg = session.observer.snapshot()[-1]
        with lock:
            st["turn_on"][key] = msg
        return {"seq": msg.get("seq")}

    def recreate_mark(name: str) -> dict:
        with lock:
            st["recreate"][name] = time.time()
        return {}

    def begin_reset() -> dict:
        # Steps 5 and 6 of the slice, in its order, in THIS process: reset_t is on the same
        # clock (time.monotonic) as NativeStub.commands[i]["t"], with which it is later merged.
        pre = session.observer.last_sim_step()
        t = time.monotonic()
        with lock:
            st["pending"] = (t, pre)
        return {"pre_reset_sim_step": pre}

    def commit_reset(reset_gen: int) -> dict:
        with lock:
            t, pre = st["pending"]
            session.reset_events.append((t, int(reset_gen), pre))
            st["pending"] = None
            return {"n_reset_events": len(session.reset_events)}

    def flush_run_dir() -> dict:
        sas._flush_partial_run_dir(session, run_dir, session.reset_events)
        return {"n_reset_events": len(session.reset_events)}

    def inject_echoes(name: str, first_command_index: int, last_command_index: int, seed: int) -> dict:
        injected = sas.inject_synthetic_echoes(
            session, int(first_command_index), int(last_command_index), seed=int(seed))
        with lock:
            st["injected"][name] = injected
        return {"n_injected": len(injected)}

    def finish_cycle(name: str, rep: int, arm: str, reset_gen: int, bridge_log_path: str,
                     reset_record_path: str, snap_out: str, verify_out: str, verify_rc: int,
                     setup: dict, flight: dict, pre_reset_sim_step: Optional[int]) -> dict:
        def leg(d: dict):
            return rb.LegBounds(int(d["first_seq"]), int(d["last_seq"]), list(d["flown"]),
                                [rb.MoveCall(**m) for m in d["moves"]])
        bounds = rb.CycleBounds(setup=leg(setup), flight=leg(flight),
                                reset_sim_step=pre_reset_sim_step, reset_seed=int(reset_gen))
        with lock:
            session.cycles.append(bounds)
            st["cycles"].append(sas.StageACycle(
                name=name, rep=int(rep), arm=arm, bounds=bounds,
                bridge_log_path=Path(bridge_log_path), reset_gen=int(reset_gen),
                reset_record_path=Path(reset_record_path), reset_snapshot_stdout=snap_out,
                reset_verify_stdout=verify_out, reset_verify_rc=int(verify_rc),
                injected_echoes=list(st["injected"].get(name, [])),
                turn_on_state_msg=st["turn_on"].get(name), recreate_timestamp=st["recreate"][name]))
            return {"n_cycles": len(st["cycles"])}

    def emit_evidence() -> dict:
        stage = sas.StageASession(session=session, cycles=list(st["cycles"]), run_dir=run_dir,
                                  control_dir=Path(control_dir))
        evidence = sas.emit_stage_a_evidence(stage, Path(ev_dir))
        return {"run_dir": str(evidence.run_dir), "control_dir": str(evidence.control_dir),
                "arm_map": str(evidence.arm_map_path), "native_log": str(evidence.native_log_path),
                "derived_sha256sums": str(evidence.derived_sha256sums),
                "n_injected_echoes": {c.name: len(c.injected_echoes) for c in st["cycles"]}}

    def shutdown() -> dict:
        return {}

    def after_shutdown() -> None:
        try:
            session.close()
        finally:
            exit_evt.set()

    handlers = {
        "ping": lambda: {"role": "N"}, "count_commands": count_commands, "last_seq": last_seq,
        "last_sim_step": last_sim_step, "turn_on_state": turn_on_state, "recreate_mark": recreate_mark,
        "begin_reset": begin_reset, "commit_reset": commit_reset, "flush_run_dir": flush_run_dir,
        "inject_echoes": inject_echoes, "finish_cycle": finish_cycle, "emit_evidence": emit_evidence,
        "shutdown": shutdown}
    return handlers, {"shutdown": after_shutdown}


def role_n(args) -> None:
    p = _Proc("N", args)
    import real_bridge as rb
    import stage_a_slice as sas
    if sas.RUN_NAME != cc.RUN_NAME:
        raise RuntimeError("stage_a_slice.RUN_NAME differs from c2ps_common.RUN_NAME")

    ev_dir, control_dir = Path(args.ev_dir), Path(args.control_dir)
    run_dir = ev_dir / "e1_server_runs" / sas.RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)

    session = sas.build_stage_a_session()
    handlers, post_reply = make_n_handlers(session, sas, rb, ev_dir, control_dir, p.exit_evt)
    server = cc.ControlServer(handlers, post_reply=post_reply).start()
    rec = p.provenance("after_setup")
    p.ready(server, rec["ok"], rec["problems"], stub_port=session.stub.port,
            run_dir=str(run_dir), control_dir=str(control_dir))
    if not rec["ok"]:
        p.exit_evt.set()
    p.finish()


# ---------------------------------------------------------------------------
# B: one bridge generation
# ---------------------------------------------------------------------------

def role_b(args) -> None:
    p = _Proc("B", args)
    import real_bridge as rb

    state: Dict[str, Any] = {"cap": None}
    if args.log_path:
        state["cap"] = AppendLogCapture(args.log_path).__enter__()
    gen = rb.BridgeGen(StubUrl(args.stub_port))

    def reset() -> dict:
        gen.reset()
        return {}

    def begin_capture(path: str) -> dict:
        state["cap"] = AppendLogCapture(path).__enter__()
        return {}

    def end_capture() -> dict:
        if state["cap"] is not None:
            state["cap"].__exit__(None, None, None)
            state["cap"] = None
        return {}

    def stop() -> dict:
        # The coordinator has already had C_k close its channels (A4) before asking for this.
        gen.close()
        return {}

    server = cc.ControlServer({"ping": lambda: {"role": "B"}, "reset": reset,
                               "begin_capture": begin_capture, "end_capture": end_capture,
                               "stop": stop}, post_reply={"stop": p.exit_evt.set}).start()
    rec = p.provenance("after_setup")
    p.ready(server, rec["ok"], rec["problems"], grpc_port=gen.port)
    if not rec["ok"]:
        p.exit_evt.set()
    p.finish()


# ---------------------------------------------------------------------------
# C: the SDK clients of one cycle
# ---------------------------------------------------------------------------

def role_c(args) -> None:
    p = _Proc("C", args)
    import real_bridge as rb
    from reachy_sdk import ReachySDK
    from reachy_ai.motion import rig_routes as R

    routes = {"PLACE_ROUTE": R.PLACE_ROUTE, "LIFT_TO_PRESENT": R.LIFT_TO_PRESENT}
    remote = RemoteSession(args.n_port)
    clients: List[Any] = []

    def client(key: Optional[str] = None, record_turn_on_state: bool = False) -> dict:
        # Steps 13/20: new client (+ sleep 0.2, as BridgeGen.client()), turn_on, sleep 0.3;
        # step 14 (setup only): the turn-on state, requested from N right after the sleep.
        c = ReachySDK(host="127.0.0.1", sdk_port=args.grpc_port, camera_port=args.grpc_port,
                      restart_port=args.grpc_port)
        time.sleep(0.2)
        clients.append(c)
        c.turn_on("r_arm")
        time.sleep(0.3)
        if record_turn_on_state:
            cc.rpc(args.n_port, "turn_on_state", {"key": key})
        return {}

    def fly(route: str, key: str) -> dict:
        moves: List["rb.MoveCall"] = []
        flown = rb._fly_or_record(remote, clients[-1].r_arm, routes[route], key, moves)
        seq = cc.rpc(args.n_port, "last_seq")          # the slice's setup_last_seq / flight_last_seq
        return {"flown": list(flown), "moves": [dataclasses.asdict(m) for m in moves],
                "last_seq": seq, "findings": dict(remote.findings)}

    def close() -> dict:
        # A4: each client's own gRPC channel first (standard grpc API), then leave.
        for c in clients:
            try:
                c._grpc_channel.close()
            except Exception:                            # noqa: BLE001 -- as BridgeGen.close()
                pass
        return {}

    server = cc.ControlServer({"ping": lambda: {"role": "C"}, "client": client, "fly": fly,
                               "close": close}, post_reply={"close": p.exit_evt.set}).start()
    rec = p.provenance("after_setup")
    p.ready(server, rec["ok"], rec["problems"])
    if not rec["ok"]:
        p.exit_evt.set()
    p.finish()


ROLES = {"N": role_n, "B": role_b, "C": role_c}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=sorted(ROLES))
    ap.add_argument("--name", required=True)
    ap.add_argument("--clone", required=True)
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--ev-dir")
    ap.add_argument("--control-dir")
    ap.add_argument("--stub-port", type=int)
    ap.add_argument("--n-port", type=int)
    ap.add_argument("--grpc-port", type=int)
    ap.add_argument("--log-path")
    args = ap.parse_args(argv)
    try:
        ROLES[args.role](args)
    except SystemExit:
        raise
    except BaseException:                                # noqa: BLE001 -- recorded, never swallowed
        tb = traceback.format_exc()
        cc.write_json_atomic(Path(args.proc_dir) / f"{args.name}.ready.json", {
            "ok": False, "error": tb[-4000:], "pid": os.getpid(), "port": 0})
        sys.stderr.write(tb)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
