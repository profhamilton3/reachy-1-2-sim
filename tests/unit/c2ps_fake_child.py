"""A FAKE child process for the c2_process_split lifecycle tests (no NativeStub, no bridge, no gRPC
server, no SDK client): it speaks the same request protocol as the real N/B/C children
(``c2ps_common.ControlServer``), writes the same ready/provenance files, and appends one JSON
line per lifecycle event to ``$C2PS_FAKE_EVENTS`` so a test can check ordering.

``$C2PS_FAKE_MODE`` (comma separated) injects faults: ``hang_fly`` (fly never answers),
``die_on_reset`` (B exits abruptly on reset), ``ignore_term`` (SIGTERM ignored), ``bad_provenance``
(ready file reports a separation problem), ``no_ready`` (never becomes ready), ``exit_nonzero``
(exit code 3 on the exit request).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "fixtures" / "goalfix_cmp"))
import c2ps_common as cc  # noqa: E402

MODES = set(filter(None, os.environ.get("C2PS_FAKE_MODE", "").split(",")))
EVENTS = os.environ.get("C2PS_FAKE_EVENTS")


def log(who: str, what: str, **extra) -> None:
    if not EVENTS:
        return
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"ns": time.monotonic_ns(), "who": who, "what": what, **extra}) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("role")
    ap.add_argument("--name", required=True)
    ap.add_argument("--clone", default="/nonexistent")
    ap.add_argument("--proc-dir", required=True)
    ap.add_argument("--ev-dir")
    ap.add_argument("--control-dir")
    ap.add_argument("--stub-port")
    ap.add_argument("--n-port")
    ap.add_argument("--grpc-port")
    ap.add_argument("--log-path")
    a = ap.parse_args()
    name, role, proc = a.name, a.role, Path(a.proc_dir)
    if "ignore_term" in MODES:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    log(name, "start", pid=os.getpid())
    if "no_ready" in MODES and role == os.environ.get("C2PS_FAKE_MODE_ROLE", role):
        time.sleep(600)
    exit_evt = threading.Event()
    seq = {"n": 0}

    def rec(what):
        def h(**kw):
            log(name, what, **{k: v for k, v in kw.items() if isinstance(v, (int, str, type(None)))})
            return {}
        return h

    handlers = {"ping": lambda: {"role": role}}
    post = {}
    extra = {}
    if role == "N":
        def last_seq():
            seq["n"] += 1
            log(name, "last_seq")
            return seq["n"]

        def begin_reset():
            log(name, "begin_reset")
            return {"pre_reset_sim_step": 5}

        def emit_evidence(**kw):
            log(name, "emit_evidence")
            run = str(Path(a.ev_dir) / "e1_server_runs" / cc.RUN_NAME)
            return {"run_dir": run, "control_dir": str(a.control_dir), "arm_map": "x", "native_log": "y"}

        handlers.update({
            "count_commands": lambda: 0, "last_seq": last_seq, "begin_reset": begin_reset,
            "recreate_mark": rec("recreate_mark"), "commit_reset": rec("commit_reset"),
            "flush_run_dir": rec("flush_run_dir"), "turn_on_state": rec("turn_on_state"),
            "inject_echoes": lambda **kw: (log(name, "inject_echoes", **kw) or {"n_injected": 3}),
            "finish_cycle": lambda **kw: (log(name, "finish_cycle", cycle=kw.get("name")) or {}),
            "emit_evidence": emit_evidence, "shutdown": rec("shutdown")})
        post["shutdown"] = exit_evt.set
        extra["stub_port"] = 4242
    elif role == "B":
        def reset():
            log(name, "reset")
            if "die_on_reset" in MODES:
                os._exit(9)
            return {}

        def stop():
            log(name, "stop")
            return {}

        handlers.update({"reset": reset, "begin_capture": rec("begin_capture"),
                         "end_capture": rec("end_capture"), "stop": stop})
        post["stop"] = exit_evt.set
        extra["grpc_port"] = 5151
    else:
        def fly(route, key):
            log(name, "fly", route=route)
            if "hang_fly" in MODES:
                time.sleep(600)
            moves = [{"pose": {}, "seconds": 1.0, "first_command_index": 10, "last_command_index": 20},
                     {"pose": {}, "seconds": 1.0, "first_command_index": 20, "last_command_index": 40}]
            flown = ["HOVER", "REST_SHUT"] if route == "PLACE_ROUTE" else ["PRESENT"]
            return {"flown": flown, "moves": moves[:len(flown)], "last_seq": 7, "findings": {}}

        def close():
            log(name, "close")
            return {}

        handlers.update({"client": rec("client"), "fly": fly, "close": close})
        post["close"] = exit_evt.set

    server = cc.ControlServer(handlers, post_reply=post).start()
    problems = ["fake separation problem"] if "bad_provenance" in MODES else []
    point = {"ok": not problems, "problems": problems}
    cc.write_json_atomic(proc / f"{name}.provenance.json", {"records": {"after_setup": point, "before_exit": point}})
    cc.write_json_atomic(proc / f"{name}.ready.json", {"ok": not problems, "port": server.port, "pid": os.getpid(),
                                                       "problems": problems, **extra})
    log(name, "ready")
    exit_evt.wait()
    log(name, "exit")
    os._exit(3 if "exit_nonzero" in MODES else 0)


if __name__ == "__main__":
    raise SystemExit(main())
