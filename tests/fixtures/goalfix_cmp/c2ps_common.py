"""Shared, stdlib-only building blocks of the U-only process-separated C2 diagnostic
(``c2_process_split.py`` driver, ``c2ps_children.py`` child entry points).

PREPARATION ONLY.  Nothing in this module starts NativeStub, a bridge, a gRPC server or an
SDK client; it imports none of ``reachy_sdk``, ``grpc``, ``websockets``, ``native_stub``,
``mujoco_remote_backend`` or ``fake_reachy_server`` (a test pins this).  It holds:

* the fixed constants (timeouts, caps, pins, process names);
* a tiny line-delimited JSON request/response protocol over loopback TCP
  (``ControlServer`` / ``rpc``), used between the coordinator and each child and directly
  from the client process C to the native-stand-in process N;
* ``Supervisor``: spawn children in their own session/process group with stdout/stderr to
  files, wait for a ready file, RPC, stop with terminate-then-kill, record exit codes and
  pids;
* per-process provenance and the separation assertions.

NO DIAGNOSTIC CAPTURE: nothing here wraps, patches or subclasses any product code.
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Fixed pins and parameters
# ---------------------------------------------------------------------------

M_PRIME_SHA = "dbc878cb1429d240b17edb8dc5b04a5cbc459be6"   # origin/main (PR #147 merged); the base
M_PRIME_TOOLS_TREE = "9ea1a47f63d351ca050b7699cc5eea68c6f7d720"
A_SHA = "8c0dad2d62dde791c8912fa723b8c2b7f190e1e8"          # label only
B_SHA = "67730a1ecf646544825fb60c00129cf46de6d307"          # bridge blobs must equal B's
BRIDGE_OPT_FILES = ("fake_reachy_server.py", "mujoco_remote_backend.py", "reset_watcher.py")
RUNTIME_TREES = ("native_mujoco", "scripts", "src", "scenes", "notebooks", "web", "ros",
                 "fake_reachy_server.py", "mujoco_remote_backend.py", "reset_watcher.py",
                 "kinematic_backend.py", "Dockerfile", "docker-compose.yml", "requirements.txt",
                 "supervisord.conf", "entrypoint.sh")
ALLOWED_NEW_PREFIXES = ("tests/fixtures/goalfix_cmp/", "tests/unit/")

N_RUNS = 7
ARM_LABELS = ("A", "B", "B", "A")
CYCLE_NAMES = tuple(f"S2-B4-c-r{i}" for i in range(1, 5))
RUN_NAME = "run_stage_a_slice"          # equals stage_a_slice.RUN_NAME (asserted in N)
ECHO_SEED_BASE = 20260925               # stage_a_slice.run_stage_a_session default

# Step timeouts (seconds). Every coordinator wait is finite; a timeout is an infrastructure failure.
PROCESS_READY_TIMEOUT_S = 60
RESET_TIMEOUT_S = 30
LEG_TIMEOUT_S = 300
N_REQUEST_TIMEOUT_S = 30
EMIT_TIMEOUT_S = 120
CHILD_EXIT_TIMEOUT_S = 30
TERM_GRACE_S = 10
RESET_VERIFY_TIMEOUT_S = 30
CLI_TIMEOUT_S = 600
COORDINATOR_TERM_GRACE_S = 20

# Caps.
RUN_CAP_S = 15 * 60
TOTAL_CAP_S = 90 * 60
OUT_CAP_BYTES = 1 * 2 ** 30
MIN_FREE_BYTES = 20 * 2 ** 30

FORBIDDEN_OUT_PREFIXES = ("/tmp", "/private/tmp")

#: Modules a process must NOT have imported at all (the coordinator's stricter rule).
COORDINATOR_FORBIDDEN_MODULES = (
    "reachy_sdk", "grpc", "websockets", "native_stub", "mujoco_remote_backend",
    "fake_reachy_server", "kinematic_backend", "real_bridge", "stage_a_slice")

#: Top-level names of modules that are "product or fixture" code: each must be imported from
#: under the clone (never from the primary checkout or anywhere else).
PRODUCT_TOPLEVEL = (
    "fake_reachy_server", "mujoco_remote_backend", "kinematic_backend", "native_stub",
    "real_bridge", "stage_a_slice", "make_fixtures", "joint_map", "reachy_ai", "tools",
    "e1_identity", "e1_stage1", "c2ps_common", "c2ps_children", "c2_process_split",
    "reset_watcher")


class InfraFailure(Exception):
    """An infrastructure/evidence failure: the only thing that stops the batch."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class StepTimeout(InfraFailure):
    def __init__(self, detail: str):
        super().__init__("step_timeout", detail)


class RpcError(Exception):
    """The peer answered with ``ok: false``."""


class ChildDied(InfraFailure):
    def __init__(self, detail: str):
        super().__init__("child_died", detail)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path, doc) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    os.replace(str(tmp), str(p))


# ---------------------------------------------------------------------------
# Line-delimited JSON over loopback TCP: one request per connection.
# ---------------------------------------------------------------------------

class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:                                   # pragma: no cover - thin
        server: "ControlServer" = self.server.owner             # type: ignore[attr-defined]
        try:
            self.request.settimeout(30)
            line = self.rfile.readline(1 << 20)
            if not line:
                return
            msg = json.loads(line)
            method = msg.get("method")
            fn = server.handlers.get(method)
            if fn is None:
                reply = {"ok": False, "error": f"unknown method {method!r}"}
            else:
                try:
                    reply = {"ok": True, "result": fn(**(msg.get("params") or {}))}
                except BaseException as exc:                    # noqa: BLE001 -- reported, never swallowed
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(reply, default=str) + "\n").encode())
            self.wfile.flush()
            after = server.post_reply.get(method)
            if after is not None and reply.get("ok"):
                after()                          # e.g. "leave now": only after the reply is on the wire
        except (OSError, ValueError):
            return


class _TCP(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ControlServer:
    """A small threaded request server: ``handlers`` maps method name -> callable(**params)."""

    def __init__(self, handlers: Mapping[str, Callable[..., Any]], host: str = "127.0.0.1",
                 post_reply: Optional[Mapping[str, Callable[[], None]]] = None):
        self.handlers = dict(handlers)
        self.post_reply = dict(post_reply or {})
        self._srv = _TCP((host, 0), _Handler)
        self._srv.owner = self                                  # type: ignore[attr-defined]
        self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True,
                                        name="c2ps-control-server")

    def start(self) -> "ControlServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def rpc(port: int, method: str, params: Optional[dict] = None, timeout: float = N_REQUEST_TIMEOUT_S,
        host: str = "127.0.0.1") -> Any:
    """One request/response.  A timeout is ``StepTimeout``; a closed connection without a reply
    is ``ChildDied``; ``ok: false`` is ``RpcError``."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps({"method": method, "params": params or {}}) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(1 << 16)
                if not chunk:
                    break
                buf += chunk
    except socket.timeout as exc:
        raise StepTimeout(f"{method} on port {port} did not answer within {timeout} s") from exc
    except OSError as exc:
        raise ChildDied(f"{method} on port {port}: {type(exc).__name__}: {exc}") from exc
    if not buf.endswith(b"\n"):
        raise ChildDied(f"{method} on port {port}: connection closed without a reply")
    reply = json.loads(buf)
    if not reply.get("ok"):
        raise RpcError(reply.get("error", "unknown error"))
    return reply.get("result")


# ---------------------------------------------------------------------------
# Provenance and separation
# ---------------------------------------------------------------------------

#: (module, qualname) of classes each role must not HOLD an instance of.
_NATIVE_STUB = ("native_stub", "NativeStub")
_OBSERVER = ("real_bridge", "Observer")
_SDK = ("reachy_sdk.reachy_sdk", "ReachySDK")
_GRPC_SERVER = ("grpc._server", "_Server")
_BACKEND = ("mujoco_remote_backend", "MujocoRemoteBackend")
_BRIDGE = ("mujoco_remote_backend", "KinematicBridge")
_SERVICERS = (("fake_reachy_server", "FakeJointService"), ("fake_reachy_server", "FakeSensorService"),
              ("fake_reachy_server", "FakeFanService"))
_BRIDGE_SIDE = (_GRPC_SERVER, _BACKEND, _BRIDGE) + _SERVICERS

FORBIDDEN_INSTANCES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "C": (_NATIVE_STUB, _OBSERVER) + _BRIDGE_SIDE,
    "B": (_SDK, _NATIVE_STUB, _OBSERVER),
    "N": (_SDK,) + _BRIDGE_SIDE,
    "coordinator": (_SDK, _NATIVE_STUB, _OBSERVER) + _BRIDGE_SIDE,
}


def find_forbidden_instances(role: str, objects: Iterable[Any]) -> List[str]:
    """Descriptions of every object in ``objects`` that is an instance (by class or any base
    class, matched on (module, qualname)) of something ``role`` must not hold."""
    forbidden = set(FORBIDDEN_INSTANCES[role])
    found: List[str] = []
    for o in objects:
        try:
            mro = type(o).__mro__
        except Exception:                                       # noqa: BLE001
            continue
        for cls in mro:
            key = (getattr(cls, "__module__", None), getattr(cls, "__qualname__", None))
            if key in forbidden:
                found.append(f"{role} holds an instance of {key[0]}.{key[1]}")
                break
    return found


def check_module_origins(clone_root: str, module_files: Mapping[str, Optional[str]]) -> List[str]:
    """Every product/fixture module (top-level name in ``PRODUCT_TOPLEVEL``) must have a
    ``__file__`` lying under the clone."""
    root = Path(clone_root).resolve()
    problems: List[str] = []
    for name, f in sorted(module_files.items()):
        if name.split(".")[0] not in PRODUCT_TOPLEVEL:
            continue
        if not f:
            continue                                            # namespace package without a file
        try:
            p = Path(f).resolve()
        except OSError:
            problems.append(f"module {name}: unresolvable file {f!r}")
            continue
        if root != p and root not in p.parents:
            problems.append(f"module {name} imported from {p}, outside the clone {root}")
    return problems


def loaded_module_files() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f is None:
            paths = list(getattr(mod, "__path__", []) or [])
            f = paths[0] if paths else None
        out[name] = f
    return out


def check_separation(role: str, objects: Optional[Iterable[Any]] = None,
                     module_names: Optional[Iterable[str]] = None) -> List[str]:
    """Separation assertions for ``role`` (a failure is an infrastructure failure): no forbidden
    instance is held; the coordinator additionally must not have imported the heavy modules."""
    objs = gc.get_objects() if objects is None else objects
    problems = find_forbidden_instances(role, objs)
    if role == "coordinator":
        names = set(sys.modules) if module_names is None else set(module_names)
        for m in COORDINATOR_FORBIDDEN_MODULES:
            if any(n == m or n.startswith(m + ".") for n in names):
                problems.append(f"coordinator imported forbidden module {m}")
    return problems


def thread_inventory() -> List[str]:
    return sorted(t.name for t in threading.enumerate())


def collect_provenance(role: str, name: str, clone_root: str, start_utc: str, *,
                       point: str, objects: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
    """The per-process record: interpreter, pid/ppid, times, ``__file__`` of every loaded module
    whose top-level name is product/fixture code (or that lies under the clone), the full sorted
    ``sys.modules`` key list, a thread inventory, and the separation/origin problems found."""
    files = loaded_module_files()
    root = str(Path(clone_root).resolve())
    product_files = {n: f for n, f in files.items()
                     if n.split(".")[0] in PRODUCT_TOPLEVEL or (f and str(f).startswith(root))}
    problems = check_module_origins(clone_root, product_files)
    problems += check_separation(role, objects)
    return {
        "role": role, "name": name, "point": point,
        "sys_executable": sys.executable, "sys_version": sys.version,
        "pid": os.getpid(), "ppid": os.getppid(),
        "start_utc": start_utc, "record_utc": utc_now(),
        "clone_root": root, "product_module_files": dict(sorted(product_files.items())),
        "sys_modules": sorted(sys.modules), "threads": thread_inventory(),
        "problems": problems, "ok": not problems,
    }


# ---------------------------------------------------------------------------
# Supervisor: children in their own sessions; files, not pipes; terminate-then-kill.
# ---------------------------------------------------------------------------

class Child:
    def __init__(self, name: str, argv: Sequence[str], popen, stdout_path: Path, stderr_path: Path,
                 ready_path: Path):
        self.name = name
        self.argv = list(argv)
        self.popen = popen
        self.pid = popen.pid
        self.stdout_path, self.stderr_path, self.ready_path = stdout_path, stderr_path, ready_path
        self.exit_code: Optional[int] = None
        self.stopped_by: Optional[str] = None                    # "exit" | "terminate" | "kill"
        self.ready: Optional[dict] = None
        self._files: List[Any] = []

    @property
    def port(self) -> int:
        assert self.ready is not None, f"{self.name} is not ready"
        return int(self.ready["port"])

    def alive(self) -> bool:
        return self.popen.poll() is None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Supervisor:
    """Owns every child of one run.  ``spawn`` is injectable for tests."""

    def __init__(self, proc_dir: Path, env: Mapping[str, str], cwd: str, *,
                 spawn: Callable[..., Any] = subprocess.Popen, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, term_grace: float = TERM_GRACE_S):
        self.term_grace = term_grace
        self.proc_dir = Path(proc_dir)
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.env = dict(env)
        self.cwd = cwd
        self._spawn, self._clock, self._sleep = spawn, clock, sleep
        self.children: List[Child] = []
        self._lock = threading.Lock()
        self.events: List[dict] = []                              # ordered lifecycle log (this process only)

    # -- bookkeeping ------------------------------------------------------
    def _event(self, what: str, name: str, **extra: Any) -> None:
        self.events.append({"seq": len(self.events), "what": what, "name": name, **extra})

    def _write_pids(self) -> None:
        write_json_atomic(self.proc_dir / "pids.json", [
            {"name": c.name, "pid": c.pid, "pgid": c.pid, "argv": c.argv} for c in self.children])

    def _write_exit_codes(self) -> None:
        write_json_atomic(self.proc_dir / "exit_codes.json", {
            c.name: {"pid": c.pid, "exit_code": c.exit_code, "stopped_by": c.stopped_by}
            for c in self.children})

    # -- lifecycle --------------------------------------------------------
    def spawn(self, name: str, argv: Sequence[str]) -> Child:
        out_p, err_p = self.proc_dir / f"{name}.stdout", self.proc_dir / f"{name}.stderr"
        ready_p = self.proc_dir / f"{name}.ready.json"
        fo, fe = open(out_p, "wb"), open(err_p, "wb")
        try:
            popen = self._spawn(list(argv), cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=fo, stderr=fe, start_new_session=True)
        except OSError as exc:
            fo.close()
            fe.close()
            raise InfraFailure("child_start", f"{name}: {type(exc).__name__}: {exc}") from exc
        ch = Child(name, argv, popen, out_p, err_p, ready_p)
        ch._files = [fo, fe]
        with self._lock:
            self.children.append(ch)
            self._write_pids()
        self._event("spawn", name, pid=ch.pid)
        return ch

    def wait_ready(self, ch: Child, timeout: float = PROCESS_READY_TIMEOUT_S) -> dict:
        deadline = self._clock() + timeout
        while True:
            if ch.ready_path.is_file():
                try:
                    ch.ready = json.loads(ch.ready_path.read_text())
                except ValueError:
                    ch.ready = None
                if ch.ready is not None:
                    if not ch.ready.get("ok", True):
                        raise InfraFailure(
                            "separation" if ch.ready.get("problems") else "child_start",
                            f"{ch.name} reported not ok: {ch.ready.get('problems') or ch.ready.get('error')}")
                    self._event("ready", ch.name)
                    return ch.ready
            if not ch.alive():
                ch.exit_code = ch.popen.poll()
                raise InfraFailure("child_start", f"{ch.name} exited with {ch.exit_code} before becoming ready")
            if self._clock() >= deadline:
                raise StepTimeout(f"{ch.name} not ready within {timeout} s")
            self._sleep(0.05)

    def call(self, ch: Child, method: str, params: Optional[dict] = None,
             timeout: float = N_REQUEST_TIMEOUT_S) -> Any:
        self._event("rpc", ch.name, method=method)
        return rpc(ch.port, method, params, timeout)

    def wait_exit(self, ch: Child, timeout: float = CHILD_EXIT_TIMEOUT_S) -> int:
        deadline = self._clock() + timeout
        while ch.alive():
            if self._clock() >= deadline:
                raise StepTimeout(f"{ch.name} did not exit within {timeout} s")
            self._sleep(0.05)
        ch.exit_code = ch.popen.poll()
        ch.stopped_by = ch.stopped_by or "exit"
        self._close_files(ch)
        self._event("exit", ch.name, exit_code=ch.exit_code)
        with self._lock:
            self._write_exit_codes()
        return ch.exit_code

    def _close_files(self, ch: Child) -> None:
        for f in ch._files:
            try:
                f.close()
            except OSError:
                pass
        ch._files = []

    def _killpg(self, ch: Child, sig: int) -> None:
        try:
            os.killpg(ch.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:                                   # pragma: no cover
            pass

    def stop(self, ch: Child, grace: Optional[float] = None) -> int:
        """Terminate then kill the child's whole process group; record the exit code."""
        grace = self.term_grace if grace is None else grace
        if ch.alive():
            self._killpg(ch, signal.SIGTERM)
            ch.stopped_by = "terminate"
            deadline = self._clock() + grace
            while ch.alive() and self._clock() < deadline:
                self._sleep(0.05)
            if ch.alive():
                self._killpg(ch, signal.SIGKILL)
                ch.stopped_by = "kill"
                try:
                    ch.popen.wait(timeout=grace)
                except subprocess.TimeoutExpired:                 # pragma: no cover
                    pass
        # A grandchild that shares the group must not survive either.
        self._killpg(ch, signal.SIGKILL)
        ch.exit_code = ch.popen.poll()
        self._close_files(ch)
        self._event("stop", ch.name, exit_code=ch.exit_code, stopped_by=ch.stopped_by)
        with self._lock:
            self._write_exit_codes()
        return ch.exit_code if ch.exit_code is not None else -9

    def shutdown_all(self) -> Dict[str, Any]:
        """On every exit path: terminate then kill every child still alive, record exit codes,
        and report any pid that is still alive afterwards (there must be none)."""
        for ch in list(self.children):
            if ch.alive() or ch.exit_code is None:
                self.stop(ch)
        survivors = [c.pid for c in self.children if pid_alive(c.pid) and c.popen.poll() is None]
        with self._lock:
            self._write_exit_codes()
        return {"exit_codes": {c.name: c.exit_code for c in self.children}, "survivors": survivors}


def install_signal_cleanup(cleanup: Callable[[], None]) -> Callable[[], None]:
    """SIGINT/SIGTERM run ``cleanup`` then exit with 128+signum.  Returns a restore function."""
    old = {}

    def handler(signum, _frame):
        try:
            cleanup()
        finally:
            os._exit(128 + signum)

    for s in (signal.SIGINT, signal.SIGTERM):
        old[s] = signal.signal(s, handler)

    def restore() -> None:
        for s, h in old.items():
            signal.signal(s, h)
    return restore


def ps_leftovers(needle: str) -> List[int]:
    """Pids of processes whose command line contains ``needle`` (the run directory path)."""
    try:
        out = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    me = os.getpid()
    pids = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and needle in parts[1] and int(parts[0]) != me:
            pids.append(int(parts[0]))
    return pids
