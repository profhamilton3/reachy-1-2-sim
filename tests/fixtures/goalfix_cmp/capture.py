"""Test-only capture shim for the B4 harness validation (prepared, not run
against any recorded evidence).

Purpose
-------
The bridge can split ONE SDK update (one ``JointsCommand`` batch) across two
emitted ``joint_command`` messages, because

  * ``FakeJointService._submit_batch`` submits a batch joint by joint, each
    ``MujocoRemoteBackend.submit_command`` taking the backend lock separately;
  * ``MujocoRemoteBackend._send`` drains ``_pending_cmds`` every 20 ms and
    builds one 21-vector (``_build_command``) from ``_last_target`` plus
    whatever is queued.

A drain that lands between two joints of one batch emits a command holding
update *u+1* for some joints and update *u* (carried) for the rest; the next
command completes *u+1*.  This shim records enough to classify every emitted
command and every SDK batch, so a C2 failure can be attributed (or not) to
this mechanism.

What is recorded (JSONL, one record per event; ``time.monotonic_ns()``, the
same clock the harness Observer uses for ``wall_time_ns``)
---------------------------------------------------------------------------
* ``gen``    bridge generation (ordinal of the ``MujocoRemoteBackend`` instance
             in construction order under capture).
* ``batch``  one ``_submit_batch`` call: generation, ``batch_id`` (monotone
             per capture), rpc name (caller frame name, if determinable),
             arrival ``t_ns``, entry count.
* ``submit`` one ``submit_command`` call: ``sid``, ``batch_id`` (``null`` =
             unattributed: no batch was current on this thread), entry
             ordinal, uid, protocol index, ``goal_position``, ``compliant``,
             ``t_ns``.  The ``JointCommand`` object is tagged by ``id()``
             and a reference is kept so the id cannot be reused.
* ``build``  one ``_build_command`` call: generation, predicted ``seq``
             (``backend._cmd_seq + 1`` at build time; verified against
             ``commands.jsonl`` by ``cross_check``, never assumed), ``t_ns``,
             the drained list in order (``sid``/``batch_id``/``entry``/
             ``index``), the returned 21-vector ``target``, and per index the
             ``source``: the ``batch_id`` of the LAST drained command that set
             ``goal_position`` for that index (last writer wins, exactly as
             ``_build_command`` does), ``-1`` if that command was
             unattributed, or ``null`` = kept from ``_last_target`` (carry).
             ``stiffen_no_goal`` lists indices named by a ``compliant=False``
             entry with no goal: ``_build_command`` may set those from the
             present position, a third source this shim does not model.

Classification (pure, unit-tested: ``classify``)
------------------------------------------------
* mixed-batch command: its non-null goal sources contain >= 2 distinct
  ``batch_id``.
* split batch: a batch whose submitted entries were drained into >= 2
  different builds.
* split-involved command: contains >= 1 entry of a split batch.
* per command class: ``mixed_batch`` | ``split_involved_not_mixed`` |
  ``neither``.  Also the arrival gap (ns) between consecutive batches of a
  generation, and for each mixed command the arrival-time span of its source
  batches.

Cross-check (``cross_check``; a capture failure if it does not hold)
--------------------------------------------------------------------
Every ``joint_command`` row of the run's ``commands.jsonl`` must join to
exactly one build record by (generation, seq) with a bit-equal 21-vector, and
vice versa; the unattributed-submit count must be 0.  Generations are
delimited in ``commands.jsonl`` as follows: the bridge's ``_cmd_seq`` restarts
at 1 for each new ``MujocoRemoteBackend``, so a generation is a maximal run of
``joint_command`` rows with STRICTLY INCREASING ``seq``; ``reset`` rows are not
delimiters (a reset inside one backend does not restart ``seq``).  The k-th
run is matched to the k-th capture generation that has >= 1 build.  A count
mismatch, a seq list that differs from the capture's predicted seqs, a
duplicate, or any unmatched row/build FAILS CLOSED (no guessing).

The Stage A slice overwrites some commands in the harness's own stub log
(``stage_a_slice.inject_synthetic_echoes`` mutates ``NativeStub.commands`` in
place for A cycles).  ``cross_check`` therefore accepts an explicit
``allowed_overrides`` mapping {joint_command row ordinal: {index: value}}: at
those (row, index) positions the row must equal the declared value; every
other index must be bit-equal.  Nothing is overridden by default.

Hard constraints honoured
-------------------------
No production code, synchronization, queue semantics or send cadence is
changed.  There is NO batch-wide lock: the shim's own bookkeeping lock is held
only for its own list/dict update, never while calling an original method.
Wrapping is by class attribute inside a context manager that restores the
original function objects on exit (also on exception).  Every wrapper calls the
original with identical arguments and returns its result unchanged.

TIMING EFFECTS (disclosed; capture-on rates are NOT capture-off rates)
----------------------------------------------------------------------
The wrappers add Python work (a monotonic-clock read, a dict/list update and,
per submit, a frame lookup) between the per-joint submits of one batch.  That
can WIDEN the window in which a 20 ms drain lands between two joints and so
RAISE the observed split rate.  The ``_build_command`` wrapper does its own
(small) work inside the call ``_send`` makes under the backend lock, slightly
delaying concurrent ``submit_command`` callers.  Rates measured with capture
on must not be quoted as capture-off rates.
"""
from __future__ import annotations

import functools
import itertools
import json
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
for _p in (_REPO, _REPO / "src", _REPO / "native_mujoco"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

UNATTRIBUTED_SOURCE = -1
N_JOINTS = 21


# ---------------------------------------------------------------------------
# The shim
# ---------------------------------------------------------------------------

class _Cur:
    """Thread-local "current batch" cursor."""
    __slots__ = ("batch_id", "n")

    def __init__(self, batch_id: int) -> None:
        self.batch_id = batch_id
        self.n = 0


class Capture:
    """Context manager; see the module docstring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()      # bookkeeping ONLY; never held across an original call
        self._events: List[dict] = []
        self._refs: List[Any] = []         # keeps tagged objects alive (id() reuse)
        self._tag: Dict[int, int] = {}     # id(JointCommand) -> sid
        self._gens: Dict[int, int] = {}    # id(backend) -> ordinal
        self._gen_refs: List[Any] = []
        self._batch_ids = itertools.count(1)
        self._sids = itertools.count(0)
        self._tls = threading.local()
        self._orig: Dict[str, Tuple[type, str, Any]] = {}
        self._active = False

    # -- install / restore ------------------------------------------------

    def __enter__(self) -> "Capture":
        if self._active:
            raise RuntimeError("Capture is already active")
        import fake_reachy_server as frs
        import mujoco_remote_backend as mrb
        targets = [
            (frs.FakeJointService, "_submit_batch", self._wrap_batch),
            (mrb.MujocoRemoteBackend, "submit_command", self._wrap_submit),
            (mrb.MujocoRemoteBackend, "_build_command", self._wrap_build),
            (mrb.MujocoRemoteBackend, "__init__", self._wrap_init),
        ]
        try:
            for cls, name, maker in targets:
                orig = cls.__dict__[name]
                self._orig[f"{cls.__name__}.{name}"] = (cls, name, orig)
                setattr(cls, name, maker(orig))
        except BaseException:
            self._restore()
            raise
        self._active = True
        return self

    def __exit__(self, *exc) -> None:
        self._restore()

    def _restore(self) -> None:
        for cls, name, orig in list(self._orig.values()):
            try:
                setattr(cls, name, orig)
            except Exception:
                pass
        self._orig.clear()
        self._active = False

    # -- helpers ----------------------------------------------------------

    def _gen_of(self, remote: Any) -> int:
        key = id(remote)
        with self._lock:
            g = self._gens.get(key)
            if g is None:
                g = len(self._gens)
                self._gens[key] = g
                self._gen_refs.append(remote)
                self._events.append({"type": "gen", "gen": g, "t_ns": time.monotonic_ns(),
                                     "late": True})
            return g

    # -- wrappers ---------------------------------------------------------

    def _wrap_init(self, orig):
        cap = self

        @functools.wraps(orig)
        def __init__(backend, *args, **kwargs):
            result = orig(backend, *args, **kwargs)
            with cap._lock:
                g = len(cap._gens)
                cap._gens[id(backend)] = g
                cap._gen_refs.append(backend)
                cap._events.append({"type": "gen", "gen": g, "t_ns": time.monotonic_ns(),
                                    "late": False})
            return result
        return __init__

    def _wrap_batch(self, orig):
        cap = self

        @functools.wraps(orig)
        def _submit_batch(svc, batch):
            t_ns = time.monotonic_ns()
            backend = getattr(svc, "_backend", None)
            remote = getattr(backend, "_r", backend)
            gen = cap._gen_of(remote)
            try:
                n_entries = len(batch.commands)
            except Exception:
                n_entries = None
            try:
                rpc = sys._getframe(1).f_code.co_name
            except Exception:
                rpc = None
            with cap._lock:
                bid = next(cap._batch_ids)
                cap._events.append({"type": "batch", "gen": gen, "batch_id": bid, "rpc": rpc,
                                    "t_ns": t_ns, "n_entries": n_entries})
            prev = getattr(cap._tls, "cur", None)
            cap._tls.cur = _Cur(bid)
            try:
                return orig(svc, batch)          # no lock held here
            finally:
                cap._tls.cur = prev
        return _submit_batch

    def _wrap_submit(self, orig):
        cap = self

        @functools.wraps(orig)
        def submit_command(backend, cmd):
            t_ns = time.monotonic_ns()
            gen = cap._gen_of(backend)
            cur = getattr(cap._tls, "cur", None)
            if cur is not None:
                bid, entry = cur.batch_id, cur.n
                cur.n += 1
            else:
                bid, entry = None, None
            idx = getattr(backend, "_uid_to_idx", {}).get(cmd.uid)
            # Recorded BEFORE the original call so the tag exists before the
            # command can be drained by a concurrent _send.
            with cap._lock:
                sid = next(cap._sids)
                cap._tag[id(cmd)] = sid
                cap._refs.append(cmd)
                cap._events.append({
                    "type": "submit", "sid": sid, "gen": gen, "batch_id": bid, "entry": entry,
                    "uid": cmd.uid, "index": idx, "goal_position": cmd.goal_position,
                    "compliant": cmd.compliant, "t_ns": t_ns})
            return orig(backend, cmd)            # no lock held here
        return submit_command

    def _wrap_build(self, orig):
        cap = self

        @functools.wraps(orig)
        def _build_command(backend, cmds):
            t_ns = time.monotonic_ns()
            gen = cap._gen_of(backend)
            seq = getattr(backend, "_cmd_seq", 0) + 1
            with cap._lock:
                sids = [cap._tag.get(id(c)) for c in cmds]
            result = orig(backend, cmds)         # no lock held here
            target = [float(v) for v in result[0]]
            with cap._lock:
                cap._events.append({"type": "build", "gen": gen, "seq": seq, "t_ns": t_ns,
                                    "sids": sids, "target": target})
            return result
        return _build_command

    # -- output -----------------------------------------------------------

    def raw_events(self) -> List[dict]:
        with self._lock:
            return list(self._events)

    def records(self) -> List[dict]:
        """Events with each build enriched (drained list, per-index source)."""
        return enrich(self.raw_events())

    def write_jsonl(self, path) -> None:
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in self.records())
        Path(path).write_text(text)


def load_jsonl(path) -> List[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def enrich(events: Sequence[dict]) -> List[dict]:
    """Attach ``drained`` and ``source`` to each raw build event."""
    submits = {e["sid"]: e for e in events if e["type"] == "submit"}
    out: List[dict] = []
    for e in events:
        if e["type"] != "build":
            out.append(dict(e))
            continue
        drained = []
        source: List[Optional[int]] = [None] * N_JOINTS
        stiffen: List[int] = []
        for sid in e["sids"]:
            s = submits.get(sid)
            if s is None:
                drained.append({"sid": None, "batch_id": None, "entry": None, "index": None,
                                "untagged": True})
                continue
            drained.append({"sid": sid, "batch_id": s["batch_id"], "entry": s["entry"],
                            "index": s["index"]})
            idx = s["index"]
            if idx is None:
                continue
            if s["goal_position"] is not None:
                # last writer wins, in drain order
                source[idx] = s["batch_id"] if s["batch_id"] is not None else UNATTRIBUTED_SOURCE
            if s["compliant"] is False and s["goal_position"] is None:
                stiffen.append(idx)
        b = dict(e)
        b.pop("sids", None)
        b["drained"] = drained
        b["source"] = source
        b["stiffen_no_goal"] = sorted(set(stiffen))
        out.append(b)
    return out


def classify(records: Sequence[dict]) -> Dict[str, Any]:
    """Pure classification over enriched records (see module docstring).

    Returns a dict with ``builds`` (one entry per build in record order, keyed
    by (gen, seq)), ``split_batches`` (sorted batch ids), ``batch_gaps_ns``
    and counts."""
    builds = [r for r in records if r["type"] == "build"]
    batches = {r["batch_id"]: r for r in records if r["type"] == "batch"}

    builds_of_batch: Dict[int, set] = {}
    for k, b in enumerate(builds):
        for d in b["drained"]:
            if d["batch_id"] is not None:
                builds_of_batch.setdefault(d["batch_id"], set()).add(k)
    split = {bid for bid, ks in builds_of_batch.items() if len(ks) >= 2}

    out_builds = []
    for b in builds:
        goal_batches = {s for s in b["source"] if s is not None}
        mixed = len(goal_batches) >= 2
        involved = any(d["batch_id"] in split for d in b["drained"] if d["batch_id"] is not None)
        klass = "mixed_batch" if mixed else ("split_involved_not_mixed" if involved else "neither")
        span_ns = None
        if mixed:
            ts = [batches[s]["t_ns"] for s in goal_batches if s in batches]
            if len(ts) == len(goal_batches):
                span_ns = max(ts) - min(ts)
        out_builds.append({
            "gen": b["gen"], "seq": b["seq"], "class": klass, "mixed": mixed,
            "split_involved": involved, "source_batches": sorted(goal_batches),
            "source_span_ns": span_ns,
        })

    gaps: List[dict] = []
    by_gen: Dict[int, List[dict]] = {}
    for r in records:
        if r["type"] == "batch":
            by_gen.setdefault(r["gen"], []).append(r)
    for g, lst in by_gen.items():
        lst = sorted(lst, key=lambda r: r["batch_id"])
        for a, c in zip(lst, lst[1:]):
            gaps.append({"gen": g, "batch_id": c["batch_id"], "prev_batch_id": a["batch_id"],
                         "gap_ns": c["t_ns"] - a["t_ns"]})

    return {
        "builds": out_builds,
        "split_batches": sorted(split),
        "batch_gaps_ns": gaps,
        "n_builds": len(builds),
        "n_batches": len(batches),
        "n_mixed": sum(1 for b in out_builds if b["mixed"]),
        "n_split_involved_not_mixed": sum(
            1 for b in out_builds if b["class"] == "split_involved_not_mixed"),
        "n_split_batches": len(split),
        "n_unattributed_submits": sum(
            1 for r in records if r["type"] == "submit" and r["batch_id"] is None),
    }


def _bits(x: float) -> bytes:
    return struct.pack("<d", float(x))


class CrossCheckResult:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.n_rows = 0
        self.n_builds = 0
        self.n_generations = 0
        self.overridden: List[dict] = []
        self.joined: Dict[Tuple[int, int], int] = {}   # (gen, seq) -> joint_command row ordinal

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {"ok": self.ok, "failures": self.failures, "n_joint_command_rows": self.n_rows,
                "n_builds": self.n_builds, "n_generations": self.n_generations,
                "overridden": self.overridden}


def cross_check(records: Sequence[dict], commands_rows: Sequence[dict], *,
                allowed_overrides: Optional[Dict[int, Dict[int, float]]] = None) -> CrossCheckResult:
    """See the module docstring ("Cross-check"). Fails closed."""
    res = CrossCheckResult()
    allowed_overrides = allowed_overrides or {}

    n_unattr = sum(1 for r in records if r["type"] == "submit" and r["batch_id"] is None)
    if n_unattr:
        res.failures.append(f"{n_unattr} unattributed submit(s) (no current batch)")

    builds = [r for r in records if r["type"] == "build"]
    res.n_builds = len(builds)
    cap_gens: Dict[int, List[dict]] = {}
    for b in builds:
        cap_gens.setdefault(b["gen"], []).append(b)
    cap_gen_order = sorted(cap_gens)
    res.n_generations = len(cap_gen_order)

    # Delimit generations in commands.jsonl: maximal runs of strictly increasing seq.
    runs: List[List[Tuple[int, dict]]] = []   # (joint_command row ordinal, row)
    prev_seq: Optional[int] = None
    ordinal = -1
    for row in commands_rows:
        if row.get("type") != "joint_command":
            continue
        ordinal += 1
        try:
            seq = int(row["seq"])
        except (KeyError, TypeError, ValueError):
            res.failures.append(f"joint_command row {ordinal} has no usable seq")
            return res
        if prev_seq is None or seq <= prev_seq:
            runs.append([])
        runs[-1].append((ordinal, row))
        prev_seq = seq
    res.n_rows = ordinal + 1

    if len(runs) != len(cap_gen_order):
        res.failures.append(
            f"generation count mismatch: commands.jsonl has {len(runs)} strictly-increasing-seq "
            f"run(s), capture has {len(cap_gen_order)} generation(s) with builds "
            "(ambiguous or missing boundary; failing closed)")
        return res

    for run, g in zip(runs, cap_gen_order):
        gb = cap_gens[g]
        row_seqs = [int(r["seq"]) for _, r in run]
        build_seqs = [b["seq"] for b in gb]
        if len(set(build_seqs)) != len(build_seqs):
            res.failures.append(f"gen {g}: duplicate predicted seq in capture builds {build_seqs}")
            continue
        if row_seqs != build_seqs:
            missing_rows = sorted(set(build_seqs) - set(row_seqs))
            missing_builds = sorted(set(row_seqs) - set(build_seqs))
            res.failures.append(
                f"gen {g}: seq mismatch; builds without a row {missing_rows}, "
                f"rows without a build {missing_builds}, "
                f"orders equal={sorted(row_seqs) == sorted(build_seqs)}")
            continue
        for (ordn, row), b in zip(run, gb):
            res.joined[(g, b["seq"])] = ordn
            tgt = row.get("target_rad")
            if not isinstance(tgt, list) or len(tgt) != N_JOINTS:
                res.failures.append(f"gen {g} seq {b['seq']}: row target_rad is not a 21-vector")
                continue
            ov = allowed_overrides.get(ordn, {})
            for i in range(N_JOINTS):
                if i in ov:
                    if _bits(tgt[i]) != _bits(ov[i]):
                        res.failures.append(
                            f"gen {g} seq {b['seq']} idx {i}: overridden value in row "
                            f"{tgt[i]!r} != declared override {ov[i]!r}")
                    else:
                        res.overridden.append({"gen": g, "seq": b["seq"], "index": i,
                                               "row_ordinal": ordn,
                                               "build_value": b["target"][i], "row_value": tgt[i]})
                elif _bits(tgt[i]) != _bits(b["target"][i]):
                    res.failures.append(
                        f"gen {g} seq {b['seq']} idx {i}: row target {tgt[i]!r} != "
                        f"built target {b['target'][i]!r}")
                    break
    return res
