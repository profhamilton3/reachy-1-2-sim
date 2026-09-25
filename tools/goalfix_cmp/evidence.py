"""Evidence loading, integrity checking, and cycle/leg slicing (plan §2.2).

Loads ``states.jsonl``/``commands.jsonl``, checks both against a session's
``SHA256SUMS`` before use, assigns each row a reset epoch and each command
its ``[t_lo, t_hi]`` bracket (see ``_simtime.py``), and slices legs from the
existing linker sidecars (``scripts/link_e1_flight.py``'s ``<log>.link.json``
output) -- this module consumes that sidecar's already-computed
``alignment`` list, it does not recompute it.

Never reads recorded evidence itself: every path is supplied by the caller.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO / "src", _REPO / "native_mujoco"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reachy_ai.motion.rig_routes import R_JOINTS, ARM7  # noqa: E402
from joint_map import JOINT_TABLE  # noqa: E402

from tools.goalfix_cmp._io import (  # noqa: E402
    IntegrityError, parse_sha256sums, verified,
)
from tools.goalfix_cmp._simtime import (  # noqa: E402
    Bracket, assign_command_epochs, assign_state_epochs, bracket_commands,
)

#: MJCF/JOINT_TABLE order for the right arm (native_mujoco/joint_map.py):
#: identical to rig_routes.R_JOINTS, asserted once at import time so any
#: future divergence between the two modules fails loudly here rather than
#: silently mis-indexing every 21-vector this module loads.
assert R_JOINTS[:7] == ARM7


class EvidenceError(ValueError):
    """Evidence is missing, malformed, or too ambiguous to use. Callers map
    this to rc=3 (evidence incomplete), never rc=0."""


def _read_jsonl(path) -> Tuple[List[dict], bool]:
    """Parsed JSON rows, plus whether the final line was truncated (and
    dropped) rather than a real parse error. A malformed NON-final line is
    a harder failure -- ``EvidenceError``, not a silent skip."""
    text = Path(path).read_text()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # trailing newline, not a truncation
    rows: List[dict] = []
    truncated = False
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                truncated = True
                continue
            raise EvidenceError(
                f"{path}: malformed JSON on line {i + 1} of {len(lines)} "
                "(not the final line -- a truncated recording only ever "
                "affects the last one)")
    return rows, truncated


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

@dataclass
class States:
    seq: np.ndarray             # the state's own monotonic seq (server-side)
    sim_step: np.ndarray
    sim_time_s: np.ndarray
    wall_time_ns: np.ndarray    # time.monotonic_ns(); locates windows only
    cmd_seq: np.ndarray         # last applied joint_command seq
    position_rad: np.ndarray    # (n, 21), JOINT_TABLE/R_JOINTS order
    compliant: np.ndarray       # (n, 21) bool
    epoch: np.ndarray           # (n,) int, from sim_step drops
    duplicate_indices: List[int]     # rows sharing (epoch, sim_step) with an earlier row
    truncated_final_line: bool

    def __len__(self) -> int:
        return len(self.sim_step)

    def right_arm_position_rad(self) -> np.ndarray:
        return self.position_rad[:, :8]


def load_states(path) -> States:
    rows, truncated = _read_jsonl(path)
    if not rows:
        raise EvidenceError(f"{path}: no usable state rows")
    try:
        seq = np.array([r["seq"] for r in rows], dtype=np.int64)
        sim_step = np.array([r["sim_step"] for r in rows], dtype=np.int64)
        sim_time_s = np.array([r["sim_time_s"] for r in rows], dtype=np.float64)
        wall_time_ns = np.array([r["wall_time_ns"] for r in rows], dtype=np.int64)
        cmd_seq = np.array([r["cmd_seq"] for r in rows], dtype=np.int64)
        position_rad = np.empty((len(rows), 21), dtype=np.float64)
        compliant = np.empty((len(rows), 21), dtype=bool)
        for i, r in enumerate(rows):
            by_name = {j["name"]: j for j in r["joints"]}
            for k, name in enumerate(_JOINT_ORDER):
                jd = by_name[name]
                position_rad[i, k] = jd["position_rad"]
                compliant[i, k] = bool(jd["compliant"])
    except (KeyError, TypeError) as exc:
        raise EvidenceError(f"{path}: state row missing an expected field: {exc}")

    epoch = assign_state_epochs(sim_step)
    duplicate_indices: List[int] = []
    for e in np.unique(epoch):
        mask = np.nonzero(epoch == e)[0]
        seen: Dict[int, int] = {}
        for idx in mask:
            s = int(sim_step[idx])
            if s in seen:
                duplicate_indices.append(int(idx))
            else:
                seen[s] = int(idx)

    return States(seq, sim_step, sim_time_s, wall_time_ns, cmd_seq,
                  position_rad, compliant, epoch, duplicate_indices, truncated)


#: JOINT_TABLE order (native_mujoco/joint_map.py), i.e. MJCF depth-first
#: order -- the order both target_rad and each state's own "joints" list
#: (matched here by name, not position) actually carry. Derived, not
#: hand-copied, so a future joint-table change fails loudly instead of
#: silently mis-indexing every 21-vector this module loads.
_JOINT_ORDER: Tuple[str, ...] = tuple(
    e.sdk_name for e in sorted(JOINT_TABLE, key=lambda e: e.mjcf_index))
assert len(_JOINT_ORDER) == 21, len(_JOINT_ORDER)
assert _JOINT_ORDER[:8] == R_JOINTS


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@dataclass
class Commands:
    kind: List[str]              # "joint_command" | "reset"
    seq: np.ndarray              # -1 for reset rows (never bracketed)
    target_rad: np.ndarray       # (n, 21); all-NaN for reset rows
    compliant: List[Optional[List[Optional[bool]]]]
    epoch: np.ndarray            # the epoch each row belongs to / ends
    reset_sim_step: List[Optional[int]]  # sim_step recorded on reset rows

    def __len__(self) -> int:
        return len(self.kind)

    def joint_command_mask(self) -> np.ndarray:
        return np.array([k == "joint_command" for k in self.kind])


def load_commands(path) -> Commands:
    rows, truncated = _read_jsonl(path)
    if not rows:
        raise EvidenceError(f"{path}: no usable command rows")
    kind: List[str] = []
    seq: List[int] = []
    target_rad = np.full((len(rows), 21), np.nan, dtype=np.float64)
    compliant: List[Optional[List[Optional[bool]]]] = []
    reset_sim_step: List[Optional[int]] = []
    for i, r in enumerate(rows):
        t = r.get("type")
        if t == "joint_command":
            kind.append("joint_command")
            seq.append(int(r["seq"]))
            tgt = r["target_rad"]
            if len(tgt) != 21:
                raise EvidenceError(
                    f"{path}: joint_command seq={r.get('seq')} has "
                    f"{len(tgt)} target_rad entries, expected 21")
            target_rad[i, :] = tgt
            compliant.append(r.get("compliant"))
            reset_sim_step.append(None)
        elif t == "reset":
            kind.append("reset")
            seq.append(-1)
            compliant.append(None)
            reset_sim_step.append(r.get("sim_step"))
        else:
            raise EvidenceError(f"{path}: unknown command type {t!r} on row {i}")
    epoch = assign_command_epochs(kind)
    return Commands(kind, np.array(seq, dtype=np.int64), target_rad, compliant,
                     epoch, reset_sim_step)


def check_epoch_counts(states: States, commands: Commands) -> None:
    """Cross-check (plan §2.2): the number of reset-derived epochs in
    ``states`` must equal the number in ``commands``. Raises
    ``EvidenceError`` (evidence incomplete) on a mismatch -- this is the
    "checked against the step drop" the plan requires, not an assumption."""
    n_state_epochs = int(states.epoch.max()) + 1 if len(states) else 0
    n_cmd_epochs = int(commands.epoch.max()) + 1 if len(commands) else 0
    if n_state_epochs != n_cmd_epochs:
        raise EvidenceError(
            "epoch count mismatch: states.jsonl implies "
            f"{n_state_epochs} epoch(s) from sim_step drops, "
            f"commands.jsonl implies {n_cmd_epochs} from reset entries")


# ---------------------------------------------------------------------------
# Evidence bundle: states + commands + brackets, integrity-checked together
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    states: States
    commands: Commands
    brackets: List[Bracket]           # one per commands row (reset rows: unplaceable)
    unplaceable_command_indices: List[int]

    def n_epochs(self) -> int:
        return int(self.states.epoch.max()) + 1 if len(self.states) else 0


def load_evidence(states_path, commands_path) -> Evidence:
    states = load_states(states_path)
    commands = load_commands(commands_path)
    check_epoch_counts(states, commands)
    jc_mask = commands.joint_command_mask()
    jc_idx = np.nonzero(jc_mask)[0]
    brackets_jc = bracket_commands(
        commands.seq[jc_idx], commands.epoch[jc_idx],
        states.sim_time_s, states.cmd_seq, states.epoch)
    brackets: List[Optional[Bracket]] = [None] * len(commands)
    for i, b in zip(jc_idx, brackets_jc):
        brackets[i] = b
    for i in np.nonzero(~jc_mask)[0]:
        brackets[i] = Bracket(int(commands.epoch[i]), None, None, None)
    unplaceable = [i for i, b in enumerate(brackets)
                   if commands.kind[i] == "joint_command" and b.unplaceable]
    return Evidence(states, commands, brackets, unplaceable)  # type: ignore[arg-type]


def verify_and_load(
    evidence_dir, states_rel: str, commands_rel: str, sha256sums_rel: str = "SHA256SUMS",
) -> Evidence:
    """Check ``states_rel``/``commands_rel`` against ``sha256sums_rel``
    (all relative to ``evidence_dir``) before loading either -- the D8
    narrow-grant path (assignment §0): no other file under ``evidence_dir``
    is ever opened."""
    sums = parse_sha256sums(Path(evidence_dir) / sha256sums_rel)
    states_path = verified(evidence_dir, states_rel, sums)
    commands_path = verified(evidence_dir, commands_rel, sums)
    return load_evidence(states_path, commands_path)


# ---------------------------------------------------------------------------
# Legs, from the existing linker sidecar's already-computed alignment
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    label: str
    epoch: int
    first_state_index: int
    last_state_index: int
    t_lo: float   # sim_time_s of the first aligned state
    t_hi: float   # sim_time_s of the last aligned state


def leg_from_sidecar(label: str, sidecar: Dict[str, Any], states: States) -> Leg:
    """A ``Leg`` from a linker sidecar's ``alignment`` list (as written by
    ``scripts/link_e1_flight.py``: a list of dicts each carrying
    ``server_seq``, the aligned state's own monotonic ``seq``). Raises
    ``EvidenceError`` (evidence incomplete) for a missing/empty alignment, a
    ``server_seq`` absent from ``states``, or a leg whose first and last
    aligned states fall in different epochs."""
    alignment = sidecar.get("alignment")
    if not alignment:
        raise EvidenceError(f"{label}: sidecar has no (or an empty) alignment list")
    seq_to_index = {int(s): i for i, s in enumerate(states.seq)}
    try:
        first_seq = int(alignment[0]["server_seq"])
        last_seq = int(alignment[-1]["server_seq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"{label}: malformed alignment entry: {exc}")
    if first_seq not in seq_to_index or last_seq not in seq_to_index:
        raise EvidenceError(
            f"{label}: alignment references a server_seq not present in states.jsonl")
    fi, li = seq_to_index[first_seq], seq_to_index[last_seq]
    if states.epoch[fi] != states.epoch[li]:
        raise EvidenceError(f"{label}: leg's aligned states cross a reset epoch")
    return Leg(label, int(states.epoch[fi]), fi, li,
               float(states.sim_time_s[fi]), float(states.sim_time_s[li]))


def check_legs_non_overlapping(legs: Sequence[Leg]) -> None:
    """Raises ``EvidenceError`` if two legs in the same epoch overlap in
    ``sim_time_s`` -- an overlapping sidecar is evidence-incomplete, per
    plan §2.2, not a silently-accepted double count."""
    by_epoch: Dict[int, List[Leg]] = {}
    for leg in legs:
        by_epoch.setdefault(leg.epoch, []).append(leg)
    for epoch, group in by_epoch.items():
        ordered = sorted(group, key=lambda l: l.t_lo)
        for a, b in zip(ordered, ordered[1:]):
            if b.t_lo < a.t_hi:
                raise EvidenceError(
                    f"epoch {epoch}: legs {a.label!r} and {b.label!r} overlap "
                    f"({a.t_lo}-{a.t_hi} vs {b.t_lo}-{b.t_hi})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    from tools.goalfix_cmp._io import write_result, RC_OK, RC_INCONCLUSIVE

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--states", default="states.jsonl")
    p.add_argument("--commands", default="commands.jsonl")
    p.add_argument("--sha256sums", default="SHA256SUMS")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    try:
        ev = verify_and_load(args.evidence_dir, args.states, args.commands, args.sha256sums)
    except (EvidenceError, IntegrityError) as exc:
        return write_result(args.out, RC_INCONCLUSIVE, {"ok": False, "reason": str(exc)})

    payload = {
        "ok": True,
        "n_states": len(ev.states),
        "n_commands": len(ev.commands),
        "n_epochs": ev.n_epochs(),
        "duplicate_state_indices": ev.states.duplicate_indices,
        "truncated_final_state_line": ev.states.truncated_final_line,
        "unplaceable_command_indices": ev.unplaceable_command_indices,
    }
    return write_result(args.out, RC_OK, payload)


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
