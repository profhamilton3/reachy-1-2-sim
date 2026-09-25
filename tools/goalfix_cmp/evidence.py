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
from tools.goalfix_cmp import _simtime as st  # noqa: E402
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
    truncated_final_line: bool = False   # T10.2: flagged, not silently dropped

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
            # T10.3: a missing `seq` or a malformed `target_rad` is
            # evidence-incomplete, never an uncaught KeyError/TypeError that
            # would escape every CLI as an untranslated traceback (rc 1, no
            # JSON -- the exact review-row bug this closes).
            try:
                row_seq = int(r["seq"])
                tgt = r["target_rad"]
                if len(tgt) != 21:
                    raise EvidenceError(
                        f"{path}: joint_command seq={r.get('seq')} has "
                        f"{len(tgt)} target_rad entries, expected 21")
                tgt_row = [float(v) for v in tgt]
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceError(
                    f"{path}: joint_command on row {i} is malformed: {exc}") from exc
            kind.append("joint_command")
            seq.append(row_seq)
            target_rad[i, :] = tgt_row
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
                     epoch, reset_sim_step, truncated_final_line=truncated)


def check_epoch_counts(states: States, commands: Commands) -> None:
    """Cross-check (plan §2.2; T3/review §3.1/M3): the number of
    reset-derived epochs in ``states`` must equal the number in
    ``commands``, and each reset row's own recorded ``sim_step`` must agree
    with the step drop it opens. Raises ``EvidenceError`` (evidence
    incomplete) on either mismatch.

    Epochs are counted as ``n_reset_rows + 1`` (T3), never
    ``commands.epoch.max() + 1``: a reset row is stamped with the epoch it
    ENDS (``_simtime.assign_command_epochs``), so a recording whose LAST
    row is a reset undercounts by one under the old ``max()+1`` rule -- a
    valid recording was rejected as evidence-incomplete for this alone (the
    previous handoff's B1 "structural finding" was this bug, not a property
    of the data; see T3's correction to that handoff).
    """
    n_state_epochs = int(states.epoch.max()) + 1 if len(states) else 0
    n_reset_rows = sum(1 for k in commands.kind if k == "reset")
    n_cmd_epochs = n_reset_rows + 1 if len(commands) else 0
    if n_state_epochs != n_cmd_epochs:
        raise EvidenceError(
            "epoch count mismatch: states.jsonl implies "
            f"{n_state_epochs} epoch(s) from sim_step drops, "
            f"commands.jsonl implies {n_cmd_epochs} from {n_reset_rows} "
            "reset row(s) (+1)")

    # Each reset row's recorded sim_step must agree with the step drop it
    # opens: strictly after the last state recorded in the epoch it
    # closes, and no further than one recording interval beyond it.
    # States are recorded every `state_every` sim_steps, not every one (a
    # real B4 s1 recording samples every 10; make_fixtures' FlightSim
    # fixtures sample every 1) -- an exact "+1" relationship only holds
    # when state_every==1. `state_every` is derived from the data itself
    # (the most common consecutive-state step within the closing epoch),
    # never hard-coded, so this holds for both. Validated against real
    # B4 s1 data (V1): a genuine reset row recorded sim_step 7 steps
    # (state_every=10) after the last sampled state of the epoch it
    # closed -- an exact "+1" check rejected this valid recording.
    reset_i = 0
    for state_i in range(1, len(states)):
        if states.sim_step[state_i] < states.sim_step[state_i - 1]:
            # states.epoch[state_i - 1] just closed; find the reset row
            # that closes the SAME epoch (the (states.epoch[state_i-1]+1)-th
            # reset row in file order, 0-indexed).
            closing_epoch = int(states.epoch[state_i - 1])
            resets_seen = 0
            reset_row_index = None
            for ci, kind in enumerate(commands.kind):
                if kind == "reset":
                    if resets_seen == closing_epoch:
                        reset_row_index = ci
                        break
                    resets_seen += 1
            if reset_row_index is None:
                raise EvidenceError(
                    f"states.jsonl drops sim_step after epoch {closing_epoch}, "
                    "but commands.jsonl has no matching reset row")
            recorded = commands.reset_sim_step[reset_row_index]
            last_step_of_closing_epoch = int(states.sim_step[state_i - 1])
            epoch_step_mask = states.epoch == closing_epoch
            epoch_steps = states.sim_step[epoch_step_mask]
            step_diffs = np.diff(epoch_steps)
            step_diffs = step_diffs[step_diffs > 0]
            state_every = int(np.min(step_diffs)) if len(step_diffs) else 1
            ok = (recorded is not None
                  and last_step_of_closing_epoch <= int(recorded) <= last_step_of_closing_epoch + state_every)
            if not ok:
                raise EvidenceError(
                    f"reset row {reset_row_index}: recorded sim_step={recorded} "
                    f"does not agree with the step drop it opens (epoch "
                    f"{closing_epoch}'s last state is sim_step="
                    f"{last_step_of_closing_epoch})")
            reset_i += 1


# ---------------------------------------------------------------------------
# Evidence bundle: states + commands + brackets, integrity-checked together
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    states: States
    commands: Commands
    brackets: List[Bracket]           # one per commands row (reset rows: unplaceable)
    unplaceable_command_indices: List[int]
    #: T2: per-epoch bridge-restart detection (container-recreate/seq-restart
    #: regime), keyed by epoch index.
    bridge_sessions: Dict[int, st.BridgeSessionInfo] = field(default_factory=dict)

    def n_epochs(self) -> int:
        return int(self.states.epoch.max()) + 1 if len(self.states) else 0


def load_evidence(states_path, commands_path) -> Evidence:
    states = load_states(states_path)
    commands = load_commands(commands_path)
    check_epoch_counts(states, commands)
    jc_mask = commands.joint_command_mask()
    jc_idx = np.nonzero(jc_mask)[0]
    bridge_sessions = st.detect_bridge_sessions(
        commands.seq[jc_idx], commands.epoch[jc_idx], states.cmd_seq, states.epoch)
    brackets_jc = bracket_commands(
        commands.seq[jc_idx], commands.epoch[jc_idx],
        states.sim_time_s, states.cmd_seq, states.epoch,
        bridge_sessions=bridge_sessions)
    brackets: List[Optional[Bracket]] = [None] * len(commands)
    for i, b in zip(jc_idx, brackets_jc):
        brackets[i] = b
    for i in np.nonzero(~jc_mask)[0]:
        brackets[i] = Bracket(int(commands.epoch[i]), None, None, None)
    unplaceable = [i for i, b in enumerate(brackets)
                   if commands.kind[i] == "joint_command" and b.unplaceable]
    return Evidence(states, commands, brackets, unplaceable,  # type: ignore[arg-type]
                     bridge_sessions=bridge_sessions)


def check_no_unplaceable_in_range(evidence: Evidence, command_indices: Sequence[int]) -> None:
    """T2 (assignment §2 T2 acceptance): an unplaceable ``joint_command``
    inside a leg or the affected segment is evidence incomplete, in every
    CLI -- never silently classified as ``carry``/``fresh`` for gating
    purposes. Raises ``EvidenceError`` naming every unplaceable index found,
    not just the first."""
    bad = [i for i in command_indices
           if evidence.commands.kind[i] == "joint_command"
           and evidence.brackets[i].unplaceable]
    if bad:
        ambiguous = [i for i in bad if evidence.brackets[i].seq_ambiguous]
        raise EvidenceError(
            f"unplaceable joint_command(s) inside range: {bad}"
            + (f" ({ambiguous} seq-ambiguous)" if ambiguous else ""))


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


def commands_in_leg(evidence: "Evidence", leg: Leg) -> List[int]:
    """T4 item 2: the ``joint_command`` global indices belonging to
    ``leg``, found BY BRACKET from the leg's own aligned-state span --
    never an operator-supplied index. A command belongs to the leg if its
    bracket's ``hi_state_index`` (the first state reporting it applied)
    falls within ``[leg.first_state_index, leg.last_state_index]``."""
    out = []
    for i, kind in enumerate(evidence.commands.kind):
        if kind != "joint_command":
            continue
        b = evidence.brackets[i]
        if b.unplaceable or b.hi_state_index is None:
            continue
        if leg.first_state_index <= b.hi_state_index <= leg.last_state_index:
            out.append(i)
    return out


def leg_turn_on_state_index(evidence: "Evidence", leg_command_indices: Sequence[int]) -> Optional[int]:
    """T4/T7: the global STATE index of the present-position reading in
    force at a leg's own ``turn_on`` -- "the goal set at turn_on" (plan
    §7.1). This is the state immediately BEFORE the leg's own first
    command was applied (its bracket's ``t_lo`` state): ``hi_state_index -
    1``, which always holds globally (states never interleave epochs, and
    a restart's own boundary state -- see ``_simtime.bracket_commands`` --
    is exactly this same "last state before" position). ``None`` if the
    leg has no (placeable) commands at all."""
    if not leg_command_indices:
        return None
    first = min(leg_command_indices)
    hi = evidence.brackets[first].hi_state_index
    return None if hi is None else hi - 1


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
