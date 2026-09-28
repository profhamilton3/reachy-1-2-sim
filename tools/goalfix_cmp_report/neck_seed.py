"""Report-only neck / seed extractor for the B4 comparison (plan rev 4 §4.2 and §4).

Per cycle and per arm LABEL (a label only; nothing here depends on it), reports

* the NECK TARGETS (indices 16-18) of each leg's first command,
* the ACTUAL NECK POSITIONS (``position_rad``, indices 16-18) at the P4 state,
  the P5 state, each leg's first-command bracket ``t_hi`` state and each leg's
  last aligned state, each with target/actual minus the native keyframe,
* the 13-joint seed deviation of each leg's first command, obtained by calling
  the shipped ``pathcheck.check_c8(targets21, keyframe21)`` and taking its
  second value (not re-implemented).

REPORT-ONLY.  No threshold, no gate, no rc-based verdict, and no change to any
approved window or carry definition.  rc 0 = every field ``ok``; rc 3 = at
least one field ``incomplete``; rc 2 = usage error.  The rc never expresses a
verdict on the values.  Reporting these figures does not show that the two
arms are equivalent (plan §4.2).

Inputs are all explicit; no provenance-bearing value has a silent default.
Evidence is loaded through ``tools.goalfix_cmp.evidence`` (checksum-verified;
reused, not re-implemented) and legs are located exactly as the shipped
``cycle`` CLI locates them (``cycle.resolve_leg`` with the cycle manifest's
sidecar names, route names and run directory; read-only reuse).

Reading of plan §4.2 "the first ``joint_command`` after ``turn_on``": this tool
reads it as ``command_indices[0]`` of the leg as the shipped tooling locates it
(the first PLACEABLE command whose applying state lies in the sidecar's aligned
span; the row ``pathcheck.check_c8`` uses for its seed figure).  The two can
differ in real evidence: (a) ``commands_in_leg`` skips unplaceable commands, so
an unplaceable command sent first would leave ``command_indices[0]`` at a
later one (this tool reports that leg as incomplete when unplaceable
``joint_command`` rows immediately precede the located row); (b) a command
inside the aligned span but before the ``turn_on`` command makes
``command_indices[0]`` a pre-``turn_on`` command (``holds.find_turn_on_command``
reports that as a lead-in violation).  ``turn_on_match_local_index`` is
reported alongside, informationally, so a difference is visible.

Completeness: every field carries ``status`` (``ok``/``incomplete``) and a
``reason``.  Missing, ambiguous (duplicate ``sim_step``, unplaceable first
command, a bracket whose state lies in another epoch) or out-of-epoch evidence
gives ``value: null`` plus the reason.  Nothing is ever zero-filled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for _p in (_REPO, _REPO / "src", _REPO / "native_mujoco", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from joint_map import JOINT_TABLE  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp._io import IntegrityError, RC_INCONCLUSIVE, RC_OK, RC_STOP, write_result, sha256_of  # noqa: E402

RC_USAGE = RC_STOP            # 2
RC_INCOMPLETE = RC_INCONCLUSIVE   # 3

NECK_IDX = (16, 17, 18)
NECK_NAMES = tuple(ev._JOINT_ORDER[i] for i in NECK_IDX)
assert NECK_NAMES == ("neck_roll", "neck_pitch", "neck_yaw"), NECK_NAMES
OTHER13 = slice(8, 21)


class UsageError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Field helpers: never zero-fill
# ---------------------------------------------------------------------------

def _ok(value: Any) -> dict:
    return {"status": "ok", "reason": "", "value": value}


def _inc(reason: str) -> dict:
    return {"status": "incomplete", "reason": reason, "value": None}


def _neck(vec: Sequence[float]) -> Dict[str, float]:
    return {n: float(vec[i]) for n, i in zip(NECK_NAMES, NECK_IDX)}


def _neck_minus(vec: Sequence[float], kf21: Sequence[float]) -> Dict[str, float]:
    return {n: float(vec[i]) - float(kf21[i]) for n, i in zip(NECK_NAMES, NECK_IDX)}


# ---------------------------------------------------------------------------
# Keyframe
# ---------------------------------------------------------------------------

def load_keyframe(path: str, name: str, expected_sha256: str,
                  source_rev: str) -> Tuple[Optional[dict], str]:
    """(keyframe dict, "") or (None, reason).  Verifies the file's sha256, parses
    ``<key name=...>``'s ``qpos`` (exactly one such key, exactly 21 finite
    values) and maps it MJCF order -> ``evidence._JOINT_ORDER`` through
    ``joint_map.JOINT_TABLE``'s ``mjcf_index``.  Never a default keyframe."""
    p = Path(path)
    if not p.is_file():
        return None, f"keyframe file {path} does not exist"
    actual = sha256_of(p)
    if actual != expected_sha256.lower():
        return None, f"keyframe sha256 mismatch (expected {expected_sha256.lower()}, got {actual})"
    try:
        root = ET.parse(str(p)).getroot()
    except ET.ParseError as exc:
        return None, f"keyframe XML does not parse: {exc}"
    keys = [k for k in root.iter("key") if k.attrib.get("name") == name]
    if len(keys) != 1:
        return None, f"expected exactly one <key name={name!r}>, found {len(keys)}"
    qpos_text = keys[0].attrib.get("qpos")
    if qpos_text is None:
        return None, f"<key name={name!r}> has no qpos attribute"
    try:
        qpos = [float(t) for t in qpos_text.split()]
    except ValueError as exc:
        return None, f"qpos is not numeric: {exc}"
    if len(qpos) != 21:
        return None, f"qpos has {len(qpos)} values, expected exactly 21"
    if not all(math.isfinite(v) for v in qpos):
        return None, "qpos has a non-finite value"
    mjcf_index = {e.sdk_name: e.mjcf_index for e in JOINT_TABLE}
    try:
        kf21 = [qpos[mjcf_index[n]] for n in ev._JOINT_ORDER]
    except (KeyError, IndexError) as exc:
        return None, f"joint mapping failed: {exc!r}"
    return {"path": str(path), "sha256": actual, "name": name, "source_rev": source_rev,
            "qpos21_joint_order": kf21}, ""


# ---------------------------------------------------------------------------
# State selection (P4 / P5)
# ---------------------------------------------------------------------------

def first_state_at_or_after(states: ev.States, epoch: int, sim_step: int) -> Tuple[Optional[int], str]:
    rows = np.nonzero(states.epoch == epoch)[0]
    if rows.size == 0:
        return None, f"no states in epoch {epoch}"
    cand = rows[states.sim_step[rows] >= sim_step]
    if cand.size == 0:
        return None, f"no state in epoch {epoch} with sim_step >= {sim_step}"
    i = int(cand[0])
    dup = rows[states.sim_step[rows] == states.sim_step[i]]
    if dup.size > 1:
        return None, (f"ambiguous: {dup.size} states share (epoch {epoch}, "
                      f"sim_step {int(states.sim_step[i])})")
    return i, ""


def _state_field(states: ev.States, i: Optional[int], why: str, epoch: int,
                 kf21: Sequence[float], what: str) -> dict:
    if i is None:
        return _inc(why or f"{what}: state not located")
    ep = int(states.epoch[i])
    if ep != epoch:
        return _inc(f"{what}: state {i} lies in epoch {ep}, expected {epoch}")
    pos = states.position_rad[i]
    if not all(math.isfinite(float(pos[k])) for k in NECK_IDX):
        return _inc(f"{what}: state {i} has a non-finite neck position")
    return _ok({"state_index": int(i), "seq": int(states.seq[i]),
                "sim_step": int(states.sim_step[i]), "epoch": ep,
                "position_rad": _neck(pos), "minus_keyframe": _neck_minus(pos, kf21)})


# ---------------------------------------------------------------------------
# Legs
# ---------------------------------------------------------------------------

_LEGS = (
    ("setup", "setup_sidecar", "PLACE_ROUTE"),
    ("flight", "flight_sidecar", "LIFT_TO_PRESENT"),
)


def _locate_leg(evd: ev.Evidence, control_dir: Path, manifest: dict, label: str,
                key: str, route_name: str, run_dir: str):
    """(LegSpec, ev.Leg, "") or (None, None, reason).  As the shipped ``cycle`` CLI."""
    try:
        sidecar = str(control_dir / manifest[key])
        route = getattr(R, route_name)
        guard = R.CRITICAL_JOINTS if route_name == "PLACE_ROUTE" else R._PRESENT_GUARD
        spec = cyc.resolve_leg(evd, sidecar, label, route, guard,
                               expected_route_name=route_name, expected_server_run_dir=run_dir)
        doc = json.loads(Path(sidecar).read_text())
        leg = ev.leg_from_sidecar(label, doc, evd.states)
    except (ev.EvidenceError, IntegrityError, OSError, ValueError, KeyError, TypeError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    return spec, leg, ""


def _leg_fields(evd: ev.Evidence, spec, leg, epoch: int, kf21: Sequence[float],
                label: str) -> Tuple[dict, dict, dict, dict]:
    """(neck_target, first_cmd_t_hi_state, last_aligned_state, seed_deviation_13)."""
    idx = list(spec.command_indices)
    first = idx[0]
    cmd_epoch = int(evd.commands.epoch[first])
    if leg.epoch != epoch or cmd_epoch != epoch:
        why = (f"{label}: leg lies in epoch {leg.epoch} (first command in epoch {cmd_epoch}), "
               f"expected {epoch}")
        return _inc(why), _inc(why), _inc(why), _inc(why)

    # Last aligned state (as the shipped tooling defines it: ev.Leg.last_state_index).
    last_state = _state_field(evd.states, leg.last_state_index, "", epoch, kf21,
                              f"{label} last aligned state")

    # An unplaceable joint_command immediately before the located first command could
    # be the true first command after turn_on: refuse rather than report the wrong row.
    j = first - 1
    unplaceable_before: List[int] = []
    while j >= 0 and evd.commands.kind[j] == "joint_command" and evd.brackets[j].unplaceable:
        unplaceable_before.append(j)
        j -= 1
    if unplaceable_before:
        why = (f"{label}: unplaceable joint_command row(s) {sorted(unplaceable_before)} immediately "
               f"precede the located first command (row {first}); the first command after turn_on "
               "is ambiguous")
        return _inc(why), _inc(why), last_state, _seed_field(evd, idx, kf21, label, epoch)

    # First command's neck target.
    tgt = evd.commands.target_rad[first]
    if not all(math.isfinite(float(tgt[k])) for k in NECK_IDX):
        neck_target = _inc(f"{label}: first command (row {first}) has a non-finite neck target")
    else:
        match = None
        try:
            match = holds.find_turn_on_command(evd, idx)
        except Exception:                                   # informational only
            match = None
        neck_target = _ok({
            "row": int(first), "seq": int(evd.commands.seq[first]), "epoch": cmd_epoch,
            "command_indices_0_is_leg_first_placeable_command": True,
            "turn_on_match_local_index": None if match is None else int(match.local_index),
            "target_rad": _neck(tgt), "minus_keyframe": _neck_minus(tgt, kf21)})

    # First command's bracket t_hi state.
    b = evd.brackets[first]
    if b.unplaceable or b.hi_state_index is None:
        t_hi_state = _inc(f"{label}: first command (row {first}) has an unplaceable bracket")
    else:
        hi = int(b.hi_state_index)
        eps = {int(b.epoch), int(evd.states.epoch[hi]), int(evd.states.epoch[max(hi - 1, 0)])}
        if len(eps) != 1:
            t_hi_state = _inc(f"{label}: first command's bracket spans epochs {sorted(eps)}")
        else:
            t_hi_state = _state_field(evd.states, hi, "", epoch, kf21,
                                      f"{label} first-command bracket t_hi state")
            if t_hi_state["status"] == "ok":
                t_hi_state["value"]["bracket_t_hi_sim_time_s"] = float(b.t_hi)
                t_hi_state["value"]["command_row"] = int(first)

    return neck_target, t_hi_state, last_state, _seed_field(evd, idx, kf21, label, epoch)


def _seed_field(evd: ev.Evidence, idx: Sequence[int], kf21: Sequence[float], label: str,
                epoch: int) -> dict:
    targets21 = evd.commands.target_rad[np.asarray(idx)]
    _res, seed_dev = pc.check_c8(targets21, kf21)      # shipped function; second value only
    if seed_dev is None or not math.isfinite(seed_dev):
        return _inc(f"{label}: no finite seed deviation from pathcheck.check_c8")
    first = np.asarray(targets21[0])
    diffs = {ev._JOINT_ORDER[i]: float(first[i]) - float(kf21[i]) for i in range(8, 21)}
    return _ok({"max_abs_rad": float(seed_dev), "first_command_row": int(idx[0]),
                "other13_minus_keyframe": diffs,
                "source": "tools.goalfix_cmp.pathcheck.check_c8 second return value"})


# ---------------------------------------------------------------------------
# Whole extraction
# ---------------------------------------------------------------------------

def _all_incomplete(payload: dict, why: str) -> None:
    payload["neck_targets"] = {"setup": _inc(why), "flight": _inc(why)}
    payload["neck_actual"] = {k: _inc(why) for k in (
        "p4", "p5", "setup_first_cmd_t_hi", "setup_last_aligned",
        "flight_first_cmd_t_hi", "flight_last_aligned")}
    payload["seed_deviation_13"] = {"setup": _inc(why), "flight": _inc(why)}


def _statuses(payload: dict) -> List[str]:
    out = []
    for grp in ("neck_targets", "neck_actual", "seed_deviation_13"):
        out.extend(f["status"] for f in payload[grp].values())
    return out


def extract(*, ev_dir: str, run: Optional[str], states_rel: Optional[str],
            commands_rel: Optional[str], sha256sums: str, control_dir: str, cycle: str,
            arm: str, epoch: int, keyframe_xml: str, keyframe_name: str,
            keyframe_sha256: str, keyframe_source_rev: str, state_every: int,
            fixed_sim_step: int) -> Tuple[dict, int]:
    states_rel = states_rel or (f"e1_server_runs/{run}/states.jsonl" if run else "states.jsonl")
    commands_rel = commands_rel or (
        f"e1_server_runs/{run}/commands.jsonl" if run else "commands.jsonl")
    payload: dict = {
        "report_only": True, "tool": "tools.goalfix_cmp_report.neck_seed", "cycle": cycle,
        "arm_label": arm, "expected_epoch": epoch, "state_every": state_every,
        "fixed_sim_step": fixed_sim_step, "states_rel": states_rel, "commands_rel": commands_rel,
        "reading_of_first_command": (
            "plan §4.2's 'first joint_command after turn_on' is read as command_indices[0] of "
            "the leg as the shipped cycle tooling locates it"),
        "note": "report-only; no threshold, gate or verdict; does not show the arms are equivalent",
    }

    kf, kf_why = load_keyframe(keyframe_xml, keyframe_name, keyframe_sha256, keyframe_source_rev)
    if kf is None:
        payload["keyframe"] = _inc(kf_why)
        _all_incomplete(payload, f"keyframe: {kf_why}")
        payload["all_ok"] = False
        return payload, RC_INCOMPLETE
    kf21 = kf["qpos21_joint_order"]
    payload["keyframe"] = {"status": "ok", "reason": "", "path": kf["path"], "sha256": kf["sha256"],
                           "name": kf["name"], "source_rev": kf["source_rev"],
                           "neck_keyframe_rad": _neck(kf21)}

    try:
        evd = ev.verify_and_load(ev_dir, states_rel, commands_rel, sha256sums)
    except (ev.EvidenceError, IntegrityError, OSError, ValueError) as exc:
        _all_incomplete(payload, f"evidence: {type(exc).__name__}: {exc}")
        payload["all_ok"] = False
        return payload, RC_INCOMPLETE

    states = evd.states
    i4, w4 = first_state_at_or_after(states, epoch, state_every)
    i5, w5 = first_state_at_or_after(states, epoch, fixed_sim_step)
    na = {"p4": _state_field(states, i4, f"P4: {w4}", epoch, kf21, "P4 state"),
          "p5": _state_field(states, i5, f"P5: {w5}", epoch, kf21, "P5 state")}

    nt: Dict[str, dict] = {}
    sd: Dict[str, dict] = {}
    control = Path(control_dir)
    manifest, m_why = None, ""
    manifest_path = control / f"cycle_{cycle}.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("cycle") != cycle:
            manifest, m_why = None, f"manifest cycle {manifest.get('cycle')!r} != {cycle!r}"
    except (OSError, ValueError) as exc:
        manifest, m_why = None, f"cycle manifest {manifest_path.name}: {type(exc).__name__}: {exc}"
    run_dir = str((Path(ev_dir) / "e1_server_runs" / run).resolve() if run
                  else Path(ev_dir).resolve())

    for label, key, route_name in _LEGS:
        if manifest is None:
            why = f"{label}: {m_why}"
            nt[label], sd[label] = _inc(why), _inc(why)
            na[f"{label}_first_cmd_t_hi"], na[f"{label}_last_aligned"] = _inc(why), _inc(why)
            continue
        spec, leg, why = _locate_leg(evd, control, manifest, label, key, route_name, run_dir)
        if spec is None:
            why = f"{label}: {why}"
            nt[label], sd[label] = _inc(why), _inc(why)
            na[f"{label}_first_cmd_t_hi"], na[f"{label}_last_aligned"] = _inc(why), _inc(why)
            continue
        nt[label], na[f"{label}_first_cmd_t_hi"], na[f"{label}_last_aligned"], sd[label] = \
            _leg_fields(evd, spec, leg, epoch, kf21, label)

    payload["neck_targets"] = nt
    payload["neck_actual"] = {k: na[k] for k in (
        "p4", "p5", "setup_first_cmd_t_hi", "setup_last_aligned",
        "flight_first_cmd_t_hi", "flight_last_aligned")}
    payload["seed_deviation_13"] = sd
    all_ok = all(s == "ok" for s in _statuses(payload))
    payload["all_ok"] = all_ok
    return payload, (RC_OK if all_ok else RC_INCOMPLETE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ev-dir", required=True)
    p.add_argument("--run", help="run directory name under <ev-dir>/e1_server_runs/ (as the cycle CLI)")
    p.add_argument("--states")
    p.add_argument("--commands")
    p.add_argument("--sha256sums", default="SHA256SUMS")
    p.add_argument("--control-dir", required=True)
    p.add_argument("--cycle", required=True)
    p.add_argument("--arm", choices=["A", "B"], required=True, help="label only")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--keyframe-xml", required=True)
    p.add_argument("--keyframe-name", required=True)
    p.add_argument("--keyframe-sha256", required=True)
    p.add_argument("--keyframe-source-rev", required=True)
    p.add_argument("--state-every", type=int, required=True)
    p.add_argument("--fixed-sim-step", type=int, required=True)
    p.add_argument("--out", required=True)
    return p


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)     # argparse exits 2 on a usage error
    try:
        if args.epoch < 0 or args.state_every < 0 or args.fixed_sim_step < 0:
            raise UsageError("--epoch, --state-every and --fixed-sim-step must be >= 0")
        h = args.keyframe_sha256.lower()
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise UsageError("--keyframe-sha256 must be 64 hex characters")
        if not args.keyframe_source_rev.strip() or not args.keyframe_name.strip():
            raise UsageError("--keyframe-source-rev and --keyframe-name must be non-empty")
    except UsageError as exc:
        print(f"usage error: {exc}", file=sys.stderr)
        try:
            write_result(args.out, RC_USAGE, {"report_only": True, "usage_error": str(exc)})
        except Exception:
            pass
        return RC_USAGE
    payload, rc = extract(
        ev_dir=args.ev_dir, run=args.run, states_rel=args.states, commands_rel=args.commands,
        sha256sums=args.sha256sums, control_dir=args.control_dir, cycle=args.cycle, arm=args.arm,
        epoch=args.epoch, keyframe_xml=args.keyframe_xml, keyframe_name=args.keyframe_name,
        keyframe_sha256=h, keyframe_source_rev=args.keyframe_source_rev,
        state_every=args.state_every, fixed_sim_step=args.fixed_sim_step)
    return write_result(args.out, rc, payload)


def main() -> int:
    return _cli()


if __name__ == "__main__":
    raise SystemExit(main())
