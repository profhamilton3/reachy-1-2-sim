"""Shared experiment-acceptance gate (PR #124 re-review, R3).

`leg.sh`/`parked.sh` already STOP when the recorder or the linker
(`link_e1_flight.py`) fails, but neither of them -- nor
`e1_tail_check.check_init`, which used to read only `contacts_recorded`
-- ever looked at what the linker's evidence actually SAYS happened
during the flight. A recorded contact and a 20 mm board displacement
both produced a `contacts_recorded: True` sidecar (the evidence about
the recording is complete) and were both reported `LEG ok` anyway
(2026-09-16 re-review, R3): `contacts_recorded=True` is a completeness
verdict, not a no-contact verdict, and nothing in the chain enforced
that distinction. `link_e1_flight.main()`'s exit codes (4/5) already
cover evidence that is missing or malformed; this module is the one
place that reads a *complete* sidecar's `contacts` and `displacement_m`
fields and decides whether the experiment they describe is acceptable.

`leg.sh`, `parked.sh`, and `e1_tail_check.check_init` (Stage 0's own
acceptance check) all call `evaluate()` (or this file's CLI) instead of
re-implementing the same read -- "one shared experiment-acceptance gate
... including Stage 0 INIT".

`evaluate()` is deliberately layered ON TOP of the linker rather than
folded into it: it never writes to the sidecar or the log, so a
rejected experiment's evidence is left exactly as the linker wrote it
and stays fully analyzable after the fact. Evidence validity and
experiment acceptance are two distinct questions, both fail-closed:

  * evidence invalid (`evidence_ok=False`, `accepted=None`) -- the
    sidecar is missing, unparseable, not an object, not tied to THIS
    recording (`sidecar['log']` names a different log -- a hand-copied
    or stale sidecar), or its `contacts_recorded`/`contacts`/
    `displacement_m` fields are not the shape a successful linker run
    guarantees. `evaluate()` cannot tell whether the experiment was
    clean, so it refuses.
  * experiment not accepted (`evidence_ok=True`, `accepted=False`) --
    the evidence is valid and complete, and it says either a contact
    happened (`contacts` non-empty) or some tracked object moved more
    than `DISPLACEMENT_TOL_M` (1 mm) between the first and last sample.
    The evidence is trusted; the experiment it describes is rejected.

Both cases make `ok=False`. Only `evidence_ok=True` and `accepted=True`
(no contacts, every displacement finite and <= `DISPLACEMENT_TOL_M`)
makes `ok=True`.

CLI: `python3 scripts/experiment_gate.py <log_path>` -- exit 0 if
accepted, 1 if evidence is invalid, 2 if evidence is valid but the
experiment is rejected. Any non-zero exit is a STOP for
`leg.sh`/`parked.sh`, which both call this right after the linker
succeeds and after the recording has already been archived to
`recorder_logs/` -- so a rejection never costs the recording its copy.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any, Dict, Optional

#: The review's own bound ("displacement_m > 1 mm -> per-board stop"),
#: and the same magnitude `link_e1_flight._SETTLED_TOL_XY_M`/
#: `_SETTLED_TOL_Z_M` use for the first-sample settled check -- this is
#: the companion bound across the whole flight (first sample to last),
#: not just at the start.
DISPLACEMENT_TOL_M: float = 0.001


def sidecar_path_for(log_path) -> pathlib.Path:
    """Same `.with_suffix(".link.json")` derivation `leg.sh` and
    `link_e1_flight.sidecar_path_for` use. Recomputed here rather than
    imported so this module stays free of `link_e1_flight`'s heavier
    dependencies (kinematics, `SceneModel`) that a shell-invoked gate
    check has no other reason to load."""
    return pathlib.Path(log_path).with_suffix(".link.json")


def _result(*, ok: bool, evidence_ok: bool, evidence_reason: str = "",
           accepted: Optional[bool] = None, reason: str = "",
           contacts_count: Optional[int] = None,
           max_displacement_m: Optional[float] = None,
           displacement_by_object_m: Optional[Dict[str, float]] = None,
           sidecar_path=None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "evidence_ok": evidence_ok,
        "evidence_reason": evidence_reason,
        "accepted": accepted,
        "reason": reason or evidence_reason,
        "contacts_count": contacts_count,
        "max_displacement_m": max_displacement_m,
        "displacement_by_object_m": displacement_by_object_m,
        "sidecar_path": str(sidecar_path) if sidecar_path is not None else None,
    }


def evaluate(log_path) -> Dict[str, Any]:
    """Judge the recorded experiment for `log_path` against its own
    linked sidecar. See the module docstring for the evidence-validity
    vs experiment-acceptance split."""
    sidecar_path = sidecar_path_for(log_path)
    try:
        text = sidecar_path.read_text()
    except OSError:
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason=f"no linked sidecar at {sidecar_path} (run "
                            "link_e1_flight.py first)")
    try:
        sidecar = json.loads(text)
    except json.JSONDecodeError as exc:
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason=f"sidecar {sidecar_path} is not valid JSON: {exc}")
    if not isinstance(sidecar, dict):
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason=f"sidecar {sidecar_path} is not a JSON object")

    log_name = pathlib.Path(log_path).name
    sidecar_log = sidecar.get("log")
    if sidecar_log != log_name:
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason=f"sidecar names log {sidecar_log!r}, not this "
                            f"recording {log_name!r} -- wrong-recording "
                            "evidence must never accept an experiment")

    if sidecar.get("contacts_recorded") is not True:
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason="contacts_recorded="
                            f"{sidecar.get('contacts_recorded')!r}, not True "
                            "-- contact evidence is missing or incomplete")

    contacts = sidecar.get("contacts")
    if not isinstance(contacts, list):
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason=f"sidecar 'contacts' is {type(contacts).__name__}, "
                            "expected a list")

    displacement = sidecar.get("displacement_m")
    if not isinstance(displacement, dict):
        return _result(
            ok=False, evidence_ok=False, sidecar_path=sidecar_path,
            evidence_reason=f"sidecar 'displacement_m' is "
                            f"{type(displacement).__name__}, expected an object")
    for oid, d in displacement.items():
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d):
            return _result(
                ok=False, evidence_ok=False, sidecar_path=sidecar_path,
                evidence_reason=f"sidecar 'displacement_m[{oid!r}]'={d!r} is "
                                "not a finite number")

    # Evidence is complete and well-formed from here on; judge the
    # experiment it describes.
    max_disp = max(displacement.values()) if displacement else 0.0
    if contacts:
        return _result(
            ok=False, evidence_ok=True, accepted=False, sidecar_path=sidecar_path,
            reason=f"{len(contacts)} contact(s) recorded during the flight -- "
                   "experiment not accepted",
            contacts_count=len(contacts), max_displacement_m=max_disp,
            displacement_by_object_m=displacement)

    over = {oid: d for oid, d in displacement.items() if d > DISPLACEMENT_TOL_M}
    if over:
        worst = max(over, key=over.get)
        return _result(
            ok=False, evidence_ok=True, accepted=False, sidecar_path=sidecar_path,
            reason=f"{worst!r} displaced {over[worst] * 1000:.2f} mm > "
                   f"{DISPLACEMENT_TOL_M * 1000:.1f} mm tolerance -- board "
                   "disturbed, experiment not accepted",
            contacts_count=0, max_displacement_m=max_disp,
            displacement_by_object_m=displacement)

    return _result(
        ok=True, evidence_ok=True, accepted=True, sidecar_path=sidecar_path,
        contacts_count=0, max_displacement_m=max_disp,
        displacement_by_object_m=displacement)


def report(res: Dict[str, Any]) -> None:
    print(f"sidecar={res['sidecar_path']}")
    if not res["evidence_ok"]:
        print(f"evidence_ok=False ({res['evidence_reason']})")
    else:
        print(f"evidence_ok=True contacts={res['contacts_count']} "
              f"max_displacement_mm={res['max_displacement_m'] * 1000:.2f}")
        if not res["accepted"]:
            print(f"reason: {res['reason']}")
    print(f"EXPERIMENT_ACCEPTED={'yes' if res['ok'] else 'NO'}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <log_path>")
    result = evaluate(sys.argv[1])
    report(result)
    if result["ok"]:
        sys.exit(0)
    sys.exit(1 if not result["evidence_ok"] else 2)


if __name__ == "__main__":
    main()
