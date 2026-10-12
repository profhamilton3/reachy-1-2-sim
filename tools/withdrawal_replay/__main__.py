"""python -m tools.withdrawal_replay --out REPORT.md [--fixtures DIR] NAME=DIR ...

Each NAME=DIR is one recorded lift (a directory holding `live/`).  Writes the
comparison of base / O1 / O2 / O1+O2 and the chosen option to REPORT.md, and
with --fixtures the extracted records (the test fixtures) to DIR/NAME.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

from . import options as O
from .rebuild import Rebuilt
from .record import dump, extract


def _f(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def report(rows, recorded, validation, choice, summary, sha):
    today = datetime.date.today().isoformat()
    out = [f"# Issue #173 offline replay — withdrawal options ({today})", "",
           f"reachy-1-2-sim at `{sha or 'worktree'}`.  Offline only: each Phase 1.5 lift's state "
           "just before the hand opened is rebuilt in the #172 offline world (FWDCenterLabSivaPool, "
           "table and rails) and every option is flown there in MuJoCo.  No simulator server.", "",
           "## Rebuild check", "",
           "| lift | plan vs recorded (deg) | commanded-pose fit (deg) | recorded drag (mm) | "
           "offline base drag (mm) | recorded withdrawal | offline base |",
           "|---|---|---|---|---|---|---|"]
    for v in validation:
        out.append(f"| {v['lift']} | {v['plan_err']:.4f} | {v['cmd_fit']:.3f} | "
                   f"{v['rec_drag']:.2f} | {v['base_drag']:.2f} | {v['rec']} | {v['base']} |")
    out += ["", "## Options, per lift", "",
            "| lift | option | refused | why | way out | final opening (deg) | "
            "min retreat clearance (mm) | residual pad–cube (mm) | release drag (mm) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['lift']} | {r['option']} | {'**yes**' if r['refused'] else 'no'} | "
                   f"{r.get('why', '')} | {r.get('kind', '—')} | "
                   f"{_f(r.get('final_opening_deg'), 1)} | "
                   f"{_f(r['min_clearance_mm']) if r['min_clearance_mm'] != float('inf') else '—'} | "
                   f"{r['residual_mm']:.2f} | {_f(r.get('drag_mm'))} |")
    out += ["", "## Summary", "",
            "| option | refusals | lifts with residual pad contact | worst residual (mm) | "
            "min retreat clearance (mm) | drag mean / max (mm) |", "|---|---|---|---|---|---|"]
    for o, s in summary.items():
        out.append(f"| {o} | {s['refusals']} / {s['lifts']} | {s['residual_lifts']} | "
                   f"{s['residual_max_mm']:.2f} | {s['min_clearance_mm']:.2f} | "
                   f"{s['drag_mean_mm']:.1f} / {s['drag_max_mm']:.1f} |")
    out += ["", f"**Chosen: {choice or 'NONE — no option reaches 0 of 10 refusals'}** "
            "(rule: 0 refusals and zero residual pad contact first, then the smallest drag).", ""]
    if recorded:
        out += ["## The new withdrawal step from the RECORDED release states", "",
                "The live release itself, not a rebuilt one: each lift posed at its recorded "
                "post-release state, then judged and flown by the new withdrawal.", "",
                "| lift | recorded withdrawal | refused | way out | opened wider | "
                "min clearance (mm) | residual (mm) |", "|---|---|---|---|---|---|---|"]
        for r in recorded:
            out.append(f"| {r['lift']} | {r['recorded']} | {'**yes**' if r['refused'] else 'no'} "
                       f"{r.get('why', '')} | {r.get('kind', '—')} | {r['opened_wider']} | "
                       f"{_f(r['min_clearance_mm']) if r['min_clearance_mm'] != float('inf') else '—'} | "
                       f"{r['residual_mm']:.2f} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("lifts", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--fixtures")
    p.add_argument("--sha", default="")
    p.add_argument("--json")
    a = p.parse_args(argv)

    rows, recorded, validation = [], [], []
    for spec in a.lifts:
        name, folder = spec.split("=", 1)
        rec = extract(name, folder)
        if a.fixtures:
            pathlib.Path(a.fixtures).mkdir(parents=True, exist_ok=True)
            dump(rec, str(pathlib.Path(a.fixtures) / f"{name}.json"))
        rb = Rebuilt(rec)
        mine = [O.run_option(rb, opt) for opt in O.OPTIONS]
        rows += mine
        r = O.withdraw_from_recorded(rb)
        r["recorded"] = "completed" if rec["recorded"]["withdrawal_clean"] else "refused"
        recorded.append(r)
        base = mine[0]
        validation.append({"lift": name, "plan_err": rb.plan_err_deg, "cmd_fit": rb.cmd_fit_deg,
                           "rec_drag": rec["recorded"]["drag_mm"], "base_drag": base["drag_mm"],
                           "rec": "completed" if rec["recorded"]["withdrawal_clean"] else "refused",
                           "base": "refused" if base["refused"] else "completed"})
        print(name, " | ".join(f"{m['option']}: {'REFUSED' if m['refused'] else m.get('kind')}"
                               f" res {m['residual_mm']:.2f} drag {m.get('drag_mm', 0):.1f}"
                               for m in mine), flush=True)
    summary = O.summarise(rows)
    choice = O.choose(summary)
    pathlib.Path(a.out).write_text(report(rows, recorded, validation, choice, summary, a.sha))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"rows": rows, "recorded": recorded,
                                                    "validation": validation,
                                                    "summary": summary, "choice": choice},
                                                   indent=1, default=str))
    print("chosen:", choice)
    return 0 if choice else 1


if __name__ == "__main__":
    sys.exit(main())
