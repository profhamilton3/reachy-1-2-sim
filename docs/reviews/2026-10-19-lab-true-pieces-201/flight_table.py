"""Tables for the #201 offline flights: reads flights/out_*.jsonl (written by
fly_batch.py) and writes flights/summary.md.

Cells: ✓ every #174 criterion true; ✗ flew to the end with a criterion
false (named in the detail table); H halted or refused during the flight
(the arm moved); R refused before motion (planning); E crashed.

Run: python3 flight_table.py
"""
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
F = HERE / "flights"
CELLS = [f"r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3)]


def load(name):
    p = F / f"out_{name}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


def mark(r):
    if "error" in r:
        return "E"
    if "refused" in r:
        return "R"
    if r.get("all_criteria"):
        return "✓"
    if r.get("halt") or r.get("refusal"):
        return "H"
    return "✗"


def failed(r):
    return [k for k, v in (r.get("criteria") or {}).items() if not v]


def why(r):
    if "error" in r:
        return r["error"]
    if "refused" in r:
        return r["refused"]
    return r.get("halt") or r.get("refusal") or ", ".join(failed(r))


def counts(rows):
    c = collections.Counter(mark(r) for r in rows)
    return ", ".join(f"{k} {c[k]}" for k in ("✓", "✗", "H", "R", "E") if c[k])


def fmt(v, nd=2):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def detail(rows, label=lambda j: f"{j.get('origin')} → {j.get('dest') or '(lift)'}"):
    out = ["| flight | result | grasp | az / pitch | error mm | tilt ° | slip mm | drag mm "
           "| way out | others moved mm | failed / halt / refusal |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        j = r["job"]
        az = "" if r.get("azimuth_deg") is None else f"{r['azimuth_deg']:.0f} / {r.get('pitch_deg')}"
        out.append(f"| {label(j)} | {mark(r)} | {r.get('grasp', '')} | {az} | "
                   f"{fmt(r.get('error_mm'))} | {fmt(r.get('tilt_deg'))} | {fmt(r.get('slip_mm'), 3)} | "
                   f"{fmt(r.get('drag_mm'))} | {r.get('way_out') or ''} | "
                   f"{fmt(r.get('others_moved_max_mm'), 4)} | "
                   f"{'' if mark(r) == '✓' else why(r)} |")
    return out


def grid(rows):
    by = {(r["job"]["origin"], r["job"]["dest"]): r for r in rows}
    out = ["| from \\ to | " + " | ".join(CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for o in CELLS:
        cells = []
        for d in CELLS:
            cells.append("—" if o == d else (mark(by[(o, d)]) if (o, d) in by else " "))
        out.append(f"| {o} | " + " | ".join(cells) + " |")
    return out


def main():
    md = ["# #201 offline flights — summary (pads condim 4; pool box 50.8 mm cube, pool cylinder 45.7 mm dia x 71.1 mm)", "",
          "Offline physics (`tools/place_sweep/fly.py`: the executor on the live model's physics, "
          "host IK mirror).  Scene FWDCenterLabSivaPool, the object upright on the origin cell, "
          "nothing else on the board unless stated.  Grasp `best` = the planner choice.  "
          "✓ every #174 criterion true; ✗ flew with a criterion false; H halted or refused in "
          "flight; R refused before motion; E crashed.", ""]
    lifts = load("final_lift")
    if lifts:
        md += ["## Lifts (grasp best)", "", "| object | " + " | ".join(CELLS) + " | counts |",
               "|---|" + "---|" * (len(CELLS) + 1)]
        for oid in sorted({r["job"]["oid"] for r in lifts}):
            mine = {r["job"]["origin"]: r for r in lifts if r["job"]["oid"] == oid}
            md.append(f"| {oid} | " + " | ".join(mark(mine[c]) if c in mine else " " for c in CELLS)
                      + f" | {counts(mine.values())} |")
        md += ["", *detail(lifts, lambda j: f"{j['oid']} {j['origin']}"), ""]
    for oid, fname, note in (("red_cube", "final_place_red_cube", ""), ("pool_box_1", "final_place_pool_box_1", ""),
                             ("pool_cyl_1", "final_place_pool_cyl_1", ""),
                             ("pool_cyl_1", "base_place_pool_cyl_1", " [pads condim 3, attribution]")):
        rows = load(fname)
        if not rows:
            continue
        flown = [r for r in rows if "refused" not in r and "error" not in r]
        md += [f"## Placements: {oid} (grasp best){note}", "", f"All 72 ordered pairs: {counts(rows)}.  "
               f"Flown: {len(flown)}; every criterion: {sum(1 for r in flown if r.get('all_criteria'))}.",
               "", *grid(rows), "", "Flown pairs:", "", *detail(flown), ""]
        ref = collections.Counter(r["refused"] for r in rows if "refused" in r)
        if ref:
            md += ["Refused before motion:", ""] + [f"- {n} × {k}" for k, n in ref.most_common()] + [""]
    for name, title in (("perturb", "Perturbations (red_cube, grasp best)"),
                        ("before_after", "Before / after: PR #197's pair r3c2 → r3c3"),
                        ("neighbours", "Neighbours on the board (red_cube)")):
        rows = load(name)
        if not rows:
            continue

        def lab(j):
            extra = []
            if j.get("cube_xy_offset"):
                extra.append(f"offset {[round(1000 * v, 1) for v in j['cube_xy_offset']]} mm")
            if j.get("yaw_deg"):
                extra.append(f"yaw {j['yaw_deg']}°")
            if j.get("others"):
                extra.append("others " + ", ".join(f"{k} on {v}" for k, v in j["others"].items()))
            return (f"{j.get('origin')} → {j.get('dest') or '(lift)'} [{j.get('grasp')}] "
                    + "; ".join(extra))
        md += [f"## {title}", "", f"Counts: {counts(rows)}.", "", *detail(rows, lab), ""]
    (F / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
