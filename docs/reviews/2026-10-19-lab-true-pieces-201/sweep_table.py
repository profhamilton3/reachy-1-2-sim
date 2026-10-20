"""Summarise the offline sweeps in sweep/*.json into sweep/summary.md."""
import json, pathlib, collections
H = pathlib.Path(__file__).resolve().parent / "sweep"
CELLS = [f"r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3)]
out = ["# Sweep summary (S = side-on, C = crane, · = refused, — = same cell)\n"]

def load(n): return json.loads((H / n).read_text())

def sym(r):
    if not r.get("feasible"): return "·"
    return {"side-on": "S", "crane": "C"}.get(r.get("used"), "?")

def reason(r):
    if r.get("feasible"): return None
    rf = r.get("refusals")
    if rf: return "; ".join(f"{k}: {v}" for k, v in rf.items() if v)
    return f"{r.get('stage')}: {r.get('reason')}"

def grid(rows):
    d = {(r["from"], r["to"]): r for r in rows}
    L = ["| from \\ to | " + " | ".join(CELLS) + " |", "|---|" + "---|" * 9]
    for a in CELLS:
        L.append(f"| **{a}** | " + " | ".join("—" if a == b else sym(d[(a, b)]) if (a, b) in d else "?" for b in CELLS) + " |")
    return L

def section(title, rows, obj):
    rs = [r for r in rows if r["object"] == obj]
    out.append(f"## {title}: {obj}\n")
    feas = [r for r in rs if r["feasible"]]
    by = collections.Counter(r.get("used") for r in feas)
    out.append(f"Feasible pairs: **{len(feas)} of {len(rs)}**; by grasp: {dict(by)}\n")
    out.extend(grid(rs)); out.append("")
    both = [r for r in feas if r.get("margins", {}).get("crane") is not None and r.get("margins", {}).get("side-on") is not None]
    if both:
        out.append(f"Both grasps feasible ({len(both)} pairs): used / margins mm side-on, crane\n")
        out.append("| from → to | used | side-on | crane |\n|---|---|---|---|")
        for r in both: out.append(f"| {r['from']} → {r['to']} | {r['used']} | {r['margins']['side-on']} | {r['margins']['crane']} |")
        out.append("")
    else:
        out.append("Pairs where both grasps were feasible: none.\n")
    t = collections.Counter(reason(r) for r in rs if not r["feasible"])
    out.append("Refusal reasons tallied:\n")
    for k, v in t.most_common(): out.append(f"- {v} x {k}")
    out.append("")

lift = load("lift_best.json")
out.append("## Lifts (grasp best)\n")
out.append("| object | " + " | ".join(CELLS) + " | feasible |\n|---|" + "---|" * 10)
for obj in dict.fromkeys(r["object"] for r in lift):
    d = {r["from"]: r for r in lift if r["object"] == obj}
    cells = []
    for c in CELLS:
        r = d.get(c)
        cells.append("?" if r is None else sym(r) + (f" {r['margins'].get(r['used'])}" if r["feasible"] else ""))
    n = sum(1 for r in d.values() if r["feasible"])
    out.append(f"| {obj} | " + " | ".join(cells) + f" | {n}/{len(d)} |")
out.append("\nLift refusal reasons:\n")
for obj in dict.fromkeys(r["object"] for r in lift):
    t = collections.Counter((c := reason(r)) for r in lift if r["object"] == obj and not r["feasible"])
    for k, v in t.most_common(): out.append(f"- {obj}: {v} x {k}")
out.append("")
for obj in ("red_cube", "pool_box_1", "pool_cyl_1"):
    section("Placement, grasp best", load(f"place_best_{obj}.json"), obj)
cr = load("place_crane.json")
for obj in dict.fromkeys(r["object"] for r in cr):
    section("Placement, crane only (main)", [dict(r, used="crane" if r["feasible"] else None) for r in cr], obj)
(H / "summary.md").write_text("\n".join(out) + "\n")
