#!/usr/bin/env python3
"""results/part*.json -> the results table (Markdown), so no number is copied by hand."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SHAPE = {"11": "block, short", "12": "block, tall", "21": "cylinder, short", "22": "cylinder, tall"}


def events(f, kind):
    return [e for e in f["events_of_interest"] if e["kind"] == kind]


def row(r):
    d, f = r["design"], r["flown"]
    name = f"PART{d} ({SHAPE[d[:2]]}, {'holed' if d[2] == '1' else 'solid'})"
    planner = {"crane": "crane", "side": "side-on"}[r["planner"]]
    envelope = r["grasp_envelope"]["outline"] if r["grasp_envelope"] else "—"
    if f is None:
        why = sorted({p["why"].split(": ", 1)[1] for p in r["refused_pairs"]})
        return [name, planner, envelope, f"not flown ({len(r['refused_pairs'])} pair(s))", "—", "—", "—",
                "planning: " + "; ".join(why)]
    cells = f"{f['origin']} → {f['dest']}"
    held = events(f, "HELD")
    if not held:
        return [name, planner, envelope, cells, "**not lifted**", "not placed", "—",
                "in flight: " + f["halt"]]
    h = held[0]
    drift = max((e.get("drift_mm", 0.0) for e in events(f, "IN_HAND")), default=0.0)
    lift = (f"{'yes' if f['criteria']['retained_height'] else 'no'}: rose {h['rise_end_mm']:.1f} mm, "
            f"slip {h['in_hand_slip_mm']:.3f} mm; carried drift ≤ {drift:.2f} mm; grip {h['grip_force_n']:.0f} N")
    place = (f"{'on target' if f['criteria']['placed_on_target'] else 'off target'}: "
             f"{f['placement_error_mm']:.1f} mm, tilt {f['tilt_deg']:.2f}°")
    ok = all(f["criteria"].values())
    note = "all 6 criteria met; back at PRESENT" if ok else (
        "failed " + ", ".join(k for k, v in f["criteria"].items() if not v) + ": " + str(f["halt"]))
    note += f"; D6 evaluator: {'success' if f['d6'].get('successful') else 'fail'}"
    return [name, planner, envelope, cells, lift, place, f"{f['release_drag_mm']} mm", note]


def main():
    head = ["Design", "Planner", "Envelope", "Cells flown", "Lift", "Place", "Release drag", "Refusal / note"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for d in ["110", "111", "120", "121", "210", "211", "220", "221"]:
        print("| " + " | ".join(row(json.loads((HERE / "results" / f"part{d}.json").read_text()))) + " |")


if __name__ == "__main__":
    main()
