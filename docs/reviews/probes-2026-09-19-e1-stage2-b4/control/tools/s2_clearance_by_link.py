"""Offline, pure geometry, existing recordings only: planned vs realised
clearance per recording / hand model / link / object for every Stage 2 B4
recording, then aggregated by route x role x link across the six repetitions.
Same method and functions as the Stage 1 retry's control/tools/clearance_by_link.py
(`scripts/measure_route_clearance.py` geometry + `link_capsules`, one link at a
time, each sample at its own measured aperture; motion recordings split into
pre-motion / moving / parked-tail by joint velocity > 2 deg/s). No server, no SDK
connection, no motion. Usage: python s2_clearance_by_link.py <REPO> <EV>
"""
import glob, json, math, os, re, statistics, sys
REPO, EV = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/scripts")
import measure_route_clearance as M
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import link_capsules, joint_path, hand_radius
from reachy_ai.scene.awareness import SceneModel

scene = SceneModel.from_yaml(f"{REPO}/scenes/e1_boards/B4_pool_box_1_r2c3.yaml")
HANDS = M.HAND_MODES
BOARD_OBJ = "pool_box_1"
SLEEP_AFFECTED_CYCLE = "S2-B4-c-r1"


def recordings():
    out = []
    for log in sorted(glob.glob(f"{EV}/control/recorder_*.log")):
        name = os.path.basename(log)[len("recorder_"):-len(".log")]
        m = re.search(r"Saved \d+ samples to (\S+)", open(log).read())
        if not m:
            continue
        path = f"{EV}/recorder_logs/" + os.path.basename(m.group(1))
        role = ("parked" if name.endswith("-parked") else "setup" if name.endswith("-setup")
                else "flight" if name.endswith("-flight") else "armon")
        cycle = name.rsplit("-", 1)[0]
        route = re.search(r"route_clearance_([A-Z_]+)_\d", path).group(1)
        out.append((name, cycle, role, route, path))
    return out


def per_link_worst(q7_grip_iter):
    out = {h: {} for h in HANDS}
    for q7, g in q7_grip_iter:
        for h in HANDS:
            for lname, p0, p1, rad in link_capsules(q7, "right", g, hand=h):
                cl = scene.clearances([(lname, p0, p1, rad)])
                d = out[h].setdefault(lname, {})
                for oid, c in cl.items():
                    if oid not in d or c.distance < d[oid]:
                        d[oid] = c.distance
    return out


def planned_iter(route, n):
    wps = R.FOOTPRINT_LEGS[route]; steps = max(2, n)
    for a, b in zip(wps, wps[1:]):
        qa = [a[j] for j in R.ARM7]; qb = [b[j] for j in R.ARM7]
        g = max(a["r_gripper"], b["r_gripper"], key=hand_radius)
        for q in joint_path(qa, qb, steps=steps):
            yield q, g


def segments(samples, thresh_deg_per_s=2.0):
    moving = []
    for i in range(1, len(samples)):
        dt = samples[i]["t"] - samples[i - 1]["t"]
        if dt <= 0:
            continue
        v = max(abs(samples[i]["joints"][j] - samples[i - 1]["joints"][j]) / dt for j in R.R_JOINTS)
        if v > thresh_deg_per_s:
            moving.append(i)
    return (moving[0], moving[-1]) if moving else None


planned_cache = {}
result = {}
for name, cycle, role, route, path in recordings():
    log = json.load(open(path)); samples = log["samples"]; sv = M.schema_version_of(log)
    q7g, assumed = M.validated_samples(samples, schema_version=sv)
    key = (route, len(samples))
    if key not in planned_cache:
        planned_cache[key] = per_link_worst(planned_iter(route, len(samples)))
    rec = {"cycle": cycle, "role": role, "route": route, "n_samples": len(samples),
           "aperture_assumed_samples": len(assumed), "sleep_affected": cycle == SLEEP_AFFECTED_CYCLE,
           "planned": planned_cache[key], "realised_all": per_link_worst(q7g)}
    if role in ("setup", "flight"):
        seg = segments(samples)
        if seg:
            i0, i1 = seg
            rec["segments"] = {"moving": [i0, i1], "t_move_start_s": round(samples[i0]["t"], 2),
                               "t_move_end_s": round(samples[i1]["t"], 2)}
            rec["realised_moving"] = per_link_worst(q7g[i0:i1 + 1])
            rec["realised_parked_tail"] = per_link_worst(q7g[i1 + 1:])
    result[name] = rec
    print("done", name, flush=True)

os.makedirs(f"{EV}/summaries", exist_ok=True)
json.dump(result, open(f"{EV}/summaries/clearance_by_link_all.json", "w"), indent=1)


# ── Aggregation: route x role x hand x link, planned - realised (cm), across reps ──
def agg(exclude_sleep):
    rows = []
    groups = {}
    for name, rec in result.items():
        if exclude_sleep and rec["sleep_affected"]:
            continue
        if rec["role"] == "armon":
            continue
        seg_key = "realised_moving" if rec["role"] in ("setup", "flight") else "realised_all"
        if seg_key not in rec:
            continue
        for h in HANDS:
            for link, objs in rec[seg_key][h].items():
                if BOARD_OBJ not in objs:
                    continue
                plan = rec["planned"][h].get(link, {}).get(BOARD_OBJ)
                real = objs[BOARD_OBJ]
                tail = rec.get("realised_parked_tail", {}).get(h, {}).get(link, {}).get(BOARD_OBJ)
                groups.setdefault((rec["route"], rec["role"], h, link), []).append((plan, real, tail))
    for (route, role, h, link), vals in sorted(groups.items()):
        plans = [p for p, _, _ in vals if p is not None]; reals = [r for _, r, _ in vals]
        deltas = [(p - r) for p, r, _ in vals if p is not None]
        tails = [t for _, _, t in vals if t is not None]
        rows.append({"route": route, "role": role, "hand": h, "link": link, "n": len(vals),
                     "planned_cm": round(100 * plans[0], 2) if plans else None,
                     "realised_min_cm": round(100 * min(reals), 2), "realised_max_cm": round(100 * max(reals), 2),
                     "delta_plan_minus_real_mean_cm": round(100 * statistics.mean(deltas), 2) if deltas else None,
                     "delta_min_cm": round(100 * min(deltas), 2) if deltas else None,
                     "delta_max_cm": round(100 * max(deltas), 2) if deltas else None,
                     "parked_tail_min_cm": round(100 * min(tails), 2) if tails else None})
    return rows


all_rows = agg(exclude_sleep=False)
excl_rows = agg(exclude_sleep=True)
json.dump({"all_reps": all_rows, "excluding_S2-B4-c-r1": excl_rows},
          open(f"{EV}/summaries/clearance_by_link_aggregate.json", "w"), indent=1)


def fmt(v): return "n/a" if v is None else f"{v:+.2f}"


lines = ["# Planned minus realised clearance to `pool_box_1`, by route x role x hand x link, across repetitions (cm)", "",
         "Pure geometry from the recorded joint samples at each sample's own measured aperture, against the B4 YAML poses "
         "(same method as Stage 1's `clearance_by_link.py`). `realised` for setups/flights is the **moving** segment "
         "(joint velocity > 2 deg/s); parked recordings use the whole 3 s recording. `delta` = planned - realised: positive "
         "means the flight came closer than the planned corridor. n = recordings in the group (6 per route/role for setups "
         "and flights; 6 or 12 for parked). Zero physics contacts on every recording; negative clearance is the conservative "
         "capsule model overlapping, not a contact. Deterministic simulator: the six repetitions of one route are near-identical "
         "re-runs of one open-loop trajectory from one reset state, so agreement across them is evidence of repeatability of "
         "this procedure on this board, NOT independent evidence of broad reliability.", ""]
for title, rows in (("All 18 cycles", all_rows), ("Excluding the sleep-affected cycle S2-B4-c-r1", excl_rows)):
    lines += [f"## {title}", "", "| route | role | hand | link | n | planned | realised min..max | delta mean (min..max) | parked-tail min |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['route']} | {r['role']} | {r['hand']} | {r['link']} | {r['n']} | {fmt(r['planned_cm'])} | "
                     f"{fmt(r['realised_min_cm'])}..{fmt(r['realised_max_cm'])} | {fmt(r['delta_plan_minus_real_mean_cm'])} "
                     f"({fmt(r['delta_min_cm'])}..{fmt(r['delta_max_cm'])}) | {fmt(r['parked_tail_min_cm'])} |")
    lines.append("")
# per-recording worst link (shells) for the sleep-affected cycle vs its siblings
lines += ["## Sleep-affected cycle S2-B4-c-r1 vs the other five c-repetitions (shells, closest link to pool_box_1, moving segment, cm)", "",
          "| recording | role | closest link | planned | realised | delta |", "|---|---|---|---|---|---|"]
for name, rec in sorted(result.items()):
    if not name.startswith("S2-B4-c-") or rec["role"] == "parked":
        continue
    seg = rec.get("realised_moving", rec["realised_all"])["shells"]
    link = min((l for l in seg if BOARD_OBJ in seg[l]), key=lambda l: seg[l][BOARD_OBJ])
    plan = rec["planned"]["shells"].get(link, {}).get(BOARD_OBJ); real = seg[link][BOARD_OBJ]
    tag = " **(sleep-affected)**" if rec["sleep_affected"] else ""
    lines.append(f"| {name}{tag} | {rec['role']} | {link} | {fmt(100*plan if plan is not None else None)} | {fmt(100*real)} | "
                 f"{fmt(100*(plan-real)) if plan is not None else 'n/a'} |")
open(f"{EV}/summaries/clearance_by_link.md", "w").write("\n".join(lines) + "\n")
print("written", f"{EV}/summaries/clearance_by_link.md")
