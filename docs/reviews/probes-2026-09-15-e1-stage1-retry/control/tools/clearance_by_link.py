"""Offline, pure-geometry: planned vs realised clearance per recording, per
hand model, per link, per object, from the Stage 1 retry recordings only.
Motion recordings are split into pre-motion / moving / parked-tail segments
by joint velocity so stationary samples are reported apart from motion."""
import json, sys, glob, os, math
REPO, EV = sys.argv[1], sys.argv[2]
sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/scripts")
import measure_route_clearance as M
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import link_capsules, joint_path, hand_radius
from reachy_ai.scene.awareness import SceneModel

scene = SceneModel.from_yaml(f"{REPO}/scenes/e1_boards/B4_pool_box_1_r2c3.yaml")
HANDS = M.HAND_MODES
RECS = [  # name, route, role
 ("parked_stage0","LOWER_TO_REST","parked"),("parked_S1a","RAISE_TO_SIDE","parked"),
 ("setup_a","RAISE_TO_SIDE","motion"),("flight_a","LOWER_TO_REST","motion"),
 ("parked_S1b","PLACE_ROUTE","parked"),("flight_b","PLACE_ROUTE","motion"),
 ("parked_S1c","PLACE_ROUTE","parked"),("setup_c","PLACE_ROUTE","motion"),("flight_c","LIFT_TO_PRESENT","motion"),
]
def logpath(name):
    t = open(f"{EV}/control/recorder_{name}.log").read()
    import re; p = re.search(r"Saved \d+ samples to (\S+)", t).group(1)
    return f"{EV}/recorder_logs/" + os.path.basename(p)

def per_link_worst(q7_grip_iter):
    out = {h: {} for h in HANDS}   # hand -> link -> object -> worst distance
    for q7, g in q7_grip_iter:
        for h in HANDS:
            for name, p0, p1, rad in link_capsules(q7, "right", g, hand=h):
                cl = scene.clearances([(name, p0, p1, rad)])
                d = out[h].setdefault(name, {})
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
    """Indices of the moving segment: first/last sample where any joint moves faster than thresh."""
    moving = []
    for i in range(1, len(samples)):
        dt = samples[i]["t"] - samples[i-1]["t"]
        if dt <= 0: continue
        v = max(abs(samples[i]["joints"][j] - samples[i-1]["joints"][j]) / dt for j in R.R_JOINTS)
        if v > thresh_deg_per_s: moving.append(i)
    if not moving: return None
    return moving[0], moving[-1]

result = {}
for name, route, role in RECS:
    log = json.load(open(logpath(name)))
    samples = log["samples"]; sv = M.schema_version_of(log)
    q7g, assumed = M.validated_samples(samples, schema_version=sv)
    rec = {"route": route, "role": role, "n_samples": len(samples), "aperture_assumed_samples": len(assumed),
           "planned": {h: {l: o for l, o in d.items()} for h, d in per_link_worst(planned_iter(route, len(samples))).items()},
           "realised_all": per_link_worst(q7g)}
    first = samples[0]["joints"]; last = samples[-1]["joints"]
    rec["start_state"] = {"r_gripper": first["r_gripper"], "r_wrist_roll": first["r_wrist_roll"], "r_wrist_pitch": first["r_wrist_pitch"]}
    rec["end_state"] = {"r_gripper": last["r_gripper"], "r_wrist_roll": last["r_wrist_roll"], "r_wrist_pitch": last["r_wrist_pitch"]}
    if role == "motion":
        seg = segments(samples)
        i0, i1 = seg
        rec["segments"] = {"pre_motion": [0, i0-1], "moving": [i0, i1], "parked_tail": [i1+1, len(samples)-1],
                           "t_move_start_s": round(samples[i0]["t"], 2), "t_move_end_s": round(samples[i1]["t"], 2)}
        rec["realised_pre_motion"] = per_link_worst(q7g[:i0])
        rec["realised_moving"] = per_link_worst(q7g[i0:i1+1])
        rec["realised_parked_tail"] = per_link_worst(q7g[i1+1:])
    result[name] = rec
os.makedirs(f"{EV}/summaries", exist_ok=True)
json.dump(result, open(f"{EV}/summaries/clearance_by_link.json", "w"), indent=1)

# Markdown summary: worst object per link per hand, planned vs realised, plus pool_box_1
def fmt(x): return f"{x*100:+.1f}" if x is not None else "  n/a"
lines = ["# Planned vs realised clearance by recording, hand model and link (cm, signed; negative = model overlap)", "",
         "Pure geometry from the recorded joint samples at each sample's own measured aperture, against the B4 YAML poses "
         "(`scripts/measure_route_clearance.py` geometry, called one link at a time). `planned` samples the commanded joint-space "
         "line between waypoints with the worst-case endpoint aperture. `all` = whole recording; motion recordings are also split "
         "into pre-motion, moving, and parked-tail segments by joint velocity (> 2 deg/s). Zero contacts were reported by the "
         "server's contact tracker on every recording; a negative number here is the conservative capsule model overlapping, "
         "not a physics contact.", ""]
for name, rec in result.items():
    lines.append(f"## {name} — {rec['route']} ({rec['role']}, {rec['n_samples']} samples)")
    if rec["role"] == "motion":
        s = rec["segments"]; lines.append(f"moving segment: samples {s['moving'][0]}–{s['moving'][1]} (t = {s['t_move_start_s']}–{s['t_move_end_s']} s); pre-motion {s['pre_motion'][1]+1} samples; parked tail {rec['n_samples']-s['parked_tail'][0]} samples")
    lines.append(f"start: gripper {rec['start_state']['r_gripper']:.1f}°, wrist_roll {rec['start_state']['r_wrist_roll']:.1f}°, wrist_pitch {rec['start_state']['r_wrist_pitch']:.1f}°; "
                 f"end: gripper {rec['end_state']['r_gripper']:.1f}°, wrist_roll {rec['end_state']['r_wrist_roll']:.1f}°, wrist_pitch {rec['end_state']['r_wrist_pitch']:.1f}°")
    lines.append("")
    cols = ["planned", "realised_all"] + (["realised_pre_motion", "realised_moving", "realised_parked_tail"] if rec["role"] == "motion" else [])
    lines.append("| hand | link | object | " + " | ".join(c.replace("realised_", "real ") for c in cols) + " |")
    lines.append("|---|---|---|" + "---|" * len(cols))
    for h in HANDS:
        links = sorted(set().union(*[set(rec[c][h]) for c in cols]))
        for l in links:
            objs = set().union(*[set(rec[c][h].get(l, {})) for c in cols])
            # worst object per link (by realised_all), plus pool_box_1
            worst_obj = min(objs, key=lambda o: rec["realised_all"][h].get(l, {}).get(o, 9)) if objs else None
            for o in ([worst_obj] if worst_obj == "pool_box_1" else [worst_obj, "pool_box_1"]):
                if o is None: continue
                vals = [rec[c][h].get(l, {}).get(o) for c in cols]
                tag = " (worst)" if o == worst_obj else ""
                lines.append(f"| {h} | {l} | {o}{tag} | " + " | ".join(fmt(v) for v in vals) + " |")
    lines.append("")
open(f"{EV}/summaries/clearance_by_link.md", "w").write("\n".join(lines))
print("\n".join(lines[:12]))
print("... written", f"{EV}/summaries/clearance_by_link.md")
