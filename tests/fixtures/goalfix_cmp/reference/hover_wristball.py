"""Looks into the remaining wrist_ball (and forearm/thumb/thumb_pad) plan-vs-realised offsets
near HOVER, to tell tracking effects apart from analysis errors. Needs campaign.py to have run
first, since it reads per_recording.csv.

Checks:
  A1  planned-side sampling convergence (400 vs 2000 steps per leg)
  A2  settled agreement: realised at the parked REST tail vs planned at the commanded REST
  A3  independent FK + sphere/box distance (no repo imports) at every worst sample
  A4  the public report(full_route=True) min-over-links agrees with campaign.py
  T1  every worst sample: phase, joint speed, joint error vs the nearest planned-path pose, and
      the clearance recovered by resetting each joint to that planned pose one at a time
Writes hover_attribution.csv and hover_checks.json, and prints a summary.
"""
import csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser("~/reachy-1-2-sim")
for p in ("/src", "/scripts", "/native_mujoco"):
    sys.path.insert(0, REPO + p)
import measure_route_clearance as M  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import link_capsules, joint_path  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402

SESS = {"B4": ("~/e1-stage2-B4-s1-2026-09-18", "B4_pool_box_1_r2c3.yaml", ["pool_box_1"]),
        "B1": ("~/e1-stage2-B1-s1-2026-09-19", "B1_evidence.yaml", ["foam_block"]),
        "B2": ("~/e1-stage2-B2-s2-2026-09-22", "B2_foam_r3c3.yaml", ["foam_block"])}
LINKS = ("wrist_ball", "forearm", "thumb", "thumb_pad")
J8 = R.ARM7 + ("r_gripper",)


def clear(scene, oid, link, J):
    q = [J[j] for j in R.ARM7]
    for c in link_capsules(q, "right", J["r_gripper"], hand="shells"):
        if c[0] == link:
            return 100 * scene.clearances([c])[oid].distance


def dense_path(route, steps=400):
    wps = M.full_route_poses(route)
    names = [M._FULL_ROUTE_START_POSE[route]] and (["START"] + [w.name for w in M._FULL_ROUTE_WAYPOINTS[route]])
    out = []
    for k, (a, b) in enumerate(zip(wps, wps[1:])):
        qa = [a[j] for j in R.ARM7]; qb = [b[j] for j in R.ARM7]
        for i, q in enumerate(joint_path(qa, qb, steps=steps)):
            d = dict(zip(R.ARM7, q)); d["r_gripper"] = M._lerp(a["r_gripper"], b["r_gripper"], i, steps)
            out.append((f"{names[k]}->{names[k+1]}", i / (steps - 1), d))
    return out


# ── independent FK (copied constants; see 2026-09-22 indep.py) ────────────
def _Rx(a): c, s = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def _Ry(a): c, s = math.cos(a), math.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def _Rz(a): c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def indep_wrist_ball(J, center, size):
    q = np.radians([J[j] for j in R.ARM7])
    sh = np.array([0, -0.19, 1.0]); Rm = _Ry(q[0]) @ _Rx(q[1]) @ _Rz(q[2]); el = sh + Rm @ [0, 0, -0.28]
    Rm = Rm @ _Ry(q[3]) @ _Rz(q[4]); wr = el + Rm @ [0, 0, -0.25]
    d = np.abs(wr - np.array(center)) - np.array(size) / 2
    return 100 * (np.linalg.norm(np.maximum(d, 0)) + min(d.max(), 0) - 0.028)


checks = {"A1_convergence": [], "A2_settled_REST": [], "A3_independent_wrist_ball": [], "A4_report_api": []}
rows = []
pr = list(csv.DictReader(open(os.path.join(HERE, "per_recording.csv"))))
agg = {(r["session"], r["route"], r["role"], r["object"], r["link"]): r
       for r in csv.DictReader(open(os.path.join(HERE, "aggregate.csv")))}
for tag, (ev, yml, oids) in SESS.items():
    ev = os.path.expanduser(ev)
    scene = SceneModel.from_yaml(f"{REPO}/scenes/e1_boards/{yml}")
    ledger = {r["leg"]: r for r in csv.DictReader(open(os.path.join(ev, "ledger.csv")))}
    paths = {}
    logs = {}
    for oid in oids:
        for route in ("PLACE_ROUTE", "RAISE_TO_SIDE", "LOWER_TO_REST", "LIFT_TO_PRESENT"):
            for link in LINKS:
                a = M.planned_clearance_by_link(route, 400, scene, link, waypoints=M.full_route_poses(route))[oid]
                b = M.planned_clearance_by_link(route, 2000, scene, link, waypoints=M.full_route_poses(route))[oid]
                checks["A1_convergence"].append([tag, route, oid, link, round(100 * a, 3), round(100 * b, 3)])
    for r in pr:
        if r["session"] != tag or r["object"] not in oids or r["link"] not in LINKS:
            continue
        leg, route, oid, link, idx = r["leg"], r["route"], r["object"], r["link"], int(r["idx"])
        if leg not in logs:
            logs[leg] = M.load_log(os.path.join(ev, "recorder_logs", ledger[leg]["log"]))["samples"]
        S = logs[leg]
        if route not in paths:
            paths[route] = dense_path(route)
        J = S[idx]["joints"]
        seg, frac, ref = min(paths[route], key=lambda p: sum((J[j] - p[2][j]) ** 2 for j in R.ARM7))
        lo, hi = max(0, idx - 2), min(len(S) - 1, idx + 2)
        speed = math.sqrt(sum((S[hi]["joints"][j] - S[lo]["joints"][j]) ** 2 for j in R.ARM7)) / (S[hi]["t"] - S[lo]["t"])
        base = clear(scene, oid, link, J)
        gains = {j: clear(scene, oid, link, dict(J, **{j: ref[j]})) - base for j in J8}
        top = max(gains, key=gains.get)
        row = dict(session=tag, leg=leg, route=route, role=r["role"], object=oid, link=link, idx=idx,
                   t_s=r["t_s"], realised_cm=round(base, 3), ref_segment=seg, ref_frac=round(frac, 2),
                   ref_cm=round(clear(scene, oid, link, ref), 3),
                   all_joints_to_ref_gain_cm=round(clear(scene, oid, link, ref) - base, 3),
                   joint_speed_dps=round(speed, 1),
                   **{f"err_{j[2:]}": round(J[j] - ref[j], 2) for j in J8},
                   **{f"gain_{j[2:]}": round(gains[j], 3) for j in J8},
                   top_joint=top[2:], top_gain_cm=round(gains[top], 3))
        rows.append(row)
        if link == "wrist_ball":
            o = scene.objects[oid]
            checks["A3_independent_wrist_ball"].append([tag, leg, idx, round(base, 3), round(indep_wrist_ball(J, o.center, o.size), 3)])
    # A2: parked tail of each REST-ending recording (last 40 samples) against planned at the commanded REST
    for leg, S in logs.items():
        if ledger[leg]["route"] not in ("PLACE_ROUTE", "LOWER_TO_REST"):
            continue
        for oid in oids:
            tail = [clear(scene, oid, "wrist_ball", s["joints"]) for s in S[-40:]]
            checks["A2_settled_REST"].append([tag, leg, oid, round(float(np.median(tail)), 3),
                                              round(clear(scene, oid, "wrist_ball", R.REST), 3)])
    # A4: report() through the public API on one PLACE_ROUTE recording
    leg = next(l for l in logs if ledger[l]["route"] == "PLACE_ROUTE")
    rep = M.report(logs[leg], "PLACE_ROUTE", f"{REPO}/scenes/e1_boards/{yml}", full_route=True)
    mine = min(float(x["realised_min_cm"]) for x in pr if x["leg"] == leg and x["object"] == oids[0])
    checks["A4_report_api"].append([tag, leg, rep["planned_route_scope"], round(100 * rep["realised"]["shells"][oids[0]], 3), mine])

with open(os.path.join(HERE, "hover_attribution.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(checks, open(os.path.join(HERE, "hover_checks.json"), "w"), indent=1)

conv = max(abs(a - b) for *_, a, b in checks["A1_convergence"])
set_ = [a - b for *_, a, b in checks["A2_settled_REST"]]
ind = max(abs(a - b) for *_, a, b in checks["A3_independent_wrist_ball"])
print(f"A1 max |planned 400 - planned 2000| = {conv:.3f} cm")
print(f"A2 settled REST realised - planned wrist_ball: min {min(set_):+.3f} max {max(set_):+.3f} cm (n={len(set_)})")
print(f"A3 max |repo - independent| wrist_ball = {ind:.4f} cm (n={len(checks['A3_independent_wrist_ball'])})")
print("A4", checks["A4_report_api"])
for tag in SESS:
    for link in LINKS:
        rs = [r for r in rows if r["session"] == tag and r["link"] == link and r["route"] in ("PLACE_ROUTE", "RAISE_TO_SIDE")]
        if not rs:
            continue
        sp = [r["err_shoulder_pitch"] for r in rs]; g = [r["gain_shoulder_pitch"] for r in rs]
        ga = [r["all_joints_to_ref_gain_cm"] for r in rs]
        from collections import Counter
        print(f"{tag} {link:10s} n={len(rs)} seg={Counter(r['ref_segment'] for r in rs).most_common(2)} "
              f"sp_err {min(sp):+.1f}..{max(sp):+.1f}  sp_gain {min(g):+.2f}..{max(g):+.2f}  all_gain {min(ga):+.2f}..{max(ga):+.2f} "
              f"top={Counter(r['top_joint'] for r in rs).most_common(2)} speed_med={np.median([r['joint_speed_dps'] for r in rs]):.1f}")
