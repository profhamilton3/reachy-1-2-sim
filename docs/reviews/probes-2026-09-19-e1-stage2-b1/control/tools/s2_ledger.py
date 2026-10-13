"""Offline ledger builder for the Stage 2 B1 s1 session (B4 copy with BOARD/
OBJECTS/session literals parameterized; two board objects on B1): one row per recorded
action (armon / parked / setup / flight), read only from the evidence tree
(control/, recorder_logs/, plan_*.json). No server, no SDK. Stage 1 retry
columns plus shape/rep/session/start_variant. Usage:
    python3 s2_ledger.py <EV>  -> writes <EV>/ledger.csv, prints a summary
"""
import csv, glob, json, os, pathlib, re, sys

EV = pathlib.Path(sys.argv[1])
CTRL = EV / "control"
BOARD_ID = "B1"
BOARD = "B1_evidence"
OBJECTS = ("soda_can", "foam_block")
BOARD_SHA = (EV / "board" / "sha256.txt").read_text().split()[0]
CODE = "e6c30b7"
SESSION = "s1"
COLS = ["cycle", "shape", "rep", "session", "role", "leg", "route", "board", "board_sha256",
        "code", "run_dir", "log", "samples", "tool", "identity_ok", "settled_pose_ok",
        "contacts", "contacts_recorded", "evidence_N_N", "covered", "seq_contiguous",
        "reset_in_window", *[f"displacement_{o}_m" for o in OBJECTS], "displacement_max_m",
        "object_disturbance_gt_1mm", "gate_accepted", "tail_target", "tail_mode",
        "tail_line", "tail_verdict", "start_variant", "droop_class", "notebook_outcome",
        "elapsed_s", "compliance_waited_s", "outcome"]


def read(p):
    try:
        return pathlib.Path(p).read_text()
    except OSError:
        return ""


def sidecar_for(name):
    log_line = re.search(r"Saved \d+ samples to (\S+)", read(CTRL / f"recorder_{name}.log"))
    if not log_line:
        return None, None
    log = pathlib.Path(log_line.group(1))
    sc = EV / "recorder_logs" / (log.stem + ".link.json")
    return log.name, (json.loads(sc.read_text()) if sc.is_file() else None)


def row(cycle, shape, rep, role, leg, route, tail_target, tail_mode, tool):
    logname, sc = sidecar_for(leg)
    r = dict.fromkeys(COLS, "")
    r.update(cycle=cycle, shape=shape, rep=rep, session=SESSION, role=role, leg=leg,
             route=route, board=BOARD, board_sha256=BOARD_SHA, code=CODE, log=logname or "",
             tool=tool, tail_target=tail_target, tail_mode=tail_mode)
    rec = read(CTRL / f"recorder_{leg}.log")
    m = re.search(r"Saved (\d+) samples", rec); r["samples"] = m.group(1) if m else ""
    if sc:
        cw = sc.get("contacts_window", {})
        disp = sc.get("displacement_m", {})
        r.update(run_dir=os.path.relpath(sc.get("server_run_dir", ""), EV),
                 identity_ok=sc.get("identity", {}).get("ok"),
                 settled_pose_ok=all(v.get("ok") for v in sc.get("settled_pose_check", {}).values()),
                 contacts=len(sc.get("contacts", [])), contacts_recorded=sc.get("contacts_recorded"),
                 evidence_N_N=f"{cw.get('states_with_contacts_key')}/{cw.get('states_in_window')}",
                 covered=cw.get("covered"), seq_contiguous=cw.get("seq_contiguous"),
                 reset_in_window=cw.get("reset_in_window"),
                 **{f"displacement_{o}_m": f"{disp.get(o, float('nan')):.3g}" for o in OBJECTS},
                 displacement_max_m=f"{max(disp.values()):.3g}" if disp else "",
                 object_disturbance_gt_1mm=any(v > 1e-3 for v in disp.values()) or not all(o in disp for o in OBJECTS))
    gate = read(CTRL / f"gate_{leg}.txt")
    r["gate_accepted"] = "yes" if "EXPERIMENT_ACCEPTED=yes" in gate else ("NO" if gate else "")
    tc = read(CTRL / f"tailcheck_{leg}_{tail_target}.txt")
    tl = re.search(r"tail_samples=.*", tc); r["tail_line"] = tl.group(0) if tl else ""
    tv = re.search(r"(PARKED_AT_\w+=\w+|STAGE0_INIT_OK=\w+)", tc); r["tail_verdict"] = tv.group(1) if tv else ""
    sv = CTRL / f"start_variant_{cycle}.json"
    if role == "parked" and sv.is_file():
        r["start_variant"] = json.loads(sv.read_text()).get("start_variant")
    done = CTRL / f"{leg}_done"
    if done.is_file():
        d = json.loads(done.read_text())
        r["notebook_outcome"] = d.get("outcome")
        r["elapsed_s"] = d.get("elapsed_s")
        cc = d.get("compliance_check") or {}
        r["compliance_waited_s"] = cc.get("waited_s", cc.get("elapsed_s", ""))
        if role in ("setup", "flight"):
            r["droop_class"] = "D0 (route completed)" if d.get("outcome") == "returned" else "see outcome"
    else:
        r["notebook_outcome"] = "n/a"; r["droop_class"] = "n/a"
    ok = (r["gate_accepted"] == "yes" and (tail_mode == "start" or "yes" in str(r["tail_verdict"]))
          and r["contacts"] == 0 and r["settled_pose_ok"] is True and r["covered"] is True
          and r["seq_contiguous"] is True and r["reset_in_window"] is False
          and (role == "parked" and r["start_variant"] == "stiff-zero" or role != "parked")
          and (role in ("parked",) or r["notebook_outcome"] == "returned"))
    if logname is None and not rec and not (CTRL / f"go_{leg}").is_file():
        r["outcome"] = "NOT FLOWN (session stopped)"   # B1 s1: control/stop before this cycle's legs
    else:
        r["outcome"] = "pass" if ok else "CHECK"
    return r


def main():
    rows = []
    rows.append(row(f"stage0-{BOARD_ID}-{SESSION}", "", "", "armon", f"stage0-{BOARD_ID}-{SESSION}-armon", "RAISE_TO_SIDE (label only)",
                    "INIT", "end", "reachy.turn_on('r_arm') (stage0 notebook)"))
    plans = sorted(glob.glob(str(EV / f"plan_S2-{BOARD_ID}-*.json")),
                   key=lambda p: (int(re.search(r"-r(\d+)", p).group(1)), re.search(rf"S2-{BOARD_ID}-(\w)-", p).group(1)))
    routes = {"a": ("RAISE_TO_SIDE", "PRESENT", "LOWER_TO_REST", "REST", "primitives.raise_to_side", "rig_motion.from_present"),
              "b": (None, None, "PLACE_ROUTE", "REST", None, "rig_motion.deploy_to_rest"),
              "c": ("PLACE_ROUTE", "REST", "LIFT_TO_PRESENT", "PRESENT", "rig_motion.deploy_to_rest", "rig_motion.to_present")}
    for p in plans:
        d = json.loads(read(p)); c, s, rep = d["cycle"], d["shape"], d["rep"]
        sr, st, fr, ft, stool, ftool = routes[s]
        rows.append(row(c, s, rep, "parked", f"{c}-parked", sr or fr, "HOME", "start", "none (reset before)"))
        if sr:
            rows.append(row(c, s, rep, "setup", f"{c}-setup", sr, st, "end", stool))
        rows.append(row(c, s, rep, "flight", f"{c}-flight", fr, ft, "end", ftool))
    with open(EV / "ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['cycle']:<14} {r['role']:<7} {r['route']:<28} contacts={r['contacts']} disp_max={r['displacement_max_m']} "
              f"gate={r['gate_accepted']} tail={r['tail_verdict']:<22} sv={r['start_variant'] or '-':<10} "
              f"nb={r['notebook_outcome']:<8} el={r['elapsed_s'] if r['elapsed_s'] == '' else round(float(r['elapsed_s']), 2)} {r['outcome']}")
    print(f"rows={len(rows)} pass={sum(r['outcome'] == 'pass' for r in rows)}")


if __name__ == "__main__":
    main()
