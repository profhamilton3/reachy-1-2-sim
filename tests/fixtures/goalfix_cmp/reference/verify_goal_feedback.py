"""Independent offline check of the bridge/SDK goal-position feedback loop (E1 Stage 2 B4 / B1 / B2 s2).

Read-only. It verifies every evidence file it opens against its session SHA256SUMS and writes only here.

Why an exact match is a valid echo test:
  * the native server records the same JSON state message it broadcasts (server.py, the state push),
    so the bridge parsed exactly the doubles that states.jsonl holds;
  * the fake server sends goal_position = position_rad as a protobuf FloatValue, which is float32
    (fake_reachy_server._sample_to_proto, mujoco_remote_backend.to_kinematic_snapshot);
  * the SDK echoes self._state['goal_position'] unchanged (Joint._pop_command), and the bridge
    forwards float(cmd.goal_position) into target_rad (_build_command).
  So an echoed target equals float32(position_rad) of an earlier state *bit for bit*. A minimum-jerk
  setpoint (deg -> rad -> float32) matching one by chance is vanishingly unlikely. The time-shifted
  control below measures that chance rate on the same data.

Per command and per right-arm joint, a target is one of:
  carry  identical to the previous command's target (the bridge re-sends last targets for joints
         not in the batch)
  echo   new, and equal to float32(position) of a state in the 500 ms before the command was applied
  fresh  new, and not such a match (a trajectory setpoint)

Hold check: gaps of >= 0.2 s with no new command (the target is constant, e.g. fly_route's
settle_s sleep) show whether a held target sags on its own.

Writes echo_summary.csv, windows.csv, gaps.csv, and summary.json.
"""
import csv, hashlib, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESS = {"B4": "~/e1-stage2-B4-s1-2026-09-18", "B1": "~/e1-stage2-B1-s1-2026-09-19",
        "B2": "~/e1-stage2-B2-s2-2026-09-22"}
J8 = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch", "r_forearm_yaw",
      "r_wrist_pitch", "r_wrist_roll", "r_gripper")
GUARD = (0, 1, 2, 3, 5)  # CRITICAL_JOINTS indices in J8
HOVER = np.radians([-40, -10, 0, -60, 0, -15, 0, 20])
LOOKBACK_NS = 500_000_000
SHIFT_NS = 20_000_000_000  # control: same test against states 20 s earlier
RX_HEAD = re.compile(rb'"wall_time_ns":(\d+),"cmd_seq":(\d+)')
RX_POS = re.compile(rb'"position_rad":([-0-9.eE+]+)')
RX_CMP = re.compile(rb'"compliant":(true|false)')


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sums_of(ev):
    return {os.path.normpath(l.split(None, 1)[1].strip().lstrip("*")): l.split()[0]
            for l in open(os.path.join(ev, "SHA256SUMS"))}


def verified(ev, rel, sums):
    p = os.path.join(ev, rel)
    assert sha(p) == sums[os.path.normpath(rel)], rel
    return p


def load_states(path):
    t, seq, pos, cmp_ = [], [], [], []
    with open(path, "rb") as f:
        for line in f:
            head = line[:2600]
            m = RX_HEAD.search(head)
            if not m:
                continue
            p = RX_POS.findall(head)[:8]
            c = RX_CMP.findall(head)[:8]
            if len(p) < 8:
                continue
            t.append(int(m.group(1))); seq.append(int(m.group(2)))
            pos.append([float(x) for x in p]); cmp_.append([x == b"true" for x in c])
    return np.array(t, dtype=np.int64), np.array(seq), np.array(pos), np.array(cmp_)


def load_cmds(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        if d.get("type") == "joint_command":
            out.append((d["seq"], d["target_rad"][:8], (d.get("compliant") or [None] * 8)[:8]))
    return out


def classify(cmds, st, spos32, t_hi, shift=0):
    """Per command, per joint: 0 carry, 1 echo, 2 fresh. Also the echo age in ms."""
    n = len(cmds)
    cls = np.zeros((n, 8), dtype=np.int8)
    age = np.full((n, 8), np.nan)
    prev = None
    for i, (_, tgt, _) in enumerate(cmds):
        tgt = np.array(tgt)
        a = int(np.searchsorted(st, t_hi[i] - shift - LOOKBACK_NS))
        b = int(np.searchsorted(st, t_hi[i] - shift, side="right"))
        win = spos32[a:b]
        for j in range(8):
            if prev is not None and tgt[j] == prev[j]:
                continue
            hit = np.nonzero(win[:, j] == tgt[j])[0] if len(win) else []
            if len(hit):
                cls[i, j] = 1
                age[i, j] = (t_hi[i] - shift - st[a + hit[-1]]) / 1e6
            else:
                cls[i, j] = 2
        prev = tgt
    return cls, age


summary, echo_rows, win_rows, gap_rows = {}, [], [], []
for tag, ev in SESS.items():
    ev = os.path.expanduser(ev)
    sums = sums_of(ev)
    run = os.listdir(os.path.join(ev, "e1_server_runs"))[0]
    base = f"e1_server_runs/{run}"
    st, sseq, spos, scmp = load_states(verified(ev, f"{base}/states.jsonl", sums))
    spos32 = spos.astype(np.float32).astype(np.float64)
    cmds = load_cmds(verified(ev, f"{base}/commands.jsonl", sums))
    cseq = np.array([c[0] for c in cmds])
    # t_hi: first state that reports this command (or a later one) applied
    first_ge = np.searchsorted(np.maximum.accumulate(sseq), cseq, side="left")
    ok = first_ge < len(st)
    cmds = [c for c, k in zip(cmds, ok) if k]
    t_hi = st[first_ge[ok]]
    cls, age = classify(cmds, st, spos32, t_hi)
    ccls, _ = classify(cmds, st, spos32, t_hi, shift=SHIFT_NS)
    tgt = np.array([c[1] for c in cmds])

    def counts(c, rows=slice(None)):
        c = c[rows]
        return {k: int((c == v).sum()) for k, v in (("carry", 0), ("echo", 1), ("fresh", 2))}

    summary[tag] = {"run": run, "states": int(len(st)), "commands": len(cmds),
                    "session": counts(cls), "control_shift_20s": counts(ccls),
                    "echo_age_ms_median": float(np.nanmedian(age)), "echo_age_ms_p95": float(np.nanpercentile(age, 95)),
                    "echo_all8_same_command": int(((cls == 1).sum(axis=1) >= 5).sum())}
    for j, name in enumerate(J8):
        cj, kj = cls[:, j], ccls[:, j]
        echo_rows.append(dict(session=tag, joint=name, carry=int((cj == 0).sum()), echo=int((cj == 1).sum()),
                              fresh=int((cj == 2).sum()), control_echo=int((kj == 1).sum()),
                              control_fresh=int((kj == 2).sum()),
                              echo_age_ms_median=round(float(np.nanmedian(age[:, j])), 1) if (cj == 1).any() else ""))

    # HOVER -> REST_SHUT windows, same definition as tolerance.py
    for r in csv.DictReader(open(verified(ev, "ledger.csv", sums))):
        if r["role"] not in ("setup", "flight") or r["route"] not in ("PLACE_ROUTE", "RAISE_TO_SIDE") or r["outcome"] != "pass":
            continue
        S = json.load(open(verified(ev, f"recorder_logs/{r['log']}", sums)))["samples"]
        t0r, t1r = S[0]["wall_time_ns"], S[-1]["wall_time_ns"]
        idx = np.nonzero((t_hi >= t0r) & (t_hi <= t1r))[0]
        T = np.degrees(tgt[idx])
        dh = np.abs(T - np.degrees(HOVER)).max(axis=1)
        near = np.nonzero(dh <= 1.0)[0]
        i0 = near[0] if len(near) else int(np.argmin(dh))
        i1 = next(i for i in range(i0, len(idx)) if T[i, 7] < 20 - 0.1)
        ils = next(i for i in range(i0, i1) if T[i, 6] >= 0.5)
        hold = idx[i0:ils]; leg = idx[ils:i1]
        # echo ratchet on shoulder_pitch during the hold: fresh echoes and their step vs the previous target
        sp_steps = []
        for k in hold:
            if cls[k, 0] == 1 and k > 0:
                sp_steps.append(np.degrees(tgt[k, 0] - tgt[k - 1, 0]))
        sp_steps = np.array(sp_steps)
        a, b = int(np.searchsorted(st, t_hi[idx[i0]])), int(np.searchsorted(st, t_hi[idx[ils]]))
        # the arm's closest approach to HOVER on shoulder_pitch during the hold, and the target in force then
        kb = a + int(np.argmin(np.abs(np.degrees(spos[a:b + 1, 0]) + 40.0)))
        ci = int(np.searchsorted(t_hi, st[kb], side="right")) - 1
        row = dict(session=tag, leg=r["leg"], route=r["route"], role=r["role"],
                   hold_cmds=len(hold), hold_echo_cmds=int((cls[hold] == 1).any(axis=1).sum()),
                   hold_fresh_cmds=int((cls[hold] == 2).any(axis=1).sum()),
                   leg_cmds=len(leg), leg_echo_cmds=int((cls[leg] == 1).any(axis=1).sum()),
                   leg_fresh_cmds=int((cls[leg] == 2).any(axis=1).sum()),
                   sp_echo_steps=len(sp_steps),
                   sp_echo_steps_sag_dir=int((sp_steps > 0).sum()),
                   sp_echo_ratchet_deg=round(float(sp_steps.sum()), 2),
                   sp_target_hold_start=round(float(np.degrees(tgt[idx[i0], 0])), 2),
                   sp_target_leg_start=round(float(np.degrees(tgt[idx[ils] - 1, 0])), 2),
                   sp_pos_hold_start=round(float(np.degrees(spos[a, 0])), 2),
                   sp_pos_leg_start=round(float(np.degrees(spos[b, 0])), 2),
                   sp_pos_best=round(float(np.degrees(spos[kb, 0])), 2),
                   sp_target_at_best=round(float(np.degrees(tgt[ci, 0])), 2),
                   sp_pos_rise_after_best=round(float(np.degrees(spos[b, 0] - spos[kb, 0])), 2),
                   sp_target_rise_after_best=round(float(np.degrees(tgt[idx[ils] - 1, 0] - tgt[ci, 0])), 2))
        # gaps with no new command inside the window (target held constant)
        W = idx[i0:i1]
        for k0, k1 in zip(W[:-1], W[1:]):
            g = (t_hi[k1] - t_hi[k0]) / 1e9
            if g < 0.2:
                continue
            ga = int(np.searchsorted(st, t_hi[k0] + 100_000_000)); gb = int(np.searchsorted(st, t_hi[k1])) - 1
            if gb <= ga:
                continue
            gap_rows.append(dict(session=tag, leg=r["leg"], gap_s=round(g, 3),
                                 phase="hold" if k1 <= idx[ils] else "leg",
                                 **{f"drift_{J8[j][2:]}_deg": round(float(np.degrees(spos[gb, j] - spos[ga, j])), 3) for j in GUARD},
                                 sp_err_to_target_deg=round(float(np.degrees(spos[gb, 0] - tgt[k0, 0])), 3),
                                 stiff=bool(~scmp[ga:gb + 1, :7].any())))
        win_rows.append(row)
    print(tag, summary[tag])

for name, rows in (("echo_summary.csv", echo_rows), ("windows.csv", win_rows), ("gaps.csv", gap_rows)):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

wr = win_rows
summary["windows"] = {
    "n": len(wr),
    "hold_cmds_with_echo_frac": round(sum(r["hold_echo_cmds"] for r in wr) / sum(r["hold_cmds"] for r in wr), 3),
    "leg_cmds_with_echo_frac": round(sum(r["leg_echo_cmds"] for r in wr) / sum(r["leg_cmds"] for r in wr), 3),
    "sp_echo_steps_sag_dir_frac": round(sum(r["sp_echo_steps_sag_dir"] for r in wr) / max(1, sum(r["sp_echo_steps"] for r in wr)), 3),
    "sp_echo_ratchet_deg_min_med_max": [float(np.min([r["sp_echo_ratchet_deg"] for r in wr])),
                                        float(np.median([r["sp_echo_ratchet_deg"] for r in wr])),
                                        float(np.max([r["sp_echo_ratchet_deg"] for r in wr]))],
}
gs = [g for g in gap_rows if g["stiff"]]
if gs:
    d = np.array([g["drift_shoulder_pitch_deg"] for g in gs]); gl = np.array([g["gap_s"] for g in gs])
    summary["held_target_gaps"] = {"n": len(gs), "gap_s_min_med_max": [float(gl.min()), float(np.median(gl)), float(gl.max())],
                                   "sp_drift_deg_min_med_max": [float(d.min()), float(np.median(d)), float(d.max())],
                                   "max_abs_drift_any_guarded_deg": float(max(abs(g[f"drift_{J8[j][2:]}_deg"]) for g in gs for j in GUARD))}
json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("windows", "held_target_gaps") if k in summary}, indent=1))
