"""Wait until the arm is quiescent: every R_JOINT's spread over the last 3 s
of states < 0.5 deg AND >= min_s elapsed since the last reset (sim_step /
500 Hz). Prints the settled pose. Exit 1 on timeout (STOP)."""
import json, math, sys, time, pathlib
run = pathlib.Path(sys.argv[1]); min_s = float(sys.argv[2]); timeout = float(sys.argv[3])
JOINTS = ["r_shoulder_pitch","r_shoulder_roll","r_arm_yaw","r_elbow_pitch","r_forearm_yaw","r_wrist_pitch","r_wrist_roll","r_gripper"]
def tail_states(n=200):
    data = (run/"states.jsonl").read_bytes().splitlines()[-n:]
    out = []
    for raw in data:
        try: out.append(json.loads(raw))
        except json.JSONDecodeError: pass
    return out
t0 = time.monotonic()
while time.monotonic() - t0 < timeout:
    st = tail_states()
    if len(st) >= 150:
        last = st[-1]; win = [s for s in st if s["wall_time_ns"] >= last["wall_time_ns"] - 3_000_000_000]
        def deg(s, j): return math.degrees(next(x["position_rad"] for x in s["joints"] if x["name"] == j))
        spread = {j: max(deg(s, j) for s in win) - min(deg(s, j) for s in win) for j in JOINTS}
        since_reset = last["sim_step"] / 500.0
        if max(spread.values()) < 0.5 and since_reset >= min_s:
            print(json.dumps({"settled_after_s": round(since_reset, 1), "sim_step": last["sim_step"], "seq": last["seq"],
                              "max_spread_deg": round(max(spread.values()), 3),
                              "pose_deg": {j: round(deg(last, j), 2) for j in JOINTS}}))
            sys.exit(0)
    time.sleep(1.0)
print("STOP: arm did not settle within timeout"); sys.exit(1)
