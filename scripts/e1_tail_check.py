"""Parked-tail check for an E1 recorder log (schema 3).

Moved here unchanged from the E1 pilot evidence directory (pilot item,
2026-09-15 matrix readiness): the matrix procedure gates every setup and
flight on `PARKED_AT_<T>`, so this checker belongs in `scripts/` with the
rest of the E1 tooling -- not only in a one-off evidence directory with no
unit test, where it could drift from `rig_routes` unnoticed. Criteria are
unchanged from the pilot's version.

Usage (from the reachy-1-2-sim checkout, so the targets come from the
reviewed code, never from this file):

    PYTHONPATH=src python3 scripts/e1_tail_check.py runs/<log>.json PRESENT|REST|REST_SHUT|HOME [window_s]
    PYTHONPATH=src python3 scripts/e1_tail_check.py --selftest

Targets are `reachy_ai.motion.rig_routes.{PRESENT,REST,REST_SHUT,HOME}`,
read at run time and printed, so the operator sees the exact pose the log
is being judged against. REST and REST_SHUT share every arm joint and
differ only in `r_gripper` (OPEN -45 vs SHUT +20), so the posture test on
`GROSS_JOINTS` alone cannot tell them apart -- the gripper criterion below
is what does, and `--selftest` proves it.

Criteria for `PARKED_AT_<TARGET>=yes` over the last `window_s` (default
3 s) of samples:
  * >= 90% of the expected 20 Hz samples are present in the window;
  * every GROSS joint is within 8 deg of the target on every sample
    (posture tolerance, the same number `rig_routes.at_pose` defaults to);
  * every one of the 8 R_JOINTS has a spread (max - min) <= 1 deg across
    the window (the arm is not moving);
  * `r_gripper` on the last sample is within 3 deg of the target's.
Wrist residuals are printed for the ledger but are not pass/fail (the
known wrist-pitch droop is data, not a parking failure).
"""
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from reachy_ai.motion import rig_routes as R  # noqa: E402

TARGETS = ("PRESENT", "REST", "REST_SHUT", "HOME")
POSTURE_TOL_DEG = 8.0
SPREAD_TOL_DEG = 1.0
GRIPPER_TOL_DEG = 3.0
SAMPLE_HZ = 20.0


def check(samples, target_name, window_s=3.0):
    if target_name not in TARGETS:
        raise SystemExit(f"target must be one of {TARGETS}, got {target_name!r}")
    tgt = getattr(R, target_name)
    t_end = samples[-1]["t"]
    tail = [s for s in samples if s["t"] >= t_end - window_s]
    n_ok = len(tail) >= 0.9 * SAMPLE_HZ * window_s
    gross_err = {j: max(abs(s["joints"][j] - tgt[j]) for s in tail) for j in R.GROSS_JOINTS}
    wrist_err = {j: max(abs(s["joints"][j] - tgt[j]) for s in tail)
                 for j in R.R_JOINTS if j not in R.GROSS_JOINTS and j != "r_gripper"}
    spread = {j: max(s["joints"][j] for s in tail) - min(s["joints"][j] for s in tail)
              for j in R.R_JOINTS}
    grip = tail[-1]["joints"]["r_gripper"]
    grip_ok = abs(grip - tgt["r_gripper"]) <= GRIPPER_TOL_DEG
    posture_ok = max(gross_err.values()) <= POSTURE_TOL_DEG
    still_ok = max(spread.values()) <= SPREAD_TOL_DEG
    ok = n_ok and posture_ok and still_ok and grip_ok
    return {
        "target": target_name, "target_pose": tgt, "tail_samples": len(tail),
        "window_s": window_s, "gross_err_deg": gross_err, "wrist_err_deg": wrist_err,
        "spread_deg": spread, "r_gripper": grip, "gripper_ok": grip_ok,
        "posture_ok": posture_ok, "still_ok": still_ok, "samples_ok": n_ok, "ok": ok,
    }


def report(res):
    print(f"target {res['target']} = {json.dumps(res['target_pose'])}")
    print(f"tail_samples={res['tail_samples']} (last {res['window_s']:.1f}s)  "
          f"max_gross_err_deg={max(res['gross_err_deg'].values()):.1f}  "
          f"max_spread_deg={max(res['spread_deg'].values()):.1f}  "
          f"r_gripper={res['r_gripper']:+.1f} (target {res['target_pose']['r_gripper']:+.1f})")
    print("gross residuals: " + ", ".join(f"{j}={v:.1f}" for j, v in res["gross_err_deg"].items()))
    print("wrist residuals (info): " + ", ".join(f"{j}={v:.1f}" for j, v in res["wrist_err_deg"].items()))
    print(f"samples_ok={res['samples_ok']} posture_ok={res['posture_ok']} "
          f"still_ok={res['still_ok']} gripper_ok={res['gripper_ok']}")
    print(f"PARKED_AT_{res['target']}={'yes' if res['ok'] else 'NO'}")


def _synthetic(pose, seconds=20.0, moving_from_s=None, gripper=None):
    out = []
    for k in range(int(seconds * SAMPLE_HZ)):
        t = k / SAMPLE_HZ
        p = dict(pose)
        if gripper is not None:
            p["r_gripper"] = gripper
        if moving_from_s is not None and t > moving_from_s:
            p["r_elbow_pitch"] += (t - moving_from_s) * 10.0
        out.append({"t": t, "wall_time_ns": 10**9 + k * 50_000_000, "joints": p})
    return out


def selftest():
    cases = [
        ("parked at REST, judged REST", _synthetic(R.REST), "REST", True),
        ("parked at REST_SHUT, judged REST", _synthetic(R.REST_SHUT), "REST", False),
        ("parked at REST, judged REST_SHUT", _synthetic(R.REST), "REST_SHUT", False),
        ("parked at REST_SHUT, judged REST_SHUT", _synthetic(R.REST_SHUT), "REST_SHUT", True),
        ("parked at PRESENT, judged PRESENT", _synthetic(R.PRESENT), "PRESENT", True),
        ("parked at PRESENT, judged REST", _synthetic(R.PRESENT), "REST", False),
        ("parked at HOME, judged PRESENT", _synthetic(R.HOME), "PRESENT", False),
        ("moving at the end, judged REST", _synthetic(R.REST, moving_from_s=17.0), "REST", False),
        ("REST arm joints, gripper half-open (-20)", _synthetic(R.REST, gripper=-20.0), "REST", False),
        ("REST, gripper -47 (within 3)", _synthetic(R.REST, gripper=-47.0), "REST", True),
        ("REST, wrist_pitch drooped 18 deg (info only)",
         [dict(s, joints=dict(s["joints"], r_wrist_pitch=R.REST["r_wrist_pitch"] - 18.0))
          for s in _synthetic(R.REST)], "REST", True),
    ]
    # the fact the whole check rests on: REST and REST_SHUT differ ONLY in the gripper
    diff = {j for j in R.R_JOINTS if R.REST[j] != R.REST_SHUT[j]}
    assert diff == {"r_gripper"}, diff
    assert R.REST["r_gripper"] == R.OPEN == -45.0 and R.REST_SHUT["r_gripper"] == R.SHUT == 20.0
    assert R.at_pose(R.REST_SHUT, R.REST, tol=8.0, joints=list(R.GROSS_JOINTS)), \
        "gross-joint posture test is expected NOT to separate REST_SHUT from REST"
    failures = 0
    for label, samples, target, expect in cases:
        got = check(samples, target)["ok"]
        mark = "ok " if got == expect else "FAIL"
        failures += got != expect
        print(f"[{mark}] {label}: judged {target} -> {'yes' if got else 'NO'} (expected {'yes' if expect else 'NO'})")
    print(f"selftest: {len(cases) - failures}/{len(cases)} passed")
    return failures == 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    log = json.load(open(sys.argv[1]))
    window = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    res = check(log["samples"], sys.argv[2], window)
    report(res)
    sys.exit(0 if res["ok"] else 1)
