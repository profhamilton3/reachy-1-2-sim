"""Generate one marker-gated notebook per Stage 1 cycle (scope rev 2 §4).
Each motion cell: waits for control/go_<leg> (written by the operator only
after every preceding gate passed) AND the leg's recorder printing 'fly the
route now'; applies the route's own start check; runs the pilot's tool; no
retry, no recovery, no reset from the notebook. control/stop disarms all."""
import json, sys, pathlib
REPO = "/Users/terrancehamilton/reachy-1-2-sim-stage1"
P = f"{REPO}/docs/reviews/probes-2026-09-15-e1-stage1-b4"
CYCLES = {
    "S1a": [("setup_a", "RAISE_TO_SIDE", "primitives.raise_to_side", "HOME->PRESENT", "PLACE_ROUTE_start"),
            ("flight_a", "LOWER_TO_REST", "rig_motion.from_present", "PRESENT->REST", "PRESENT")],
    "S1b": [("flight_b", "PLACE_ROUTE", "rig_motion.deploy_to_rest", "HOME->REST", "PLACE_ROUTE_start")],
    "S1c": [("setup_c", "PLACE_ROUTE", "rig_motion.deploy_to_rest", "HOME->REST", "PLACE_ROUTE_start"),
            ("flight_c", "LIFT_TO_PRESENT", "rig_motion.to_present", "REST->PRESENT", "REST")],
}
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
cycle = sys.argv[1]; legs = CYCLES[cycle]
cells = [md(f"# E1 Stage 1 cycle {cycle} — B4, simulator only, guard absent by omission (2026-09-15)\n\n"
            f"Code `{REPO}` = `main` `6bca12e`. Legs: " + ", ".join(f"`{l[0]}` {l[1]} via `{l[2]}` ({l[3]})" for l in legs) +
            ". One attempt each; a `stop` marker or a failed start check means no motion."),
 code(f'''# Cell 1 — connect with literals; hygiene shown
import os, sys, time, json, pathlib, traceback
sys.path.insert(0, "{REPO}/src"); sys.path.insert(0, "{REPO}/scripts")
CTRL = pathlib.Path("{P}/control"); RECORD_ROOT = "{P}/e1_server_runs"
SCENE = "{REPO}/scenes/e1_boards/B4_pool_box_1_r2c3.yaml"; LEAD_IN_S = 3.0; CYCLE = "{cycle}"
print("REACHY env in this kernel:", {{k: v for k, v in os.environ.items() if k.upper().startswith("REACHY")}})
assert "REACHY_IP" not in os.environ and "REACHY_ENABLE_MOTION" not in os.environ, "shell hygiene violated"
from reachy_sdk import ReachySDK
HOST, PORT = "localhost", 50051
reachy = ReachySDK(host=HOST, sdk_port=PORT)
print(f"ReachySDK(host={{HOST!r}}, sdk_port={{PORT}}) connected at wall {{time.time_ns()}} mono {{time.monotonic_ns()}}")
print("python:", sys.executable)
'''),
 code('''# Cell 2 — motion-client binding check: the recorder's identity function on THIS SDK object
import e1_identity
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion import primitives
from reachy_ai.tasks import rig_motion
def _pose(): return {name: float(getattr(reachy.r_arm, name).present_position) for name in R.R_JOINTS}
ident = e1_identity.verify_simulator_identity(host=HOST, port=PORT, scene_path=SCENE, record_root=RECORD_ROOT, read_sdk_joints=_pose)
d = ident.as_dict(); print(json.dumps(d, indent=2, default=str))
BINDING_OK = bool(ident.ok)
(CTRL / (f"binding_ok_{CYCLE}" if BINDING_OK else f"binding_FAIL_{CYCLE}")).write_text(json.dumps(d, default=str))
print("BINDING_OK =", BINDING_OK); print("present:", {k: round(v, 1) for k, v in _pose().items()})
def wait_for(pred, timeout_s, period=0.25):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if (CTRL / "stop").exists(): return "stop"
        if pred(): return "ready"
        time.sleep(period)
    return "timeout"
def start_check(kind):
    p = _pose(); here = R.posture_of(p)
    if kind == "PLACE_ROUTE_start": ok, why = rig_motion.check_start(reachy.r_arm, R.PLACE_ROUTE)
    elif kind == "PRESENT": ok = R.at_pose(p, R.PRESENT, tol=12.0, joints=list(R.GROSS_JOINTS)); why = "" if ok else "not at PRESENT (gross, 12 deg)"
    elif kind == "REST": ok = R.at_pose(p, R.REST, tol=12.0, joints=list(R.GROSS_JOINTS)); why = "" if ok else "not at REST (gross, 12 deg)"
    return {"kind": kind, "ok": bool(ok), "why": why, "posture_of": here, "pose": {k: round(v, 1) for k, v in p.items()}}
PREV_OK = BINDING_OK
'''),
]
for name, route, tool, desc, check in legs:
    fn = {"primitives.raise_to_side": "primitives.raise_to_side(reachy.r_arm)",
          "rig_motion.from_present": "rig_motion.from_present(reachy.r_arm, on_phase=lambda *a: phases.append([time.monotonic_ns(), *map(str, a)]))",
          "rig_motion.deploy_to_rest": "rig_motion.deploy_to_rest(reachy.r_arm, on_phase=lambda *a: phases.append([time.monotonic_ns(), *map(str, a)]))",
          "rig_motion.to_present": "rig_motion.to_present(reachy.r_arm, on_phase=lambda *a: phases.append([time.monotonic_ns(), *map(str, a)]))"}[tool]
    cells.append(code(f'''# Leg {name}: {route} via {tool} ({desc}) — one attempt, gated on go_{name} + the recorder
LEG = {{"leg": "{name}", "route": "{route}", "tool": "{tool}", "cycle": CYCLE}}
go = wait_for(lambda: (CTRL / "go_{name}").exists(), 1800) if PREV_OK else "not_eligible"
LEG["go"] = go; print("go:", go)
rec = wait_for(lambda: (CTRL / "recorder_{name}.log").exists() and "fly the route now" in (CTRL / "recorder_{name}.log").read_text(), 900) if go == "ready" else go
LEG["recorder_status"] = rec; print("recorder status:", rec)
if rec == "ready":
    time.sleep(LEAD_IN_S)
    LEG["start_check"] = start_check("{check}"); print("start check:", LEG["start_check"])
if rec == "ready" and LEG["start_check"]["ok"]:
    LEG["t_start_mono_ns"] = time.monotonic_ns(); LEG["t_start_wall_ns"] = time.time_ns()
    phases = []
    reachy.turn_on("r_arm")
    try:
        ret = {fn}
        LEG["outcome"] = "returned"; LEG["returned"] = ret
    except Exception as exc:
        LEG["outcome"] = f"EXC {{type(exc).__name__}}: {{exc}}"; traceback.print_exc()
    LEG["phases"] = phases
    LEG["t_end_mono_ns"] = time.monotonic_ns(); LEG["t_end_wall_ns"] = time.time_ns()
    LEG["elapsed_s"] = (LEG["t_end_mono_ns"] - LEG["t_start_mono_ns"]) / 1e9
    LEG["end_pose"] = {{k: round(v, 1) for k, v in _pose().items()}}
    print("outcome:", LEG["outcome"], "elapsed %.1f s" % LEG["elapsed_s"]); print("returned:", LEG.get("returned")); print("end pose:", LEG["end_pose"])
else:
    LEG["outcome"] = "not_attempted"
PREV_OK = LEG["outcome"] == "returned"
(CTRL / "{name}_done").write_text(json.dumps(LEG)); print(json.dumps(LEG, default=str))
'''))
cells.append(code('''# Final read-only state; no further motion
print("final pose:", {k: round(v, 1) for k, v in _pose().items()}); print("done at wall", time.time_ns())
'''))
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "e1venv", "display_name": "e1venv", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = pathlib.Path(P) / f"e1_stage1_{cycle}.ipynb"; out.write_text(json.dumps(nb, indent=1)); print(out)
