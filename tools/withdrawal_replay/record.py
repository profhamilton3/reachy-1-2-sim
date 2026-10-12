"""What one recorded lift holds that the replay needs, as a small JSON record.

A recording (`live/states.jsonl`, `live/panel/panel-task-final.json`,
`live/exec/events-preflight.jsonl`) is ~30 MB; the replay needs two states and
a few events.  `extract` reduces it to a record that can be committed as a
test fixture, and every later step reads only the record.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Dict, List, Mapping, Sequence

OID = "red_cube"


def _clock_offset(events: Sequence[Mapping[str, Any]],
                  states: Sequence[Mapping[str, Any]]) -> float:
    from reachy_ai.evaluation.lift_replay import clock_offset
    return clock_offset(events, states, OID)


def _gripper_deg(state: Mapping[str, Any]) -> float:
    return math.degrees(next(j for j in state["joints"]
                             if j["name"] == "r_gripper")["position_rad"])


def _slim(state: Mapping[str, Any]) -> Dict[str, Any]:
    return {"host_time": state["host_time"], "sim_step": state["sim_step"],
            "joints": [{k: j[k] for k in ("name", "position_rad", "velocity_rad_s",
                                          "compliant")} for j in state["joints"]],
            "objects": [{k: o[k] for k in ("object_id", "pos_xyz", "quat_wxyz")}
                        for o in state["objects"]]}


def release_start(states: Sequence[Mapping[str, Any]], t_supported: float):
    """The last state before the gripper starts opening after SUPPORTED."""
    after = [s for s in states if s["host_time"] >= t_supported - 0.05]
    g0 = _gripper_deg(after[0])
    for prev, s in zip(after, after[1:]):
        if g0 - _gripper_deg(s) > 0.5:
            return prev
    raise ValueError("no gripper opening after SUPPORTED")


def extract(name: str, folder: str) -> Dict[str, Any]:
    f = pathlib.Path(folder)
    task = json.loads((f / "live/panel/panel-task-final.json").read_text())
    with open(f / "live/states.jsonl") as fh:
        states = [json.loads(line) for line in fh if line.strip()]
    pre = [json.loads(line) for line in open(f / "live/exec/events-preflight.jsonl")]
    crane = task["execution_evidence"]["crane"]
    events = crane["events"]
    off = _clock_offset(events, states)
    sup = next(e for e in events if e["kind"] == "SUPPORTED")
    rel = next(e for e in events if e["kind"] == "RELEASED")
    start = release_start(states, sup["t"] - off)
    post = min((s for s in states if s["host_time"] >= rel["t"] - off),
               key=lambda s: s["host_time"])
    wd = crane.get("withdrawal") or {}
    kept = [e for e in events if e["kind"] in ("CORRECTION", "MEASURE",
                                               "WITHDRAWAL_CORRECTION",
                                               "WITHDRAWAL_JUDGED", "SUPPORTED",
                                               "RELEASED")]
    return {
        "name": name,
        "recorded": {"withdrawal_clean": bool(wd.get("clean")),
                     "corrected": any(e["kind"] == "WITHDRAWAL_CORRECTION"
                                      for e in events) and bool(wd.get("clean")),
                     "drag_mm": 1000 * math.dist(rel["object"][:2],
                                                 next(o for o in start["objects"]
                                                      if o["object_id"] == OID)["pos_xyz"][:2]),
                     "judged": next((e["route"] for e in events
                                     if e["kind"] == "WITHDRAWAL_JUDGED"), None)},
        "preflight_object": next(e for e in pre if e["kind"] == "object"),
        "events": kept,
        "clock_offset_s": off,
        "release_start": _slim(start),
        "post_release": _slim(post),
    }


def load(path: str) -> Dict[str, Any]:
    return json.loads(pathlib.Path(path).read_text())


def dump(record: Mapping[str, Any], path: str) -> None:
    pathlib.Path(path).write_text(json.dumps(record, separators=(",", ":")))


def corrections(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [e for e in record["events"] if e["kind"] == "CORRECTION"]
