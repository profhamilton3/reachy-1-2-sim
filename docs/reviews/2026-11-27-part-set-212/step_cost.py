#!/usr/bin/env python3
"""Physics step time of FWDCenterLabSivaPool by which parts reach the end faces (#212).

    python3 step_cost.py

Times mj_step only, with every object parked, after 1000 settling steps.  Each
variant is made from the committed scene by changing part heights (or removing a
block's bore ring) in memory.

A BENCHMARK, NOT A TEST: the numbers move with the machine and its load.  What
gates a step-cost regression is the contact-count test in
tests/unit/test_part_set_scene.py (owner decision D3).
"""
import copy, pathlib, sys, time

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "native_mujoco"))

import mujoco
from contact_model import apply_contact_model
from objects import build_scene_model_xml
from scene_io import load_scene

MODEL = str(REPO / "native_mujoco" / "model" / "reachy_1_2.xml")
RELIEF = 0.0001


def variant(doc, rings, blocks):
    """Full height for the listed parts of a ring cylinder (16 parts) / a holed
    block (20: walls 0-3, bore ring 4-19), 0.2 mm less for the rest."""
    doc = copy.deepcopy(doc)
    for o in doc["objects"]:
        g = o["geometry"]
        if g["kind"] != "compound":
            continue
        full = max(p["size"][2] for p in g["parts"])
        keep = rings if len(g["parts"]) == 16 else blocks
        g["parts"] = [dict(p, size=[p["size"][0], p["size"][1],
                                    full if i in keep else full - 2 * RELIEF])
                      for i, p in enumerate(g["parts"])]
    return doc


def run(label, doc, n=1000):
    model = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc, MODEL))
    apply_contact_model(model, None)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = model.jnt_qposadr[j]
            data.qpos[a:a + 7] = model.qpos0[a:a + 7]
    for _ in range(1000):
        mujoco.mj_step(model, data)
    rounds = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            mujoco.mj_step(model, data)
        rounds.append(1000 * (time.perf_counter() - t) / n)
    print(f"{label:52s} {sorted(rounds)[1]:.3f} ms/step  ncon={data.ncon:3d} "
          f"noslip={model.opt.noslip_iterations} (budget {1000 * model.opt.timestep:.1f} ms)")


base = load_scene(REPO / "scenes" / "FWDCenterLabSivaPool.yaml")
without = dict(base, objects=[o for o in base["objects"]
                              if "part-set" not in (o.get("tags") or [])])
run("without the PART set (the scene before #212)", without)
square = dict(base, objects=[
    dict(o, geometry=dict(o["geometry"], parts=o["geometry"]["parts"][:4]))
    if o["geometry"]["kind"] == "compound" and len(o["geometry"]["parts"]) == 20 else o
    for o in base["objects"]])
run("A: every face flat", variant(base, set(range(16)), set(range(20))))
run("B: ring on 4 of 16 segments (85 % standing area)", variant(base, {0, 4, 8, 12}, set(range(4))))
run("C: B, block on its 2 long walls", variant(base, {0, 4, 8, 12}, {0, 1}))
run("D without the blocks' bore rings (square bore)", square)
run("D as committed (owner D1 + D2)  <- the scene", base)
