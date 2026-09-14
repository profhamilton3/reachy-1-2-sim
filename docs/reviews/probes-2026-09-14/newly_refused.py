"""Which boards the corrected tube newly refuses (legacy >= 0, corrected < 0):
(A) six scene objects x nine grid cells x every guarded route; (B) a 6 cm cube on
the 4 cm / 224-position grid, worst over any guarded route, with the true MJCF
clearance at each flipped position.  Run from the repo root:

    PYTHONPATH=src:native_mujoco python3 docs/reviews/probes-2026-09-14/newly_refused.py

Writes newly_refused.json next to itself (gitignored).  Offline: mj_forward only.
"""
import math, sys, json, numpy as np, mujoco
sys.path.insert(0,"tests/unit"); sys.path.insert(0,"src"); sys.path.insert(0,"native_mujoco")
from dataclasses import replace
import test_footprint_boards as T
from objects import build_scene_model_xml; from scene_io import load_scene; from placement import ObjectPlacer
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import hand_radius, joint_path, link_capsules
from reachy_ai.scene.awareness import SceneModel

def legacy_hand_radius(g):
    if g is None: g=-68.8
    y=-0.037+0.038*math.sin(math.radians(g)); return max(math.hypot(0.025,0.046), math.hypot(0.012,abs(y)+0.010))

# per-route leg samples with the guard's aperture rule (A2)
LEGS={}
for route,wps in R.FOOTPRINT_LEGS.items():
    LEGS[route]=[]
    for a,b in zip(wps,wps[1:]):
        gg=max(a["r_gripper"],b["r_gripper"],key=hand_radius)
        for q in joint_path([a[j] for j in R.ARM7],[b[j] for j in R.ARM7],13): LEGS[route].append((q,gg))

def worst(model,oid,route,hand,radius_fn=None):
    w=1e9
    for q,gg in LEGS[route]:
        caps=link_capsules(q,"right",gg,hand=hand)
        if radius_fn: caps=[(n,p0,p1,radius_fn(gg)) if n=="hand" else (n,p0,p1,r) for n,p0,p1,r in caps]
        w=min(w,model.clearances(caps,ids=[oid])[oid].distance)
    return w

# A. real objects x 9 cells x each guarded route
print("== A. scene objects on grid cells: legacy>=0 and new<0 (per route) ==")
cells=[f"cell_r{r}c{c}" for r in (1,2,3) for c in (1,2,3)]
objs=["red_cube","blue_cylinder","soda_can","foam_block","pool_box_1","pool_cyl_1"]
flipsA=[]; n_boards=0
for oid in objs:
    for cell in cells:
        model=T._board({oid:T._cell_xy(cell)}); n_boards+=1
        for route in R.FOOTPRINT_LEGS:
            lo=worst(model,oid,route,"tube",legacy_hand_radius); nw=worst(model,oid,route,"tube"); sh=worst(model,oid,route,"shells")
            if lo>=0 and nw<0: flipsA.append((oid,cell,route,lo,nw,sh))
            if nw<0 and lo<0 and route=="PLACE_ROUTE": pass
print(f"boards x routes checked: {n_boards} x {len(R.FOOTPRINT_LEGS)}; flips: {len(flipsA)}")
for f in flipsA: print(f"  {f[0]:14s} {f[1]:10s} {f[2]:16s} legacy {f[3]*100:+.1f}  new {f[4]*100:+.1f}  shells {f[5]*100:+.1f}")
print("already refused under legacy (any route), for context:")
for oid in objs:
    for cell in cells:
        model=T._board({oid:T._cell_xy(cell)})
        lo=min(worst(model,oid,r,"tube",legacy_hand_radius) for r in R.FOOTPRINT_LEGS)
        if lo<0: print(f"  {oid:14s} {cell:10s} legacy {lo*100:+.1f}")

# B. 224-grid 6cm cube, worst over ANY guarded route, with MJCF reference at flipped positions
print("== B. 6 cm cube on the 4 cm grid, worst over any guarded route ==")
doc=load_scene(T._SCENE_PATH); m=mujoco.MjModel.from_xml_string(build_scene_model_xml(doc)); d=mujoco.MjData(m); mujoco.mj_resetData(m,d)
for jid in range(m.njnt):
    if m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE: a=m.jnt_qposadr[jid]; d.qpos[a:a+7]=m.qpos0[a:a+7]
mujoco.mj_forward(m,d); placer=ObjectPlacer(m,d,doc)
OID="pool_box_1"; placer.reshape(OID,size=[0.06]*3)
ob=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,OID); og=int(m.body_geomadr[ob]); oadr=int(m.jnt_qposadr[int(m.body_jntadr[ob])])
jq={n:int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in R.R_JOINTS}
arm_geoms=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ("r_upper_arm_col","r_forearm_col","r_thumb_body","r_finger_body","r_thumb_col","r_finger_col","r_wrist_ball")]
tb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"table_top"); tg=int(m.body_geomadr[tb]); tc=d.geom_xpos[tg]; th=m.geom_size[tg]
xs=np.arange(tc[0]-th[0]+0.03, tc[0]+th[0]-0.03+1e-9, 0.04); ys=np.arange(tc[1]-th[1]+0.03, tc[1]+th[1]-0.03+1e-9, 0.04)
scene=SceneModel.from_yaml(T._SCENE_PATH); scene._objects[OID]=replace(scene._objects[OID],size=(0.06,)*3)
top=scene.table_surface_z
allsamples=[s for route in LEGS for s in LEGS[route]]
flipsB=[]; n_new_refused=0; n_legacy_refused=0
for x in xs:
    for y in ys:
        scene.update_poses({OID:(float(x),float(y),top+0.031)})
        lo=nw=sh=1e9
        for q,gg in allsamples:
            caps=link_capsules(q,"right",gg); nw=min(nw,scene.clearances(caps,ids=[OID])[OID].distance)
            lo=min(lo,scene.clearances([(n,p0,p1,legacy_hand_radius(gg)) if n=="hand" else (n,p0,p1,r) for n,p0,p1,r in caps],ids=[OID])[OID].distance)
            sh=min(sh,scene.clearances(link_capsules(q,"right",gg,hand="shells"),ids=[OID])[OID].distance)
        n_new_refused+= nw<0; n_legacy_refused+= lo<0
        if lo>=0 and nw<0:
            # MJCF reference at this position over the same samples
            d.qpos[oadr:oadr+3]=(float(x),float(y),top+0.031); d.qpos[oadr+3:oadr+7]=(1,0,0,0)
            ref=1e9
            for q,gg in allsamples:
                for n,v in zip(R.ARM7,q): d.qpos[jq[n]]=math.radians(v)
                d.qpos[jq["r_gripper"]]=math.radians(gg); mujoco.mj_forward(m,d)
                ref=min(ref,min(T._mjcf_distance(m,d,g,og) for g in arm_geoms))
            flipsB.append((float(x),float(y),lo,nw,sh,ref))
print(f"positions {len(xs)*len(ys)}: refused legacy {n_legacy_refused}, new {n_new_refused}, newly refused {len(flipsB)}")
for x,y,lo,nw,sh,ref in flipsB:
    print(f"  x={x:.2f} y={y:.2f}  legacy {lo*100:+.1f}  new {nw*100:+.1f}  shells {sh*100:+.1f}  MJCF ref {ref*100:+.1f}")
json.dump({"A":flipsA,"B":flipsB,"grid":{"xs":xs.tolist(),"ys":ys.tolist()}}, open(__import__("os").path.join(__import__("os").path.dirname(__file__),"newly_refused.json"),"w"), indent=1)
