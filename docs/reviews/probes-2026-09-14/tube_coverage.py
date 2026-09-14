"""Tube coverage vs the compiled MJCF over the r_gripper x r_wrist_roll grid,
the demonstrated misses, the shells-capsule axis margins, and the fixture table
(corrected tube / legacy tube / shells).  Run from the repo root:

    PYTHONPATH=src:native_mujoco python3 docs/reviews/probes-2026-09-14/tube_coverage.py

Offline: mj_forward only, no server, no SDK, no motion.
"""
import math, sys, numpy as np, mujoco
sys.path.insert(0,"tests/unit"); sys.path.insert(0,"src"); sys.path.insert(0,"native_mujoco")
from dataclasses import replace
import test_footprint_boards as T
from objects import build_scene_model_xml; from scene_io import load_scene; from placement import ObjectPlacer
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import hand_radius, joint_path, link_capsules, link_frames, _hand_radius_at
from reachy_ai.scene.awareness import SceneModel

def legacy_hand_radius(g):
    if g is None: g = -68.8
    y = -0.037 + 0.038*math.sin(math.radians(g)); return max(math.hypot(0.025,0.046), math.hypot(0.012, abs(y)+0.010))

doc=load_scene(T._SCENE_PATH); m=mujoco.MjModel.from_xml_string(build_scene_model_xml(doc)); d=mujoco.MjData(m); mujoco.mj_resetData(m,d)
for jid in range(m.njnt):
    if m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE: a=m.jnt_qposadr[jid]; d.qpos[a:a+7]=m.qpos0[a:a+7]
mujoco.mj_forward(m,d)
jq={n:int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in R.R_JOINTS}
G=lambda n: mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n)
def corners(g):
    gid=G(g); h=m.geom_size[gid]; Rm=d.geom_xmat[gid].reshape(3,3); c=d.geom_xpos[gid]
    return [c+Rm@np.array([sx*h[0],sy*h[1],sz*h[2]]) for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)]
def seg_dist(p,a,b):
    ab=b-a; t=np.clip(np.dot(p-a,ab)/np.dot(ab,ab),0,1); return np.linalg.norm(p-(a+t*ab))
def set_arm(q7,g):
    for n,v in zip(R.ARM7,q7): d.qpos[jq[n]]=math.radians(v)
    d.qpos[jq["r_gripper"]]=math.radians(g); mujoco.mj_forward(m,d)

# 1. coverage over grid: HOVER with roll override x gripper
print("== coverage: worst (corner dist - radius) over roll x gripper grid, HOVER base ==")
worst_excess=-1; worst_slack=1; wb=None
for roll in range(-45,46,5):
    for g in [-68.75]+list(range(-65,21,5))+[20.05]:
        q7=[R.HOVER[j] for j in R.ARM7]; q7[6]=roll; set_arm(q7,g)
        caps={c[0]:c for c in link_capsules(q7,"right",g)}; _,a,b,r=caps["hand"]; a=np.array(a); b=np.array(b)
        dmax=max(seg_dist(p,a,b) for gn in ("r_thumb_body","r_finger_body","r_thumb_col","r_finger_col") for p in corners(gn))
        ball=seg_dist(d.geom_xpos[G("r_wrist_ball")],a,b)+0.028
        ex=max(dmax,ball)-r; worst_excess=max(worst_excess,ex); 
        if r-dmax<worst_slack: worst_slack=r-dmax; wb=(roll,g)
print(f"max excess (must be <=0): {worst_excess:.2e} m; tightness: min(radius - farthest corner) = {worst_slack:.2e} at {wb}")

# 2. demonstrated misses now
print("== demonstrated misses, now (margin cm, + = inside) ==")
for pose,g,geom in [("HOVER",-68.8,"r_finger_col"),("HOVER",-68.8,"r_finger_body"),("HOVER",-45,"r_finger_col"),("HOVER",-45,"r_finger_body"),("REST",-68.8,"r_finger_col"),("REST",-68.8,"r_finger_body"),("REST",-45,"r_finger_col"),("REST",-45,"r_finger_body"),("REST",0,"r_finger_col"),("REST",0,"r_finger_body"),("REST_SHUT",-68.8,"r_finger_col"),("PRESENT",-68.8,"r_finger_col"),("HOME",-68.8,"r_finger_col")]:
    q7=[getattr(R,pose)[j] for j in R.ARM7]; set_arm(q7,g)
    _,a,b,r=[c for c in link_capsules(q7,"right",g) if c[0]=="hand"][0]; a=np.array(a); b=np.array(b)
    print(f"  {pose:9s} {g:6.1f} {geom:14s} margin {(r-max(seg_dist(p,a,b) for p in corners(geom)))*100:6.2f}")

# 3. shells-capsule margins (old test's metric) now
print("== tube radius - shells capsule axis-distance (surface / centreline), cm ==")
for pose in ("REST","REST_SHUT","HOVER","PRESENT","HOME"):
    q7=[getattr(R,pose)[j] for j in R.ARM7]
    for g in (-68.8,-45.0,0.0,20.0):
        _s,_e,wrist,Rm=link_frames(q7); ax=Rm@np.array([0,0,-1.0]); ax/=np.linalg.norm(ax); rt=hand_radius(g,q7[6])
        ws=wc=1e9
        for name,p0,p1,r in link_capsules(q7,"right",g,hand="shells"):
            if name in("upper_arm","forearm"): continue
            for p in (np.array(p0),np.array(p1)):
                rel=p-wrist; perp=np.linalg.norm(rel-np.dot(rel,ax)*ax); ws=min(ws,rt-perp-r); wc=min(wc,rt-perp)
        print(f"  {pose:9s} {g:6.1f} surface {ws*100:6.2f}  centreline {wc*100:6.2f}")

# 4. fixture table
print("== fixture table (tube new / tube legacy / shells), cm ==")
def worst_any(model,oid,hand,radius_fn=None):
    worst=1e9
    for name,wps in R.FOOTPRINT_LEGS.items():
        for a_,b_ in zip(wps,wps[1:]):
            qa=[a_[j] for j in R.ARM7]; qb=[b_[j] for j in R.ARM7]; gg=max(a_["r_gripper"],b_["r_gripper"],key=hand_radius)
            for q in joint_path(qa,qb,13):
                caps=link_capsules(q,"right",gg,hand=hand)
                if radius_fn: caps=[(n,p0,p1,radius_fn(gg)) if n=="hand" else (n,p0,p1,r) for n,p0,p1,r in caps]
                worst=min(worst,model.clearances(caps,ids=[oid])[oid].distance)
    return worst
boards={"evidence":({"soda_can":T._cell_xy("cell_r1c1"),"foam_block":T._cell_xy("cell_r2c3")},"foam_block"),
        "incident":({"foam_block":(T._cell_xy("cell_r2c3")[0],T._cell_xy("cell_r2c3")[1]-0.07)},"foam_block"),
        "foam_r3c3":({"foam_block":T._cell_xy("cell_r3c3")},"foam_block"),
        "soda_r2c3":({"soda_can":T._cell_xy("cell_r2c3")},"soda_can")}
for name,(objs,oid) in boards.items():
    model=T._board(objs)
    print(f"  {name:10s} tube_new {worst_any(model,oid,'tube')*100:6.2f}  tube_legacy {worst_any(model,oid,'tube',legacy_hand_radius)*100:6.2f}  shells {worst_any(model,oid,'shells')*100:6.2f}")
