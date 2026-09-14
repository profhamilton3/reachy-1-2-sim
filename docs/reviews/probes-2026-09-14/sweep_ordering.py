"""The review's 224-position PLACE_ROUTE-tail sweep under the corrected tube:
tube / shells / MJCF reference (all hand geoms incl. collision pads) at the same
aperture, plus the reference-at-commanded-aperture artefact for comparison.  Run
from the repo root:

    PYTHONPATH=src:native_mujoco python3 docs/reviews/probes-2026-09-14/sweep_ordering.py

Offline: mj_forward only.
"""
import math, sys, numpy as np, mujoco
sys.path.insert(0,"tests/unit"); sys.path.insert(0,"src"); sys.path.insert(0,"native_mujoco")
import test_footprint_boards as T
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.kinematics import hand_radius, joint_path, link_capsules
from reachy_ai.scene.awareness import SceneModel
from dataclasses import replace
from placement import ObjectPlacer
from objects import build_scene_model_xml; from scene_io import load_scene

doc = load_scene(T._SCENE_PATH); m = mujoco.MjModel.from_xml_string(build_scene_model_xml(doc)); d = mujoco.MjData(m)
mujoco.mj_resetData(m,d)
for jid in range(m.njnt):
    if m.jnt_type[jid]==mujoco.mjtJoint.mjJNT_FREE:
        a=m.jnt_qposadr[jid]; d.qpos[a:a+7]=m.qpos0[a:a+7]
mujoco.mj_forward(m,d); placer=ObjectPlacer(m,d,doc)
OID="pool_box_1"; placer.reshape(OID,size=[0.06]*3)
ob=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,OID); og=int(m.body_geomadr[ob]); oadr=int(m.jnt_qposadr[int(m.body_jntadr[ob])])
jq={n:int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in R.R_JOINTS}
arm_geoms=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ("r_upper_arm_col","r_forearm_col","r_thumb_body","r_finger_body","r_thumb_col","r_finger_col","r_wrist_ball")]
tb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"table_top"); tg=int(m.body_geomadr[tb]); tc=d.geom_xpos[tg]; th=m.geom_size[tg]
xs=np.arange(tc[0]-th[0]+0.03, tc[0]+th[0]-0.03+1e-9, 0.04); ys=np.arange(tc[1]-th[1]+0.03, tc[1]+th[1]-0.03+1e-9, 0.04)
scene=SceneModel.from_yaml(T._SCENE_PATH); scene._objects[OID]=replace(scene._objects[OID],size=(0.06,)*3)
leg=R.FOOTPRINT_LEGS["PLACE_ROUTE"]; samples=[]
for a,b in zip(leg,leg[1:]):
    qa=[a[j] for j in R.ARM7]; qb=[b[j] for j in R.ARM7]; ga,gb=a["r_gripper"],b["r_gripper"]; gg=max(ga,gb,key=hand_radius)
    for k,q in enumerate(joint_path(qa,qb,13)): samples.append((q, ga+(k/12.0)*(gb-ga), gg))
print("gripper per leg (cmd endpoints -> gg):", [(a["r_gripper"],b["r_gripper"],max(a["r_gripper"],b["r_gripper"],key=hand_radius)) for a,b in zip(leg,leg[1:])])
top_z=scene.table_surface_z
sr_cmd=[]; sr_gg=[]; ts=[]; tr_gg=[]
for x in xs:
    for y in ys:
        d.qpos[oadr:oadr+3]=(float(x),float(y),top_z+0.031); d.qpos[oadr+3:oadr+7]=(1,0,0,0); mujoco.mj_forward(m,d)
        scene.update_poses({OID:(float(x),float(y),top_z+0.031)})
        tw=sw=rc=rg=None
        for q,gcmd,gg in samples:
            ct=scene.clearances(link_capsules(q,"right",gg),ids=[OID])[OID].distance
            cs=scene.clearances(link_capsules(q,"right",gg,hand="shells"),ids=[OID])[OID].distance
            tw=ct if tw is None else min(tw,ct); sw=cs if sw is None else min(sw,cs)
            for n,v in zip(R.ARM7,q): d.qpos[jq[n]]=math.radians(v)
            for gref,store in ((gcmd,"c"),(gg,"g")):
                d.qpos[jq["r_gripper"]]=math.radians(gref); mujoco.mj_forward(m,d)
                ref=min(T._mjcf_distance(m,d,g,og) for g in arm_geoms)
                if store=="c": rc=ref if rc is None else min(rc,ref)
                else: rg=ref if rg is None else min(rg,ref)
        ts.append(tw-sw); sr_cmd.append(sw-rc); sr_gg.append(sw-rg); tr_gg.append(tw-rg)
print(f"positions {len(ts)}")
print(f"tube>shells: {sum(1 for g in ts if g>1e-6)}  max {max(ts)*100:.2f} cm")
print(f"shells>ref (ref @ gcmd, as committed): {sum(1 for g in sr_cmd if g>1e-6)}  max {max(sr_cmd)*100:.2f} cm")
print(f"shells>ref (ref @ gg, same aperture): {sum(1 for g in sr_gg if g>1e-6)}  max {max(sr_gg)*100:.2f} cm")
tol=0.002
print(f"tube>shells (>2mm): {sum(1 for g in ts if g>tol)}  max {max(ts)*100:.2f} cm")
print(f"tube>ref   (>2mm): {sum(1 for g in tr_gg if g>tol)}  max {max(tr_gg)*100:.2f} cm")
print(f"shells tighter than tube by >1cm: {sum(1 for g in ts if g < -0.01)}")
print(f"tube over-conservatism vs ref: max {max(-g for g in tr_gg)*100:.2f} cm")
