"""URDF (native_mujoco/model/reachy.urdf) right-arm chain: torso -> wrist / r_gripper_thumb / right_tip."""
import numpy as np
def R(ax,a):
    c,s=np.cos(a),np.sin(a); T=np.eye(4)
    if ax=='x': T[1:3,1:3]=[[c,-s],[s,c]]
    if ax=='y': T[0,0],T[0,2],T[2,0],T[2,2]=c,s,-s,c
    if ax=='z': T[0:2,0:2]=[[c,-s],[s,c]]
    return T
def Tr(x,y,z): T=np.eye(4); T[:3,3]=(x,y,z); return T
def fk(qdeg, upto='tip'):
    q=np.radians(qdeg)
    T=Tr(0,-0.19,0)@R('y',q[0])@R('x',q[1])@R('z',q[2])@Tr(0,0,-0.28)@R('y',q[3])@R('z',q[4])@Tr(0,0,-0.25)@R('y',q[5])
    if upto=='wrist': return T
    T=T@Tr(0,0,-0.0325)@R('x',q[6])
    if upto=='thumb': return T
    return T@Tr(0,-0.012,-0.085)
