"""Damped least-squares 6-D IK on urdf_fk.fk, with optional joint clamping (degrees)."""
import numpy as np
from urdf_fk import fk
def err(q,T):
    F=fk(q); Re=T[:3,:3]@F[:3,:3].T
    w=0.5*np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])
    return np.r_[T[:3,3]-F[:3,3], 0.1*w]          # metres, 0.1 m per rad
def solve(T, seed, lim=None, iters=600):
    q=np.array(seed,float)
    for _ in range(iters):
        e=err(q,T)
        if np.linalg.norm(e)<1e-7: break
        J=np.zeros((6,7)); h=1e-4
        for i in range(7):
            d=np.zeros(7); d[i]=h; J[:,i]=(err(q+d,T)-e)/h
        dq=-J.T@np.linalg.solve(J@J.T+1e-6*np.eye(6), e)
        q=q+np.clip(dq,-5,5)
        if lim is not None: q=np.clip(q,lim[:,0],lim[:,1])
    F=fk(q); Re=T[:3,:3]@F[:3,:3].T
    ang=np.degrees(np.arccos(np.clip((np.trace(Re)-1)/2,-1,1)))
    return q, 1000*np.linalg.norm(F[:3,3]-T[:3,3]), ang
