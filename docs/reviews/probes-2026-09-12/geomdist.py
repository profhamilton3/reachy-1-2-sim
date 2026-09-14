"""Geometry distances between MuJoCo geoms that do not depend on MuJoCo's
collision routines.

mujoco.mj_geomDistance (3.11, native CCD on) agrees with these to <1e-4 m at
most poses, but returned an exact 0.0 for a box-box pair at isolated samples
of an otherwise ~5 cm-clear sweep (see probe_evidence_board.txt, first run).
The probes therefore use this module for every reported number and keep
mj_geomDistance only as a cross-check.

Method: sample the TARGET geom's surface (box: 6 faces x N x N; cylinder:
wall + both caps), and take the minimum closed-form distance from those
points to the ARM primitive (capsule: point-to-segment minus radius; sphere:
point-to-centre minus radius; box: point-to-OBB).  For box-box pairs both
directions are sampled.  Resolution is set by N (default 41 on a 7 cm face
-> ~1.8 mm spacing; the reported error is at most half that for convex
targets).  Negative values mean penetration, but only to the depth the
sampled surface points fall inside the arm primitive -- fine for "is it
inside", not a true penetration depth.
"""
import numpy as np
import mujoco

BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)


def surface_points(m, d, g, n=41):
    """World points on the surface of box/cylinder/sphere/capsule geom g."""
    t = int(m.geom_type[g])
    h = np.asarray(m.geom_size[g], dtype=float)
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    u = np.linspace(-1.0, 1.0, n)
    pts = []
    if t == BOX:
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            A, B = np.meshgrid(u, u)
            for s in (-1.0, 1.0):
                loc = np.zeros((A.size, 3))
                loc[:, i] = s * h[i]
                loc[:, j] = A.ravel() * h[j]
                loc[:, k] = B.ravel() * h[k]
                pts.append(loc)
    elif t == CYLINDER:
        r, hz = h[0], h[1]
        th = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        T, Z = np.meshgrid(th, u * hz)
        pts.append(np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(),
                             Z.ravel()], axis=1))
        rr = np.linspace(0, r, n // 2 + 1)
        T, RR_ = np.meshgrid(th, rr)
        for s in (-1.0, 1.0):
            pts.append(np.stack([RR_.ravel() * np.cos(T).ravel(),
                                 RR_.ravel() * np.sin(T).ravel(),
                                 np.full(T.size, s * hz)], axis=1))
    elif t in (SPHERE, CAPSULE):
        r = h[0]
        hz = h[1] if t == CAPSULE else 0.0
        phi = np.linspace(0, np.pi, n)
        th = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        P, T = np.meshgrid(phi, th)
        sp = np.stack([r * np.sin(P).ravel() * np.cos(T).ravel(),
                       r * np.sin(P).ravel() * np.sin(T).ravel(),
                       r * np.cos(P).ravel()], axis=1)
        for s in (-1.0, 1.0) if hz else (0.0,):
            q = sp.copy()
            q[:, 2] += s * hz
            pts.append(q)
        if hz:
            Z, T = np.meshgrid(u * hz, th)
            pts.append(np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(),
                                 Z.ravel()], axis=1))
    else:
        raise ValueError(f"unsupported geom type {t}")
    loc = np.concatenate(pts)
    return c + loc @ R.T


def point_to_geom(m, d, pts, g):
    """Closed-form signed distance from world points to primitive geom g."""
    t = int(m.geom_type[g])
    h = np.asarray(m.geom_size[g], dtype=float)
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    loc = (pts - c) @ R            # world -> local
    if t == BOX:
        outside = np.maximum(np.abs(loc) - h, 0.0)
        dout = np.linalg.norm(outside, axis=1)
        inside = np.max(np.abs(loc) - h, axis=1)      # <= 0 when inside
        return np.where(dout > 0, dout, inside)
    if t == SPHERE:
        return np.linalg.norm(loc, axis=1) - h[0]
    if t == CAPSULE:
        z = np.clip(loc[:, 2], -h[1], h[1])
        seg = loc.copy()
        seg[:, 2] -= z
        return np.linalg.norm(seg, axis=1) - h[0]
    if t == CYLINDER:
        rad = np.linalg.norm(loc[:, :2], axis=1) - h[0]
        ax = np.abs(loc[:, 2]) - h[1]
        outside = np.linalg.norm(np.stack([np.maximum(rad, 0),
                                           np.maximum(ax, 0)], 1), axis=1)
        inside = np.maximum(rad, ax)
        return np.where(outside > 0, outside, inside)
    raise ValueError(f"unsupported geom type {t}")


def distance(m, d, arm_geom, target_geom, n=41):
    """Min distance from arm primitive to sampled target surface (m)."""
    best = float(point_to_geom(m, d, surface_points(m, d, target_geom, n),
                               arm_geom).min())
    if int(m.geom_type[arm_geom]) == BOX:
        # box-box: sample both ways, the corners of either can be closest
        best = min(best, float(point_to_geom(
            m, d, surface_points(m, d, arm_geom, n), target_geom).min()))
    return best


def mj_distance(m, d, g1, g2, distmax=1.0):
    fromto = np.zeros(6)
    return float(mujoco.mj_geomDistance(m, d, g1, g2, distmax, fromto))
