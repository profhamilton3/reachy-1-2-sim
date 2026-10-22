"""Exact signed distances for the box and capsule pairs, in numpy, so a clearance never rests on
`mj_geomDistance`'s box-box reading. In MuJoCo 3.11 it returns a spurious 0 for box pairs whenever the pair is closer
than distmax, at any distmax (a finger box 2.93 cm from a 50.8 mm cube reads 0.0 at distmax 0.03), so Step 1's guard
could read a box pair too close or, when the 3 cm read was right and the 6 cm read spurious, too far.

Separated pairs: the minimum over the feature pairs that can hold the closest points of two convex polytopes (each
vertex against the other solid, and edge against edge), which is exact. Overlapping boxes: minus the separating-axis
penetration depth. A capsule is its segment swept by its radius.
"""
import itertools

import numpy as np

_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(_SIGNS[i] != _SIGNS[j]) == 1]


class Box:
    def __init__(self, centre, rot, half):
        self.c = np.asarray(centre, dtype=float)
        self.r = np.asarray(rot, dtype=float).reshape(3, 3)    # columns: the box's axes in the world
        self.h = np.asarray(half, dtype=float)

    def vertices(self):
        return self.c + (_SIGNS * self.h) @ self.r.T

    def edges(self):
        v = self.vertices()
        return v[[i for i, _ in _EDGES]], v[[j for _, j in _EDGES]]


def point_box(points, box):
    """Distance from each point to the solid box (0 inside)."""
    local = (np.atleast_2d(points) - box.c) @ box.r
    return np.linalg.norm(np.maximum(np.abs(local) - box.h, 0.0), axis=1)


def segment_segment(p1, q1, p2, q2):
    """Distances between segments p1q1[i] and p2q2[j] for all pairs, shape (len(p1), len(p2)) (Ericson 5.1.9)."""
    p1, q1, p2, q2 = (np.atleast_2d(x)[:, None, :] if k < 2 else np.atleast_2d(x)[None, :, :]
                      for k, x in enumerate((p1, q1, p2, q2)))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)
    denom = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom > 1e-12 * a * e, np.clip((b * f - c * e) / denom, 0.0, 1.0), 0.0)
        t = (b * s + f) / e
        s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    diff = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
    return np.linalg.norm(diff, axis=-1)


def _sat_overlap(a, b):
    """Smallest overlap of the two boxes' projections over the 15 separating axes; negative when separated."""
    axes = [a.r[:, i] for i in range(3)] + [b.r[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            n = np.cross(a.r[:, i], b.r[:, j])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n / norm)
    t = b.c - a.c
    best = np.inf
    for axis in axes:
        ra = np.sum(a.h * np.abs(a.r.T @ axis))
        rb = np.sum(b.h * np.abs(b.r.T @ axis))
        best = min(best, ra + rb - abs(t @ axis))
    return best


def box_box(a, b):
    overlap = _sat_overlap(a, b)
    if overlap >= 0.0:
        return -overlap
    ea, eb = a.edges(), b.edges()
    return float(min(point_box(a.vertices(), b).min(), point_box(b.vertices(), a).min(),
                     segment_segment(ea[0], ea[1], eb[0], eb[1]).min()))


def _segment_box_hits(p, q, box):
    """Whether the segment pq crosses the solid box (slab test in the box frame)."""
    lp, lq = (p - box.c) @ box.r, (q - box.c) @ box.r
    d = lq - lp
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(lp[k]) > box.h[k]:
                return False
            continue
        u0, u1 = sorted(((-box.h[k] - lp[k]) / d[k], (box.h[k] - lp[k]) / d[k]))
        t0, t1 = max(t0, u0), min(t1, u1)
        if t0 > t1:
            return False
    return True


def capsule_box(p, q, radius, box):
    """Signed distance from the capsule (segment pq, radius) to the box. When the segment itself enters the box the
    depth is at least the radius, and -radius is returned."""
    if _segment_box_hits(p, q, box):
        return -float(radius)
    e = box.edges()
    seg = min(point_box(np.array([p, q]), box).min(), segment_segment(p, q, e[0], e[1]).min())
    return float(seg - radius)
