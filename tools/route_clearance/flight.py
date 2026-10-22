"""Physics flights of a route (design section 4.2's player, offline): motors on, minimum-jerk goals streamed at
STREAM_HZ from the present read-back to each leg's pose, then a hold. Every contact of a right-arm geom is kept, with
its leg and time; following and arrival errors are per leg; every placed piece's largest displacement is kept.
"""
import math

import mujoco
import numpy as np

from .route import ARM7
from .world import JOINTS

STREAM_HZ = 50
SETTLE_S = 1.0           # motors on at the start pose before the first leg (Step 1: 500 steps)
HOLD_S = 1.0             # after each leg, before the arrival read (Step 1's B2_GRIP_TUNE flight)
GROSS = ARM7[:4]         # following error is judged on these (design section 4.3)
GUARD = ARM7[:4] + ("r_wrist_pitch",)   # arrival is judged on these: the sim's CRITICAL_JOINTS
WEAK = ("r_forearm_yaw", "r_wrist_roll")   # arrival reported only


def min_jerk(s):
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def leg_samples(leg, hz=STREAM_HZ):
    """[(t, q)] along the leg's minimum-jerk path at the stream rate, both ends included."""
    n = max(1, int(round(leg.duration_s * hz)))
    out = []
    for i in range(n + 1):
        b = min_jerk(i / n)
        out.append((i / hz, {j: leg.q_from[j] + (leg.q_to[j] - leg.q_from[j]) * b for j in JOINTS}))
    return out


def place_pieces(world, layout):
    """layout: {body: (x, y)}. The piece stands upright on the board, 1 mm above it, before the settle."""
    d = world.data
    for body, (x, y) in sorted(layout.items()):
        a = world.body_qadr(body)
        d.qpos[a:a + 3] = [x, y, world.board_top + world.piece_half_height(body) + 0.001]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]


def _contacts(world):
    pairs = set()
    d = world.data
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = world.gname(int(c.geom1)), world.gname(int(c.geom2))
        if a.startswith("r_") or b.startswith("r_"):
            pairs.add("/".join(sorted((a, b), key=lambda n: (not n.startswith("r_"), n))))
    return pairs


def fly(world, legs, layout=None):
    """Fly every leg in order, continuously. Returns {"settle": {...}, "legs": [...], "moved_mm": {...}}.

    Contacts are recorded per physics step as {pair: [first_t, last_t, steps, dev_start, dev_to]}: t in seconds from
    the leg's start (the hold runs on past the leg's duration), and the arm's largest joint deviation (deg, the seven
    arm joints) from the leg's start and end poses over the contact's steps. The settle's contacts use the first leg's
    start pose for both."""
    layout = layout or {}
    core, m, d = world.core, world.model, world.data
    dt = float(m.opt.timestep)
    tick = int(round(1.0 / (STREAM_HZ * dt)))
    core.reset()
    world.set_arm(legs[0].q_from)
    place_pieces(world, layout)
    mujoco.mj_forward(m, d)
    core.controller.sync_targets_to_current(d)
    for n in JOINTS:
        core.controller.set_compliant("act_" + n, False)

    def present():
        return {n: math.degrees(d.qpos[world.adr[n]]) for n in JOINTS}

    def deviation(q):
        return max(abs(math.degrees(d.qpos[world.adr[j]]) - q[j]) for j in ARM7)

    def note(store, t, q_start, q_to):
        pairs = _contacts(world)
        if not pairs:
            return
        dev_start, dev_to = deviation(q_start), deviation(q_to)
        for pair in pairs:
            row = store.setdefault(pair, [t, t, 0, 0.0, 0.0])
            row[1] = t
            row[2] += 1
            row[3] = max(row[3], dev_start)
            row[4] = max(row[4], dev_to)

    settle = {}
    for k in range(int(round(SETTLE_S / dt))):
        core.advance()
        note(settle, (k + 1) * dt, legs[0].q_from, legs[0].q_from)
    start = {body: d.qpos[world.body_qadr(body):world.body_qadr(body) + 3].copy() for body in layout}
    moved = {body: 0.0 for body in layout}

    def track():
        for body in layout:
            a = world.body_qadr(body)
            moved[body] = max(moved[body], float(np.linalg.norm(d.qpos[a:a + 3] - start[body])))

    rows = []
    for leg in legs:
        q0 = present()
        n = max(1, int(round(leg.duration_s * STREAM_HZ)))
        goal = dict(q0)
        follow, contacts, step = 0.0, {}, 0
        for i in range(n):
            b = min_jerk((i + 1) / n)
            goal = {j: q0[j] + (leg.q_to[j] - q0[j]) * b for j in JOINTS}
            for j in JOINTS:
                core.controller.set_goal_position("act_" + j, math.radians(goal[j]))
            for _ in range(tick):
                core.advance()
                step += 1
                cur = present()
                follow = max(follow, max(abs(cur[j] - goal[j]) for j in GROSS))
                note(contacts, step * dt, leg.q_from, leg.q_to)
                track()
        for _ in range(int(round(HOLD_S / dt))):
            core.advance()
            step += 1
            note(contacts, step * dt, leg.q_from, leg.q_to)
            track()
        cur = present()
        rows.append({"follow_deg": follow,
                     "arrival_deg": max(abs(cur[j] - leg.q_to[j]) for j in GUARD),
                     "arrival_weak_deg": max(abs(cur[j] - leg.q_to[j]) for j in WEAK),
                     "contacts": contacts})
    return {"settle": settle, "legs": rows, "moved_mm": {b: v * 1000 for b, v in moved.items()}}
