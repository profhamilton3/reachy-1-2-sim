"""One offline crane flight (`tools.place_sweep.fly`) and the numbers #195 needs from it:
the measured tracking error at each MEASURE, the corrections asked for, the outcome and the
#174 criteria.  Caps can be opened (``--cap-lat/--cap-vert`` mm) to SEE the correction a
flight needs; that is a measurement, not a setting.

PYTHONPATH=.:src:native_mujoco python docs/reviews/2026-10-19-crane-retune-195/fly_probe.py \
    --origin r3c2 [--dest r2c1] [--tilt 30 --lean 59.52 --face x] [--place-tilt .. --place-lean ..] [--out X.json]
"""
import argparse, json, sys
from dataclasses import replace
import tools.place_sweep  # noqa: F401
from tools.place_sweep import fly as F
from reachy_ai.tasks import crane_pick_live as CP
from reachy_ai.tasks import crane_place_live as PL


CONTACTS = []


def watch_contacts():
    """Record every robot geom that touches the object (first time per geom, with the
    sim time and the phase so far): pads are expected at the close, nothing else is."""
    import mujoco
    orig = F._Clock.sleep

    def sleep(self, dt):
        orig(self, dt)
        w = self.world
        m, d = w.m, w.d
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, F.OID)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if bid not in (b1, b2):
                continue
            other = c.geom2 if b1 == bid else c.geom1
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]) or ''
            if not body.startswith(('r_', 'l_', 'torso', 'head', 'neck')):
                continue                         # the table, the board, other objects
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or f'{body}/geom{other}'
            if c.dist < 0 and not any(x['geom'] == name for x in CONTACTS):
                CONTACTS.append({'geom': name, 't': round(self.t, 3), 'depth_mm': round(-1000 * c.dist, 3)})
    F._Clock.sleep = sleep


def summarise(run):
    if 'refused' in run:
        return {'refused': run['refused']}
    ev = run['events']
    meas = [dict(label=kw['label'], rung=kw['rung'], gap_error_mm=kw['gap_error_mm'],
                 joint_offset_deg=kw['joint_offset_deg']) for k, kw in ev if k == 'MEASURE']
    corr = [dict(label=kw['label'], shift_mm=kw['shift_mm'], cumulative_mm=kw['cumulative_mm'],
                 because=kw['because']) for k, kw in ev if k == 'CORRECTION']
    sd = [kw for k, kw in ev if k.startswith('SETDOWN')]
    out = run['outcome']
    return {'halt': out.get('halt'), 'final': run['final'], 'criteria': run['criteria'],
            'error_mm': run['error_mm'], 'tilt_deg': run['tilt_deg'], 'drag_mm': run['drag_mm'],
            'others_moved_max_mm': run['others_moved_max_mm'],
            'pad_contact_after_release_mm': run['pad_contact_after_release_mm'],
            'withdrawal': run['withdrawal'], 'way_out': run['way_out'], 'aborted': run['aborted'],
            'setdown_margin': run['setdown_margin'], 'replaced': out.get('replaced'),
            'place': {k: v for k, v in (out.get('place') or {}).items() if not isinstance(v, (list, dict))},
            'measure': meas, 'corrections': corr, 'setdown_events': sd[:12], 'sim_s': run['sim_s'],
            'object_contacts': CONTACTS,
            'phases': [(kw.get('name'), None) for k, kw in ev if k == 'phase']}


def run_job(j):
    """One flight from a dict of the options below (missing keys: defaults)."""
    _install()
    CONTACTS.clear()
    F.OID = j.get('obj', 'red_cube')
    dest = j.get('dest')
    spec = replace(CP.CraneSpec(), **{k: v for k, v in (('face_axis', j.get('face')),
                                                         ('tilt_deg', j.get('tilt')),
                                                         ('lean_bearing_deg', j.get('lean')))
                                      if v is not None})
    if dest:
        base = PL.PLACE_GRASP
        spec = replace(base, **{k: v for k, v in (('face_axis', j.get('place_face')),
                                                  ('tilt_deg', j.get('place_tilt')),
                                                  ('lean_bearing_deg', j.get('place_lean')))
                                if v is not None})
    lim = CP.CraneLimits()
    if j.get('cap_lat'):
        lim = replace(lim, max_lateral_correction_m=j['cap_lat'] / 1000)
    if j.get('cap_vert'):
        lim = replace(lim, max_vertical_correction_m=j['cap_vert'] / 1000)
    place_spec = None
    if j.get('retreat') is not None or j.get('backoff') is not None:
        place_spec = replace(PL.PlaceSpec(), **{k: v for k, v in (
            ('release_retreat_m', j.get('retreat')), ('release_backoff_m', j.get('backoff')))
            if v is not None})
    abort = None
    if j.get('abort_on'):
        abort = lambda ev, k=j['abort_on']: any(e == k for e, _ in ev)
    run = F.fly(cube_xy_offset=(j.get('dx', 0.0), j.get('dy', 0.0)), yaw_deg=j.get('yaw', 0.0),
                dest=dest, crane_spec=spec, origin='cell_' + j.get('origin', 'r2c2'),
                limits=lim, abort_when=abort, place_spec=place_spec)
    return summarise(run)


_installed = []


def _install():
    if not _installed:
        watch_contacts()
        _installed.append(True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--origin', default='r2c2'); a.add_argument('--dest')
    a.add_argument('--obj', default='red_cube')
    a.add_argument('--face'); a.add_argument('--tilt', type=float); a.add_argument('--lean', type=float)
    a.add_argument('--place-face'); a.add_argument('--place-tilt', type=float)
    a.add_argument('--place-lean', type=float)
    a.add_argument('--cap-lat', type=float); a.add_argument('--cap-vert', type=float)
    a.add_argument('--dx', type=float, default=0.0); a.add_argument('--dy', type=float, default=0.0)
    a.add_argument('--yaw', type=float, default=0.0)
    a.add_argument('--out')
    o = a.parse_args()
    s = dict(args=vars(o), **run_job({k: v for k, v in vars(o).items() if v is not None}))
    txt = json.dumps(s, default=str)
    if o.out:
        open(o.out, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
