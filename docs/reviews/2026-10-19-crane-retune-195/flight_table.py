"""Print a flight batch (`fly_batch.py` JSONL) as a table: outcome, the #174 criteria, the
placement error, the hover corrections (largest cumulative lateral / vertical), the first
measured tracking error, and every robot geom that touched the object.

python docs/reviews/2026-10-19-crane-retune-195/flight_table.py FLIGHTS.jsonl [--md]
"""
import json, math, sys

KEYS = ('placed_on_target', 'retained_height', 'slip_within_resolution', 'clean_placement',
        'clean_withdrawal', 'returned_to_present')
rows = sorted((json.loads(l) for l in open(sys.argv[1])), key=lambda r: r['job'].get('tag', ''))
md = '--md' in sys.argv
if md:
    print('| flight | outcome | criteria met | error mm | drag mm | correction lat / vert mm | hover gap error mm | object touched by |')
    print('|---|---|---|---|---|---|---|---|')
lat_max = vert_max = 0.0
ok_n = 0
for r in rows:
    c = r.get('criteria') or {}
    if r.get('refused'):
        outcome = 'refused: ' + r['refused'][:70]
    elif r.get('error'):
        outcome = 'error: ' + r['error'][:70]
    elif r.get('halt'):
        outcome = 'halt: ' + r['halt'][:70]
    else:
        outcome = 'completed'
    met = [k for k in KEYS if k in c and c[k]]
    allok = bool(c) and all(c.values()) and not r.get('halt')
    ok_n += allok
    lat = max((math.hypot(x['cumulative_mm'][0], x['cumulative_mm'][1]) for x in r.get('corrections', [])), default=0.0)
    vert = max((abs(x['cumulative_mm'][2]) for x in r.get('corrections', [])), default=0.0)
    lat_max, vert_max = max(lat_max, lat), max(vert_max, vert)
    gap = next((m['gap_error_mm'] for m in r.get('measure', []) if m['label'] == 'hover after pause'), None)
    touched = sorted({x['geom'] for x in r.get('object_contacts', [])})
    err = r.get('error_mm')
    errs = '' if err is None or not r.get('job', {}).get('dest') else f'{err:.1f}'
    drag = '' if r.get('drag_mm') is None else f"{r['drag_mm']:.1f}"
    crit = 'all' if allok else ('-' if not c else ','.join(k for k in KEYS if k in c and not c[k]) + ' false')
    line = [r['job'].get('tag', ''), outcome, crit, errs, drag, f'{lat:.2f} / {vert:.2f}', str(gap), ' '.join(touched)]
    print(('| ' + ' | '.join(line) + ' |') if md else '  '.join(line))
print(f'\n{ok_n}/{len(rows)} flights met every criterion; largest cumulative hover correction: '
      f'lateral {lat_max:.2f} mm, vertical {vert_max:.2f} mm')
