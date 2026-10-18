"""Summarise fly_point/*.jsonl (fly_wave_point.py point runs): usage: python3 fly_point_summary.py docs/reviews/2026-10-17-october-layout-192/fly_point"""
import json, sys
for f in ["none", "pool_box_1", "foam_block", "red_cube", "pool_cyl_1", "blue_cylinder", "soda_can"]:
    rows = [json.loads(l) for l in open(sys.argv[1] + "/" + f + ".jsonl") if l.startswith("{")]
    print("==", f)
    for r in rows:
        if "leg" not in r: continue
        res = r.get("result"); err = r.get("error")
        worst = min(r["worst_mjcf_cm"].values()) if r["worst_mjcf_cm"] else None
        moved = max(r["moved_mm"].values()) if r["moved_mm"] else 0
        objc = [c for c in r["robot_contacts"] if not any(x == "table_top" for x in c)]
        if res and isinstance(res, list) and res and isinstance(res[0], list) and res[0] and str(res[0][0]).startswith("point"):
            res = [res[0][1].replace("point: ", ""), res[0][2].replace("point: ", ""), res[1]]
        print(f"  {r['leg']:28s} {'OK ' + str(res) if res is not None else 'ERR ' + err[:150]}  worst {worst}  moved {moved}  other-contacts {objc}")
