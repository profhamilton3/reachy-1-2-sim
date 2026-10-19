set -x
cd /Users/terrancehamilton/reachy-1-2-sim-180
E=/Users/terrancehamilton/reachy-1-2-sim-180/docs/reviews/2026-10-19-side-on-grasp-180
python3 -m tools.place_sweep --grasp best --lift --objects red_cube,pool_box_1,pool_cyl_1,blue_cylinder,foam_block,soda_can --procs 7 --out $E/sweep/lift_best.md --json $E/sweep/lift_best.json
for O in red_cube pool_box_1 pool_cyl_1; do
python3 -m tools.place_sweep --grasp best --objects $O --procs 7 --out $E/sweep/place_best_$O.md --json $E/sweep/place_best_$O.json
done
python3 -m tools.place_sweep --grasp crane --objects red_cube,pool_box_1 --procs 7 --out $E/sweep/place_crane.md --json $E/sweep/place_crane.json
echo STEP1DONE
