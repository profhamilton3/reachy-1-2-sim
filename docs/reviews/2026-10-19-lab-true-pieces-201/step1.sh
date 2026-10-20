# #201 preflight sweeps (planning only, no physics), as #180's step1.sh.
set -x
cd "$(dirname "$0")/../../.."
E=docs/reviews/2026-10-19-lab-true-pieces-201
P=${PROCS:-2}
mkdir -p $E/sweep
python3 -m tools.place_sweep --grasp best --lift --objects red_cube,pool_box_1,pool_cyl_1,blue_cylinder,foam_block,soda_can --procs $P --out $E/sweep/lift_best.md --json $E/sweep/lift_best.json
for O in red_cube pool_box_1 pool_cyl_1; do
python3 -m tools.place_sweep --grasp best --objects $O --procs $P --out $E/sweep/place_best_$O.md --json $E/sweep/place_best_$O.json
done
python3 -m tools.place_sweep --grasp crane --objects red_cube,pool_box_1 --procs $P --out $E/sweep/place_crane.md --json $E/sweep/place_crane.json
python3 -m tools.place_sweep --grasp crane --lift --objects red_cube,pool_box_1 --procs $P --out $E/sweep/lift_crane.md --json $E/sweep/lift_crane.json
echo STEP1DONE
