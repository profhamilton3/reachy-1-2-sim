cd /Users/terrancehamilton/reachy-1-2-sim-180/docs/reviews/2026-10-19-side-on-grasp-180
for j in final_lift final_place_red_cube final_place_pool_box_1 final_place_pool_cyl_1 perturb before_after neighbours; do
 python3 fly_batch.py flights/jobs_$j.json flights/out_$j.jsonl --procs 7 > flights/log_$j.txt 2>&1
done
echo FLIGHTSDONE > flights/done.txt
