# #201 re-fly after the closed-loop release (src/reachy_ai/motion/primitives.py
# release_to): the same jobs as run_all.sh, into out_release_*.jsonl.
cd "$(dirname "$0")/.."
P=${PROCS:-5}
for j in final_lift final_place_pool_cyl_1 final_place_pool_box_1 final_place_red_cube before_after perturb neighbours; do
 python3 fly_batch.py flights/jobs_$j.json flights/out_release_$j.jsonl --procs $P > flights/log_release_$j.txt 2>&1
done
echo RELEASEDONE > flights/done_release.txt
