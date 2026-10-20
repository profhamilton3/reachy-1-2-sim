# #201 re-fly: pads condim 4 (the MJCF), Siva's pool sizes.  The last batch
# flies the pool cylinder placements with the pads back at condim 3 (the
# contact override in fly_batch.py), to separate the size from the contact.
cd "$(dirname "$0")/.."
P=${PROCS:-5}
for j in final_lift final_place_pool_cyl_1 final_place_pool_box_1 final_place_red_cube before_after perturb neighbours base_place_pool_cyl_1; do
 python3 fly_batch.py flights/jobs_$j.json flights/out_$j.jsonl --procs $P > flights/log_$j.txt 2>&1
done
echo FLIGHTSDONE > flights/done.txt
