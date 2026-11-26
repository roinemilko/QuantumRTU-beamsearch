#!/usr/bin/env bash
# The rest of run_hp_queue.sh after it was stopped on 2026-11-26 at 21:38 (during the cuda-fr p = 0.001 rerun),
# plus the S3 rerun with the unpruned RT baseline. Same rules: sequential, tagged outputs, failures logged.
#   setsid nohup bash scripts/run_hp_queue_rest.sh > .tmp/hp_queue_rest.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source env.sh > /dev/null 2>&1
free_mib() { python -c "import cupy as cp; print(int(cp.cuda.runtime.memGetInfo()[0] / 2**20))" 2>/dev/null || echo 0; }
stamp() { echo "[$(date '+%F %T')] host=$(hostname) free=$(free_mib)MiB load=$(cut -d' ' -f1-3 /proc/loadavg) $*"; }
run() { stamp "start: $*"; python "$@" || stamp "FAILED: $*"; stamp "done: $*"; }

run scripts/bench_gpu.py 10000 cuda-fr --cases "bb144_z_p0.001=beam8_230+beam32_340+beam64_640" --tag p0.001rerun --fp32-only
for case in bb144_z_p0.002_10k bb144_z_p0.003_10k; do
  for backend in cuda cuda-fr rt; do run scripts/bench_batch_throughput.py "$backend" "$case" --case "$case"; done
done
for backend in cuda cuda-fr; do run scripts/bench_batch_throughput.py "$backend" p0.001rerun --case bb144_z_p0.001; done
run scripts/bench_frontier_scene.py results/frontier_scene_v2.json
stamp "queue finished"
