#!/usr/bin/env bash
# Benchmark queue for the higher-p comparison (2026-11-26), run on the GPU host:
#   setsid nohup bash scripts/run_hp_queue.sh > .tmp/hp_queue.log 2>&1 < /dev/null &
# Waits until the GPU has >= MIN_FREE_MIB free (another user's job may hold it), then runs every job
# sequentially. All outputs use new names (tags), so earlier results are never overwritten:
#   results/{gpu,cuda-fr,rt}_baseline_hp.json          latency (10k shots, batch 1) + quick throughput
#   results/{gpu,cuda-fr}_baseline_p0.001rerun.json    same-session reference at p = 0.001
#   results/throughput_{cuda,cuda-fr,rt}_<case>.json    batch sweeps (same protocol as throughput_cuda.json)
# A failing job is logged and the queue continues.
set -u
cd "$(dirname "$0")/.."
source env.sh > /dev/null 2>&1
MIN_FREE_MIB=${MIN_FREE_MIB:-8000}
GIVE_UP_S=${GIVE_UP_S:-86400}

free_mib() { python -c "import cupy as cp; print(int(cp.cuda.runtime.memGetInfo()[0] / 2**20))" 2>/dev/null || echo 0; }
stamp() { echo "[$(date '+%F %T')] free=$(free_mib)MiB load=$(cut -d' ' -f1-3 /proc/loadavg) $*"; }

start=$(date +%s)
until [ "$(free_mib)" -ge "$MIN_FREE_MIB" ]; do
  if [ $(( $(date +%s) - start )) -ge "$GIVE_UP_S" ]; then stamp "gave up waiting for a free GPU"; exit 1; fi
  sleep 120
done
stamp "GPU free, starting"

HP="bb144_z_p0.002_10k=beam8_230+beam32_340,bb144_z_p0.003_10k=beam8_230+beam32_340"
LOW="bb144_z_p0.001=beam8_230+beam32_340+beam64_640"
run() { stamp "start: $*"; python "$@" || stamp "FAILED: $*"; stamp "done: $*"; }

# Latency at batch 1 on the same 10k shots (fp32): plain CUDA, cached CUDA, RT.
for backend in cuda cuda-fr rt; do run scripts/bench_gpu.py 10000 "$backend" --cases "$HP" --tag hp --fp32-only; done
# Same-session reference at p = 0.001 (the published numbers come from earlier sessions and driver 580.196.07).
for backend in cuda cuda-fr; do run scripts/bench_gpu.py 10000 "$backend" --cases "$LOW" --tag p0.001rerun --fp32-only; done
# Batch sweeps, beam8, same protocol as results/throughput_cuda.json.
for case in bb144_z_p0.002_10k bb144_z_p0.003_10k; do
  for backend in cuda cuda-fr rt; do run scripts/bench_batch_throughput.py "$backend" "$case" --case "$case"; done
done
for backend in cuda cuda-fr; do run scripts/bench_batch_throughput.py "$backend" p0.001rerun --case bb144_z_p0.001; done
stamp "queue finished"
