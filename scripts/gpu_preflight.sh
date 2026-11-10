#!/usr/bin/env bash
# Read-only checks for a rented GPU box: RT cores, driver, CUDA, OptiX runtime, disk.
# Usage (from the laptop):  ssh <host> 'bash -s' < scripts/gpu_preflight.sh
# Changes nothing on the machine. Prints one PASS/WARN/FAIL line per check.

set -u
pass() { printf 'PASS  %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; }

if ! command -v nvidia-smi >/dev/null; then fail "nvidia-smi missing: no NVIDIA driver"; exit 1; fi
gpu=$(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader | head -1)
pass "GPU: $gpu"
case "$gpu" in
  *A100*|*H100*|*H200*|*B200*|*B300*|*GB200*|*GB300*|*V100*) fail "this GPU has no RT cores";;
  *) pass "GPU model has RT cores (RTX/L-series/A10/A6000 class)";;
esac
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$driver" -ge 570 ] && pass "driver major $driver >= 570 (OptiX 9.x, Blackwell ok)" || warn "driver major $driver < 570: use OptiX 8.x headers"

optix=$(ldconfig -p 2>/dev/null | grep -E 'libnvoptix\.so\.1' | head -1)
rtcore=$(ldconfig -p 2>/dev/null | grep -E 'libnvidia-rtcore' | head -1)
[ -n "$optix" ] && pass "OptiX runtime: $optix" || fail "libnvoptix.so.1 not found (Ubuntu: package libnvidia-gl-$driver*)"
[ -n "$rtcore" ] && pass "RT core lib: $rtcore" || warn "libnvidia-rtcore not in ldconfig cache"
python3 - <<'EOF' 2>/dev/null && pass "libnvoptix.so.1 loads via dlopen" || fail "libnvoptix.so.1 does not load"
import ctypes; ctypes.CDLL("libnvoptix.so.1")
EOF

if command -v nvcc >/dev/null; then pass "nvcc: $(nvcc --version | tail -1)"; else
  ls -d /usr/local/cuda*/bin/nvcc >/dev/null 2>&1 && warn "nvcc not on PATH; found $(ls -d /usr/local/cuda*/bin/nvcc | head -1)" || warn "no CUDA toolkit (CuPy wheels still work; NVRTC comes with them)"
fi
command -v dpkg >/dev/null && dpkg -l 2>/dev/null | awk '/libnvidia-gl|nvidia-driver|nvidia-headless/ {print "INFO  pkg " $2 " " $3}'

pass "python: $(python3 --version 2>&1)"
python3 -c "import venv, ensurepip" 2>/dev/null && pass "python venv available" || warn "python3-venv missing (uv can replace it)"
command -v git >/dev/null && pass "git present" || fail "git missing"
pass "CPUs: $(nproc), RAM: $(free -g | awk '/Mem:/ {print $2}') GB, disk free: $(df -h ~ | awk 'NR==2 {print $4}')"
