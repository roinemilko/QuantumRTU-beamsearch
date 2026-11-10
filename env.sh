# Source before working: keeps every cache and tool state inside the project dir.
export PQC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="$PQC_ROOT/.cache/uv"
export PIP_CACHE_DIR="$PQC_ROOT/.cache/pip"
export MPLCONFIGDIR="$PQC_ROOT/.cache/matplotlib"
export MYPY_CACHE_DIR="$PQC_ROOT/.cache/mypy"
export PYTHONPYCACHEPREFIX="$PQC_ROOT/.cache/pycache"
# IonQ's decoder allocates 4*beam_width*nnz doubles per decode call; with glibc defaults that
# can come from fresh mmap pages and adds ~10 ms/shot of page faults on WSL2 (not decoding work).
export MALLOC_MMAP_THRESHOLD_=1073741824
export MALLOC_TRIM_THRESHOLD_=1073741824
source "$PQC_ROOT/.venv/bin/activate" 2>/dev/null || true
