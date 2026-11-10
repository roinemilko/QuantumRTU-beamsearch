#!/usr/bin/env bash
# Idempotent environment setup on the GPU host, entirely inside the project dir (no sudo).
# Usage on the host:  cd ~/pqc && bash scripts/gpu_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools .cache
if [ ! -x .tools/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh
fi
source env.sh
[ -d .venv ] || uv venv -q --python /usr/bin/python3 .venv
source env.sh
uv pip install -q -r external/BeamSearchDecoder/requirements.txt pytest "cupy-cuda13x>=14"
uv pip install -q -e .
(cd external/BeamSearchDecoder/decoder && python setup.py build_ext --inplace > "$PQC_ROOT/.cache/ionq_build.log" 2>&1)
echo "$PQC_ROOT/external/BeamSearchDecoder/decoder" > "$(python -c 'import site; print(site.getsitepackages()[0])')/ionq_beamsearch.pth"
python -c "import cupy, beam_search_decoder, rtbeam; d = cupy.cuda.Device(); print('cupy', cupy.__version__, 'cc', d.compute_capability, cupy.cuda.runtime.getDeviceProperties(0)['name'].decode())"
