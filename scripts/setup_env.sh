#!/usr/bin/env bash
# Create an isolated Python 3.12 env for this repo, independent of the machine's default Python
# (Kaggle and Colab kernels are Python 3.13 now; LeRobot 0.6.1 needs >= 3.12).
#
#   bash scripts/setup_env.sh [VENV_DIR]
#
# Env overrides:
#   TORCH_INDEX  wheel index for torch (default cu128; cu126 for GPUs below compute capability 7.5,
#                e.g. P100/V100, which the cu128 wheels do not cover; cpu when no driver is found)
#   TRT_SPEC     TensorRT wheel spec (default tensorrt-cu12==10.16.*); WITH_TRT=0 skips it
#   WITH_MUJOCO  1 installs mujoco for the rendering check in env_check.sh (default 1)
set -euo pipefail

VENV=${1:-${VENV:-$HOME/venvs/smolvla}}
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)
if [ -n "$DRIVER" ]; then
  CU=cu128
  if [ -n "$CC" ] && [ "$CC" -lt 75 ] 2>/dev/null; then CU=cu126; fi
  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/$CU}
else
  TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}
fi
echo "driver=${DRIVER:-none} compute_cap=${CC:-?} torch_index=$TORCH_INDEX venv=$VENV"

uv python install 3.12
uv venv "$VENV" --python 3.12 --allow-existing
pip_in() { VIRTUAL_ENV="$VENV" uv pip install "$@"; }

# torch first from the CUDA index; LeRobot's own torch pin (>=2.7,<2.12) is then already satisfied.
pip_in "torch==2.11.*" "torchvision==0.26.*" --index-url "$TORCH_INDEX"
pip_in "lerobot[smolvla]==0.6.1" onnx onnxscript onnxruntime matplotlib
if [ -n "$DRIVER" ] && [ "${WITH_TRT:-1}" = 1 ]; then
  # The PyPI tensorrt-*-libs packages are stubs that fetch the real wheels from pypi.nvidia.com.
  # TensorRT 11 removed the weakly typed FP16 flag; trt_vision.py's default mode works on 10 and 11.
  pip_in "${TRT_SPEC:-tensorrt-cu12==10.16.*}" --extra-index-url https://pypi.nvidia.com
fi
if [ "${WITH_MUJOCO:-1}" = 1 ]; then
  pip_in mujoco
fi
"$VENV/bin/python" - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__)
if torch.cuda.is_available():
    cc = "sm_%d%d" % torch.cuda.get_device_capability(0)
    print("gpu", torch.cuda.get_device_name(0), cc, "| wheel archs", torch.cuda.get_arch_list())
    print("kernel check:", (torch.ones(4, device="cuda") * 2).sum().item() == 8.0)
EOF
echo "done: use $VENV/bin/python"
