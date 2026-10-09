#!/usr/bin/env bash
# Week-1 gate: what is this machine, and can it render MuJoCo headless (needed later for LIBERO)?
#
#   bash scripts/env_check.sh [PYTHON]     (default: $HOME/venvs/smolvla/bin/python)
#
# Prints the system Python and driver, the venv's torch/CUDA/bf16 support, which GL libraries
# exist, and MuJoCo renders/s with EGL and with OSMesa. Save the output with the results.
set -uo pipefail
PY=${1:-$HOME/venvs/smolvla/bin/python}
here=$(cd "$(dirname "$0")" && pwd)

echo "== system"
echo "date_utc: $(date -u +%FT%TZ)"
echo "system_python: $(python3 --version 2>&1)"
uname -srm
nproc | sed 's/^/vcpus: /'
free -g | awk '/Mem:/ {print "ram_gb: "$2}'
df -h . | awk 'NR==2 {print "disk_free_here: "$4}'

echo "== gpu"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv 2>/dev/null \
  || echo "no nvidia-smi"

echo "== torch ($PY)"
"$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name} cc {p.major}.{p.minor} mem {p.total_memory / 2**30:.1f} GB")
    print("wheel archs", torch.cuda.get_arch_list(), "| kernel check", (torch.ones(4, device="cuda") * 2).sum().item() == 8.0)
    print("bf16 native", torch.cuda.is_bf16_supported(including_emulation=False),
          "| with emulation", torch.cuda.is_bf16_supported(including_emulation=True))
EOF

echo "== GL libraries"
for lib in libEGL_nvidia libGLX_nvidia libEGL.so libOSMesa libvulkan; do
  # grep -q would exit early and SIGPIPE ldconfig, which pipefail then reports as "missing"
  if ldconfig -p 2>/dev/null | grep "$lib" >/dev/null; then echo "found   $lib"; else echo "missing $lib"; fi
done

echo "== MuJoCo headless rendering"
for backend in egl osmesa; do
  timeout 120 "$PY" "$here/mujoco_render_check.py" "$backend" 2>&1 | tail -1
done
