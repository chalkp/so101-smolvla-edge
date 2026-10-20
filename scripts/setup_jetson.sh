#!/usr/bin/env bash
# Install so101-smolvla-edge on a Jetson Orin with uv.  Usage: scripts/setup_jetson.sh [jp7|jp6]
# Without an argument the flavour comes from /etc/nv_tegra_release (L4T R36 = JetPack 6, R38+ = 7).
#
#   jp7  JetPack 7.x: CUDA 13, Ubuntu 24.04, Python 3.12, torch from download.pytorch.org/whl/cu132
#        (stock aarch64 wheels, one SBSA build for Orin and Thor; their sm_80 kernels run on the
#        Orin's sm_87)
#   jp6  JetPack 6.x: CUDA 12.6, Ubuntu 22.04, Python 3.10, torch from pypi.jetson-ai-lab.io
set -euo pipefail
cd "$(dirname "$0")/.."

flavour="${1:-}"
if [[ -z "$flavour" ]]; then
  if [[ -f /etc/nv_tegra_release ]]; then
    major=$(sed -n 's/^# R\([0-9]*\).*/\1/p' /etc/nv_tegra_release)
    if (( major >= 38 )); then flavour=jp7; else flavour=jp6; fi
    echo "L4T R${major}: using '${flavour}'"
  else
    echo "not a Jetson (no /etc/nv_tegra_release);" \
      "on a desktop GPU or an Intel NPU laptop use pixi:" \
      "pixi install  /  pixi install -e npu  (uv fallback: uv sync --extra desktop --extra dev)" >&2
    exit 1
  fi
fi
case "$flavour" in
  jp7) py=3.12 ;;
  jp6) py=3.10 ;;
  *) echo "flavour must be jp7 or jp6" >&2; exit 1 ;;
esac

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --python "$py" --extra "$flavour" --extra trt

# JetPack installs the TensorRT python bindings for the system Python only. Link just that package
# into the venv: all of dist-packages would shadow the venv's numpy and friends.
sys_trt=$(/usr/bin/python3 -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))" \
  2>/dev/null || true)
venv_py=$(.venv/bin/python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
sys_py=$(/usr/bin/python3 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
if [[ -n "$sys_trt" && "$venv_py" == "$sys_py" ]]; then
  site=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  mkdir -p .venv/trt-site && ln -sfn "$sys_trt" .venv/trt-site/tensorrt
  echo "$PWD/.venv/trt-site" > "$site/system_tensorrt.pth"
  echo "TensorRT: linked $sys_trt into the venv"
else
  echo "TensorRT python bindings not linked (system: '${sys_trt:-none}'," \
    "python $sys_py vs venv $venv_py); --vision trt unavailable"
fi

# No library paths to export: the CUDA 13 wheels find their own libraries in site-packages
# (JetPack's CUDA is only there for nvcc), and so101_smolvla_edge preloads the libcudss that
# JetPack 6's torch needs.
uv run so101-smolvla-edge doctor

cat <<'MSG'

Next:
  * maximum clocks for benchmarks / runs (as root):  sudo nvpmodel -m $(grep -oP 'POWER_MODEL ID=\K[0-9]+(?= NAME=MAXN)' /etc/nvpmodel.conf | head -1) && sudo jetson_clocks
    (the MAXN mode's id differs per module: 2 = MAXN_SUPER on Orin Nano, 0 = MAXN on AGX Orin;
    mode 0 on an Orin Nano is 15 W, GPU capped at 612 MHz; check with: nvpmodel -q)
    `doctor` reports whether the GPU clock is pinned; `sudo jetson_clocks` pins it.
  * serial + camera access:  sudo usermod -aG dialout,video $USER   (log in again)
  * copy a bundle here and check it:
      uv run so101-smolvla-edge verify <bundle>  &&  uv run so101-smolvla-edge bench <bundle>
  * fastest path (the default kernels are Triton, no compile step):
      uv run so101-smolvla-edge build-trt <bundle> --precision bf16 int8
          (TensorRT vision engines, once per device; int8 needs <bundle>/trt/act_clip.json)
      uv run so101-smolvla-edge bench <bundle> --vision trt --trt-precision int8
MSG
