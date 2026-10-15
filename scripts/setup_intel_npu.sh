#!/usr/bin/env bash
# Set up the Intel NPU (Core Ultra) for `--backend openvino` without root: the environment, and
# Intel's user-space driver unpacked next to it.
#
#   scripts/setup_intel_npu.sh [DIR]          # DIR defaults to <env>/npu-stack
#
# The environment is pixi's `npu` (.pixi/envs/npu) when pixi is installed, else a uv venv (.venv).
#
# OpenVINO reaches the NPU through three layers. This script provides the third:
#
#   kernel     intel_vpu (mainline) -> /dev/accel/accel0, which your user must be able to open
#              (group `render`, or the ACL a desktop session grants the logged-in user)
#   loader     Level Zero's libze_loader.so.1 and the compiler's libtbb.so.12, from Ubuntu:
#              sudo apt install libze1 libtbb12
#   driver     libze_intel_npu.so and its graph compiler, from Intel's .debs at
#              https://github.com/intel/linux-npu-driver/releases
#
# With root you can `dpkg -i` those .debs as their release notes say instead, and so101_smolvla_edge
# needs nothing more. Without root the two .debs that hold libraries are unpacked into DIR, and
# so101_smolvla_edge/_npu_libs.py points the loader at
# DIR/usr/lib/x86_64-linux-gnu/libze_intel_npu.so.1. The firmware .deb is left out: the kernel
# loads firmware from /lib/firmware, and Ubuntu's linux-firmware has it. For a DIR other than the
# default, export SO101_NPU_STACK=DIR.
#
# NPU_DRIVER_VERSION picks the release. The default is the one tested on a Panther Lake laptop
# with Ubuntu 26.04, kernel 7.0 and OpenVINO 2026.4.
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE=${NPU_DRIVER_VERSION:-v1.38.0}

[ "$(uname -m)" = x86_64 ] || { echo "Intel's NPU driver is x86_64 only"; exit 1; }
if command -v pixi >/dev/null; then
    pixi install -e npu
    env=.pixi/envs/npu
    run=(pixi run -e npu)
elif command -v uv >/dev/null; then
    uv sync --python 3.12 --extra npu --extra dev
    env=.venv
    run=(uv run --no-sync)
else
    echo "install pixi (https://pixi.sh) or uv (https://docs.astral.sh/uv/) first"; exit 1
fi
DIR=${1:-$env/npu-stack}

if [ ! -e /dev/accel/accel0 ]; then
    echo "!! no /dev/accel/accel0: this machine has no NPU, or its kernel has no intel_vpu driver."
    echo "   The OpenVINO backend still runs on the CPU plugin (--ov-device CPU)."
    exit 0
fi

lib="$DIR/usr/lib/x86_64-linux-gnu"
if [ -e "$lib/libze_intel_npu.so.1" ]; then
    echo "driver already unpacked in $DIR (delete it to fetch $RELEASE again)"
else
    # sourced in a subshell: os-release sets VERSION and more
    tag="ubuntu$(. /etc/os-release && echo "${VERSION_ID//./}")"
    api="https://api.github.com/repos/intel/linux-npu-driver/releases/tags/$RELEASE"
    urls=$(curl -fsSL "$api" | grep -o '"browser_download_url": *"[^"]*"' | cut -d'"' -f4)
    url=$(grep -- "-$tag\.tar\.gz$" <<<"$urls" || grep -- '-ubuntu2404\.tar\.gz$' <<<"$urls" || true)
    [ -n "$url" ] || { echo "no Ubuntu archive in release $RELEASE ($api)"; exit 1; }
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    echo "fetching $url"
    curl -fSL --retry 3 -o "$tmp/driver.tar.gz" "$url"
    mkdir "$tmp/debs"
    tar --no-same-owner -xzf "$tmp/driver.tar.gz" -C "$tmp/debs"
    mkdir -p "$DIR"
    n=0
    while IFS= read -r deb; do          # the libraries only: no firmware, no debug symbols
        dpkg-deb -x "$deb" "$DIR"
        echo "unpacked $(basename "$deb")"
        n=$((n + 1))
    done < <(find "$tmp/debs" -name 'intel-level-zero-npu_*.deb' -o -name 'intel-driver-compiler-npu_*.deb')
    [ "$n" = 2 ] || { echo "expected 2 .debs in the archive, found $n"; exit 1; }
    echo "$RELEASE" >"$DIR/VERSION"
fi

for need in libze_loader.so.1:libze1 libtbb.so.12:libtbb12; do
    # not grep -q: it would close the pipe early and pipefail would report ldconfig's SIGPIPE
    ldconfig -p | grep "${need%%:*}" >/dev/null || [ -e "$lib/${need%%:*}" ] ||
        echo "!! ${need%%:*} is missing: sudo apt install ${need##*:}"
done
[ -r /dev/accel/accel0 ] && [ -w /dev/accel/accel0 ] ||
    echo "!! /dev/accel/accel0 is not accessible: sudo usermod -aG render \$USER (then log in again)"

if [ "$DIR" != "$env/npu-stack" ]; then
    echo "not the default place: export SO101_NPU_STACK=$(realpath "$DIR")"
    export SO101_NPU_STACK=$(realpath "$DIR")
fi
"${run[@]}" so101-smolvla-edge doctor | sed -n '/^openvino/,/^cv2/p' | grep -v '^cv2'
