#!/usr/bin/env bash
# Download the ONNX Runtime C/C++ SDK (headers and libonnxruntime) that the C++ runtime builds against.
#
#   bash cpp/scripts/fetch_onnxruntime.sh [--version 1.31.0] [--flavor cpu|cuda12|cuda13] [--source auto|github|nuget]
#                                         [--dest DIR]
#
# The default destination, cpp/third_party/onnxruntime, is where cpp/CMakeLists.txt looks when ONNXRUNTIME_ROOT
# is not set. The official release archive comes from GitHub. For the CPU flavor the NuGet package
# Microsoft.ML.OnnxRuntime carries the same library, so --source auto falls back to it where GitHub release
# downloads are blocked. The cuda flavors also need CUDA and cuDNN of the matching major version at run time.
set -euo pipefail

VERSION=1.31.0
FLAVOR=cpu
SOURCE=auto
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/onnxruntime"
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION=$2; shift 2 ;;
    --flavor) FLAVOR=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --dest) DEST=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$(uname -m)" in
  x86_64) ARCH=x64; NUGET_RID=linux-x64 ;;
  aarch64) ARCH=aarch64; NUGET_RID=linux-arm64 ;;
  *) echo "no ONNX Runtime release for $(uname -m)" >&2; exit 1 ;;
esac
case "$FLAVOR" in
  cpu) NAME="onnxruntime-linux-$ARCH-$VERSION" ;;
  cuda12 | cuda13)
    if [ "$ARCH" != x64 ]; then
      echo "GPU archives are x86_64 only; on Jetson, use the ONNX Runtime build that matches JetPack" >&2
      exit 1
    fi
    NAME="onnxruntime-linux-x64-gpu_$FLAVOR-$VERSION" ;;
  *) echo "unknown flavor $FLAVOR: choose cpu, cuda12 or cuda13" >&2; exit 2 ;;
esac
case "$SOURCE" in
  auto | github | nuget) ;;
  *) echo "unknown source $SOURCE: choose auto, github or nuget" >&2; exit 2 ;;
esac
if [ "$SOURCE" = nuget ] && [ "$FLAVOR" != cpu ]; then
  echo "--source nuget provides the cpu flavor only" >&2
  exit 2
fi

# Whether $1 holds what cpp/CMakeLists.txt builds against: the C++ API header and the shared library.
is_sdk() { [ -f "$1/include/onnxruntime_cxx_api.h" ] && [ -e "$1/lib/libonnxruntime.so" ]; }

if is_sdk "$DEST" && [ "$(cat "$DEST/VERSION_NUMBER" 2>/dev/null)" = "$VERSION" ] &&
  [ "$(cat "$DEST/FLAVOR" 2>/dev/null)" = "$FLAVOR" ]; then
  echo "$DEST already holds ONNX Runtime $VERSION ($FLAVOR)"
  exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
GITHUB_URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$NAME.tgz"
NUGET_URL="https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$VERSION/microsoft.ml.onnxruntime.$VERSION.nupkg"

# Both run as conditions, where set -e does not stop them, so each step returns on failure itself. Each leaves a
# complete SDK in $STAGE or fails.
from_github() {
  curl -fsSL --retry 3 -o "$TMP/ort.tgz" "$GITHUB_URL" || return 1
  tar -xzf "$TMP/ort.tgz" -C "$TMP" || return 1
  STAGE="$TMP/$NAME"
  is_sdk "$STAGE" || { echo "$GITHUB_URL did not unpack to an ONNX Runtime SDK" >&2; return 1; }
}

# Rearranges the package into the release archive's layout: include/, lib/, VERSION_NUMBER.
from_nuget() {
  local tool
  for tool in unzip objdump; do
    command -v "$tool" >/dev/null || { echo "$tool is needed to unpack the NuGet package" >&2; return 1; }
  done
  curl -fsSL --retry 3 -o "$TMP/ort.nupkg" "$NUGET_URL" || return 1
  unzip -q "$TMP/ort.nupkg" 'build/native/include/*' "runtimes/$NUGET_RID/native/*" -d "$TMP/nuget" || return 1
  STAGE="$TMP/stage"
  mkdir -p "$STAGE/include" "$STAGE/lib" || return 1
  cp -R "$TMP/nuget/build/native/include/." "$STAGE/include/" || return 1
  cp "$TMP/nuget/runtimes/$NUGET_RID/native/"*.so* "$STAGE/lib/" || return 1
  # Executables record the library's SONAME (libonnxruntime.so.1), which the package does not ship as a file.
  local soname
  soname=$(objdump -p "$STAGE/lib/libonnxruntime.so" | awk '$1 == "SONAME" {print $2}') || return 1
  if [ -n "$soname" ] && [ ! -e "$STAGE/lib/$soname" ]; then
    ln -s libonnxruntime.so "$STAGE/lib/$soname" || return 1
  fi
  echo "$VERSION" >"$STAGE/VERSION_NUMBER" || return 1
  is_sdk "$STAGE" || { echo "$NUGET_URL did not unpack to an ONNX Runtime SDK" >&2; return 1; }
}

STAGE=
if [ "$SOURCE" != nuget ] && from_github; then
  echo "downloaded $GITHUB_URL"
elif [ "$SOURCE" = github ] || [ "$FLAVOR" != cpu ]; then
  echo "could not download and unpack $GITHUB_URL" >&2
  exit 1
else
  [ "$SOURCE" = auto ] && echo "could not use the GitHub release archive; trying NuGet" >&2
  from_nuget || { echo "could not download and unpack $NUGET_URL" >&2; exit 1; }
  echo "downloaded $NUGET_URL"
fi

echo "$FLAVOR" >"$STAGE/FLAVOR"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$STAGE" "$DEST"
echo "ONNX Runtime $VERSION ($FLAVOR) is in $DEST"
echo "cmake picks it up from cpp/third_party/onnxruntime, or pass -DONNXRUNTIME_ROOT=$DEST"
