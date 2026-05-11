#!/usr/bin/env bash
# Launcher for build_dataset.py inside WSL / Linux.
#
# - Self-locates build_dataset.py next to this script (works wherever the repo is checked out).
# - Activates the Piper training venv from $TTS_VENV (default: ~/TTS/piper1-gpl/.venv).
# - Sets LD_LIBRARY_PATH so faster-whisper / CTranslate2 finds CUDA 12 libs
#   (PyTorch installs CUDA 13, but the prebuilt ctranslate2 wheel is built against CUDA 12).
# - Logs to $TTS_DATA_ROOT/$TTS_DATASET/pipeline.log so progress can be tailed.
#
# Override env vars to point at a different dataset / venv:
#   TTS_VENV       -> path to Piper venv         (default: ~/TTS/piper1-gpl/.venv)
#   TTS_DATA_ROOT  -> data root                  (default: ~/TTS/data)
#   TTS_DATASET    -> dataset subdir name        (default: merve)
#   TTS_LANGUAGE   -> Whisper language code      (default: tr)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TTS_VENV="${TTS_VENV:-$HOME/TTS/piper1-gpl/.venv}"
TTS_DATA_ROOT="${TTS_DATA_ROOT:-$HOME/TTS/data}"
TTS_DATASET="${TTS_DATASET:-merve}"
TTS_LANGUAGE="${TTS_LANGUAGE:-tr}"
export TTS_DATA_ROOT TTS_DATASET TTS_LANGUAGE

# Pick up CUDA 12 libs from inside the venv site-packages.
SITE_PACKAGES="$(find "$TTS_VENV/lib" -maxdepth 2 -name 'site-packages' -type d | head -1)"
if [[ -z "$SITE_PACKAGES" ]]; then
  echo "ERROR: could not find site-packages under $TTS_VENV/lib" >&2
  exit 1
fi
export LD_LIBRARY_PATH="\
$SITE_PACKAGES/nvidia/cublas/lib:\
$SITE_PACKAGES/nvidia/cudnn/lib:\
$SITE_PACKAGES/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck disable=SC1090,SC1091
source "$TTS_VENV/bin/activate"

LOG="$TTS_DATA_ROOT/$TTS_DATASET/pipeline.log"
mkdir -p "$(dirname "$LOG")"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee "$LOG"
echo "TTS_DATA_ROOT=$TTS_DATA_ROOT  TTS_DATASET=$TTS_DATASET  TTS_LANGUAGE=$TTS_LANGUAGE" | tee -a "$LOG"
python3 -u "$SCRIPT_DIR/build_dataset.py" 2>&1 | tee -a "$LOG"
