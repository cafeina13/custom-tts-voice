#!/usr/bin/env bash
# Launcher for `piper.train fit` (fine-tune a Piper voice).
#
# - Activates the Piper venv from $TTS_VENV.
# - Sets LD_LIBRARY_PATH so any CUDA libs the venv ships are picked up
#   (matches what run_pipeline.sh does).
# - Defaults are tuned for an RTX 4070 Laptop (8 GB VRAM) doing a
#   *fine-tune* (not from-scratch) of a 22 kHz "medium" Piper voice.
# - Outputs go under $TTS_OUTPUT_DIR. PyTorch Lightning writes checkpoints
#   to <output_dir>/lightning_logs/version_*/checkpoints/.
#
# Override via env vars:
#   TTS_VENV            -> path to Piper venv     (~/TTS/piper1-gpl/.venv)
#   TTS_DATA_ROOT       -> dataset root           (~/TTS/data)
#   TTS_DATASET         -> dataset subdir         (merve)
#   TTS_LANGUAGE        -> espeak voice code      (tr)
#   TTS_VOICE_NAME      -> name for the trained voice
#                                                 (tr_TR-${TTS_DATASET}-medium)
#   TTS_CHECKPOINT_DIR  -> dir with base.ckpt + config.json to fine-tune from
#                                                 (~/TTS/checkpoints/tr_TR-dfki-medium)
#   TTS_OUTPUT_DIR      -> where Lightning writes logs + checkpoints
#                                                 (~/TTS/output/$TTS_VOICE_NAME)
#   TTS_BATCH_SIZE      -> training batch size    (12 — safe for 8 GB VRAM
#                                                  with mixed precision)
#   TTS_MAX_STEPS       -> stop after N optimizer steps      (10000)
#   TTS_NUM_WORKERS     -> dataloader workers     (4)
#   TTS_PRECISION       -> Lightning precision    (16-mixed)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TTS_VENV="${TTS_VENV:-$HOME/TTS/piper1-gpl/.venv}"
TTS_DATA_ROOT="${TTS_DATA_ROOT:-$HOME/TTS/data}"
TTS_DATASET="${TTS_DATASET:-merve}"
TTS_LANGUAGE="${TTS_LANGUAGE:-tr}"
TTS_VOICE_NAME="${TTS_VOICE_NAME:-tr_TR-${TTS_DATASET}-medium}"
TTS_CHECKPOINT_DIR="${TTS_CHECKPOINT_DIR:-$HOME/TTS/checkpoints/tr_TR-dfki-medium}"
TTS_OUTPUT_DIR="${TTS_OUTPUT_DIR:-$HOME/TTS/output/$TTS_VOICE_NAME}"
TTS_BATCH_SIZE="${TTS_BATCH_SIZE:-12}"
TTS_MAX_STEPS="${TTS_MAX_STEPS:-10000}"
TTS_NUM_WORKERS="${TTS_NUM_WORKERS:-4}"
TTS_PRECISION="${TTS_PRECISION:-16-mixed}"

DATASET_DIR="$TTS_DATA_ROOT/$TTS_DATASET"

# Sanity-check the prerequisites before doing anything expensive.
if [[ ! -f "$DATASET_DIR/metadata.csv" ]]; then
  echo "ERROR: dataset metadata not found at $DATASET_DIR/metadata.csv" >&2
  echo "Run scripts/run_pipeline.sh first." >&2
  exit 1
fi
if [[ ! -f "$TTS_CHECKPOINT_DIR/base.ckpt" ]] || [[ ! -f "$TTS_CHECKPOINT_DIR/config.json" ]]; then
  echo "ERROR: base checkpoint or config missing under $TTS_CHECKPOINT_DIR" >&2
  echo "Need: base.ckpt + config.json (download from rhasspy/piper-checkpoints on HF)." >&2
  exit 1
fi

# Pick up CUDA libs from inside the venv site-packages.
#
# PyTorch installed via pip ships CUDA 13 runtime libs under nvidia/cu13/lib.
# When run_pipeline.sh ran earlier it put nvidia/cuda_nvrtc/lib (a *CUDA 12*
# nvrtc shipped for ctranslate2/faster-whisper) on LD_LIBRARY_PATH. That cu12
# directory does not contain libnvrtc-builtins.so.13.0, so a stale
# LD_LIBRARY_PATH ordering can shadow PyTorch's cu13 lookup and crash JIT
# kernel compilation. Put cu13 first; keep cu12 dirs after for completeness.
SITE_PACKAGES="$(find "$TTS_VENV/lib" -maxdepth 2 -name 'site-packages' -type d | head -1)"
if [[ -z "$SITE_PACKAGES" ]]; then
  echo "ERROR: could not find site-packages under $TTS_VENV/lib" >&2
  exit 1
fi
export LD_LIBRARY_PATH="\
$SITE_PACKAGES/nvidia/cu13/lib:\
$SITE_PACKAGES/nvidia/cublas/lib:\
$SITE_PACKAGES/nvidia/cudnn/lib:\
$SITE_PACKAGES/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck disable=SC1090,SC1091
source "$TTS_VENV/bin/activate"

mkdir -p "$TTS_OUTPUT_DIR" "$DATASET_DIR/cache"
LOG="$TTS_OUTPUT_DIR/train.log"

# Default: start from the base checkpoint. To resume an in-progress training
# instead, set TTS_CKPT_PATH to the saved Lightning checkpoint, e.g.:
#   TTS_CKPT_PATH=~/TTS/output/tr_TR-merve-medium/lightning_logs/version_0/checkpoints/last.ckpt \
#     bash scripts/train.sh
# Lightning's --ckpt_path restores both weights AND trainer state (current
# step, optimizer, scheduler), so TTS_MAX_STEPS counts toward the same
# global target across sessions.
CKPT_PATH="${TTS_CKPT_PATH:-$TTS_CHECKPOINT_DIR/base.ckpt}"

{
  echo "=========================="
  echo "Piper fine-tune launch"
  date
  echo "VOICE_NAME    : $TTS_VOICE_NAME"
  echo "DATASET_DIR   : $DATASET_DIR"
  echo "CKPT_PATH     : $CKPT_PATH"
  echo "OUTPUT_DIR    : $TTS_OUTPUT_DIR"
  echo "BATCH_SIZE    : $TTS_BATCH_SIZE"
  echo "MAX_STEPS     : $TTS_MAX_STEPS  (global total, not per-session)"
  echo "NUM_WORKERS   : $TTS_NUM_WORKERS"
  echo "PRECISION     : $TTS_PRECISION"
  echo "=========================="
} | tee "$LOG"

python3 -u "$SCRIPT_DIR/train_runner.py" fit \
  --data.voice_name "$TTS_VOICE_NAME" \
  --data.csv_path "$DATASET_DIR/metadata.csv" \
  --data.audio_dir "$DATASET_DIR/wavs/" \
  --data.espeak_voice "$TTS_LANGUAGE" \
  --data.cache_dir "$DATASET_DIR/cache/" \
  --data.config_path "$TTS_CHECKPOINT_DIR/config.json" \
  --data.batch_size "$TTS_BATCH_SIZE" \
  --data.num_workers "$TTS_NUM_WORKERS" \
  --model.sample_rate 22050 \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.precision "$TTS_PRECISION" \
  --trainer.max_steps "$TTS_MAX_STEPS" \
  --trainer.default_root_dir "$TTS_OUTPUT_DIR" \
  --ckpt_path "$TTS_CHECKPOINT_DIR/base.ckpt" \
  2>&1 | tee -a "$LOG"
