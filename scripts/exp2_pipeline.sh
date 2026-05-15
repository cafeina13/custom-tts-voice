#!/usr/bin/env bash
set -e
echo "[exp2] START $(date -Is)"
echo "[exp2] === re-cut dataset (200ms pad + foreign-word filter) ==="
bash /mnt/c/Users/ZERO/Documents/GitHub/TTS/scripts/run_pipeline.sh
echo "[exp2] END pipeline $(date -Is)"
