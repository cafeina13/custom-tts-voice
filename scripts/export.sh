#!/usr/bin/env bash
# Export a Piper training checkpoint (.ckpt) into the two-file bundle
# SherpaTTS expects on Android: a metadata-embedded model.onnx + tokens.txt.
#
# Two steps:
#   1. `python -m piper.train.export_onnx`  →  raw Piper-format .onnx
#   2. `scripts/sherpa_add_meta.py`         →  embed sherpa-onnx metadata
#                                              and emit tokens.txt
#
# Override via env vars / args:
#   $1                 -> checkpoint .ckpt to export   (REQUIRED)
#   TTS_VENV           -> Piper venv     (~/TTS/piper1-gpl/.venv)
#   TTS_CHECKPOINT_DIR -> dir containing the base config.json from training
#                        (~/TTS/checkpoints/tr_TR-dfki-medium)
#   TTS_DEPLOY_ROOT    -> root for exported voices    (~/TTS/deploy)
#   TTS_LANG_TAG       -> locale-style language tag for filename
#                        (tr_TR)
#   TTS_VOICE_NAME     -> voice name in the filename  (merve)
#   TTS_VOICE_KIND     -> voice quality tier          (medium)
#   TTS_LANG_3         -> 3-letter language code for SherpaTTS `lang` file
#                        (tur)
#   TTS_DISPLAY_NAME   -> display name in SherpaTTS `lang` file
#                        (Merve)
#
# Output goes to $TTS_DEPLOY_ROOT/<tag>/, containing:
#   <lang>-<name>-<kind>.onnx   (with metadata embedded)
#   <lang>-<name>-<kind>.onnx.json
#   tokens.txt
#   lang                        (line 1: 3-letter code, line 2: display name)
#
# The <tag> subdir is derived from the checkpoint filename so multiple
# checkpoints exported from the same voice don't collide.
set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/checkpoint.ckpt>" >&2
  exit 1
fi
CKPT="$1"
if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TTS_VENV="${TTS_VENV:-$HOME/TTS/piper1-gpl/.venv}"
TTS_CHECKPOINT_DIR="${TTS_CHECKPOINT_DIR:-$HOME/TTS/checkpoints/tr_TR-dfki-medium}"
TTS_DEPLOY_ROOT="${TTS_DEPLOY_ROOT:-$HOME/TTS/deploy}"
TTS_LANG_TAG="${TTS_LANG_TAG:-tr_TR}"
TTS_VOICE_NAME="${TTS_VOICE_NAME:-merve}"
TTS_VOICE_KIND="${TTS_VOICE_KIND:-medium}"
TTS_LANG_3="${TTS_LANG_3:-tur}"
TTS_DISPLAY_NAME="${TTS_DISPLAY_NAME:-Merve}"

# Subdir = ckpt basename stripped of .ckpt. Keeps 15k and snapshot separate.
TAG="$(basename "$CKPT" .ckpt)"
OUT_DIR="$TTS_DEPLOY_ROOT/$TAG"
ONNX_BASE="${TTS_LANG_TAG}-${TTS_VOICE_NAME}-${TTS_VOICE_KIND}"

# PyTorch CUDA 13 libs first (same fix as train.sh).
SITE_PACKAGES="$(find "$TTS_VENV/lib" -maxdepth 2 -name 'site-packages' -type d | head -1)"
export LD_LIBRARY_PATH="\
$SITE_PACKAGES/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck disable=SC1090,SC1091
source "$TTS_VENV/bin/activate"

mkdir -p "$OUT_DIR"

echo "=== Step 1: Piper .ckpt → .onnx ==="
# Use export_runner.py instead of `python -m piper.train.export_onnx` so
# torch.onnx.export gets dynamo=False forced; the new dynamo exporte
# can't handle VITS's data-dependent guards in rational_quadratic_spline.
python3 -u "$SCRIPT_DIR/export_runner.py" \
  --checkpoint "$CKPT" \
  --output-file "$OUT_DIR/$ONNX_BASE.onnx"

echo "=== Step 2: copy training config next to the .onnx ==="
cp "$TTS_CHECKPOINT_DIR/config.json" "$OUT_DIR/$ONNX_BASE.onnx.json"

echo "=== Step 3: sherpa-onnx metadata + tokens.txt ==="
python3 -u "$SCRIPT_DIR/sherpa_add_meta.py" \
  --out-dir "$OUT_DIR" \
  --name "$TTS_VOICE_NAME" \
  --kind "$TTS_VOICE_KIND" \
  --lang "$TTS_LANG_TAG"

echo "=== Step 4: write SherpaTTS lang file ==="
{
  echo "$TTS_LANG_3"
  echo "$TTS_DISPLAY_NAME"
} > "$OUT_DIR/lang"

echo "=== Done. Files in $OUT_DIR: ==="
ls -lh "$OUT_DIR"
