#!/usr/bin/env python3
"""
Convert a Piper-exported `.onnx` (paired with its `.onnx.json` config) into
the sherpa-onnx-compatible format used by SherpaTTS on Android.

This is a minimally-modified port of the official sherpa-onnx helper at
https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/piper/add_meta_data.py
— same logic, but parameterised so you can point at any directory rathe
than relying on the cwd.

What it does, in order:

1. Reads `<lang>-<name>-<kind>.onnx.json` (Piper's config, copied next to
   the exported `.onnx`).
2. Generates `tokens.txt` from the config's `phoneme_id_map` — this is the
   vocabulary file SherpaTTS / sherpa-onnx need at inference time.
3. Embeds a small block of metadata directly into the `.onnx` (model type,
   language, voice, sample rate, espeak flag, etc.) so sherpa-onnx can pick
   the right phonemizer at load time.

After this script runs, the `.onnx` plus `tokens.txt` in `--out-dir` are
exactly the files SherpaTTS expects to find unde
`sdcard/Android/data/org.woheller69.ttsengine/files/modelDir/`.
"""

import argparse
import json
import sys
from pathlib import Path

import onnx
from iso639 import Lang


def add_meta_data(filename: Path, meta_data: dict) -> None:
    """Embed key/value metadata into the ONNX model file in place."""
    model = onnx.load(str(filename))
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, str(filename))


def generate_tokens(config: dict, tokens_path: Path) -> None:
    id_map = config["phoneme_id_map"]
    with tokens_path.open("w", encoding="utf-8") as f:
        for s, i in id_map.items():
            if s == "\n":
                continue
            if isinstance(i, list):
                i = i[0]
            f.write(f"{s} {i}\n")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory containing the exported .onnx and .onnx.json.")
    parser.add_argument("--name", required=True,
                        help="Voice name (e.g. 'merve').")
    parser.add_argument("--kind", required=True,
                        help="Voice quality tier (e.g. 'medium').")
    parser.add_argument("--lang", required=True,
                        help="Language tag, locale-style (e.g. 'tr_TR').")
    args = parser.parse_args()

    base = args.out_dir / f"{args.lang}-{args.name}-{args.kind}"
    onnx_path = base.with_suffix(".onnx")
    cfg_path = base.with_suffix(".onnx.json")

    if not onnx_path.exists():
        print(f"ERROR: missing {onnx_path}", file=sys.stderr)
        return 1
    if not cfg_path.exists():
        print(f"ERROR: missing {cfg_path}", file=sys.stderr)
        return 1

    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    tokens_path = args.out_dir / "tokens.txt"
    generate_tokens(config, tokens_path)
    print(f"wrote {tokens_path}")

    sample_rate = config["audio"]["sample_rate"]
    if sample_rate == 22500:
        sample_rate = 22050

    if "lang_code" in config:
        voice = config["lang_code"]
    else:
        voice = config["espeak"]["voice"]

    has_g2pw = 0
    has_espeak = 1
    if config.get("phoneme_type") == "pinyin" and voice == "zh":
        has_espeak = 0
        has_g2pw = 1

    lang_iso = Lang(args.lang.split("_")[0])
    meta_data = {
        "model_type": "vits",
        "comment": "piper",
        "language": lang_iso.name,
        "voice": voice,
        "version": 1,
        "has_espeak": has_espeak,
        "has_g2pw": has_g2pw,
        "n_speakers": config["num_speakers"],
        "sample_rate": sample_rate,
    }
    add_meta_data(onnx_path, meta_data)
    print(f"embedded meta into {onnx_path}: {meta_data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
