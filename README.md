# Custom Turkish TTS Voices

An end-to-end pipeline that fine-tunes custom Turkish [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voices on PC and deploys them to Android phones. The trained voices are standard `.onnx` files — usable in Piper itself on desktop, anywhere [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) is, or in the [SherpaTTS](https://github.com/woheller69/ttsengine) Android app where they become available to the system "Read Aloud" accessibility feature.

The pipeline is voice/engine-agnostic — Turkish is just the first target because publicly available Turkish Piper voices are scarce (only `tr_TR-dfki-medium` ships officially).

**Status:** all stages working. Dataset prep, fine-tune, ONNX export, and on-device deployment are all implemented. Two experiments completed; quality tuning (checkpoint-ladder A/B testing) ongoing.

## Why this exists

The stock voices bundled with most TTS engines are limited and get repetitive fast — and for languages other than English the selection is genuinely thin. Modern zero-shot voice-cloning models (one short reference clip → any voice) exist but are too heavy to run on a phone, so the realistic path is: **fine-tune** a small TTS model (Piper / VITS) on PC with a few hours of single-speaker audio, export it to `.onnx`, and load it in whichever runtime you need.

Most individual pieces of this chain are documented somewhere. Stitching them together — especially for non-English Piper voices going to Android — is what this repo provides.

## Pipeline

```
YouTube source (one consistent narrator, no music, no guests)
        │  yt-dlp (original-language audio track only — avoid auto-dub)
        ▼
Raw WAV episodes
        │  faster-whisper large-v3 + Silero VAD
        │   - transcribe + skip silence/music
        │   - pad ±200 ms around each segment for natural attack/release
        ▼
Per-segment text + timestamps
        │  filter: duration 1.5–15 s, no-speech-prob < 0.5, non-empty text
        │  optional: drop segments containing foreign-language tokens
        │  librosa: resample 22050 Hz mono
        ▼
LJSpeech-style dataset (metadata.csv + 22 kHz WAV chunks)
        │  piper.train fit --ckpt_path base.ckpt
        │   - save a checkpoint every 1k steps for A/B selection
        ▼
Trained Piper voice (.ckpt)
        │  sherpa-onnx Piper conversion + tokens/lang metadata
        ▼
model.onnx + tokens.txt + lang
        │  ADB / "Install from SD Card" in SherpaTTS
        ▼
Voice available in SherpaTTS → used by Android Read Aloud
```

## Tech stack

- **TTS architecture:** [VITS](https://arxiv.org/abs/2106.06103) (end-to-end neural TTS) via [Piper](https://github.com/OHF-Voice/piper1-gpl)
- **Phonemization:** [espeak-ng](https://github.com/espeak-ng/espeak-ng) (text → IPA), embedded in Piper via a C bridge
- **Speech recognition for dataset prep:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running Whisper large-v3 with Silero VAD
- **Audio download:** [yt-dlp](https://github.com/yt-dlp/yt-dlp) with explicit original-language track selection
- **Audio processing:** librosa + soundfile
- **Foreign-word detection:** [hunspell](http://hunspell.github.io/) + a language dictionary (e.g. `hunspell-tr`)
- **Training:** PyTorch + PyTorch Lightning (everything Piper ships with)
- **Inference on phone:** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via [SherpaTTS](https://github.com/woheller69/ttsengine)
- **Host environment:** WSL2 Ubuntu 24.04 on Windows 11 with an NVIDIA GPU. Plain Linux works too.

## Repo layout

```
.
├── scripts/
│   ├── run_pipeline.sh         # Dataset stage: venv + env + LD_LIBRARY_PATH, calls build_dataset.py
│   ├── build_dataset.py        # Whisper transcribe + VAD split + padding + filter + resample → LJSpeech CSV
│   ├── scan_foreign_words.py   # Optional: build drop/review token lists from transcripts
│   ├── train.sh                # Training stage: launches piper.train fit via train_runner.py
│   ├── train_runner.py         # Thin shim over piper.train: filters legacy hparams, injects per-1k-step checkpoint callback
│   ├── export.sh               # Export stage: .ckpt → sherpa-onnx model.onnx + tokens.txt + lang
│   ├── export_runner.py        # Python side of the export
│   └── sherpa_add_meta.py      # Adds sherpa-onnx required metadata to the exported ONNX
└── .gitignore                  # Excludes datasets, checkpoints, audio, ONNX models
```

What's intentionally **not** in this repo:

- Training audio (sourced from YouTube; not redistributable).
- Trained voice models. Distributing a model trained on a real person's voice without their explicit consent is ethically and legally fraught — most jurisdictions treat voice as protected (personality rights / GDPR / similar). If you train your own voice for your own phone, that's your business. Republishing the model is a different question.
- The Piper base checkpoint (downloaded from Hugging Face during setup).

## Working layout outside this repo

The scripts run inside WSL2 / Linux. They expect this layout on the **Linux** filesystem (NOT NTFS / `/mnt/c/...` — performance, and Whisper hates Windows path handling):

```
~/TTS/
├── piper1-gpl/                              # cloned Piper training repo (with .venv)
├── checkpoints/<lang>-<name>/               # base checkpoint to fine-tune from
│   ├── config.json
│   └── base.ckpt
├── output/<voice_name>/                     # Lightning writes logs + ckpts here during training
└── data/<dataset>/                          # one folder per voice you're training
    ├── raw/         <video_id>.wav          # full episodes you downloaded
    ├── transcripts/ <video_id>.json         # Whisper output (cached — expensive to recompute)
    ├── wavs/        <video_id>_NNNN.wav     # 22050 Hz mono utterance chunks
    ├── metadata.csv                         #  "<filename>|<transcript>" per line
    ├── auto_drop_tokens.txt                 # optional foreign-word filter input
    ├── review_tokens.txt                    # optional foreign-word filter input (after human review)
    └── pipeline.log
```

Scripts are parameterized via environment variables — defaults match the layout above:

| Variable                  | Default                              | Meaning                                       |
|---------------------------|--------------------------------------|-----------------------------------------------|
| `TTS_VENV`                | `~/TTS/piper1-gpl/.venv`             | Path to the Piper Python virtualenv           |
| `TTS_DATA_ROOT`           | `~/TTS/data`                         | Parent dir for all per-voice datasets         |
| `TTS_DATASET`             | `merve`                              | Subdir under `$TTS_DATA_ROOT` to use          |
| `TTS_LANGUAGE`            | `tr`                                 | Whisper / espeak language code                |
| `TTS_VOICE_NAME`          | `tr_TR-${TTS_DATASET}-medium`        | Name used in checkpoint / export filenames    |
| `TTS_CHECKPOINT_DIR`      | `~/TTS/checkpoints/tr_TR-dfki-medium`| Dir with `base.ckpt` + `config.json`          |
| `TTS_OUTPUT_DIR`          | `~/TTS/output/$TTS_VOICE_NAME`       | Where Lightning writes logs + checkpoints     |
| `TTS_MAX_STEPS`           | `10000`                              | Stop after N optimizer steps (global total)   |
| `TTS_BATCH_SIZE`          | `12`                                 | Safe for ~8 GB VRAM with fp16                 |
| `TTS_NUM_WORKERS`         | `4`                                  | Dataloader workers                            |
| `TTS_PRECISION`           | `16-mixed`                           | Lightning precision                           |
| `TTS_CKPT_EVERY_N_STEPS`  | `1000`                               | Snapshot interval — feeds the A/B ladder      |
| `TTS_CKPT_KEEP_ALL`       | `1`                                  | Keep every snapshot, not just the best        |
| `TTS_CKPT_PATH`           | `$TTS_CHECKPOINT_DIR/base.ckpt`      | Override to resume from a saved Lightning ckpt|
| `TTS_SEGMENT_PADDING_S`   | `0.2`                                | Padding around Whisper VAD segments (seconds) |

## Setup (one-time, inside WSL2 Ubuntu or plain Linux)

```bash
# System packages
sudo apt-get install -y build-essential cmake ninja-build python3-venv \
                        ffmpeg sox hunspell hunspell-tr

# Clone Piper training repo and create venv
mkdir -p ~/TTS && cd ~/TTS
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train]'
pip install scikit-build scikit-build-core
python3 setup.py build_ext --inplace      # builds embedded espeak-ng + espeakbridge
./build_monotonic_align.sh                # Cython kernel from the VITS paper

# Dataset prep tooling
pip install yt-dlp faster-whisper

# faster-whisper / CTranslate2 needs CUDA 12 libs even if PyTorch installs CUDA 13
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

# Sherpa export (for the export stage)
pip install sherpa-onnx onnx
```

Grab the base Turkish Piper checkpoint:

```bash
mkdir -p ~/TTS/checkpoints/tr_TR-dfki-medium && cd ~/TTS/checkpoints/tr_TR-dfki-medium
# Download base.ckpt + config.json from rhasspy/piper-checkpoints on Hugging Face
# (tr_TR-dfki-medium is currently the only official Turkish Piper voice)
```

## Stage 1 — dataset prep

1. Pick a `TTS_DATASET` name.
2. Put single-speaker audio you have rights to use into `~/TTS/data/<TTS_DATASET>/raw/`. Example with yt-dlp:
   ```bash
   mkdir -p ~/TTS/data/<TTS_DATASET>/raw && cd ~/TTS/data/<TTS_DATASET>/raw
   yt-dlp -f 'ba[language=tr]/ba' --extract-audio --audio-format wav \
          -o '%(id)s.%(ext)s' <URL1> <URL2> ...
   ```
   The `language=tr` filter matters — YouTube auto-dub creates AI-translated audio tracks you almost certainly don't want to train on. Check available tracks with `yt-dlp -F <URL>` first.
3. (Optional but recommended for code-switched content) scan for foreign words and build the filter lists:
   ```bash
   python3 scripts/scan_foreign_words.py
   # Edit ~/TTS/data/<TTS_DATASET>/review_tokens.txt
   # (delete lines for words you want to KEEP — i.e. legit native words flagged as false positives)
   ```
4. Run the dataset stage:
   ```bash
   bash scripts/run_pipeline.sh
   # or with overrides
   TTS_DATASET=other_voice TTS_LANGUAGE=en bash scripts/run_pipeline.sh
   ```
5. Tail progress:
   ```bash
   tail -f ~/TTS/data/<TTS_DATASET>/pipeline.log
   ```

Output: `metadata.csv` + a `wavs/` directory of 22050 Hz mono chunks, ready to feed `piper.train`.

Transcripts are cached per video in `transcripts/<video_id>.json` — re-cutting the dataset (different padding / different filter) skips Whisper and just re-segments. Costs seconds instead of an hour.

## Stage 2 — training

```bash
bash scripts/train.sh
# or with overrides
TTS_MAX_STEPS=10000 bash scripts/train.sh
```

This launches `piper.train fit` via a small shim (`train_runner.py`) that:

- Filters out hparam keys from the base checkpoint that current Piper doesn't recognize (older checkpoints carry extra fields that crash `jsonargparse`).
- Injects a `ModelCheckpoint` callback saving every `$TTS_CKPT_EVERY_N_STEPS` steps. By default this keeps **all** snapshots, not just the best — the point is to produce a quality ladder you can A/B test.

To resume a previous run from its `last.ckpt` (continues the global step counter, optimizer state, and scheduler — does not restart):

```bash
TTS_CKPT_PATH=~/TTS/output/<voice>/lightning_logs/version_0/checkpoints/last.ckpt \
TTS_MAX_STEPS=15000 \
bash scripts/train.sh
```

Watch training via TensorBoard:

```bash
source ~/TTS/piper1-gpl/.venv/bin/activate
tensorboard --logdir ~/TTS/output/<voice>/lightning_logs --port 6006
# → http://localhost:6006
```

## Stage 3 — export to sherpa-onnx

```bash
bash scripts/export.sh ~/TTS/output/<voice>/lightning_logs/version_0/checkpoints/step_10000.ckpt
```

Produces `model.onnx`, `tokens.txt`, and a `lang` text file in a fresh export dir. Loop over the saved `step_*.ckpt` files to export the whole ladder for on-device A/B testing.

## Stage 4 — deploy to Android

SherpaTTS has an "Install from SD Card" UI that picks up a directory containing the three exported files. Either:

- **Manual:** copy the export dir to `sdcard/Android/data/org.woheller69.ttsengine/files/modelDir/<voice_name>/` via USB / file manager, then in SherpaTTS pick it from the model list.
- **ADB:**
  ```bash
  adb push <export_dir> /sdcard/Android/data/org.woheller69.ttsengine/files/modelDir/
  # then restart SherpaTTS so it scans
  ```

After installing, set the system "Preferred TTS engine" to SherpaTTS in Android Settings → Accessibility → Text-to-speech output, and pick your voice in SherpaTTS.

## Picking the right checkpoint

VITS fine-tuning quality does not move monotonically with step count. Past a certain point the model starts to memorize the training set and the voice gets robotic / over-articulated. Symptom: training loss keeps dropping while validation loss flattens or rises.

`val_loss` in TensorBoard is the canary, but VITS has stochastic components in validation (latent sampling from a flow / posterior) so any single point has ±1–3 units of noise — watch the trend over multiple points, not one drop.

The most reliable test is your ears. Export several checkpoints (e.g. 5k, 7k, 9k, 10k), install them all on the phone, generate the same sentence with each, pick the one that sounds best, throw away the rest.

## Notes from the trenches

Things that bit and how they were fixed — collected so the next person doesn't have to learn them the same way.

- **YouTube auto-dub is AI-generated.** A channel can have multiple audio tracks (original + auto-dubbed). Without explicit format selection (`yt-dlp -f 'ba[language=tr]/ba'`), you may silently download the auto-dub and unknowingly clone an AI voice instead of the real narrator. Always check tracks with `yt-dlp -F`.
- **Audiobook narrators sound too oratorical for notification reading.** Documentary/essay-style narrators with conversational prosody work better.
- **Background music in the source survives training.** Vocal separation (Demucs/UVR) helps as a last resort, but a clean source produces strictly better results — the model otherwise picks up subtle separation artifacts and reproduces them in the trained voice.
- **Single-speaker discipline matters.** Any contamination (movie clips, guest interviews, ads) gets averaged in as inconsistency. Aggressively curate.
- **Whisper's VAD cuts segments too tightly by default.** Without padding, the first phoneme of each segment is partly clipped and the model never learns proper word onsets → trained voice "swallows" the start of words. Add ~200 ms padding on both sides of each Whisper segment (`TTS_SEGMENT_PADDING_S`).
- **Foreign words wreck phoneme/audio alignment.** A Turkish-language espeak phonemizer reads "Frankenstein" with Turkish letter-to-sound rules, but the speaker actually said it English-style. That phoneme-vs-audio mismatch injects noise into training and contributes to mumbling on otherwise-clean native words. `scan_foreign_words.py` flags non-native tokens via hunspell + Unicode category checks; review the produced `review_tokens.txt` to keep legit native words that were false-flagged.
- **Save every-1k checkpoints, not just the best.** Lightning's default `ModelCheckpoint` keeps one. You want the whole ladder so you can listen-test and find the perceptual sweet spot — which often disagrees with the lowest val_loss point.
- **`.onnx` is just a container.** Two `.onnx` TTS models from different architectures (Piper VITS, Supertonic, etc.) are not interchangeable — the consuming app has to expect specific input/output tensor shapes.
- **Cross-language fine-tuning is risky.** Phoneme sets differ between Piper voices for different languages, so starting from an English Piper checkpoint to train a Turkish voice (because the English base is "higher quality") usually produces a worse result than starting from the only-Turkish-medium base. Stick to same-language bases.
- **CUDA library mix.** Piper's PyTorch installs CUDA 13 runtime libs; `faster-whisper`'s CTranslate2 wheel needs CUDA 12. The launcher scripts set `LD_LIBRARY_PATH` so both find what they need — don't remove that or one of them stops working.
- **Stay on the Linux filesystem.** Running the pipeline against `/mnt/c/...` (the Windows NTFS mount) is dramatically slower for the many-small-files workload Whisper produces. Keep `~/TTS/` on the WSL ext4 filesystem.

## License

Code in this repo: to be decided (likely MIT or similar permissive). Audio data and trained models are not part of this repo and are subject to whatever rights the source audio carries plus the speaker's right to their own voice.
