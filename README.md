# Custom Turkish TTS Voices

A learning project: a pipeline that fine-tunes custom Turkish [Piper](https://github.com/OHF-Voice/piper1-gpl) text-to-speech voices on PC. The trained voices are `.onnx` files — usable anywhere a compatible TTS engine can load them: Piper itself on desktop, [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) bindings, or Android via the [SherpaTTS](https://github.com/woheller69/ttsengine) app.

The first deployment target is an Android phone, using the system "Read Aloud" accessibility feature so notifications are read in a custom voice instead of a stock one but the pipeline is voice/engine-agnostic.

**Status:** work in progress. Dataset pipeline implemented; fine-tuning + on-device deployment to follow.

## Why this project

The stock voices bundled with most TTS engines get repetitive fast. Modern zero-shot voice-cloning models (one short reference clip → any voice) exist but are too heavy to run on a phone, so the realistic path is: **fine-tune** a small TTS model (Piper / VITS) on PC, export it to `.onnx`, and load it in whichever runtime you need.

## Pipeline

```
YouTube source (one consistent narrator)
        │  yt-dlp (original-language audio track only)
        ▼
Raw WAV episodes
        │  faster-whisper large-v3 + Silero VAD  (transcribe + skip silence/music)
        ▼
Per-segment text + timestamps
        │  filter: duration 1.5–15 s, no-speech-prob < 0.5, non-empty text
        │  librosa: resample 22050 Hz mono
        ▼
LJSpeech-style dataset  (`metadata.csv` + 22 kHz WAV chunks)
        │  piper-train: fine-tune from `tr_TR-dfki-medium` checkpoint
        ▼
Trained Piper voice (`.ckpt`)
        │  sherpa-onnx Piper conversion script
        ▼
`model.onnx` + `tokens.txt`
        │  ADB push to `sdcard/Android/data/org.woheller69.ttsengine/files/modelDir/`
        ▼
Voice available in SherpaTTS → used by Android Read Aloud
```

## Tech stack

- **TTS architecture:** [VITS](https://arxiv.org/abs/2106.06103) (end-to-end neural TTS) via [Piper](https://github.com/OHF-Voice/piper1-gpl)
- **Phonemization:** [espeak-ng](https://github.com/espeak-ng/espeak-ng) (text → IPA phonemes), embedded in Piper via a C bridge
- **Speech recognition (for dataset prep):** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) running OpenAI Whisper large-v3, with Silero VAD for speech/silence segmentation
- **Audio download:** [yt-dlp](https://github.com/yt-dlp/yt-dlp) with explicit original-language track selection (avoids YouTube auto-dub)
- **Audio processing:** librosa + soundfile
- **Inference runtime (on phone):** [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) via [SherpaTTS](https://github.com/woheller69/ttsengine) Android app
- **Training environment:** WSL2 Ubuntu 24.04 on Windows 11, GPU, PyTorch + Lightning

## Repo layout

```
.
├── scripts/
│   ├── build_dataset.py     # Whisper transcribe + VAD split + resample → LJSpeech CSV
│   └── run_pipeline.sh      # Linux/WSL launcher: sets LD_LIBRARY_PATH, env vars, venv
└── .gitignore               # Excludes data, models, and other things that shouldn't be public
```

What's **not** in this repo:
- The training audio dataset (sourced from YouTube; not redistributable).
- The Piper base checkpoint (downloaded from Hugging Face during setup).
- The trained voice model.

## Working layout (outside this repo)

The scripts run inside WSL2/Linux. They expect this directory layout on the **Linux** filesystem (NOT NTFS / `/mnt/c/...`, for performance):

```
~/TTS/                                       # work root in WSL
├── piper1-gpl/                              # cloned Piper training repo (with .venv)
├── checkpoints/<lang>-<name>/               # pretrained base checkpoint to fine-tune from
│   ├── config.json
│   └── base.ckpt
└── data/<dataset_name>/                     # one folder per voice you're training
    ├── raw/         <video_id>.wav          # full episodes you downloaded (~48 kHz)
    ├── transcripts/ <video_id>.json         # Whisper output per episode (cached)
    ├── wavs/        <video_id>_NNNN.wav     # 22050 Hz mono utterance chunks
    ├── metadata.csv                         #  "<filename>|<transcript>" per line
    └── pipeline.log                         # tailable progress log
```

The scripts are parameterized via environment variables — defaults match the layout above:

| Variable        | Default                      | Meaning                                |
|-----------------|------------------------------|----------------------------------------|
| `TTS_VENV`      | `~/TTS/piper1-gpl/.venv`     | Path to the Piper Python virtualenv    |
| `TTS_DATA_ROOT` | `~/TTS/data`                 | Parent dir for all per-voice datasets  |
| `TTS_DATASET`   | `merve`                      | Subdir under `$TTS_DATA_ROOT` to use   |
| `TTS_LANGUAGE`  | `tr`                         | Whisper / espeak language code         |

## Setup (one-time, inside WSL2 Ubuntu or Linux)

```bash
# System packages
sudo apt-get install -y build-essential cmake ninja-build python3-venv ffmpeg sox

# Clone Piper training repo and create venv
mkdir -p ~/TTS && cd ~/TTS
git clone https://github.com/OHF-Voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train]'
pip install scikit-build scikit-build-core
python3 setup.py build_ext --inplace      # builds embedded espeak-ng + espeakbridge
./build_monotonic_align.sh                # Cython extension from the VITS paper

# Dataset prep tooling
pip install yt-dlp faster-whisper

# faster-whisper needs CUDA 12 libs (PyTorch installs CUDA 13)
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Running the dataset prep

1. Decide a `TTS_DATASET` name (anything, e.g. `my_set`).
2. Download single-speaker audio you have rights to use into `~/TTS/data/<TTS_DATASET>/raw/`. Example with yt-dlp (replace URLs):
   ```bash
   mkdir -p ~/TTS/data/<TTS_DATASET>/raw && cd ~/TTS/data/<TTS_DATASET>/raw
   yt-dlp -f 'ba[language=tr]/ba' --extract-audio --audio-format wav \
          -o '%(id)s.%(ext)s' <URL1> <URL2> ...
   ```
   The `language=tr` filter is **important** (change what suits you) — YouTube auto-dub creates AI-translated audio tracks that you almost certainly don't want to train on.
3. Run the pipeline from wherever this repo is checked out:
   ```bash
   bash scripts/run_pipeline.sh
   ```
   Or with overrides:
   ```bash
   TTS_DATASET=other_voice TTS_LANGUAGE=en bash scripts/run_pipeline.sh
   ```
4. Tail progress in another terminal:
   ```bash
   tail -f ~/TTS/data/<TTS_DATASET>/pipeline.log
   ```

Output: `metadata.csv` + a `wavs/` directory of 22050 Hz mono chunks, ready to feed `piper.train`.

## Notes I learned along the way

- **YouTube auto-dub is AI-generated.** A channel can have multiple audio tracks (original + auto-dubbed). Without explicit format selection (`yt-dlp -f 'ba[language=tr]/ba'`), you might silently download the auto-dub and unknowingly clone an AI voice instead of the real narrator. Always check the format list with `yt-dlp -F`.
- **Audiobook narrators often sound too oratorical for notification reading.** Documentary/essay-style narrators with conversational prosody work better.
- **Vocal separation (Demucs/UVR) is OK as a last resort**, but a source with *no* background music produces strictly better fine-tuning results — the model picks up subtle separation artifacts and reproduces them in the trained voice.
- **`.onnx` is just a container.** Two `.onnx` TTS models from different architectures (Piper VITS, Supertonic, etc.) are not interchangeable — the consuming app has to expect the right input/output tensor shapes.
- **Single-speaker data discipline matters.** Any contamination (movie clips, guest interviews, ads) gets averaged into the trained voice as inconsistency.

## License

To be decided.
