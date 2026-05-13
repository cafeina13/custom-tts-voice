# Report — custom Turkish TTS voice project

A "what happened" write-up of the first two milestones of this project: building the dataset / training pipeline, and producing a first fine-tuned voice.

This is a learning project, written so it can also serve as a portfolio piece. The honest framing throughout: I (the author) had **zero prior TTS knowledge** going in. I used Claude Code as a pair-programming assistant; every design decision below was made through dialogue, but the technical concepts and trade-offs are real.

---

## 1. Goal

Replace the stock Turkish voice on my Galaxy S24+'s accessibility "Read Aloud" feature with a custom-trained voice — specifically, fine-tune a Piper TTS model on the voice of one Turkish YouTuber (Merve Taştan) and deploy it to my phone via [SherpaTTS](https://github.com/woheller69/ttsengine).

Notifications get read aloud through my earbuds; replacing the voice is a daily-quality-of-life improvement plus an excuse to learn the full TTS pipeline end-to-end.

## 2. Architecture, big picture

```
YouTube source (one consistent narrator)
        │  yt-dlp (original-language audio track only — NOT auto-dub)
        ▼
Raw WAV episodes (~5.7 h of Merve, 48 kHz)
        │  faster-whisper large-v3 + Silero VAD  (transcribe + split)
        ▼
Per-segment text + timestamps
        │  filter: 1.5–15 s, no-speech-prob < 0.5, non-empty text
        │  librosa: resample 22050 Hz mono, slice chunks
        ▼
LJSpeech-style dataset (4,405 utterances, metadata.csv + 22 kHz WAVs)
        │  piper.train.fit: fine-tune from tr_TR-dfki-medium
        ▼
Trained Piper voice (.ckpt)
        │  piper.train.export_onnx + sherpa-onnx metadata
        ▼
model.onnx + tokens.txt (sherpa-onnx format, ~61 MB)
        │  USB / "Install from SD Card" in SherpaTTS Android app
        ▼
Voice loaded on phone → used by system Read Aloud
```

## 3. Key concepts I learned

**TTS architectures, ranked by suitability for this project:**
- **VITS** (Piper's architecture) — end-to-end, fine-tunable on a few hours of data, runs on a phone. Good fit.
- **Zero-shot voice cloning** (OmniVoice, XTTS, ElevenLabs) — short reference clip clones any voice, but models are too heavy for a phone. Wrong tool here.
- **Singing voice synthesis** (DiffSinger, So-VITS-SVC) — for cloning a singer to sing, not speak. Wrong domain.

**Phonemization.** Neural TTS models don't read raw letters; they read **phonemes** (IPA sound units). Piper uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) compiled in-tree to turn `"Merhaba"` into `[m ˈɛ r h a b a]`. Each phoneme then maps to an integer ID via a per-voice `phoneme_id_map` baked into the model config. Without correct phonemes the voice would mispronounce everything.

**Why fine-tune rather than train from scratch.** Piper's `tr_TR-dfki-medium` base was trained for 1.49 M steps on a large multi-hour Turkish corpus. With only 5.7 h of single-speaker data, I can't beat that. But I *can* "specialize" the converged base to sound like Merve specifically with a few thousand fine-tune steps — much cheaper, much better-quality result than from-scratch.

**The training data has to match what the production runtime expects.**
- 22050 Hz mono WAV — Piper's medium tier expects this. Speech energy is mostly below 8 kHz; 22 kHz Nyquist captures it cleanly while halving file size vs music's 44.1 kHz.
- LJSpeech CSV format (`filename|transcript`).
- IPA phonemes via espeak-ng using the *same* `phoneme_id_map` as the base voice — otherwise we're producing tokens the base model doesn't know about.

**ONNX is just a container, not a format.** A `.onnx` file is the model's computation graph + weights packaged for portable inference. Two TTS models packaged as `.onnx` (Piper-VITS vs Supertonic vs whatever) are not interchangeable — the *consuming* app has to expect the right input/output tensor shapes. This is why SherpaTTS needs a "sherpa-onnx Piper" `.onnx` specifically, not just any Piper `.onnx`.

**VAD (Voice Activity Detection)** = a model that tags chunks of audio as "speech" or "not speech." Silero VAD is the standard choice; faster-whisper bundles it for skipping silence/music while transcribing. Useful upstream of dataset prep, harmful if it cuts too aggressively (see section 6).

**Mixed precision and VRAM.** Training in `16-mixed` (fp16 activations, fp32 weights) halves VRAM for the activation tensors and roughly doubles training throughput on modern NVIDIA GPUs. On my RTX 4070 Laptop's 8 GB VRAM, this was the difference between batch size 4 (slow, unstable) and batch size 12 (fast, stable).

## 4. Decisions that mattered, and why

| Decision | What was picked | Why |
|---|---|---|
| Engine on phone | SherpaTTS (sherpa-onnx) | Loads custom `.onnx` voices via SD card or ADB; the alternatives (Piper Android, Supertonic) were either harder to find or used incompatible model formats |
| Base model | `tr_TR-dfki-medium` | The only viable Turkish Piper base on Hugging Face; fine-tuning was vastly cheaper than training from scratch |
| Voice source | Merve Taştan YouTube channel | Single narrator, no background music, conversational tone (not oratorical like audiobooks), ~5 h of recent content with consistent voice |
| Why not audiobooks | Rejected | Most audiobook narrators sound too "performed" for notification reading. Documentary / essay narrators sound more natural |
| Why not vocal separation | Skipped | Demucs/UVR introduces subtle artifacts; cleaner to pick a source that's already music-free |
| Original audio guard | `yt-dlp -f 'ba[language=tr]/ba'` | YouTube auto-dub silently switches voices to an AI translation. Without explicit language pinning, I'd have cloned the AI dub voice instead of Merve |
| Training framework | PyTorch + Lightning (via Piper's own CLI) | Standard; Lightning's `--ckpt_path` handles full state restoration cleanly across sessions |
| Batch size | 12 | Safe ceiling for 8 GB VRAM with `16-mixed` precision |
| Max steps target (exp 1) | 15,000 | Conservative first guess; turned out to be past the overfitting peak |
| WSL vs native Windows | WSL2 Ubuntu | Piper training repo expects Linux build tools and shell scripts; not worth fighting on Windows |

## 5. Issues encountered and how they were resolved

These are the kind of "real ML engineering" problems that don't show up in tutorials but eat hours in practice. Worth keeping for the portfolio value.

1. **CTranslate2 needs CUDA 12 libs, PyTorch installs CUDA 13.** faster-whisper crashed with `libcublas.so.12 not found`. Fixed by `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` and adding them to `LD_LIBRARY_PATH` in the run-pipeline launcher. PyTorch's own CUDA 13 libs are kept separately under `nvidia/cu13/lib`.
2. **Then training itself broke because LD_LIBRARY_PATH put cu12 first.** PyTorch tried to JIT-compile a fused tanh-sigmoid kernel via NVRTC and looked for `libnvrtc-builtins.so.13.0`, which wasn't on the path. Fix: put `nvidia/cu13/lib` *before* the cu12 dirs in the training launcher's `LD_LIBRARY_PATH`.
3. **PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`.** That broke loading the rhasspy Piper base checkpoint, which contains `pathlib.PosixPath` objects in its hyper-parameter dict. Fix: a tiny wrapper script (`scripts/train_runner.py`) that allowlists the path types via `torch.serialization.add_safe_globals` before delegating to `piper.train`.
4. **The same base checkpoint also has 50+ stale Lightning-trainer-args in its `hyper_parameters`.** Things like `tpu_cores`, `auto_lr_find`, `sample_bytes` from a 3-year-old Lightning. The current Piper VITS model class doesn't accept them and Lightning's CLI errored with `Subcommand 'fit' does not accept option 'model.sample_bytes'`. Fix: in the same wrapper, monkey-patch `torch.load` to filter `hyper_parameters` down to the keys the current model class understands.
5. **PyTorch 2.6 also flipped `torch.onnx.export`'s default backend to "dynamo".** That backend trips on a VITS data-dependent guard inside `rational_quadratic_spline`. Fix: another wrapper (`scripts/export_runner.py`) that forces `dynamo=False` so we get the legacy TorchScript exporter, which Piper was designed for.
6. **Sherpa-onnx's metadata script needed `iso639` and `onnxscript`** as new transitive deps. Just `pip install` them in the venv.
7. **CRLF line endings on Windows broke shell scripts.** When `train.sh` ran for the first time, bash spat `$'\r': command not found` errors for every line. Cause: git's Windows-default `core.autocrlf=true` converted LF to CRLF on checkout. Fix: a `.gitattributes` that pins `*.sh` and `*.py` to `eol=lf`, plus `tr -d '\r'` (not `sed -i 's/\r$//'`, which has its own quoting issues on Windows) to clean the on-disk copies.
8. **VAD over-trimming caused boundary glitches.** Whisper's VAD draws very tight chunk boundaries to maximize speech content. The model learned "speech starts at sample 0" → it later swallows the first phoneme and emits boundary noise during inference. Fix in experiment 2: pad each chunk by 200 ms on each side, **clamped to the midpoint of the gap to the adjacent Whisper segment** (so padding never overlaps into another utterance — that would create transcript/audio mismatches and produce different artifacts).

## 6. Experiment 1 results

Trained for 15,000 steps in ~65 minutes on the RTX 4070 mobile (much faster than my pre-run estimate of 10-12 hours — Piper VITS is smaller than I assumed, and the step counter includes both the generator and discriminator updates).

Saved a snapshot at step 12,502 mid-run for comparison.

Exported both checkpoints to sherpa-onnx format, installed both as "Merve 15k" and "Merve 12.5k" via SherpaTTS's "Install from SD Card" UI, and listened to both.

**Verdict — not good enough yet, two distinct issues:**

1. **15k is more robotic than 12.5k.** Textbook overfitting signal: the curve plateaus somewhere before 12.5k, after which more training makes the model *worse* (memorizing quirks of the 4,405 utterances rather than generalizing). 15k was past the peak.
2. **Both voices "swallow" the starts of words and produce noise around chunk boundaries.** Independent of overfitting. The model was trained on chunks with no leading or trailing silence (Whisper VAD cut tight), so it never learned natural attack/release.

Both issues are addressable. The first by training fewer steps + saving a denser ladder of intermediate checkpoints so we can pick the actual peak. The second by re-cutting the dataset with 200 ms of padded silence around each chunk.

## 7. Experiment 2 plan (queued)

- Wipe the chunked audio and Piper's preprocessing cache. Keep the cached Whisper transcripts (they don't change).
- Re-cut chunks with 200 ms padding per side, clamped to gap midpoints (so chunks never overlap).
- Retrain to 10k steps (not 15k), saving a checkpoint every 1k steps via a `ModelCheckpoint` injected by `scripts/train_runner.py`.
- Export every saved checkpoint to sherpa-onnx, install all of them on the phone, A/B listen, pick the perceived sweet spot.

If the sweet-spot voice still has artifacts, the next move would be dataset cleanup (drop chunks with low Whisper confidence, look for systematic transcription errors) and possibly moving up to the "high" Piper quality tier — but that's a much bigger model and we'd need to revisit hardware budget.

## 8. What I'd do differently next time

- **Save intermediate checkpoints from step 1.** Lightning's default keeps only the latest. Knowing the quality-vs-step curve is the most important signal; I shouldn't have to retrain to recover it.
- **Listen-test the dataset before training.** I trusted Whisper's transcripts implicitly. Even a 20-clip random spot-check might have caught the tight-VAD issue earlier — I'd have heard the swallowed word-starts in the chunks themselves.
- **Start with a smaller training budget.** 15k steps was a guess. 5-10k with a ladder is more informative for the same compute.
- **A/B against the base before deploying.** Comparing my fine-tune to the unmodified `tr_TR-dfki-medium` would have told me immediately whether my training *improved* or *degraded* the base. That diagnostic step was skipped initially and would have changed my interpretation of "not good."

## 9. To elevate this from learning project to real project

If I wanted to take this seriously beyond personal use:

- **License the data path properly.** YouTube ToS doesn't permit redistribution of audio; even for personal use the trained model lives in a grey area. Sourcing public-domain LibriVox / Common Voice / commissioned recordings would make the artifact shareable.
- **Add evaluation metrics.** Right now "good or bad" is subjective. Mel-cepstral distortion, character error rate of a Whisper round-trip, MOS-style listening tests, would give actual numbers.
- **Train on a "high" quality voice tier.** Piper's "medium" trades fidelity for size and speed. A "high" model with the same fine-tuning pipeline would sound better on phone.
- **Automate the deploy loop.** A script that exports the best checkpoint, ADB-pushes it to the phone, and restarts SherpaTTS would let me iterate in seconds instead of minutes.
- **Package the dataset prep into something reproducible.** Right now it depends on Whisper-large-v3 running locally; for a real project I'd containerize and pin versions.
- **Multi-language from day one.** The repo is already parameterized via env vars, but I never actually validated the English path (`tr_TR-dfki-medium` is hardcoded as the base in a couple places).

---

*Written 2026-05-14, after experiment 1 wrapped and before experiment 2 began.*
