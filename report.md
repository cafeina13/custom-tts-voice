# Report — custom Turkish TTS voice project

A "what happened" write-up of three iterative experiments on building a custom Turkish text-to-speech voice and deploying it to an Android phone. Covers the pipeline, the architectural decisions, the practical bugs and gotchas, and the iteration loop driven by listening tests.

This is a learning project, written so it can also serve as a portfolio piece. The honest framing: I had **zero prior TTS knowledge** when I started. I used Claude Code as a pair-programming assistant. Every decision below was made through dialogue, but the trade-offs are real and the bugs are real.

---

## 1. Goal

Replace the stock Turkish voice on my Android phone's system "Read Aloud" feature with a custom-trained voice — fine-tune a Piper TTS model on a single Turkish narrator's voice and deploy it via [SherpaTTS](https://github.com/woheller69/ttsengine), which loads custom `.onnx` voices and registers itself as a system TTS engine.

Daily-quality-of-life motivation (notifications read aloud through earbuds) plus an excuse to learn the whole pipeline end-to-end.

## 2. Architecture, big picture

```
YouTube source (one consistent narrator, no music, no guests)
        │  yt-dlp (original-language audio track only — NOT auto-dub)
        ▼
Raw WAV episodes (~5.7 h, 48 kHz)
        │  faster-whisper large-v3 + Silero VAD  (transcribe + split)
        │  optional: pad ±N ms around each segment for proper attack/release
        ▼
Per-segment text + timestamps
        │  filter: 1.5–15 s, no-speech-prob < 0.5, non-empty text
        │  optional: drop segments containing foreign-language tokens
        │  librosa: resample 22050 Hz mono, slice chunks
        ▼
LJSpeech-style dataset (metadata.csv + 22 kHz WAVs)
        │  piper.train.fit: fine-tune from tr_TR-dfki-medium
        │  save every-1k-step checkpoint for A/B ladder
        ▼
Trained Piper voice (.ckpt)
        │  piper.train.export_onnx + sherpa-onnx metadata embedding
        ▼
model.onnx + tokens.txt (sherpa-onnx format, ~61 MB)
        │  USB / "Install from SD Card" in SherpaTTS Android app
        ▼
Voice loaded on phone → used by system Read Aloud
```

## 3. Concepts I learned along the way

**TTS architectures, ranked by suitability for this project:**

- **VITS** (Piper's architecture) — end-to-end, fine-tunable on a few hours of data, runs on a phone. Right tool.
- **Zero-shot voice cloning** (XTTS, ElevenLabs-style) — short reference clip clones any voice, but the models are too heavy for a phone. Wrong tool here.
- **Singing voice synthesis** (DiffSinger, So-VITS-SVC) — for cloning a singer to sing, not speak. Wrong domain.

**Phonemization.** Neural TTS models don't read raw letters; they read **phonemes** (IPA sound units). Piper uses [espeak-ng](https://github.com/espeak-ng/espeak-ng) compiled in-tree to turn `"Merhaba"` into `[m ˈɛ r h a b a]`. Each phoneme then maps to an integer ID via a per-voice `phoneme_id_map` baked into the model config. Without correct phonemes the voice would mispronounce everything.

**Why fine-tune rather than train from scratch.** Piper's `tr_TR-dfki-medium` base was trained for 1.49 M steps on a large multi-hour Turkish corpus. With only ~5 h of single-speaker data, I can't beat that from zero. But I *can* "specialize" the converged base to sound like one specific speaker with a few thousand fine-tune steps — much cheaper, much better-quality result.

**Training data has to match what the production runtime expects.**
- 22050 Hz mono WAV (Piper's medium tier).
- LJSpeech CSV format (`filename|transcript`).
- IPA phonemes via espeak-ng using the *same* `phoneme_id_map` as the base voice.

**ONNX is just a container.** A `.onnx` file is the model's computation graph + weights packaged for portable inference. Two TTS models packaged as `.onnx` (Piper-VITS vs Supertonic vs whatever) are not interchangeable — the *consuming* app has to expect the right input/output tensor shapes. SherpaTTS needs a "sherpa-onnx Piper" `.onnx` specifically, not just any Piper `.onnx`.

**VAD (Voice Activity Detection).** A model that tags chunks of audio as "speech" or "not speech." Silero VAD is the standard choice; faster-whisper bundles it for skipping silence/music while transcribing. Useful upstream of dataset prep, harmful if it cuts too tightly (see section 6).

**Mixed precision and VRAM.** Training in `16-mixed` (fp16 activations, fp32 weights) halves activation VRAM and roughly doubles throughput on modern NVIDIA GPUs. On an 8 GB laptop GPU this was the difference between batch size 4 (slow, unstable) and batch size 12 (fast, stable).

**VITS validation loss is stochastic.** VITS samples from a learned posterior at inference, including during validation. The val_loss curve has ±1–3 units of noise from this even with identical weights — never read a single point, only the trend across many.

## 4. Decisions that mattered, and why

| Decision | What was picked | Why |
|---|---|---|
| Engine on phone | SherpaTTS (sherpa-onnx) | Loads custom `.onnx` voices via SD card or ADB; alternatives (Piper Android, Supertonic) were harder to find or used incompatible model formats |
| Base model | `tr_TR-dfki-medium` | The only viable Turkish Piper base on Hugging Face; fine-tuning was vastly cheaper than training from scratch and avoids the phoneme-set mismatch of cross-language transfer |
| Voice source | A single Turkish YouTube narrator with no background music, conversational tone (not oratorical like audiobooks), ~5 h of recent content with consistent voice | Same-speaker discipline matters; conversational prosody fits notification reading better than audiobook performance |
| Vocal separation | Skipped | Demucs/UVR introduces subtle artifacts; cleaner to pick a source that's already music-free |
| Original-audio guard | `yt-dlp -f 'ba[language=tr]/ba'` | YouTube auto-dub silently switches the track to an AI translation. Without explicit language pinning, the dataset would be the AI dub voice, not the real narrator |
| Training framework | PyTorch + Lightning (via Piper's CLI) | Standard; Lightning's `--ckpt_path` handles full state restoration cleanly across sessions |
| Batch size | 12 | Safe ceiling for 8 GB VRAM with `16-mixed` precision |
| Save policy | Every 1k steps, keep all snapshots | The whole point is to find the perceptual sweet spot. Lightning's default "keep best by val_loss" is wrong here because val_loss does not predict perceptual quality in VITS |
| WSL vs native Windows | WSL2 Ubuntu | Piper expects Linux build tools and shell scripts; not worth fighting on Windows |

## 5. Bugs and gotchas encountered

The real-engineering side that doesn't appear in tutorials.

1. **CTranslate2 needs CUDA 12 libs, PyTorch installs CUDA 13.** faster-whisper crashed with `libcublas.so.12 not found`. Fixed by `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` and adding them to `LD_LIBRARY_PATH` in the run-pipeline launcher.
2. **Then training itself broke because LD_LIBRARY_PATH put cu12 first.** PyTorch tried to JIT-compile a fused tanh-sigmoid kernel via NVRTC and looked for `libnvrtc-builtins.so.13.0`, which wasn't on the path. Fix: put `nvidia/cu13/lib` *before* the cu12 dirs in the training launcher.
3. **PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`.** That broke loading the base checkpoint, which contains `pathlib.PosixPath` objects in its hyper-parameter dict. Fix: a wrapper (`scripts/train_runner.py`) that allowlists the path types via `torch.serialization.add_safe_globals` before delegating to `piper.train`.
4. **The same base checkpoint has 50+ stale Lightning-trainer-args in its `hyper_parameters`.** Things like `tpu_cores`, `auto_lr_find`, `sample_bytes` from a 3-year-old Lightning. The current Piper VITS model class rejects them. Fix: same wrapper monkey-patches `torch.load` to filter `hyper_parameters` to keys the current model class understands.
5. **PyTorch 2.6 also flipped `torch.onnx.export`'s default to "dynamo".** That trips on a VITS data-dependent guard inside `rational_quadratic_spline`. Fix: an export wrapper (`scripts/export_runner.py`) that forces `dynamo=False`.
6. **CRLF line endings on Windows broke shell scripts.** First `train.sh` run spat `$'\r': command not found` errors per line. Cause: git's Windows-default `core.autocrlf=true`. Fix: `.gitattributes` pinning `*.sh` and `*.py` to `eol=lf`, plus `tr -d '\r'` (not `sed -i 's/\r$//'`, which has its own quoting issues on Windows) to clean on-disk copies.
7. **`train.sh` had a self-bug: env var `TTS_CKPT_PATH` was respected by the resolver but then the actual `--ckpt_path` argument was hardcoded to `base.ckpt`.** A "resume" attempt silently started a fresh fine-tune from scratch and was about to overwrite the experiment-2 checkpoint ladder before it was killed. Fix: pass the resolved variable through to the command. Caught only because the launch-time process state was inspected — would not have shown in the script's own log.
8. **Lightning writes resumed runs to a new `version_N/` subdir, not back into the original.** So `step_11000.ckpt` after a resume from `last.ckpt` did not land where the previous `step_10000.ckpt` was. Watch out when ls-ing the checkpoint dir to confirm progress.

## 6. Experiment 1 — first fine-tune (15 k steps, tight VAD)

Trained for 15 000 steps in ~65 minutes on the RTX 4070 Laptop. Saved a snapshot at 12 502 mid-run for comparison.

Exported both checkpoints to sherpa-onnx, installed as "Merve 15k" and "Merve 12.5k" via SherpaTTS, listened to both.

**Verdict — not good, two distinct issues:**

1. **15 k sounded more robotic than 12.5 k.** Textbook overfitting: the perceptual peak was somewhere before 12.5 k, after which more training made the model *worse* (memorizing quirks of the dataset rather than generalizing).
2. **Both voices "swallowed" word starts and produced noise around chunk boundaries.** Independent of overfitting. The model trained on chunks with no leading or trailing silence (Whisper VAD cut tight), so it never learned natural attack/release.

3rd suspected contributor: **foreign words in transcripts.** Names like "Frankenstein", "Casablanca", "Café" — phonemized by espeak-tr with Turkish letter-to-sound rules, but the speaker pronounced them in the source language. Audio/phoneme mismatch injects training noise.

## 7. Experiment 2 — fixes from experiment 1's diagnosis

Three changes from experiment 1:

1. **Symmetric 200 ms padding** added around each Whisper VAD segment so the model learns natural attack and release. Padding clamped to the midpoint of the gap to the adjacent segment, so two padded chunks never overlap.
2. **Foreign-word filter.** A new `scripts/scan_foreign_words.py` classifies tokens via Unicode category checks and `hunspell-tr` dictionary lookup, producing two lists in the dataset dir: `auto_drop_tokens.txt` (clearly non-Turkish, drop unreviewed) and `review_tokens.txt` (uncertain, edited by hand to remove false positives). `build_dataset.py` drops any segment whose transcript contains a token in either list. ~28 % of segments filtered, leaving 3 075 clean utterances from the original ~4 400.
3. **Save every 1 k steps and keep all snapshots** via a `ModelCheckpoint` callback injected by `train_runner.py`. The whole point is to find the perceptual peak; can't do that without the ladder.

Trained from `base.ckpt` to 10 000 steps, then resumed from `last.ckpt` to 15 000. Exported 6 k, 9 k, 11 k, 13 k, 15 k for A/B testing.

**Verdict — much better, but two new specific issues identified:**

1. **VITS tail hallucination.** All checkpoints produced random gibberish after the last word. Fixed at inference time, not training: terminal punctuation (`.` / `!` / `?`) makes espeak insert an end-of-sentence phoneme, which gives the duration predictor a clean stop signal. Without punctuation the model over-extends past the last phoneme and fills the slot with prior-sampled noise. *Always end input with punctuation.*
2. **Leading-frame mumble** ("chewing in her mouth" before the first word, ~2/10 utterances). This one isn't fixable at inference — sherpa-onnx exposes only speed and pitch to the host app, not `noise_scale` / `noise_scale_w`, and `sherpa_add_meta.py` doesn't bake them into the ONNX. Diagnosed as a side effect of the symmetric 200 ms padding: the leading 200 ms of each clip contains breath / lip-noise / non-silence wind-up, so the model learned "before phonemes start, generate some non-silent audio." When deployed, it sometimes does exactly that.

Voice quality clearly improved across the step ladder — audible progression from 6 k → 15 k. So 15 k was *not* past the perceptual peak this time (cleaner data shifted the peak later than experiment 1's curve). But the leading-frame issue is structural, not training-amount-dependent.

## 8. Experiment 3 — asymmetric padding (queued)

Single change from experiment 2: split `TTS_SEGMENT_PADDING_S` into `TTS_SEGMENT_PAD_LEAD_S` and `TTS_SEGMENT_PAD_TAIL_S`. Target: `lead = 0.05 s`, `tail = 0.2 s`.

- The 0.2 s tail kept the experiment-2 fix for word-end clarity.
- The 0.05 s lead-in is small enough that the model can't reliably learn "generate non-silent audio before phonemes" while still being big enough that fast word onsets aren't clipped.

Warm-start from `step_6000.ckpt` rather than `base.ckpt`:

- At step 6 k the voice has clearly emerged in A/B but the leading-pad pattern isn't deeply baked yet.
- LR is still ~50–65 % of its starting value (vs ~15 % at step 15 k), so updates are large enough to actually shift learned behavior on near-identical data.
- Cuts training time roughly in half vs full retrain. If it fails, the safe fallback is full retrain from `base.ckpt` on the same asymmetric-padded dataset.

Target `TTS_MAX_STEPS=15000`. Listen-test from step ~9 k onwards (giving the model 3 k+ steps to shift behavior under the new data distribution).

If experiment 3 succeeds, the iteration loop stops here. If not, fall back to full retrain and look at deeper interventions (smaller `noise_scale` baked into the export, dataset-side silence trimming, possibly the "high" Piper quality tier with a bigger model).

## 9. What I'd do differently

Lessons that carry forward beyond this one project:

- **Save intermediate checkpoints from step 1.** Lightning's default keeps only the latest. Quality-vs-step is the most important diagnostic; shouldn't have to retrain to recover it. (Fixed in experiment 2; lesson permanent.)
- **Listen-test the dataset before training.** Even a 20-clip random spot-check would have caught the tight-VAD issue in experiment 1 by hearing the swallowed word-starts in the chunks themselves, no training needed.
- **A/B against the base before deploying.** Comparing a fine-tune to the unmodified base voice would have told me immediately whether training *improved* or *degraded* the base. Skipping this step changed my interpretation of "not good" in experiment 1.
- **`val_loss` does not equal perceptual quality.** VITS has stochastic validation and the loss has a noise floor of ±1–3 units. Use it to detect divergence (sustained drift, not single bumps), not to pick the best snapshot. That's what the ears are for.
- **Verify "resume" actually resumed.** Inspect the live process command line, not just the script's log echo. The `train.sh` resume bug masked itself in the log for the first few minutes.
- **Asymmetric problems need asymmetric solutions.** Symmetric padding was a fast fix for a one-sided problem (word-start clipping) and introduced a new one-sided artifact (leading mumble). Default to fixing only the side that's actually broken.

## 10. To elevate this from learning project to real project

- **Data path with proper consent.** YouTube ToS doesn't permit redistribution of audio; even for personal use the trained model lives in a grey area. Sourcing public-domain LibriVox / Common Voice / commissioned recordings (with the speaker's explicit consent) would make the trained artifact shareable. Doing this on someone else's voice without permission would not be OK regardless of how good the result is.
- **Add evaluation metrics.** Right now quality assessment is subjective. Mel-cepstral distortion, character error rate of a Whisper round-trip on synthesized audio, MOS-style listening tests would give actual numbers.
- **Train on a "high" quality voice tier.** Piper's "medium" trades fidelity for size. A "high" model with the same fine-tuning pipeline would sound better on phone, at the cost of larger ONNX / slower inference.
- **Automate the deploy loop.** A script that exports a checkpoint, ADB-pushes it to the phone, and restarts SherpaTTS would let me iterate in seconds instead of minutes.
- **Package the dataset prep reproducibly.** Currently depends on Whisper-large-v3 running locally with a specific CUDA toolchain. Containerizing with pinned versions would survive across machines.
- **Validate the multi-language path.** Scripts are env-var parameterised but `tr_TR-dfki-medium` is hardcoded as the base in a couple of places. English / other-language fine-tunes would surface what else needs to be made generic.

---

*Iteratively updated as experiments completed; first draft after experiment 1, current draft after experiment 2 and experiment 3 plan.*
