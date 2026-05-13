import json
import os
from pathlib import Path
from faster_whisper import WhisperModel
import soundfile as sf
import librosa

# Layout (override with env vars):
#   TTS_DATA_ROOT          -> root dir containing per-dataset subdirs   (default: ~/TTS/data)
#   TTS_DATASET            -> dataset subdir to process                 (default: merve)
#   TTS_LANGUAGE           -> Whisper language code                     (default: tr)
#   TTS_SEGMENT_PADDING_S  -> seconds of audio kept BEFORE/AFTER each
#                             Whisper-VAD segment when slicing chunks.
#                             0.0 = original tight cut (Whisper's exact
#                             timestamps); 0.2 = 200 ms padding each side
#                             so the model learns proper attack/release.
#                             Default: 0.2 (introduced in experiment 2
#                             after experiment 1's tight cuts caused
#                             swallowed word starts and boundary noise).
DATA_ROOT = Path(os.environ.get('TTS_DATA_ROOT', str(Path.home() / 'TTS' / 'data')))
DATASET = os.environ.get('TTS_DATASET', 'merve')
LANGUAGE = os.environ.get('TTS_LANGUAGE', 'tr')
SEGMENT_PADDING_S = float(os.environ.get('TTS_SEGMENT_PADDING_S', '0.2'))

BASE = DATA_ROOT / DATASET
RAW = BASE / 'raw'
TRANS = BASE / 'transcripts'
WAVS = BASE / 'wavs'
TRANS.mkdir(exist_ok=True, parents=True)
WAVS.mkdir(exist_ok=True, parents=True)
METADATA = BASE / 'metadata.csv'

TARGET_SR = 22050
MIN_DUR, MAX_DUR = 1.5, 15.0
MIN_TEXT_LEN = 5
MAX_NO_SPEECH_PROB = 0.5

print('Loading whisper-large-v3 on CUDA...', flush=True)
model = WhisperModel('large-v3', device='cuda', compute_type='float16')

entries = []
for wav_path in sorted(RAW.glob('*.wav')):
    vid = wav_path.stem
    trans_json = TRANS / f'{vid}.json'

    if trans_json.exists():
        print(f'[{vid}] cached transcript, loading...', flush=True)
        data = json.loads(trans_json.read_text(encoding='utf-8'))
    else:
        print(f'[{vid}] transcribing...', flush=True)
        segments, info = model.transcribe(
            str(wav_path),
            language=LANGUAGE,
            vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 500},
            beam_size=5,
        )
        seg_list = []
        for s in segments:
            seg_list.append({
                'start': s.start, 'end': s.end,
                'text': s.text.strip(),
                'no_speech_prob': s.no_speech_prob,
            })
        data = {'language': info.language, 'duration': info.duration, 'segments': seg_list}
        trans_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'  -> {len(seg_list)} segments, duration={info.duration:.1f}s', flush=True)

    print(f'[{vid}] resampling audio to {TARGET_SR} Hz...', flush=True)
    audio, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)

    kept, dropped_dur, dropped_speech, dropped_text = 0, 0, 0, 0
    for i, seg in enumerate(data['segments']):
        dur = seg['end'] - seg['start']
        text = seg['text'].strip()
        if dur < MIN_DUR or dur > MAX_DUR:
            dropped_dur += 1; continue
        if seg.get('no_speech_prob', 0) > MAX_NO_SPEECH_PROB:
            dropped_speech += 1; continue
        if len(text) < MIN_TEXT_LEN:
            dropped_text += 1; continue
        # Pad each chunk by up to SEGMENT_PADDING_S on each side.
        #
        # IMPORTANT: clamp the padding to the midpoint of the gap to the
        # adjacent Whisper segment, so two padded chunks never cover the
        # same audio. Without the clamp, a chunk's padded "tail" could
        # include the first syllable of the next utterance — but the
        # next chunk's transcript would already claim that syllable, so
        # the current chunk would learn a phantom phoneme it can't see
        # in its own transcript. That bleed-over is one cause of the
        # boundary-glitch artifacts experiment 1 had.
        prev_end = data['segments'][i - 1]['end'] if i > 0 else 0.0
        next_start = (
            data['segments'][i + 1]['start']
            if i + 1 < len(data['segments'])
            else float('inf')
        )
        pad_before = max(0.0, min(SEGMENT_PADDING_S, (seg['start'] - prev_end) / 2))
        pad_after = max(0.0, min(SEGMENT_PADDING_S, (next_start - seg['end']) / 2))
        s_idx = max(0, int((seg['start'] - pad_before) * TARGET_SR))
        e_idx = min(len(audio), int((seg['end'] + pad_after) * TARGET_SR))
        chunk = audio[s_idx:e_idx]
        out_name = f'{vid}_{i:04d}.wav'
        sf.write(WAVS / out_name, chunk, TARGET_SR, subtype='PCM_16')
        entries.append((out_name, text))
        kept += 1
    print(f'  [{vid}] kept={kept} | dropped: dur={dropped_dur} speech={dropped_speech} text={dropped_text}', flush=True)

with METADATA.open('w', encoding='utf-8') as f:
    for name, text in entries:
        f.write(f'{name}|{text}\n')

print(f'\n=== Total utterances: {len(entries)} ===', flush=True)
print(f'metadata.csv: {METADATA}', flush=True)
print(f'wavs/: {WAVS}', flush=True)
