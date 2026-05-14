"""Scan cached Whisper transcripts for likely-foreign words.

Why: Piper's espeak-tr phonemizer reads every token with Turkish letter-to-sound
rules. Foreign words (Frankenstein, Casablanca, ...) get a Turkish phoneme
sequence that doesn't match the speaker's actual English/French/etc.
pronunciation in the audio. That mismatch is one suspected cause of the
"mumbled / off prosody" output from experiment 1.

This script estimates how big the problem is BEFORE we commit to filtering.
It does NOT modify the dataset.

Buckets:
  1. non_tr_letters  -> token contains q/w/x or non-Turkish accented letters
                        (high confidence foreign)
  2. proper_noun     -> capitalized token appearing mid-sentence
                        (likely proper noun; mix of foreign + Turkish names)

Run:
  python3 scripts/scan_foreign_words.py
  TTS_DATASET=merve python3 scripts/scan_foreign_words.py
"""
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(os.environ.get('TTS_DATA_ROOT', str(Path.home() / 'TTS' / 'data')))
DATASET = os.environ.get('TTS_DATASET', 'merve')
TRANS = DATA_ROOT / DATASET / 'transcripts'
AUTO_DROP_OUT = DATA_ROOT / DATASET / 'auto_drop_tokens.txt'
REVIEW_OUT = DATA_ROOT / DATASET / 'review_tokens.txt'

TR_LOWER = set('abcdefghijklmnopqrstuvwxyzçğıiöşü')
TR_LOWER -= set('qwx')
TR_UPPER = set(c.upper() for c in TR_LOWER) | {'İ'}
TR_LETTERS = TR_LOWER | TR_UPPER

# Unicode letter class so foreign accented words (Café, Zdzisław, naïve)
# stay as a single token instead of fragmenting into pieces like "Caf"+"aw".
# A fragmented foreign word produces multiple bogus entries in the unknown
# list and looks like a tokenizer false positive during review.
TOKEN_RE = re.compile(r"[^\W\d_]+(?:[''][^\W\d_]+)*", re.UNICODE)
SENT_END_RE = re.compile(r'[.!?…]')
# Turkish writes number+suffix as "1920'ler", "1764'te", "2024'teki" etc.
# Stripping these before tokenizing prevents orphan suffix tokens (te, ler,
# da, deki, ...) from showing up as "unknown words".
NUMBER_SUFFIX_RE = re.compile(r"\d+[''][^\W\d_]+", re.UNICODE)
# Turkish writes proper-noun case suffixes after an apostrophe:
# "William'ın", "Walpole'da", "Highway'in". Strip everything from the
# apostrophe onward so we look up the bare proper noun in hunspell.
APOSTROPHE_RE = re.compile(r"['']")


def strip_suffix(token: str) -> str:
    return APOSTROPHE_RE.split(token, 1)[0]


def hunspell_unknown(tokens: set[str]) -> set[str]:
    """Return the subset of tokens that hunspell-tr does NOT recognize."""
    if not shutil.which('hunspell'):
        print('  (hunspell not installed; skipping dictionary bucket)')
        return set()
    payload = '\n'.join(sorted(tokens)) + '\n'
    proc = subprocess.run(
        ['hunspell', '-d', 'tr_TR', '-l'],
        input=payload, capture_output=True, text=True, check=True,
    )
    return {line for line in proc.stdout.splitlines() if line}


def classify(token: str) -> str | None:
    """Return bucket name if token looks foreign, else None."""
    if not token:
        return None
    for ch in token:
        if ch.isalpha() and ch not in TR_LETTERS:
            return 'non_tr_letters'
    return None


def split_sentences(text: str) -> list[str]:
    parts, cur = [], []
    for ch in text:
        cur.append(ch)
        if SENT_END_RE.match(ch):
            parts.append(''.join(cur))
            cur = []
    if cur:
        parts.append(''.join(cur))
    return parts


def main() -> None:
    files = sorted(TRANS.glob('*.json'))
    if not files:
        print(f'No transcripts in {TRANS}')
        return

    seg_total = 0
    seg_with_nontr = 0
    seg_with_proper = 0
    nontr_tokens: Counter = Counter()
    nontr_bare: Counter = Counter()
    proper_tokens: Counter = Counter()
    # Per-segment lookup tokens (apostrophe-stripped) for the hunspell pass.
    seg_lookup_tokens: list[set[str]] = []
    all_lookup_tokens: set[str] = set()
    bare_counts: Counter = Counter()

    for fp in files:
        data = json.loads(fp.read_text(encoding='utf-8'))
        for seg in data['segments']:
            seg_total += 1
            text = NUMBER_SUFFIX_RE.sub(' ', seg['text'].strip())
            seg_has_nontr = False
            seg_has_proper = False
            seg_tokens: set[str] = set()

            for sentence in split_sentences(text):
                tokens = TOKEN_RE.findall(sentence)
                for idx, tok in enumerate(tokens):
                    bare = strip_suffix(tok)
                    # Skip 1-letter tokens (e.g. "P" from "H.P. Lovecraft").
                    # They aren't reviewable on their own and any segment
                    # containing them almost always also contains the full
                    # foreign word that triggered the abbreviation.
                    if bare and len(bare) >= 2:
                        seg_tokens.add(bare)
                        all_lookup_tokens.add(bare)
                        bare_counts[bare] += 1
                    bucket = classify(tok)
                    if bucket == 'non_tr_letters':
                        nontr_tokens[tok] += 1
                        if bare:
                            nontr_bare[bare] += 1
                        seg_has_nontr = True
                    if idx > 0 and tok[0].isupper() and any(c.islower() for c in tok[1:]):
                        proper_tokens[tok] += 1
                        seg_has_proper = True

            seg_lookup_tokens.append(seg_tokens)
            if seg_has_nontr:
                seg_with_nontr += 1
            if seg_has_proper:
                seg_with_proper += 1

    print(f'Running hunspell-tr on {len(all_lookup_tokens)} unique tokens...')
    unknown = hunspell_unknown(all_lookup_tokens)
    seg_with_unknown = sum(1 for s in seg_lookup_tokens if s & unknown)
    unknown_token_counts: Counter = Counter()
    for s in seg_lookup_tokens:
        for t in s & unknown:
            unknown_token_counts[t] += 1

    print(f'Dataset: {DATASET}   transcripts scanned: {len(files)}')
    print(f'Total segments: {seg_total}')
    print()
    print('Bucket 1 - non-Turkish letters (q/w/x or foreign accents)')
    print(f'  segments affected: {seg_with_nontr}  ({100 * seg_with_nontr / seg_total:.2f}%)')
    print(f'  unique tokens: {len(nontr_tokens)}   total occurrences: {sum(nontr_tokens.values())}')
    print('  top 30:')
    for tok, n in nontr_tokens.most_common(30):
        print(f'    {n:4d}  {tok}')
    print()
    print('Bucket 2 - capitalized mid-sentence (proper nouns; mix of TR + foreign)')
    print(f'  segments affected: {seg_with_proper}  ({100 * seg_with_proper / seg_total:.2f}%)')
    print(f'  unique tokens: {len(proper_tokens)}   total occurrences: {sum(proper_tokens.values())}')
    print('  top 40:')
    for tok, n in proper_tokens.most_common(40):
        print(f'    {n:4d}  {tok}')
    print()
    print('Bucket 3 - hunspell-tr unknown (apostrophe suffix stripped)')
    print(f'  segments affected: {seg_with_unknown}  ({100 * seg_with_unknown / seg_total:.2f}%)')
    print(f'  unique unknown tokens: {len(unknown)}')
    print('  top 60:')
    for tok, n in unknown_token_counts.most_common(60):
        print(f'    {n:4d}  {tok}')

    # Split into two files so the human only reviews the ambiguous bucket.
    #   auto_drop_tokens.txt -> L-flagged (non-Turkish letters or non-Latin
    #     scripts). These cannot be standard Turkish, so they're auto-dropped
    #     without review. Includes Whisper hallucinations in foreign scripts.
    #   review_tokens.txt    -> H-only (hunspell-unknown but uses Turkish
    #     letters). Mix of real foreign names (Fred, Victor, Frankenstein)
    #     and Turkish words hunspell doesn't know (ortacag, boylelikle).
    #     Human review decides.
    auto_rows, review_rows = [], []
    for tok in set(nontr_bare) | set(unknown):
        n = bare_counts[tok]
        if tok in nontr_bare:
            flags = 'LH' if tok in unknown else 'L-'
            auto_rows.append((n, flags, tok))
        else:
            review_rows.append((n, '-H', tok))
    auto_rows.sort(key=lambda r: (-r[0], r[2].lower()))
    review_rows.sort(key=lambda r: (-r[0], r[2].lower()))

    auto_header = (
        '# AUTO-DROP tokens from dataset: ' + DATASET + '\n'
        '# Tokens with non-Turkish letters or non-Latin scripts. These\n'
        '# cannot be standard Turkish, so the build will drop any segment\n'
        '# containing them WITHOUT human review. Includes Whisper\n'
        '# hallucinations in Cyrillic/Hebrew/CJK/etc.\n'
        '# Format: <count>\\t<flags>\\t<token>, sorted by count desc.\n'
    )
    review_header = (
        '# REVIEW tokens from dataset: ' + DATASET + '\n'
        '# Hunspell-tr does not know these, but they only use Turkish\n'
        '# letters. Mix of:\n'
        '#   * real foreign names (Fred, Victor, Frankenstein) -> KEEP in list\n'
        '#   * Turkish words hunspell missed (ortacag, boylelikle, arasılık)\n'
        '#     OR foreign names the speaker reads with Turkish phonetics\n'
        '#     (judgement call) -> DELETE the line\n'
        '# Surviving lines join auto_drop_tokens.txt as the final drop set.\n'
        '# Format: <count>\\t<flags>\\t<token>, sorted by count desc.\n'
    )

    AUTO_DROP_OUT.write_text(
        auto_header + '\n'.join(f'{n}\t{f}\t{t}' for n, f, t in auto_rows) + '\n',
        encoding='utf-8',
    )
    REVIEW_OUT.write_text(
        review_header + '\n'.join(f'{n}\t{f}\t{t}' for n, f, t in review_rows) + '\n',
        encoding='utf-8',
    )
    # Clean up the old combined file if it's lying around.
    old_combined = DATA_ROOT / DATASET / 'unknown_tokens.txt'
    if old_combined.exists():
        old_combined.unlink()
    print()
    print(f'AUTO-DROP list (no review needed): {AUTO_DROP_OUT}')
    print(f'  {len(auto_rows)} tokens')
    print(f'REVIEW list (your eyes needed):   {REVIEW_OUT}')
    print(f'  {len(review_rows)} tokens')


if __name__ == '__main__':
    main()
