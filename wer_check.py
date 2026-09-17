"""Transcribe the lecture at three Whisper sizes and score each against the reference."""
import re
import sys
import time

sys.path.insert(0, ".")
from modules import audio_stt

BASE = r"..\Testing data\Naive Bayes test data used\Lecture video and audio"
AUDIO = BASE + r"\Lecture video.mp4"
REFERENCE = BASE + r"\Lecture transcript.txt"


def normalise(text):
    """Lower case, drop punctuation, collapse whitespace, so the three sizes
    are scored the same way as the base figure already in the report."""
    text = text.lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()


def score(reference_words, hypothesis_words):
    """Levenshtein distance over word lists, with the edit types counted."""
    rows, cols = len(reference_words) + 1, len(hypothesis_words) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference_words[i - 1] == hypothesis_words[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1,
                              table[i][j - 1] + 1,
                              table[i - 1][j - 1] + cost)
    subs = dels = ins = hits = 0
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] and \
                reference_words[i - 1] == hypothesis_words[j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif j > 0 and table[i][j] == table[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1
    wer = (subs + dels + ins) / len(reference_words) * 100
    return wer, subs, dels, ins, hits


reference = normalise(open(REFERENCE, encoding="utf-8").read())
print(f"reference words: {len(reference)}\n")

for size in ["tiny", "base", "small"]:
    start = time.perf_counter()
    text = audio_stt.transcribe(AUDIO, model_size=size)
    seconds = time.perf_counter() - start
    with open(f"whisper_{size}.txt", "w", encoding="utf-8") as handle:
        handle.write(text)
    wer, subs, dels, ins, hits = score(reference, normalise(text))
    print(f"{size:6} WER {wer:5.2f}%  hits {hits}  sub {subs}  del {dels}  "
          f"ins {ins}  time {seconds:.0f}s")