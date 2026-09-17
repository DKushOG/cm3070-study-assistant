"""Automatic measures for the generated notes, computed over every saved campaign run.

Run from the project folder with the virtual environment active:

    python auto_measures.py

Reports, for each of the four input modes, the three measures described in Section 5.3:
key-term coverage, a faithfulness proxy and a novel n-gram rate. Everything is computed
from files already on disk, so no model is called and nothing is regenerated.
"""
import csv
import glob
import os
import re
import statistics as st

TOPIC = os.path.join("topics", "bayes")
RUNS = os.path.join("outputs", "bayes")
OUT = os.path.join("outputs", "auto_measures.csv")

SOURCES = {
    "transcript": "transcript_sample.txt",
    "slide_text": "slide_text_sample.txt",
    "notes": "notes_sample.txt",
}
MODE_SOURCES = {
    "Transcript only": ["transcript"],
    "Slide text only": ["slide_text"],
    "Combined transcript and slide text": ["transcript", "slide_text"],
    "Transcript, slide text and notes": ["transcript", "slide_text", "notes"],
}
STOP = set("""a an and are as at be been but by for from had has have if in into is it its of on
or that the their then there these they this to was were what when which who will with you your
""".split())


def words(text):
    """Lower case, strip punctuation, drop stop words and very short tokens."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 3 and w not in STOP]


def ngrams(tokens, n=3):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def notes_of(path):
    """The revision notes section of a saved run, without the header or the quiz."""
    text = open(path, encoding="utf-8", errors="replace").read()
    if "=== REVISION NOTES ===" not in text:
        return ""
    body = text.split("=== REVISION NOTES ===", 1)[1]
    return body.split("=== QUIZ QUESTIONS ===", 1)[0]


def mode_of(path):
    first = open(path, encoding="utf-8", errors="replace").readline()
    return first.replace("MODE:", "").strip()


source_words = {k: words(open(os.path.join(TOPIC, v), encoding="utf-8").read())
                for k, v in SOURCES.items()}
all_source = [w for ws in source_words.values() for w in ws]

# a domain term is any content word appearing at least twice across the supplied sources
key_terms = {w for w in set(all_source) if all_source.count(w) >= 2}
print(f"key terms identified across the three sources: {len(key_terms)}\n")

rows = []
for path in sorted(glob.glob(os.path.join(RUNS, "*.txt"))):
    note = notes_of(path)
    if not note.strip():
        continue
    mode = mode_of(path)
    seen = set(MODE_SOURCES.get(mode, list(SOURCES)))
    supplied = [w for k in seen for w in source_words[k]]
    supplied_set = set(supplied)
    nw = words(note)
    if not nw:
        continue
    coverage = 100 * len(key_terms & set(nw)) / len(key_terms)
    unsupported = [w for w in nw if w not in supplied_set]
    faithfulness = 100 * (1 - len(unsupported) / len(nw))
    novel = ngrams(nw) - ngrams(supplied)
    novelty = 100 * len(novel) / max(1, len(ngrams(nw)))
    rows.append(dict(mode=mode, file=os.path.basename(path),
                     key_term_coverage=round(coverage, 1),
                     faithfulness=round(faithfulness, 1),
                     novel_trigrams=round(novelty, 1)))

with open(OUT, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print(f"{'mode':36} {'coverage':>9} {'faithful':>9} {'novel':>7}   runs")
for mode in MODE_SOURCES:
    got = [r for r in rows if r["mode"] == mode]
    if not got:
        continue
    print(f"{mode:36} "
          f"{st.mean(r['key_term_coverage'] for r in got):8.1f}% "
          f"{st.mean(r['faithfulness'] for r in got):8.1f}% "
          f"{st.mean(r['novel_trigrams'] for r in got):6.1f}%   {len(got)}")
print(f"\nWritten to {OUT}")
