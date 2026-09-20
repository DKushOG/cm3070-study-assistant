"""Compare candidate vision models on the slides that defeat the incumbent.

Why this exists
---------------
`qwen3-vl:4b` cannot be made to answer reliably. Four separate mechanisms for
suppressing its reasoning were tested and every one was ignored:

    chat_template_kwargs enable_thinking   ignored (OpenAI endpoint)
    /no_think prompt convention            ignored
    reasoning_effort                       ignored
    think: false on Ollama's native API    ignored (0.34.1)

The last is decisive. With `think` false the model produced 14088 completion
tokens and 62166 characters of reasoning; with `think` true it produced
14086 and 62158. The control did nothing at all.

Reply caps are honoured, but capping does not help. Capped at 1500 tokens the
model spent all 1500 thinking and returned nothing, in 17 seconds instead of
187. That bounds the cost of a failure without ever producing an answer.

Worse, the failure is not deterministic. Deck 04 page 6 was called three
times at temperature 0 and consumed 14641, 4825 and 10014 tokens, failing
once and succeeding twice. The same slide, the same settings, three different
outcomes. A model that answers a given page only sometimes cannot be the
extraction path for a tool a student relies on.

So the remedy is a different model, and the project brief asks for evidence
of testing and rejecting models rather than an assertion. This script is that
evidence.

Method
------
Every candidate sees the same nine pages, rendered once each at the same DPI
and shared between models, so any difference is the model's and not the
input's. That is the same fairness rule as evaluation/compare_extractors.py.

The pages are chosen to span the corpus rather than to flatter anyone: three
that defeated the incumbent, two dense text slides, two diagram-heavy slides
and two that are almost entirely image with no text layer at all.

Calls go through Ollama's native API so that `num_ctx` and `num_predict` can
actually be set, which the OpenAI compatible endpoint does not allow. A reply
cap is applied to every model equally, so a model that loops is recorded as
failing at a known cost instead of stalling the comparison for three minutes.

What is recorded, per model and page: whether a reply came back at all, how
long it took, how many tokens were generated, how many words were returned,
and the reply itself for reading afterwards. Format is what this measures.
Quality is for the manual column, as everywhere else in this project.

Usage, from the repository root:

    python bakeoff_vision.py                 # models already pulled
    python bakeoff_vision.py --pull          # download missing ones first

Roughly ten to fifteen minutes once the models are present. Downloads are
several gigabytes each and are not started unless --pull is given.
"""
import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

CORPUS = Path("..") / "Evaluation corpus"

# The incumbent is included so the comparison has a baseline row rather than
# a remembered one. Candidates are non-reasoning vision models small enough
# to run locally, which is the project's standing constraint.
DEFAULT_MODELS = [
    "qwen3-vl-ctx16k",      # incumbent, reasoning, the one being replaced
    "qwen2.5vl:3b",         # same family, no reasoning step
    "qwen2.5vl:7b",         # larger sibling, if it fits
    "minicpm-v:8b",         # strong on document and slide text
    "granite3.2-vision:2b",  # purpose-built for charts and diagrams
]

# Nine pages spanning the corpus. The first three are the ones that defeated
# the incumbent, so a candidate that cannot do them is no better.
PAGES = [
    ("01_AI_L8_control", "CM3020 Artificial Intelligence_lecture 8-2.pptx", 13),
    ("04_GP_T7_extreme", "CM2030_GP_PPT7-CourseraWeeks13&14.pptx", 6),
    ("02_AI_L6_mixed", "CM3020 Artificial Intelligence_lecture 6.pptx", 19),
    ("01_AI_L8_control", "CM3020 Artificial Intelligence_lecture 8-2.pptx", 5),
    ("01_AI_L8_control", "CM3020 Artificial Intelligence_lecture 8-2.pptx", 28),
    ("02_AI_L6_mixed", "CM3020 Artificial Intelligence_lecture 6.pptx", 6),
    ("03_ISP_L10_image_heavy", "CM3065 Lesson 10.pdf", 9),
    ("03_ISP_L10_image_heavy", "CM3065 Lesson 10.pdf", 2),
    ("04_GP_T7_extreme", "CM2030_GP_PPT7-CourseraWeeks13&14.pptx", 4),
]

# Applied to every model equally. Generous for a slide transcription, which
# the successful calls finished in 2000 to 3000 tokens, and low enough that a
# looping model is recorded as failing in about half a minute.
NUM_PREDICT = 3000
NUM_CTX = 16384

CSV_COLUMNS = [
    "Model", "Deck", "Page",
    "Quality (1 to 5)", "Notes",
    "answered", "done_reason", "seconds", "eval_tokens",
    "think_chars", "reply_words", "reply_chars", "output_file",
]


class Log:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8")

    def __call__(self, message=""):
        stamped = (f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
                   if message else "")
        print(stamped)
        sys.stdout.flush()
        self.handle.write(stamped + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def ollama_host(base_url):
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    return host.rstrip("/")


def installed_models(binary):
    try:
        listing = subprocess.run([binary, "list"], capture_output=True,
                                 text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return set()
    names = set()
    for line in (listing.stdout or "").splitlines()[1:]:
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def call_model(host, model, instruction, image_b64):
    """One native call with the shared caps. Never raises."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction,
                      "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/chat", data=data,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8")[:200]
        except Exception:
            pass
        return {"error": f"HTTP {error.code}: {body}",
                "seconds": time.perf_counter() - started}
    except Exception as error:
        return {"error": str(error)[:200],
                "seconds": time.perf_counter() - started}
    took = time.perf_counter() - started
    message = reply.get("message") or {}
    return {
        "content": (message.get("content") or "").strip(),
        "thinking": message.get("thinking") or "",
        "done_reason": reply.get("done_reason", "?"),
        "eval_count": reply.get("eval_count", ""),
        "seconds": took,
    }


def render_pages(log):
    """Render every page once, shared across all models.

    Returns [(deck_label, page_number, base64_png)]. A deck that cannot be
    converted is reported and skipped rather than ending the run.
    """
    from modules import slides_pptx
    import fitz

    rendered = []
    by_deck = {}
    for folder, filename, number in PAGES:
        by_deck.setdefault((folder, filename), []).append(number)

    for (folder, filename), numbers in by_deck.items():
        deck = CORPUS / folder / filename
        if not deck.is_file():
            log(f"  deck missing, skipping: {deck}")
            continue
        workspace = None
        pdf_path = deck
        try:
            if deck.suffix.lower() == slides_pptx.PPTX_SUFFIX:
                workspace = tempfile.TemporaryDirectory()
                pdf_path = Path(slides_pptx.convert_to_pdf(str(deck),
                                                           workspace.name))
            document = fitz.open(str(pdf_path))
            try:
                for number in sorted(numbers):
                    if number > len(document):
                        continue
                    pixmap = document[number - 1].get_pixmap(dpi=150)
                    rendered.append((folder, number, base64.b64encode(
                        pixmap.tobytes("png")).decode("ascii")))
            finally:
                document.close()
        except Exception as error:
            log(f"  could not render {folder}: {str(error)[:160]}")
        finally:
            if workspace:
                workspace.cleanup()
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--out", default="outputs/vision_bakeoff")
    parser.add_argument("--pull", action="store_true",
                        help="Download any candidate that is not installed. "
                             "Several gigabytes each.")
    args = parser.parse_args(argv)

    os.environ.setdefault("VISION_MODEL", "qwen3-vl-ctx16k")
    import config
    from modules import slide_vision

    out_dir = Path(args.out)
    (out_dir / "replies").mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / "bakeoff.log")
    log("=" * 68)
    log("Vision model bake-off")
    log("=" * 68)

    host = ollama_host(config.OPENAI_BASE_URL)
    binary = shutil.which("ollama")
    if binary is None:
        log("ABORT: 'ollama' not found on PATH.")
        log.close()
        return 1

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    present = installed_models(binary)
    models = []
    for model in wanted:
        if model in present or f"{model}:latest" in present:
            models.append(model)
            continue
        if not args.pull:
            log(f"  not installed, skipping: {model}")
            log(f"      install it with:  ollama pull {model}")
            continue
        log(f"  pulling {model}, this may take several minutes...")
        try:
            pulled = subprocess.run([binary, "pull", model],
                                    capture_output=True, text=True,
                                    timeout=3600, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            log(f"      pull failed to start: {error}")
            continue
        if pulled.returncode != 0:
            log(f"      pull failed: "
                f"{(pulled.stderr or '').strip()[:200]}")
            continue
        log(f"      pulled {model}")
        models.append(model)

    if not models:
        log("ABORT: no candidate models are available. Re-run with --pull, "
            "or pull them by hand using the commands above.")
        log.close()
        return 1
    log(f"models under test: {', '.join(models)}")

    log("rendering the shared page set...")
    rendered = render_pages(log)
    if not rendered:
        log("ABORT: no pages could be rendered.")
        log.close()
        return 1
    log(f"  {len(rendered)} pages rendered once, shared across all models")

    instruction = slide_vision.VISION_INSTRUCTION
    csv_path = out_dir / "bakeoff.csv"
    handle = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    handle.flush()

    totals = {}
    for model in models:
        log("-" * 68)
        log(f"model: {model}")
        answered = 0
        seconds_total = 0.0
        words_total = 0
        for deck, number, image_b64 in rendered:
            result = call_model(host, model, instruction, image_b64)
            row = {column: "" for column in CSV_COLUMNS}
            row["Model"] = model
            row["Deck"] = deck
            row["Page"] = number
            row["seconds"] = f"{result['seconds']:.2f}"
            seconds_total += result["seconds"]

            if "error" in result:
                row["answered"] = "error"
                row["done_reason"] = result["error"][:120]
                log(f"  {deck[:22]:<22} p{number:<3} ERROR "
                    f"{result['error'][:70]}")
            else:
                content = result["content"]
                words = len(content.split())
                row["done_reason"] = result["done_reason"]
                row["eval_tokens"] = result["eval_count"]
                row["think_chars"] = len(result["thinking"])
                row["reply_words"] = words
                row["reply_chars"] = len(content)
                row["answered"] = "yes" if words > 0 else "no"
                if words:
                    answered += 1
                    words_total += words
                    safe = f"{model}_{deck}_p{number:03d}".replace(
                        ":", "-").replace("/", "-")
                    path = out_dir / "replies" / f"{safe}.txt"
                    path.write_text(content, encoding="utf-8")
                    row["output_file"] = str(path)
                log(f"  {deck[:22]:<22} p{number:<3} "
                    f"{result['seconds']:>6.1f}s "
                    f"done={result['done_reason']:<7} "
                    f"tok={str(result['eval_count']):<6} "
                    f"words={words}")
            writer.writerow(row)
            handle.flush()

        totals[model] = {
            "answered": answered,
            "of": len(rendered),
            "mean_seconds": seconds_total / len(rendered),
            "mean_words": (words_total / answered) if answered else 0,
        }

    handle.close()

    log("=" * 68)
    log("Summary")
    log("=" * 68)
    log(f"{'model':<24} {'answered':>10} {'mean s/page':>12} {'mean words':>11}")
    for model, figures in totals.items():
        log(f"{model:<24} {figures['answered']:>4}/{figures['of']:<5} "
            f"{figures['mean_seconds']:>12.1f} {figures['mean_words']:>11.0f}")
    log("")
    log(f"sheet  : {csv_path}")
    log(f"replies: {out_dir / 'replies'}")
    log("")
    log("Read the answered column first. A model that does not answer every")
    log("page is not a candidate, however good its prose. Then read a few")
    log("replies side by side and fill in the Quality column by hand, because")
    log("word count cannot tell a description from a hallucination.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
