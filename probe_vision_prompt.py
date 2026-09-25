"""Can the replacement model be made to describe diagrams every time?

The bake-off answered the reliability question. `qwen2.5vl:7b` answered all
nine pages, every one finishing cleanly rather than at a cap, at 17 seconds a
page. The incumbent answered five.

Reading the replies raises a second question the summary table could not.
The whole justification for a vision model over Tesseract is that it can
describe a diagram, not merely transcribe the words around it. On the colour
sensitivity graph, a page with no text layer at all, `qwen2.5vl:7b` produced
a real description: the axes, the three cone response curves, the rod curve,
the wavelength range and the colour spectrum beneath. That is exactly the
capability claimed.

On the video codec slide it produced transcription only, and no description,
although that slide carries three labelled figures. The instruction asks for
both parts. The model supplied both on one page and one part on another.

So the model can describe. It does not reliably choose to. That is a
prompting and decoding problem rather than a capability problem, and this
project has already solved one of those: C1 moved the quiz format from a
request in the prompt into a JSON schema enforced at decode time, and format
validity went from 4 of 48 to 48 of 48.

This script tests whether the same move works here, on the same pages.

  P1  the current instruction, unchanged            the baseline
  P2  a firmer instruction naming both sections     is wording enough?
  P3  a JSON schema with both fields required       as used for the quiz

Ollama's native API accepts a `format` object, which is a schema enforced at
decode time rather than a request. If P3 produces a description on every page
where P1 and P2 do not, the vision path gets the same treatment as the quiz
path and one technique covers both.

Pages are chosen as two where description is essential and two where it is
not, so a schema that forces description onto a plain bulleted slide shows up
as a cost rather than hiding.

Usage, from the repository root:

    python probe_vision_prompt.py

About five minutes.
"""
import argparse
import base64
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CORPUS = Path("..") / "Evaluation corpus"

PAGES = [
    # Description essential: a labelled figure and a graph with no text layer.
    ("03_ISP_L10_image_heavy", "CM3065 Lesson 10.pdf", 9),
    ("04_GP_T7_extreme", "CM2030_GP_PPT7-CourseraWeeks13&14.pptx", 4),
    # Description not essential: ordinary bulleted slides.
    ("01_AI_L8_control", "CM3020 Artificial Intelligence_lecture 8-2.pptx", 5),
    ("02_AI_L6_mixed", "CM3020 Artificial Intelligence_lecture 6.pptx", 11),
]

# P1, the instruction currently in modules/slide_vision.py.
PROMPT_CURRENT = (
    "You are extracting the content of a single lecture slide image.\n"
    "1. Transcribe all visible text on the slide verbatim, preserving its "
    "reading order.\n"
    "2. Then describe any diagrams, figures, charts or images on the slide, "
    "including what they show and any labels or relationships.\n"
    "Do not add facts that are not present on the slide. If the slide has no "
    "diagram, transcribe the text only."
)

# P2. The escape clause in the last sentence of P1 is the suspect: a model
# can satisfy the instruction by deciding there is no diagram. This version
# removes the escape and requires the section to exist even when empty.
PROMPT_FIRM = (
    "You are extracting the content of a single lecture slide image.\n"
    "Return exactly two sections, both always present, in this order.\n\n"
    "TEXT:\n"
    "Transcribe all visible text on the slide verbatim, preserving reading "
    "order.\n\n"
    "VISUALS:\n"
    "Describe every diagram, figure, chart, photograph or illustration on "
    "the slide, including what it shows, its labels, and the relationships "
    "it depicts. If the slide genuinely contains no visual element beyond "
    "text, write exactly: none.\n\n"
    "Do not add facts that are not present on the slide."
)

# P3. The same requirement expressed as a schema rather than as a request,
# which is the C1 technique. Both fields are required, so the decoder has no
# token path to a reply that omits the description.
SCHEMA = {
    "type": "object",
    "properties": {
        "slide_text": {
            "type": "string",
            "description": "All visible text on the slide, verbatim, in "
                           "reading order.",
        },
        "visual_description": {
            "type": "string",
            "description": "What every diagram, figure, chart or image on "
                           "the slide shows, including labels and "
                           "relationships. The word none if there are none.",
        },
        "has_visual": {"type": "boolean"},
    },
    "required": ["slide_text", "visual_description", "has_visual"],
}

NUM_PREDICT = 3000
NUM_CTX = 16384


def call(host, model, prompt, image_b64, schema=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt,
                      "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT},
    }
    if schema is not None:
        payload["format"] = schema
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
            body = error.read().decode("utf-8")[:160]
        except Exception:
            pass
        return {"error": f"HTTP {error.code}: {body}",
                "seconds": time.perf_counter() - started}
    except Exception as error:
        return {"error": str(error)[:160],
                "seconds": time.perf_counter() - started}
    return {
        "content": ((reply.get("message") or {}).get("content") or "").strip(),
        "done_reason": reply.get("done_reason", "?"),
        "eval_count": reply.get("eval_count", ""),
        "seconds": time.perf_counter() - started,
    }


def described(text, schema_mode):
    """Did this reply actually contain a description, and how long is it?

    For the schema variant the answer is read from the field. For the text
    variants it is read from the section marker, falling back to looking for
    descriptive language, because a model that writes a description without
    the header has still done the work.
    """
    if schema_mode:
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return False, 0, "unparseable JSON"
        value = str(payload.get("visual_description", "")).strip()
        if not value or value.lower() in ("none", "none."):
            return False, 0, "field says none"
        return True, len(value.split()), ""
    upper = text.upper()
    for marker in ("VISUALS:", "DESCRIPTION", "2."):
        index = upper.find(marker)
        if index >= 0:
            tail = text[index + len(marker):].strip()
            if tail and tail.lower().lstrip(": ").startswith("none"):
                return False, 0, "says none"
            if len(tail.split()) >= 8:
                return True, len(tail.split()), ""
    return False, 0, "no description section"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save", default="outputs/vision_prompt_probe")
    args = parser.parse_args(argv)

    os.environ.setdefault("VISION_MODEL", args.model)
    import config
    from modules import slides_pptx
    import fitz

    host = config.OPENAI_BASE_URL.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    host = host.rstrip("/")

    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {args.model}   native: {host}/api/chat\n")
    print(f"{'page':<28} {'variant':<12} {'s':>6} {'done':<7} "
          f"{'desc':>5} {'desc words':>11}")
    print("-" * 78)

    tally = {"P1 current": 0, "P2 firm": 0, "P3 schema": 0}
    attempted = 0

    for folder, filename, number in PAGES:
        deck = CORPUS / folder / filename
        if not deck.is_file():
            print(f"{folder} p{number}: deck missing")
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
                if number > len(document):
                    continue
                pixmap = document[number - 1].get_pixmap(dpi=args.dpi)
                image_b64 = base64.b64encode(
                    pixmap.tobytes("png")).decode("ascii")
            finally:
                document.close()
        except Exception as error:
            print(f"{folder} p{number}: render failed: {str(error)[:80]}")
            continue
        finally:
            if workspace:
                workspace.cleanup()

        attempted += 1
        label = f"{folder[:20]} p{number}"
        for variant, prompt, schema in (
                ("P1 current", PROMPT_CURRENT, None),
                ("P2 firm", PROMPT_FIRM, None),
                ("P3 schema", PROMPT_FIRM, SCHEMA)):
            result = call(host, args.model, prompt, image_b64, schema)
            if "error" in result:
                print(f"{label:<28} {variant:<12} "
                      f"{result['seconds']:>6.1f} ERROR {result['error'][:40]}")
                continue
            ok, words, why = described(result["content"],
                                       schema is not None)
            if ok:
                tally[variant] += 1
            print(f"{label:<28} {variant:<12} {result['seconds']:>6.1f} "
                  f"{result['done_reason']:<7} {'yes' if ok else 'NO':>5} "
                  f"{words:>11}" + (f"   ({why})" if why else ""))
            safe = f"{label}_{variant}".replace(" ", "_").replace(":", "-")
            (save_dir / f"{safe}.txt").write_text(result["content"],
                                                  encoding="utf-8")
        print()

    print("-" * 78)
    print(f"Pages with a real description, out of {attempted}:")
    for variant, count in tally.items():
        print(f"  {variant:<12} {count}/{attempted}")
    print()
    print("If P3 is complete and P1 is not, the schema is doing the work that")
    print("the prompt could not, which is the C1 result reproduced on a")
    print("second model and a second subsystem. If P2 is already complete,")
    print("the wording was the whole problem and no schema is needed, which")
    print("is the cheaper outcome and worth reporting as such.")
    print()
    print(f"Replies saved under {save_dir} for reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
