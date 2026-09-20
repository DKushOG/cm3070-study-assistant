"""Can the vision model's reasoning be switched off, and if not, bounded?

Where this stands. Pinning num_ctx to 16384 worked exactly as intended: the
ceiling moved from 4096 to 16384 and two of the four gate pages, which
previously returned nothing, now answer normally. Deck 01 page 5 used 3093
tokens and returned 141 words, page 28 used 2192 and returned 236.

But two pages did not. Deck 01 page 13 generated 14081 completion tokens and
deck 04 page 6 generated 14641, both stopping at finish_reason "length" with
an empty reply. 2303 + 14081 is 16384. The new ceiling is binding in exactly
the way the old one was.

That changes the diagnosis. Fourteen thousand tokens of reasoning about a
single lecture slide is not a window that is slightly too small, it is the
model looping. Raising the window again would buy longer loops and calls of
three minutes or more, not answers. The remedy has to be to stop the model
reasoning, or failing that to stop it wasting three minutes discovering that
it will not finish.

The earlier probe established that neither `chat_template_kwargs` with
`enable_thinking` false nor the `/no_think` prompt convention has any effect
through Ollama's OpenAI compatible endpoint. What that probe did not test is
Ollama's **own** API, which takes a top level `think` field the compatibility
shim does not expose, and an `options` block the shim ignores.

This script tests that directly, on the two pages that actually fail, so the
answer comes from the hard cases rather than from an easy one.

Variants:

  N1  native API, think disabled            the hoped-for fix
  N2  native API, think enabled             a control, to show N1 changed something
  N3  native API, think disabled, capped    is a reply cap honoured natively
  N4  native API, think enabled, capped     bounds a runaway to a known cost
  O1  OpenAI endpoint, reasoning_effort     whether the shim exposes any control

Nothing here is destructive and no results are overwritten. Takes a few
minutes, most of it spent waiting for the runaway cases to give up.

Usage, from the repository root:

    python probe_vision_native.py

Add --pages or --deck to test others. The defaults are the two known failures.
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

# The two pages that defeated the 16384 token window.
DEFAULT_TARGETS = [
    ("01_AI_L8_control", "CM3020 Artificial Intelligence_lecture 8-2.pptx", 13),
    ("04_GP_T7_extreme", "CM2030_GP_PPT7-CourseraWeeks13&14.pptx", 6),
]


def post(url, payload, timeout=400):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def native_call(host, model, instruction, image_b64, label,
                think=None, num_predict=None, num_ctx=16384):
    """One call to Ollama's own /api/chat, reporting what came back.

    Ollama's native API takes `think` as a top level boolean and honours an
    `options` block. Neither is reachable through the OpenAI compatible
    endpoint, which is the reason for testing here at all.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": instruction,
                      "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    if think is not None:
        payload["think"] = think
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    started = time.perf_counter()
    try:
        reply = post(f"{host}/api/chat", payload)
    except urllib.error.HTTPError as error:
        body = ""
        try:
            body = error.read().decode("utf-8")[:160]
        except Exception:
            pass
        print(f"  {label:<34} REJECTED {error.code}: {body}")
        return None
    except Exception as error:
        print(f"  {label:<34} FAILED: {str(error)[:140]}")
        return None
    took = time.perf_counter() - started

    message = reply.get("message") or {}
    content = (message.get("content") or "").strip()
    thinking = (message.get("thinking") or "")
    print(f"  {label:<34} {took:>6.1f}s  done={reply.get('done_reason','?'):<7}"
          f" eval={reply.get('eval_count','?'):<6}"
          f" think_chars={len(thinking):<6} reply_words={len(content.split())}")
    if content:
        print(f"       reply starts: {content[:80]!r}")
    return reply


def openai_call(model, instruction, data_url, label, **extra):
    """One call through the OpenAI compatible endpoint, for comparison."""
    try:
        from openai import OpenAI
    except ImportError:
        print(f"  {label:<34} SKIPPED: openai package not importable")
        return
    import config
    client = OpenAI(api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL, timeout=400)
    started = time.perf_counter()
    try:
        reply = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": data_url}}]}],
            temperature=0, **extra)
    except Exception as error:
        print(f"  {label:<34} REJECTED: {str(error)[:140]}")
        return
    took = time.perf_counter() - started
    choice = reply.choices[0]
    content = (choice.message.content or "").strip()
    usage = getattr(reply, "usage", None)
    print(f"  {label:<34} {took:>6.1f}s  finish={choice.finish_reason or '?':<7}"
          f" eval={getattr(usage,'completion_tokens','?'):<6}"
          f" think_chars={'?':<6} reply_words={len(content.split())}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Defaults to VISION_MODEL from the configuration.")
    parser.add_argument("--deck", default=None)
    parser.add_argument("--pages", default=None,
                        help="Comma separated, used with --deck.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    os.environ.setdefault("VISION_MODEL", "qwen3-vl-ctx16k")
    import config
    from modules import slide_vision, slides_pptx
    import fitz

    model = args.model or config.VISION_MODEL
    host = config.OPENAI_BASE_URL.rstrip("/")
    if host.endswith("/v1"):
        host = host[:-3]
    host = host.rstrip("/")

    print(f"model   : {model}")
    print(f"native  : {host}/api/chat")
    print(f"openai  : {config.OPENAI_BASE_URL}")
    try:
        print(f"ollama  : version {get(host + '/api/version').get('version')}")
    except Exception as error:
        print(f"ollama  : could not read version ({str(error)[:80]})")
    print()

    if args.deck:
        targets = [(Path(args.deck).stem, None, int(p))
                   for p in (args.pages or "1").split(",") if p.strip()]
        deck_paths = {Path(args.deck).stem: Path(args.deck)}
    else:
        targets = DEFAULT_TARGETS
        deck_paths = {folder: CORPUS / folder / filename
                      for folder, filename, _ in DEFAULT_TARGETS}

    instruction = slide_vision.VISION_INSTRUCTION

    for entry in targets:
        label, _filename, page_number = entry
        deck = deck_paths[label]
        if not Path(deck).is_file():
            print(f"deck missing: {deck}\n")
            continue

        workspace = None
        pdf_path = Path(deck)
        if pdf_path.suffix.lower() == slides_pptx.PPTX_SUFFIX:
            workspace = tempfile.TemporaryDirectory()
            pdf_path = Path(slides_pptx.convert_to_pdf(str(deck),
                                                       workspace.name))
        document = fitz.open(str(pdf_path))
        try:
            if page_number > len(document):
                print(f"{label} page {page_number}: out of range\n")
                continue
            pixmap = document[page_number - 1].get_pixmap(dpi=args.dpi)
            raw = pixmap.tobytes("png")
            image_b64 = base64.b64encode(raw).decode("ascii")
            data_url = "data:image/png;base64," + image_b64

            print(f"--- {label} page {page_number} "
                  f"({pixmap.width}x{pixmap.height}) ---")
            native_call(host, model, instruction, image_b64,
                        "N1 native, think off", think=False)
            native_call(host, model, instruction, image_b64,
                        "N2 native, think on", think=True)
            native_call(host, model, instruction, image_b64,
                        "N3 native, think off, cap 1500",
                        think=False, num_predict=1500)
            native_call(host, model, instruction, image_b64,
                        "N4 native, think on, cap 3000",
                        think=True, num_predict=3000)
            openai_call(model, instruction, data_url,
                        "O1 openai, reasoning_effort low",
                        reasoning_effort="low")
            print()
        finally:
            document.close()
            if workspace:
                workspace.cleanup()

    print("Reading the result:")
    print("  N1 answers and N2 does not -> thinking is the cause and the")
    print("     native API can switch it off. The vision path moves to it.")
    print("  N1 rejected -> this Ollama build predates the think field;")
    print("     upgrading Ollama is then the cheapest route.")
    print("  N1 still empty -> the model cannot do this page at all, and the")
    print("     answer is a different vision model, not another setting.")
    print("  N3 or N4 answering -> a reply cap is honoured natively, which at")
    print("     minimum bounds a runaway to a known cost instead of 3 minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
