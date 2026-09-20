"""Find out why the vision model returns nothing on 44 per cent of slides.

The C3 instrumentation pass called qwen3-vl:4b once per slide across 111
pages. Forty-nine of them came back completely empty, and two more came back
cut off in mid-sentence. Every one of those fifty-one calls took almost
exactly the same time: 19.0 to 19.4 seconds for three of the decks, 23.8 to
24.2 seconds for the fourth. Successful calls ranged from 3.5 to 18 seconds.

A near-constant duration is the signature of a budget being spent, not of
content-dependent work. Something is stopping generation at a fixed point.
What is not yet known is what that budget is, and that matters because the
fix is different in each case:

  * a token cap on the reply (num_predict / max_tokens), in which case the
    cap must be raised;
  * a context window too small for the image plus the reply, which is the
    same fault already found and fixed for llama3.2, in which case num_ctx
    must be pinned for this model too;
  * the model spending its whole budget on internal reasoning tokens before
    emitting any answer, which qwen3 models do by default, in which case
    thinking must be switched off rather than the budget raised.

The third would explain the pattern best. A page that produced partial text
would be one where reasoning happened to finish early, and a page that
produced nothing would be one where it did not, which is exactly the mixture
observed. But that is a hypothesis, and guessing wrong here costs another
half hour of generation to discover.

This script settles it by asking the endpoint directly. For each page it
prints the finish reason, the completion token count, the length of any
reasoning the model returned separately, and the length of the actual reply.
`finish_reason` alone distinguishes the cases: "length" means a cap was hit,
"stop" means the model chose to end.

It then retries the same page under four variations, so the fix is chosen
from evidence rather than from the most plausible story.

Usage, from the repository root:

    python probe_vision.py --deck "..\\Evaluation corpus\\01_AI_L8_control\\CM3020 Artificial Intelligence_lecture 8-2.pptx" --pages 5,13,28

Page 5 returned nothing, page 13 returned nothing, page 28 was cut off
mid-sentence. Any mixture of known-failing and known-good pages works.
"""
import argparse
import tempfile
import time
from pathlib import Path

import config
from modules import slide_vision, slides_pptx

# Trimmed from the real instruction so a probe reply is short enough to read
# in a terminal, but the same shape: transcribe, then describe.
INSTRUCTION = slide_vision.VISION_INSTRUCTION


def reasoning_text(message):
    """Return whatever the endpoint put in a separate reasoning field.

    Different servers name this differently and older ones do not send it at
    all, so every known spelling is tried and a miss returns empty rather
    than raising. If this comes back long while the reply is empty, the
    model spent its budget thinking and the fix is to stop it thinking.
    """
    for name in ("reasoning", "reasoning_content", "thinking"):
        value = getattr(message, name, None)
        if value:
            return str(value)
    extra = getattr(message, "model_extra", None) or {}
    for name in ("reasoning", "reasoning_content", "thinking"):
        if extra.get(name):
            return str(extra[name])
    return ""


def run_variant(client, label, url, **overrides):
    """Issue one call and report what came back, without raising.

    A variant that the endpoint rejects is a result too: it means that
    control is not available here and is not the fix.
    """
    kwargs = {
        "model": config.VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": overrides.pop("prompt", INSTRUCTION)},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }],
        "temperature": 0,
    }
    kwargs.update(overrides)
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as error:
        print(f"  {label:<26} REJECTED: {str(error)[:110]}")
        return
    elapsed = time.perf_counter() - started
    choice = response.choices[0]
    message = choice.message
    content = (message.content or "").strip()
    thinking = reasoning_text(message)
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", "?")
    completion_tokens = getattr(usage, "completion_tokens", "?")
    print(f"  {label:<26} {elapsed:>6.1f}s  finish={choice.finish_reason or '?':<6} "
          f"prompt_tok={prompt_tokens:<6} completion_tok={completion_tokens:<6} "
          f"think={len(thinking):<6} reply_words={len(content.split())}")
    if content:
        print(f"      reply starts: {content[:90]!r}")
    if thinking and not content:
        print(f"      thinking starts: {thinking[:90]!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--pages", default="5",
                        help="Comma separated 1-based page numbers.")
    parser.add_argument("--dpi", type=int, default=config.VISION_DPI)
    args = parser.parse_args(argv)

    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("Activate the project venv first.")
    import fitz

    pages = [int(p) for p in args.pages.split(",") if p.strip()]
    deck = Path(args.deck)
    if not deck.is_file():
        raise SystemExit(f"Deck not found: {deck}")

    print(f"model: {config.VISION_MODEL}   endpoint: {config.OPENAI_BASE_URL}")
    print(f"deck : {deck.name}   dpi: {args.dpi}\n")

    client = OpenAI(api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL,
                    timeout=300)

    workspace = None
    pdf_path = deck
    if deck.suffix.lower() == slides_pptx.PPTX_SUFFIX:
        workspace = tempfile.TemporaryDirectory()
        print("converting to PDF with LibreOffice...")
        pdf_path = Path(slides_pptx.convert_to_pdf(str(deck), workspace.name))

    document = fitz.open(str(pdf_path))
    try:
        for number in pages:
            if number < 1 or number > len(document):
                print(f"page {number}: out of range\n")
                continue
            page = document[number - 1]
            import base64
            pixmap = page.get_pixmap(dpi=args.dpi)
            url = ("data:image/png;base64,"
                   + base64.b64encode(pixmap.tobytes("png")).decode("ascii"))
            size_kb = len(url) * 3 // 4 // 1024
            print(f"--- page {number}  (rendered {pixmap.width}x{pixmap.height}, "
                  f"about {size_kb} KB) ---")

            # A: exactly what the instrumentation pass sent.
            run_variant(client, "A baseline", url)
            # B: same, with a generous explicit reply budget. If A finished
            #    with "length" and B succeeds, the cap was the whole story.
            run_variant(client, "B max_tokens=2048", url, max_tokens=2048)
            # C: thinking switched off through the chat template. Supported
            #    by Ollama and vLLM for qwen3; a rejection here is itself an
            #    answer.
            run_variant(client, "C no thinking (template)", url,
                        extra_body={"chat_template_kwargs":
                                    {"enable_thinking": False}})
            # D: a larger context window for this model, the same fix that
            #    was needed for llama3.2. An image is expensive in tokens and
            #    the default allocation may not leave room for a reply.
            run_variant(client, "D num_ctx=16384", url,
                        extra_body={"options": {"num_ctx": 16384}})
            # E: the qwen convention of asking in the prompt itself, as a
            #    fallback if C is not supported by this server build.
            run_variant(client, "E /no_think in prompt", url,
                        prompt="/no_think\n" + INSTRUCTION)
            print()
    finally:
        document.close()
        if workspace:
            workspace.cleanup()

    print("Reading the result:")
    print("  finish=length on A  -> a budget was hit, not the model stopping.")
    print("  think large, reply empty on A -> reasoning consumed the budget;")
    print("     whichever of C or E returns a reply is the fix.")
    print("  only D fixes it -> the context window, same fault as llama3.2,")
    print("     and the fix is a Modelfile pinning num_ctx for this model.")
    print("  only B fixes it -> a plain reply cap; raise max_tokens in the")
    print("     client and accept the extra generation time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
