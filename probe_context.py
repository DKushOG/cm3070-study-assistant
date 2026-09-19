"""Find the context window Ollama actually allocates, as opposed to the one
the model declares.

`ollama show llama3.2` reports a context length of 131072. That is the
architecture's maximum. Ollama allocates its own default at load time unless a
num_ctx is set, and when a prompt exceeds it the **beginning** of the prompt is
discarded silently. No error is raised and no warning is printed, so an
oversized context simply loses its earliest content.

This matters here because the assembled context for the largest deck is around
6,000 tokens. If the allocated window is smaller than that, the earliest source
block is dropped before the model ever sees it, and any comparison between
decks measures truncation rather than input mode.

Method: place a distinctive codeword at the very start of the prompt, pad to a
target length, then ask for the codeword at the end. If the window is large
enough the model can answer. If the prompt has been truncated from the front,
the codeword is gone and it cannot. Testing at increasing lengths brackets the
real window.

Usage:  python probe_context.py
"""
import time

# Read the model the project is actually configured to use rather than
# hard-coding one. Probing a different model from the one the pipeline calls
# measures nothing, which is a mistake this script made on its first run.
try:
    import config
    MODEL = config.OPENAI_MODEL
    BASE_URL = config.OPENAI_BASE_URL
except Exception:
    MODEL = "llama3.2"
    BASE_URL = "http://localhost:11434/v1"
NEEDLE = "ZEPHYR-7731"

# Word counts to test. Tokens run roughly 1.3 times words for English prose.
TARGETS = [400, 1500, 3000, 5000, 8000]

FILLER = (
    "The lecture continued with a discussion of routine administrative "
    "matters and the usual reminders about deadlines and reading lists. "
)


def build_prompt(words):
    head = (f"IMPORTANT: The secret codeword is {NEEDLE}. "
            "Remember it, you will be asked for it at the end.\n\n")
    body = []
    count = 0
    while count < words:
        body.append(FILLER)
        count += len(FILLER.split())
    tail = ("\n\nWhat was the secret codeword given at the very start of "
            "this message? Reply with the codeword alone and nothing else.")
    return head + "".join(body) + tail


def main():
    try:
        from openai import OpenAI
    except ImportError:
        print("Activate the project venv first.")
        return

    client = OpenAI(api_key="ollama", base_url=BASE_URL, timeout=300)

    print(f"probing model: {MODEL}  at {BASE_URL}\n")
    print(f"{'words':>7} {'~tokens':>8} {'recalled':>10}  reply")
    print("-" * 64)
    last_good = None
    first_bad = None
    for words in TARGETS:
        prompt = build_prompt(words)
        actual = len(prompt.split())
        try:
            started = time.time()
            reply = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = (reply.choices[0].message.content or "").strip()
            ok = NEEDLE in text.upper()
            if ok:
                last_good = actual
            elif first_bad is None:
                first_bad = actual
            print(f"{actual:>7} {int(actual*1.3):>8} {'yes' if ok else 'NO':>10}"
                  f"  {text[:40]!r}  ({time.time()-started:.0f}s)")
        except Exception as error:
            print(f"{actual:>7} {int(actual*1.3):>8} {'ERROR':>10}  {error}")
            break

    print()
    if first_bad is None:
        print("The codeword survived at every length tested. The allocated")
        print("window is at least", int(TARGETS[-1] * 1.3), "tokens, which is")
        print("comfortably above the largest deck. No change needed.")
    else:
        print(f"Recall held to about {last_good} words and failed by "
              f"{first_bad} words.")
        print(f"That places the allocated window near "
              f"{int((last_good or 0)*1.3)} tokens, well below the 131072 the")
        print("model declares. The largest deck would be truncated silently.")
        print()
        print("Fix it by pinning num_ctx explicitly. See PIN_CONTEXT.md.")


if __name__ == "__main__":
    main()
