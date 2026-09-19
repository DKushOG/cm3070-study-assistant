# Pinning the context window

## The problem

`ollama show llama3.2` reports a context length of 131072. That is the
architecture's maximum, not what is allocated. The Parameters block in that
same output lists only stop tokens, with **no num_ctx**, which means nothing in
the modelfile pins it and the server default applies.

When a prompt exceeds the allocated window, Ollama discards the **front** of
the prompt to make it fit. Nothing is raised and nothing is logged to the
client. The earliest source block simply never reaches the model.

The context builder assembles sources in a fixed order: transcript, then slide
text, then student notes. So under truncation it is always the transcript that
is eaten first, which would make the transcript-bearing modes quietly
degenerate into slide-only modes while still being labelled as combined.

## Why it matters now rather than before

| Dataset | Assembled words | Approximate tokens |
|---|---|---|
| CM3060 (existing) | about 2,900 | about 3,800 |
| Deck 01, AI L8 | about 3,000 | about 3,900 |
| Deck 02, AI L6 | about 4,500 | **about 5,900** |
| Deck 03, ISP L10 | about 2,800 | about 3,600 |

Deck 02 is half again larger than anything run so far. If the allocated window
is 2048 or 4096, the existing results sat near the edge and deck 02 falls off
it. The four decks would then not be comparable with each other, which is the
entire point of collecting them.

## The fix

Pin it in a derived model rather than relying on an environment variable. A
derived model is self-documenting, travels with the repository, and cannot be
lost from a shell that was restarted.

Create a file called `Modelfile` in the project root:

    FROM llama3.2
    PARAMETER num_ctx 16384

Then build and point the project at it:

    ollama create llama3.2-ctx16k -f Modelfile
    ollama show llama3.2-ctx16k

The Parameters block should now list `num_ctx 16384`. Set the model in `.env`:

    OPENAI_MODEL=llama3.2-ctx16k

16384 was chosen with headroom rather than to fit exactly. The quiz call is the
larger of the two, because it receives the assembled context **plus** the
generated notes, so the worst case is roughly 5,900 plus about 800 for the
notes, near 6,700 tokens. Doubling that leaves room for a longer deck later
without revisiting the decision.

## What has to be re-run, and what does not

**Re-run:** every generation campaign, meaning notes and quiz. That includes
the CM3060 structured and legacy results, so the whole comparison sits on one
setting. This is cheap, the full structured 2x2 took about six minutes.

**Do not re-run:** anything to do with extraction. Transcription and slide
extraction never touch the text model, so word error rates and the extractor
comparison are unaffected by this and stay valid.

## Worth reporting

This belongs in the implementation chapter. A model declaring a 131,072 token
maximum while the runtime silently allocates a fraction of it, and truncates
from the front without warning, is exactly the class of gap between stated
capability and measured behaviour the evaluation already discusses. Finding it
before the multi-deck campaign rather than after is the kind of check that
distinguishes a measured result from a lucky one.
