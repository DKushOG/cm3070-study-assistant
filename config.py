"""Central configuration for the study assistant.

All settings come from environment variables with sensible defaults for a
free local Ollama setup, so the same code runs against a local model or a
hosted OpenAI compatible endpoint without modification.

The reproducibility and long-input controls (LLM_TEMPERATURE, LLM_SEED and
MAX_CONTEXT_WORDS) are all unset by default, so the default configuration
reproduces the exact behaviour evaluated in the Preliminary Project Report.
"""
import os

# The test package sets this before importing anything, because clearing the
# environment is not enough on its own: load_dotenv() would read .env and put
# the cleared variables straight back, so a developer whose .env sets
# LLM_TEMPERATURE would run a different suite from a reviewer whose .env does
# not. Honoured here rather than in the tests because this is the only place
# the file is read.
if os.getenv("CM3070_DISABLE_DOTENV", "").strip().lower() not in (
        "1", "true", "yes", "on"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv is convenient but not required. Plain environment
        # variables work the same way.
        pass


def _get_float(name, default=None):
    """Parse an optional float environment variable.

    An unset or blank variable returns the default, so a setting can be
    genuinely optional rather than always carrying a value. This is what
    lets the reproducibility controls default to "not set at all".
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name, default=None):
    """Parse an optional integer environment variable (see _get_float)."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.2")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Vision-language model for slide-image extraction, the project's third
# pre-trained model (image data space). Swappable through the .env file like
# every other model here, so the extractor comparison in the Evaluation
# chapter is a configuration change rather than a rewrite. Default is a small
# local Ollama VLM that reads slide text and describes diagrams in one pass.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:4b")

# Slide-rendering controls for the vision path. DPI trades render quality
# against speed; the page cap stops a long deck from running away with one
# model call per page. Both are overridable from the environment.
VISION_DPI = _get_int("VISION_DPI", 150)
VISION_MAX_PAGES = _get_int("VISION_MAX_PAGES", 20)

# Upper bound on the tokens one slide extraction may generate. A vision model
# that loses its way on a dense slide can otherwise generate until it exhausts
# the context window, which was measured at up to 14641 tokens and nearly
# three minutes for a single page, returning nothing at the end of it. Slides
# that extract successfully finish well under 1000 tokens, so this is generous
# while still turning a runaway into a fast, detectable failure rather than a
# stall. Set to 0 or blank to impose no cap.
VISION_NUM_PREDICT = _get_int("VISION_NUM_PREDICT", 3000)

# How many times a slide that comes back empty or truncated is retried before
# it is reported as unreadable. One retry costs little and recovers a page
# that failed for a transient reason. Set to 0 to disable retrying.
VISION_MAX_EMPTY_RETRIES = _get_int("VISION_MAX_EMPTY_RETRIES", 1)

# Per-slide extraction routing. Off by default so the whole-deck behaviour
# stays the reference condition for the before-and-after comparison, exactly
# as QUIZ_STRUCTURED does for the quiz path. When on, each slide is sent to
# the vision model only when the routing rule judges it worth the call.
SLIDE_ROUTING = os.getenv("SLIDE_ROUTING", "").strip().lower() in (
    "1", "true", "yes", "on")

# The two routing thresholds, exposed so an ablation can vary them without
# editing code. Their defaults live in modules/slide_routing.py alongside the
# evidence for them, and are used when these are unset.
SLIDE_ROUTING_PICTURE_THRESHOLD = _get_float("SLIDE_ROUTING_PICTURE_THRESHOLD")
SLIDE_ROUTING_WORDS_THRESHOLD = _get_int("SLIDE_ROUTING_WORDS_THRESHOLD")

# Reproducibility controls, both unset by default. When unset they are not
# sent to the API at all, so generation reproduces the exact sampling
# behaviour of the evaluated prototype and existing baseline results stay
# valid. Set them (e.g. LLM_TEMPERATURE=0 with a fixed LLM_SEED) to make a
# run repeatable when isolating an input-mode effect from run-to-run
# sampling noise in the evaluation.
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE")
LLM_SEED = _get_int("LLM_SEED")

# Request timeout in seconds for a single model call. Defaults to 600.0,
# which matches the OpenAI client's own default, so behaviour is unchanged
# from before a timeout was passed explicitly. Lower it to fail faster on a
# stalled endpoint.
LLM_TIMEOUT = _get_float("LLM_TIMEOUT", 600.0)

# Maximum attempts for the quiz call. Defaults to 1, meaning generate once
# and never retry, which is exactly the behaviour evaluated in the
# Preliminary Project Report. Raising it lets a quiz that fails format
# validation be regenerated with the identical unchanged prompt. Validation
# itself always runs, so the failure rate is measured even on the default
# setting.
QUIZ_MAX_ATTEMPTS = _get_int("QUIZ_MAX_ATTEMPTS", 1)

# Structured quiz generation. Off by default so the legacy prompt-and-check
# path stays the reference condition for the before-and-after comparison, and
# so an existing deployment keeps the behaviour that was evaluated. When on,
# the quiz call is issued under a JSON schema that makes the four required
# fields structurally mandatory rather than merely requested.
QUIZ_STRUCTURED = os.getenv("QUIZ_STRUCTURED", "").strip().lower() in (
    "1", "true", "yes", "on")

# Optional per-source word cap for the prompt context. Unset by default,
# meaning no limit and today's exact behaviour. When set, each source
# (transcript, slide text, notes) is independently truncated to at most this
# many words so a long lecture transcript cannot overrun the local model's
# context window. Truncation is always reported, never silent.
MAX_CONTEXT_WORDS = _get_int("MAX_CONTEXT_WORDS")
