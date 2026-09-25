"""Central configuration for the study assistant.

Every setting comes from an environment variable, with defaults for a local
Ollama install, so the same code runs against a local model or a hosted
OpenAI compatible endpoint with no edits.

LLM_TEMPERATURE, LLM_SEED and MAX_CONTEXT_WORDS are unset by default. An
unset control is left out of the request, so the default configuration makes
the same request as the version that was evaluated.
"""
import os

# The test package sets this before importing anything. Clearing the
# environment is not enough on its own, because load_dotenv() would read .env
# and put the variables straight back, so a developer whose .env sets
# LLM_TEMPERATURE would run a different suite from anyone else. Checked here
# because this is the only place the file is read.
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

    An unset or blank variable returns the default, so a setting can be left
    out altogether rather than always carrying a value.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name, default=None):
    """Parse an optional integer environment variable, as _get_float does."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.2")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Vision-language model for reading slide images. Changed in the .env file
# like every other model here, so comparing extractors is a setting change
# rather than a rewrite. The default is the model chosen by the candidate
# comparison. The earlier default, qwen3-vl:4b, was dropped because it reasons
# before answering, ignored the settings that should turn that off, and
# returned nothing for some slides on some runs.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

# Slide-rendering controls for the vision path. DPI trades render quality
# against speed, and the page cap limits how many model calls one deck can
# make. The cap used to be 20, which dropped the end of any longer deck
# without saying so, and now sits well above the length of a real lecture.
VISION_DPI = _get_int("VISION_DPI", 150)
VISION_MAX_PAGES = _get_int("VISION_MAX_PAGES", 200)

# Upper limit on the tokens one slide extraction may generate. A vision model
# that loses its way on a dense slide can otherwise keep going until the
# context window runs out. One page was measured at 14,641 tokens and nearly
# three minutes, and returned nothing. Slides that extract properly finish well
# under 1,000 tokens, so this limit is generous and turns a runaway into a
# quick failure instead of a stall. Set to 0 or blank for no cap.
VISION_NUM_PREDICT = _get_int("VISION_NUM_PREDICT", 3000)

# How many times a slide that comes back empty or cut short is retried before
# it is reported as unreadable. One retry costs little and recovers a page that
# failed for a passing reason. Set to 0 to switch retrying off.
VISION_MAX_EMPTY_RETRIES = _get_int("VISION_MAX_EMPTY_RETRIES", 1)

# Per-slide extraction routing. The interface offers routing as its first and
# default method and the evaluation harnesses call it directly, so this flag
# does not switch routing on or off anywhere. It is kept so an existing .env
# that sets it still loads. The whole-deck vision method stays available beside
# routing as the comparison condition.
SLIDE_ROUTING = os.getenv("SLIDE_ROUTING", "").strip().lower() in (
    "1", "true", "yes", "on")

# The two routing thresholds, exposed so an ablation can vary them without
# editing code. The defaults live in modules/slide_routing.py beside the
# reasoning for them, and are used when these are unset.
SLIDE_ROUTING_PICTURE_THRESHOLD = _get_float("SLIDE_ROUTING_PICTURE_THRESHOLD")
SLIDE_ROUTING_WORDS_THRESHOLD = _get_int("SLIDE_ROUTING_WORDS_THRESHOLD")

# Sampling controls, both unset by default. When unset they are not sent to
# the API, so generation behaves as it did in the evaluated version and the
# earlier baseline results stay comparable. Set them, for example
# LLM_TEMPERATURE=0 with a fixed LLM_SEED, to make a run repeatable when
# separating an input-mode effect from run-to-run sampling noise.
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE")
LLM_SEED = _get_int("LLM_SEED")

# Request timeout in seconds for one model call. The default of 600.0 matches
# the OpenAI client's own default, so passing it explicitly changes nothing.
# Lower it to fail faster on a stalled endpoint.
LLM_TIMEOUT = _get_float("LLM_TIMEOUT", 600.0)

# Maximum attempts for the quiz call. The default of 1 generates once with no
# retry, which is the behaviour that was evaluated. Raising it regenerates a
# quiz that fails format validation, using the same prompt. Validation runs
# either way, so the failure rate is recorded even on the default setting.
QUIZ_MAX_ATTEMPTS = _get_int("QUIZ_MAX_ATTEMPTS", 1)

# Structured quiz generation. Off by default so the original prompt-and-check
# path stays the reference condition for the comparison and an existing setup
# keeps the behaviour that was evaluated. When on, the quiz call is made under
# a JSON schema, so the four fields are required by the format rather than
# only asked for in the prompt.
QUIZ_STRUCTURED = os.getenv("QUIZ_STRUCTURED", "").strip().lower() in (
    "1", "true", "yes", "on")

# The interface is the finished system, so it uses structured generation
# unless QUIZ_STRUCTURED is set to an off value. The batch runner above keeps
# the original path as its default, so the reference condition does not move.
# The user can still switch the interface back from the sidebar.
QUIZ_STRUCTURED_IN_APP = os.getenv("QUIZ_STRUCTURED", "1").strip().lower() in (
    "1", "true", "yes", "on")

# Optional per-source word cap for the prompt context. Unset by default, which
# means no limit. When set, each source is shortened on its own to at most this
# many words, so a long transcript does not fill the model's context window on
# its own. Any shortening is reported.
MAX_CONTEXT_WORDS = _get_int("MAX_CONTEXT_WORDS")
