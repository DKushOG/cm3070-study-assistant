"""Test package setup.

config.py reads its settings from the environment when it is imported, so a
shell still carrying the variables from an evaluation campaign would change
what the tests exercise. The variables are cleared here, in the package that
is imported before any test module and therefore before config.py is first
imported by the code under test.

A test that wants a non-default setting passes it as an argument instead.
"""
import os

# Every variable config.py reads. Cleared unconditionally: the tests assert
# default behaviour, and anything that needs a different value passes it in.
_AMBIENT_SETTINGS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "WHISPER_MODEL",
    "VISION_MODEL",
    "VISION_DPI",
    "VISION_MAX_PAGES",
    "VISION_NUM_PREDICT",
    "VISION_MAX_EMPTY_RETRIES",
    "SLIDE_ROUTING",
    "SLIDE_ROUTING_PICTURE_THRESHOLD",
    "SLIDE_ROUTING_WORDS_THRESHOLD",
    "LLM_TEMPERATURE",
    "LLM_SEED",
    "LLM_TIMEOUT",
    "QUIZ_MAX_ATTEMPTS",
    "QUIZ_STRUCTURED",
    "MAX_CONTEXT_WORDS",
    "OUTPUT_DIR",
)

for _name in _AMBIENT_SETTINGS:
    os.environ.pop(_name, None)

# Clearing the environment is necessary but not sufficient. config.py calls
# load_dotenv(), which would read .env and put every one of those variables
# straight back, so the suite would silently test whatever the author's .env
# happened to contain. Setting LLM_TEMPERATURE=0 there, which the evaluation
# campaigns needed, is enough on its own to change what the reproducibility
# tests exercise. This flag tells config.py to skip the file entirely.
os.environ["CM3070_DISABLE_DOTENV"] = "1"
