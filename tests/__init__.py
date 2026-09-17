"""Test package initialisation.

This module exists to make the suite hermetic. config.py reads its settings
from the environment at import time, so a shell that still carries the
variables from an evaluation campaign (QUIZ_STRUCTURED, QUIZ_MAX_ATTEMPTS,
LLM_TEMPERATURE and so on) would silently change what the tests exercise. A
suite whose result depends on ambient shell state is not a suite, and a
reviewer cloning the repository must get the same outcome as the author.

The variables are cleared here rather than inside individual tests because
this package is imported before any test module, and therefore before
config.py is first imported by the code under test. Tests that want a
non-default setting pass it explicitly as an argument instead.
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
