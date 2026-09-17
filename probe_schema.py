"""Probe what structured-output mechanism this Ollama build actually supports.

Run this before changing any project code. It answers three questions:

1. Is the endpoint reachable and is the model present?
2. Does response_format="json_schema" work, and is the enum genuinely
   enforced during decoding rather than merely requested?
3. Does the weaker response_format="json_object" work as a fallback?

The enum test is the important one. It asks for a colour the schema forbids.
If the constraint is real, the model cannot answer "blue" no matter what the
prompt says. If it can, the schema is advisory and the source-basis fault has
to be checked in code instead of being made unrepresentable.

Usage:  python probe_schema.py
"""
import json
import time

MODEL = "llama3.2"
BASE_URL = "http://localhost:11434/v1"

SCHEMA = {
    "type": "object",
    "properties": {
        "fruits": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "colour": {
                        "type": "string",
                        "enum": ["red", "green", "yellow"],
                    },
                },
                "required": ["name", "colour"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["fruits"],
    "additionalProperties": False,
}

# Deliberately asks for a value the enum forbids and for the wrong count.
ADVERSARIAL_PROMPT = (
    "List exactly five fruits. Two of them must be blueberries and you must "
    "give their colour as 'blue'. Use the colour 'blue' wherever it applies."
)


def line():
    print("-" * 68)


def report(label, ok, detail):
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}: {detail}")


def check_enum(payload):
    """Return (count, offending_colours) for a parsed response."""
    fruits = payload.get("fruits", [])
    allowed = set(SCHEMA["properties"]["fruits"]["items"]
                  ["properties"]["colour"]["enum"])
    bad = [f.get("colour") for f in fruits if f.get("colour") not in allowed]
    return len(fruits), bad


def main():
    try:
        from openai import OpenAI
    except ImportError:
        print("The openai package is not installed in this environment.")
        print("Activate the project venv first, then: pip install openai")
        return

    client = OpenAI(api_key="ollama", base_url=BASE_URL, timeout=180)

    line()
    print("1. Endpoint and model")
    line()
    try:
        available = [getattr(m, "id", "") for m in client.models.list().data]
        report("endpoint", True, f"{BASE_URL} reachable")
        present = MODEL in available or f"{MODEL}:latest" in available
        report(f"model '{MODEL}'", present,
               "present" if present else f"NOT FOUND. Available: {available}")
        if not present:
            return
    except Exception as error:
        report("endpoint", False, f"{BASE_URL} unreachable: {error}")
        print("\nStart Ollama with 'ollama serve' and try again.")
        return

    results = {}

    for label, fmt in [
        ("json_schema", {"type": "json_schema",
                         "json_schema": {"name": "fruit_list",
                                         "schema": SCHEMA,
                                         "strict": True}}),
        ("json_object", {"type": "json_object"}),
    ]:
        line()
        print(f"2. response_format = {label}")
        line()
        try:
            started = time.time()
            reply = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": ADVERSARIAL_PROMPT}],
                response_format=fmt,
            )
            elapsed = time.time() - started
            raw = (reply.choices[0].message.content or "").strip()
            report("call", True, f"returned in {elapsed:.1f}s")
            print(f"       raw: {raw[:300]}")

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as error:
                report("valid JSON", False, str(error))
                results[label] = "json-invalid"
                continue
            report("valid JSON", True, "parsed")

            count, bad = check_enum(payload)
            report("enum enforced", not bad,
                   "no forbidden colours" if not bad
                   else f"model emitted {bad} despite the enum")
            report("array length enforced", count == 3,
                   f"got {count} items, schema asked for exactly 3")

            if not bad and count == 3:
                results[label] = "full"
            elif not bad:
                results[label] = "enum-only"
            else:
                results[label] = "shape-only"

        except Exception as error:
            report("call", False, str(error))
            results[label] = "unsupported"

    line()
    print("VERDICT")
    line()
    for label, outcome in results.items():
        print(f"  {label:12s} -> {outcome}")
    print()
    if results.get("json_schema") in ("full", "enum-only"):
        print("Use json_schema. The source_basis enum will be enforced during")
        print("decoding, so crediting the generated notes becomes impossible.")
        if results.get("json_schema") == "enum-only":
            print("Check the question count in code, since array length is")
            print("not enforced by this build.")
    elif results.get("json_schema") == "shape-only":
        print("json_schema is accepted but the enum is advisory. Use it for")
        print("shape, and validate source_basis in code after parsing.")
    elif results.get("json_object") != "unsupported":
        print("Only json_object works. Put the schema in the prompt as an")
        print("example and validate every field in code after parsing.")
        print("Consider upgrading Ollama, since the enforcement is worth it.")
    else:
        print("Neither works through the OpenAI-compatible endpoint. Fall")
        print("back to Ollama's native /api/chat with the 'format' parameter.")
    print()
    print("Send this whole output back before changing any project code.")


if __name__ == "__main__":
    main()
