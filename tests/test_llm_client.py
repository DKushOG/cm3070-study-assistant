"""Tests for the model client and its sampling settings.

Part of the model client group. No test here makes a network call.
"""
import unittest

import config
from core.llm_client import LLMClient


class RequestKwargsTests(unittest.TestCase):
    """Temperature and seed are sent only when they are set.

        Zero is a real temperature and is sent, while None is left out.

    """

    def test_temperature_and_seed_omitted_when_unset(self):
        client = LLMClient(model="m")
        # Force the unset state regardless of any .env on the dev machine.
        client.temperature = None
        client.seed = None
        kwargs = client._request_kwargs("hello")
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("seed", kwargs)
        self.assertEqual(kwargs["model"], "m")
        self.assertEqual(kwargs["messages"],
                         [{"role": "user", "content": "hello"}])

    def test_temperature_and_seed_passed_when_set(self):
        client = LLMClient(temperature=0.7, seed=42)
        kwargs = client._request_kwargs("hi")
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["seed"], 42)

    def test_temperature_zero_is_still_sent(self):
        """0 is a meaningful value (greedy decoding), not 'unset', so it must
        be passed through while an unset seed is still omitted."""
        client = LLMClient(temperature=0)
        client.seed = None
        kwargs = client._request_kwargs("hi")
        self.assertIn("temperature", kwargs)
        self.assertEqual(kwargs["temperature"], 0)
        self.assertNotIn("seed", kwargs)


class ChatRequestIdentityTests(unittest.TestCase):
    """A default request matches the one the earlier prototype sent."""

    def test_chat_request_is_byte_identical_to_baseline(self):
        """Guards the constraint that adding the vision path must not change
        the request chat() sends. With no settings, the request is exactly
        the two-key text request the evaluated prototype used."""
        client = LLMClient(model="llama3.2")
        client.temperature = None
        client.seed = None
        self.assertEqual(client._request_kwargs("hello"), {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "hello"}],
        })


class ImageRequestKwargsTests(unittest.TestCase):
    """A multimodal request has the right shape and the same settings rule."""

    def test_multimodal_content_structure(self):
        client = LLMClient(model="qwen3-vl:4b")
        client.temperature = None
        client.seed = None
        kwargs = client._image_request_kwargs(
            "describe",
            ["data:image/png;base64,AAA", "data:image/png;base64,BBB"])
        self.assertEqual(kwargs["model"], "qwen3-vl:4b")
        content = kwargs["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "describe"})
        self.assertEqual(content[1], {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAA"}})
        self.assertEqual(content[2]["image_url"]["url"],
                         "data:image/png;base64,BBB")
        self.assertEqual(len(content), 3)

    def test_image_request_honours_settings_when_set(self):
        client = LLMClient(temperature=0, seed=5)
        kwargs = client._image_request_kwargs(
            "x", ["data:image/png;base64,AAA"])
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(kwargs["seed"], 5)

    def test_image_request_omits_settings_when_unset(self):
        client = LLMClient()
        client.temperature = None
        client.seed = None
        kwargs = client._image_request_kwargs(
            "x", ["data:image/png;base64,AAA"])
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("seed", kwargs)


class PreflightTests(unittest.TestCase):
    """The availability check returns a message rather than raising.

        An offline machine or a missing package has to give a clear message,
        not an error part way through a long extraction.

    """

    class _FakeModels:
        def __init__(self, ids):
            self._ids = ids

        def list(self):
            data = [type("Model", (), {"id": name})() for name in self._ids]
            return type("Page", (), {"data": data})()

    def _client_with(self, ids):
        client = LLMClient(model="qwen3-vl:4b")
        client._client = type(
            "FakeOpenAI", (),
            {"models": self._FakeModels(ids),
             "with_options": lambda self, **kwargs: self})()
        return client

    def test_available_model_reports_ok(self):
        ok, message = self._client_with(["llama3.2", "qwen3-vl:4b"]).check_model_available()
        self.assertTrue(ok)
        self.assertIn("qwen3-vl:4b", message)

    def test_missing_model_names_the_pull_command(self):
        ok, message = self._client_with(["llama3.2"]).check_model_available()
        self.assertFalse(ok)
        self.assertIn("ollama pull qwen3-vl:4b", message)

    def test_untagged_model_matches_its_latest_tag(self):
        client = LLMClient(model="llama3.2")
        client._client = type(
            "FakeOpenAI", (),
            {"models": self._FakeModels(["llama3.2:latest"]),
             "with_options": lambda self, **kwargs: self})()
        ok, _message = client.check_model_available()
        self.assertTrue(ok)

    def test_unreachable_endpoint_degrades_gracefully(self):
        client = LLMClient(model="qwen3-vl:4b")

        class Boom:
            def list(self):
                raise OSError("connection refused")

        client._client = type(
            "FakeOpenAI", (),
            {"models": Boom(),
             "with_options": lambda self, **kwargs: self})()
        ok, message = client.check_model_available()
        self.assertFalse(ok)
        self.assertIn("Could not reach", message)
        self.assertIn("ollama serve", message)


class ClientLifecycleTests(unittest.TestCase):
    """The OpenAI client is built once per instance and honours the timeout."""

    def test_openai_client_is_built_once_per_instance(self):
        """A cached client is reused rather than rebuilt per call. Setting the
        cache to a sentinel proves _get_client returns it without reimporting
        or reconstructing, which is the once-per-instance behaviour."""
        client = LLMClient()
        sentinel = object()
        client._client = sentinel
        self.assertIs(client._get_client(), sentinel)
        self.assertIs(client._get_client(), sentinel)

    def test_timeout_defaults_from_config_and_can_be_overridden(self):
        self.assertEqual(LLMClient().timeout, config.LLM_TIMEOUT)
        self.assertEqual(LLMClient(timeout=5).timeout, 5)


if __name__ == "__main__":
    unittest.main()
