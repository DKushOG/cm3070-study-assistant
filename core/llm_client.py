"""Thin client for any OpenAI compatible chat completion endpoint.

The openai package is imported on first use, so the rest of the code and the
whole test suite can be imported on a machine without the package or without
network access. Tests inject a fake client instead, see tests/fakes.py.

Temperature and seed are optional controls. They are sent only when set, so
the default configuration makes the same request as the earlier prototype.
Ollama's OpenAI compatible endpoint accepts both, which is what lets the
evaluation separate an input-mode effect from run-to-run sampling noise.
"""
import config


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None,
                 temperature=None, seed=None, timeout=None):
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        # None means "not set", so the parameter is left out of the request
        # rather than sent as a default. Note that 0 is a real value, greedy
        # decoding, and only None counts as unset.
        self.temperature = (temperature if temperature is not None
                            else config.LLM_TEMPERATURE)
        self.seed = seed if seed is not None else config.LLM_SEED
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        # Built once per instance and cached here rather than once per call,
        # so a batch run reuses one client and its connection pool.
        self._client = None

    def _apply_settings(self, kwargs):
        """Attach the optional sampling settings to a request.

        Shared by the text and image request builders, so both treat
        temperature and seed the same way and both leave them out when unset.
        Kept in one place so the vision path does not drift from the text path.
        """
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def _request_kwargs(self, prompt, response_format=None):
        """Assemble the keyword arguments for one text completion call.

        Kept separate, with no network dependency, so the tests can check that
        temperature and seed are passed through only when set without needing
        a real model.

        response_format follows the same "None means unset" rule and is left
        out unless a caller asks for it, so the original quiz path and the
        structured quiz path differ in one thing only.
        """
        kwargs = self._apply_settings({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        })
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _image_request_kwargs(self, prompt, images, max_tokens=None):
        """Assemble the keyword arguments for one multimodal call.

        Uses the OpenAI multimodal content array: one text part followed by
        one image_url part per image, each image a base64 data URL. Ollama's
        OpenAI compatible endpoint accepts this shape, so the vision model
        reuses the same client.

        max_tokens follows the same "None means unset" rule as temperature and
        seed. It is there because a vision model that loses its way on a dense
        slide can otherwise keep generating until the context window runs out.
        One page was measured at 14,641 tokens and nearly three minutes, and
        returned nothing at the end of it.
        """
        content = [{"type": "text", "text": prompt}]
        for url in images:
            content.append({"type": "image_url", "image_url": {"url": url}})
        kwargs = self._apply_settings({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        })
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return kwargs

    def _get_client(self):
        """Return the cached OpenAI client, building it on first use so the
        module can still be imported without the openai package."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "The openai package is required for model calls. "
                    "Install it with: pip install openai"
                ) from error
            self._client = OpenAI(api_key=self.api_key,
                                  base_url=self.base_url,
                                  timeout=self.timeout)
        return self._client

    def chat(self, prompt, response_format=None):
        """Send a single user message and return the reply text.

        Works with OpenAI and with any OpenAI compatible endpoint such as a
        local Ollama server, so the model can be changed in the .env file
        rather than in the code.

        Pass response_format to constrain decoding, for example a JSON schema
        for the quiz call. Leaving it out sends the plain request.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                **self._request_kwargs(prompt, response_format))
        except Exception as error:
            raise RuntimeError(f"Model call failed: {error}") from error
        return (response.choices[0].message.content or "").strip()

    def check_model_available(self, timeout=5.0):
        """Check whether this model is usable before a long run starts.

        Returns (ok, message) rather than raising, so a machine that is offline
        or missing the openai package gets a clear message instead of an error
        part way through an extraction. It lists the endpoint's models rather
        than generating anything, so it makes no inference call. A False result
        names the model and the command to pull it, and the interface and the
        comparison harness show that message.
        """
        try:
            client = self._get_client()
            try:
                # A short timeout for the probe only. The SDK returns a copy,
                # so the real client's timeout is left alone.
                client = client.with_options(timeout=timeout)
            except (AttributeError, TypeError):
                pass
            available = [getattr(item, "id", "")
                         for item in client.models.list().data]
        except RuntimeError as error:
            # The openai package is missing. The hint is already in the error.
            return False, str(error)
        except Exception as error:
            return False, (
                f"Could not reach the model endpoint at {self.base_url} "
                f"({error}). For Ollama, start it with 'ollama serve'.")
        # Ollama reports tagged names such as 'qwen2.5vl:7b', so an untagged
        # request is treated as matching its ':latest' tag.
        wanted = {self.model, f"{self.model}:latest"}
        if wanted & set(available):
            return True, f"Model '{self.model}' is available."
        return False, (
            f"Model '{self.model}' is not available at {self.base_url}. "
            f"Pull it with: ollama pull {self.model}")

    def chat_with_images(self, prompt, images, max_tokens=None,
                         with_meta=False):
        """Send a prompt plus one or more images and return the reply text.

        images is a list of base64 data URLs, for example
        "data:image/png;base64,..". This is the vision path used for slide
        images. It reuses the same client, timeout and sampling settings as
        chat() and leaves chat() itself alone.

        with_meta returns (text, meta) instead of text alone, where meta holds
        the finish reason and the completion token count. It is off by default,
        so existing callers are unaffected.

        The metadata matters more than it looks. A vision model that runs out
        of room returns an empty string and raises nothing, and an empty string
        looks the same as a blank slide. Before the finish reason was reported,
        49 of 111 slides were recorded as extracted when nothing had come back.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                **self._image_request_kwargs(prompt, images, max_tokens))
        except Exception as error:
            raise RuntimeError(f"Model call failed: {error}") from error
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if not with_meta:
            return text
        usage = getattr(response, "usage", None)
        meta = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        return text, meta
