"""Thin client for any OpenAI compatible chat completion endpoint.

The openai package is imported lazily on first use so that the rest of the
codebase, including the whole test suite, can be imported and run on a
machine without the package or without network access. Tests inject a fake
client instead (see tests/fakes.py), which is what makes the orchestration
layer testable without a model.

Temperature and seed are optional reproducibility controls: they are sent to
the API only when set, so the default configuration reproduces the exact
sampling behaviour evaluated in the Preliminary Project Report. Ollama's
OpenAI compatible endpoint supports both parameters, which is what lets the
evaluation separate an input-mode effect from run-to-run sampling noise.
"""
import config


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None,
                 temperature=None, seed=None, timeout=None):
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        # None means "not set": the parameter is omitted from the request
        # rather than sent as a default value, so an unset control reproduces
        # today's output exactly. Note that 0 is a real value (greedy
        # decoding) and is deliberately kept, only None is treated as unset.
        self.temperature = (temperature if temperature is not None
                            else config.LLM_TEMPERATURE)
        self.seed = seed if seed is not None else config.LLM_SEED
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        # The OpenAI client is built once per instance and cached here rather
        # than once per call, so a batch evaluation reuses a single client
        # (and its connection pool) across every mode and call.
        self._client = None

    def _apply_settings(self, kwargs):
        """Attach the optional reproducibility settings to a request.

        Shared by the text and image request builders so both honour
        temperature and seed identically, and both omit them when unset.
        Factored out (rather than duplicated) precisely so adding the vision
        path cannot drift from the text path.
        """
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def _request_kwargs(self, prompt):
        """Assemble the keyword arguments for one text completion call.

        Kept as a separate method with no network dependency so the tests
        can assert that temperature and seed are passed through only when
        set, without a real model. This is the seam the reproducibility
        tests exercise, and its output must stay byte-identical to the
        request the evaluated prototype sent.
        """
        return self._apply_settings({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        })

    def _image_request_kwargs(self, prompt, images):
        """Assemble the keyword arguments for one multimodal call.

        Uses the OpenAI multimodal content array: one text part followed by
        one image_url part per image, where each image is a base64 data URL.
        Ollama's OpenAI-compatible endpoint accepts this shape, so a vision
        model reuses the same client rather than needing new architecture.
        """
        content = [{"type": "text", "text": prompt}]
        for url in images:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return self._apply_settings({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        })

    def _get_client(self):
        """Return the cached OpenAI client, importing and building it on
        first use so this module stays importable without the package."""
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

    def chat(self, prompt):
        """Send a single user message and return the reply text.

        Works with OpenAI as well as any OpenAI compatible endpoint such as
        a local Ollama server, so the model can be swapped through the .env
        file without changing the code.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                **self._request_kwargs(prompt))
        except Exception as error:
            raise RuntimeError(f"Model call failed: {error}") from error
        return (response.choices[0].message.content or "").strip()

    def check_model_available(self, timeout=5.0):
        """Cheap pre-flight probe: is this model actually usable right now?

        Returns (ok, message) and never raises, so an offline machine or a
        missing openai package degrades to a clear message instead of an
        exception part-way through a long extraction. It lists the endpoint's
        models rather than generating anything, so it costs no inference and
        does not slow the normal path. A False here names the model and how to
        pull it, which is what the interface and the comparison harness show.
        """
        try:
            client = self._get_client()
            try:
                # A short timeout for the probe only; the SDK returns a copy,
                # leaving the real client's timeout untouched.
                client = client.with_options(timeout=timeout)
            except (AttributeError, TypeError):
                pass
            available = [getattr(item, "id", "")
                         for item in client.models.list().data]
        except RuntimeError as error:
            # The openai package is missing; the hint is already in the error.
            return False, str(error)
        except Exception as error:
            return False, (
                f"Could not reach the model endpoint at {self.base_url} "
                f"({error}). For Ollama, start it with 'ollama serve'.")
        # Ollama reports tagged names such as 'qwen3-vl:4b'; treat an
        # untagged request as matching its ':latest' tag.
        wanted = {self.model, f"{self.model}:latest"}
        if wanted & set(available):
            return True, f"Model '{self.model}' is available."
        return False, (
            f"Model '{self.model}' is not available at {self.base_url}. "
            f"Pull it with: ollama pull {self.model}")

    def chat_with_images(self, prompt, images):
        """Send a prompt plus one or more images and return the reply text.

        images is a list of base64 data URLs (e.g. "data:image/png;base64,..").
        This is the vision path used for slide-image extraction; it reuses the
        same client, timeout and reproducibility settings as chat(), and does
        not alter chat() itself, so existing text generation stays unchanged.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                **self._image_request_kwargs(prompt, images))
        except Exception as error:
            raise RuntimeError(f"Model call failed: {error}") from error
        return (response.choices[0].message.content or "").strip()
