"""Test doubles for the LLM client.

The fake records every prompt it receives and returns queued replies in
order, which lets the tests verify orchestration behaviour, including the
sequential dependency of the quiz call on the generated notes, without any
model or network access.

It also exposes the same settings attributes as the real LLMClient (model,
base_url, temperature, seed), all optional, so tests can check that the
orchestrator records them in the saved output without touching a real model.
chat_with_images mirrors the real vision method and records the prompt and
image URLs it received, so the vision-path tests can assert the request
without a real model or any real image.
"""


class FakeLLMClient:
    def __init__(self, replies=None, model=None, base_url=None,
                 temperature=None, seed=None, reachable=True,
                 finish_reasons=None):
        self.prompts = []
        self.image_calls = []
        self.seeds_used = []
        self.response_formats = []
        # Recorded per image call so a test can assert that the extraction
        # path passes a token cap, without needing a real model to enforce it.
        self.max_tokens_used = []
        # Consumed in step with the image calls. Supplying "length" simulates
        # a page that stopped because it ran out of room, which is the failure
        # the vision path now has to detect rather than pass through.
        self._finish_reasons = list(finish_reasons) if finish_reasons else []
        self.reachable = reachable
        self._replies = list(replies) if replies else ["FAKE NOTES",
                                                       "FAKE QUIZ"]
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed

    def _next_reply(self):
        index = min(len(self.prompts) - 1, len(self._replies) - 1)
        return self._replies[index]

    def chat(self, prompt, response_format=None):
        self.prompts.append(prompt)
        # Record the response_format each call was issued with, so a test can
        # assert that the structured path attaches a schema and the legacy
        # path attaches nothing at all. None entries mean an unconstrained
        # call, which is what the evaluated prototype sent.
        self.response_formats.append(response_format)
        # Record the seed in force for this call. The retry path issues later
        # attempts through a shallow copy carrying a different seed, so this is
        # what lets a test assert that the seeds varied while the prompt bytes
        # did not. The list is shared with any copy, so calls land in order.
        self.seeds_used.append(self.seed)
        return self._next_reply()

    def chat_with_images(self, prompt, images, max_tokens=None,
                         with_meta=False):
        """Mirror the real vision method, including its metadata mode.

        Replies still advance with every call, retries included, so a test
        can queue an empty first reply and a good second one and assert that
        the retry recovered the page.
        """
        self.image_calls.append((prompt, list(images)))
        self.prompts.append(prompt)
        self.response_formats.append(None)
        self.max_tokens_used.append(max_tokens)
        reply = self._next_reply()
        if not with_meta:
            return reply
        if self._finish_reasons:
            position = min(len(self.image_calls) - 1,
                           len(self._finish_reasons) - 1)
            finish = self._finish_reasons[position]
        else:
            finish = "stop"
        return reply, {"finish_reason": finish, "completion_tokens": None}

    def check_model_available(self, timeout=5.0):
        """Stand in for the real pre-flight probe without any network call."""
        if self.reachable:
            return True, f"Model '{self.model}' is available."
        return False, f"Model '{self.model}' is not available."
