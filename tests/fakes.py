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
                 temperature=None, seed=None, reachable=True):
        self.prompts = []
        self.image_calls = []
        self.seeds_used = []
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

    def chat(self, prompt):
        self.prompts.append(prompt)
        # Record the seed in force for this call. The retry path issues later
        # attempts through a shallow copy carrying a different seed, so this is
        # what lets a test assert that the seeds varied while the prompt bytes
        # did not. The list is shared with any copy, so calls land in order.
        self.seeds_used.append(self.seed)
        return self._next_reply()

    def chat_with_images(self, prompt, images):
        self.image_calls.append((prompt, list(images)))
        self.prompts.append(prompt)
        return self._next_reply()

    def check_model_available(self, timeout=5.0):
        """Stand in for the real pre-flight probe without any network call."""
        if self.reachable:
            return True, f"Model '{self.model}' is available."
        return False, f"Model '{self.model}' is not available."
