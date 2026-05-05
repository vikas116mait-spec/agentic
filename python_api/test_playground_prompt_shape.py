from __future__ import annotations

import unittest
from unittest.mock import patch

from python_api.services import run_playground_prompt
from python_api.store import empty_state


class PlaygroundPromptShapeTests(unittest.TestCase):
    def test_run_playground_prompt_single_turn_uses_only_latest_user_message(self) -> None:
        state = empty_state()
        captured_calls: list[tuple[list[dict[str, str]], str, str | None]] = []

        def fake_run_model_messages(messages: list[dict[str, str]], model: str, provider: str | None = None) -> str:
            captured_calls.append((messages, model, provider))
            return f"{model}-reply"

        def fake_update_state(mutator):
            return mutator(state)

        with patch("python_api.services.run_model_messages", side_effect=fake_run_model_messages), patch(
            "python_api.services.update_state",
            side_effect=fake_update_state,
        ):
            result = run_playground_prompt(
                "",
                "demo-model",
                base_provider="ollama",
                messages=[
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Need help"},
                ],
                system_prompt="Be concise.",
                single_turn=True,
            )

        self.assertEqual(result["baseOutput"], "demo-model-reply")
        self.assertEqual(
            captured_calls[0][0],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Need help"},
            ],
        )
        self.assertEqual(state["playground_runs"][0]["prompt"], "Need help")

    def test_run_playground_prompt_uses_structured_messages(self) -> None:
        state = empty_state()
        captured_calls: list[tuple[list[dict[str, str]], str, str | None]] = []

        def fake_run_model_messages(messages: list[dict[str, str]], model: str, provider: str | None = None) -> str:
            captured_calls.append((messages, model, provider))
            return f"{model}-reply"

        def fake_update_state(mutator):
            return mutator(state)

        with patch("python_api.services.run_model_messages", side_effect=fake_run_model_messages), patch(
            "python_api.services.update_state",
            side_effect=fake_update_state,
        ):
            result = run_playground_prompt(
                "",
                "demo-model",
                base_provider="ollama",
                messages=[
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Tell me more"},
                ],
                system_prompt="Be concise.",
            )

        self.assertEqual(result["baseOutput"], "demo-model-reply")
        self.assertIsNone(result["tunedOutput"])
        self.assertEqual(len(captured_calls), 1)
        self.assertEqual(
            captured_calls[0][0],
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Tell me more"},
            ],
        )
        self.assertEqual(state["playground_runs"][0]["prompt"], "Tell me more")

    def test_run_playground_prompt_keeps_flat_prompt_compatibility(self) -> None:
        state = empty_state()
        captured_calls: list[tuple[list[dict[str, str]], str, str | None]] = []

        def fake_run_model_messages(messages: list[dict[str, str]], model: str, provider: str | None = None) -> str:
            captured_calls.append((messages, model, provider))
            return "ok"

        def fake_update_state(mutator):
            return mutator(state)

        with patch("python_api.services.run_model_messages", side_effect=fake_run_model_messages), patch(
            "python_api.services.update_state",
            side_effect=fake_update_state,
        ):
            result = run_playground_prompt(
                "Summarize this",
                "demo-model",
                base_provider="openai",
            )

        self.assertEqual(result["baseOutput"], "ok")
        self.assertEqual(
            captured_calls[0][0],
            [{"role": "user", "content": "Summarize this"}],
        )
        self.assertEqual(state["playground_runs"][0]["prompt"], "Summarize this")


if __name__ == "__main__":
    unittest.main()
