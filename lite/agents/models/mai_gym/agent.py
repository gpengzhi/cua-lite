"""MAI-Gym agents — one-step creation via AgentRegistry.

MAI-Gym-9B is a Qwen3.5-9B fine-tune with the stock Qwen3.5 chat template, so
the agents inherit :class:`Qwen3_5BaseAgent`: ``adapter.enable_thinking`` (on
for this family) opens the ``<think>`` block, and the inherited parse step
moves the reply's reasoning out of the replay sidecar into
``reasoning_content``. The family delta is ``preserve_thinking=True`` (both
platforms): MAI-Gym evaluates with ``token_native_history`` (paper Appendix
A.5), replaying each past reply verbatim, so every past assistant turn keeps
its ``<think>`` block instead of the template's default of dropping it.

Unparseable replies take the shared parse-failure final. Upstream instead
regenerates up to three times before ``terminate(status="failure")``; that
retry is not reproduced.

Usage::

    agent = AgentRegistry.get(
        "mai_gym@desktop@use",
        processor=processor,
        generate_fn=generate_fn,
        password="password",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lite.agents.models.qwen3_5.agent import Qwen3_5BaseAgent


@dataclass
class MAIGymBaseAgent(Qwen3_5BaseAgent):
    """Shared MAI-Gym prompt building: native thinking on, history reasoning kept."""

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Build the prompt with native thinking on and history reasoning kept."""
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.adapter.enable_thinking,
            preserve_thinking=True,
        )


@dataclass
class MAIGymMobileUseAgent(MAIGymBaseAgent, key="mai_gym@mobile@use"):
    """Mobile GUI-use registry entry."""


@dataclass
class MAIGymDesktopUseAgent(MAIGymBaseAgent, key="mai_gym@desktop@use"):
    """Desktop computer-use registry entry."""


__all__ = [
    "MAIGymBaseAgent",
    "MAIGymDesktopUseAgent",
    "MAIGymMobileUseAgent",
]
