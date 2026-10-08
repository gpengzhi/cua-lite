"""Holo4 agent registry entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lite.agents.models.qwen3_5.agent import Qwen3_5BaseAgent


@dataclass
class Holo4BaseAgent(
    Qwen3_5BaseAgent,
    key=r"holo4\.base(@(desktop|browser|mobile)@use)?",
):
    """Holo4 base agent with native thinking-template controls."""

    reasoning_effort: str | None = None

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Build the Holo4 chat template prompt without replaying old reasoning."""
        if self.processor is None:
            raise RuntimeError("agent.processor is not set")
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": getattr(self.adapter, "enable_thinking", True),
            "preserve_thinking": False,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self.processor.apply_chat_template(messages, **kwargs)


@dataclass
class Holo4DesktopUseAgent(
    Holo4BaseAgent,
    key=r"holo4@(desktop|browser)@use",
):
    """Holo4 desktop/browser use registry entry."""


@dataclass
class Holo4MobileUseAgent(Holo4BaseAgent, key="holo4@mobile@use"):
    """Holo4 mobile use registry entry."""


__all__ = ["Holo4BaseAgent", "Holo4DesktopUseAgent", "Holo4MobileUseAgent"]
