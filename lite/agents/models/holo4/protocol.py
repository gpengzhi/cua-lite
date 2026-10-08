"""Holo4 history compaction policy."""

from __future__ import annotations

import copy
import dataclasses

from lite.agents.core.protocol import BaseProtocol
from lite.core import LiteMessage
from lite.core.messages import (
    ASSISTANT_ROLE,
    TEXT_PART,
    USER_ROLE,
    group_into_turns,
    message_has_image,
    peel_system_message,
)


@dataclasses.dataclass
class Holo4HistoryProtocol(BaseProtocol, key="holo4.history"):
    """Keep recent screenshot turns and durable assistant narration.

    Older screenshots, tool output, tool calls, and reasoning are evicted. The
    initial task instruction and ordinary assistant text survive compaction.
    """

    max_n_images: int = 4

    def process_messages(
        self,
        messages: list[LiteMessage],
        **kwargs,
    ) -> list[LiteMessage]:
        if not messages:
            return []
        messages = copy.deepcopy(messages)
        if self.max_n_images <= 0:
            return messages

        system_message, content = peel_system_message(messages)
        turns = group_into_turns(content)
        image_turns = [
            i
            for i, turn in enumerate(turns)
            if any(message_has_image(message) for message in turn["observations"])
        ]
        if len(image_turns) <= self.max_n_images:
            return messages

        compact_before = image_turns[-self.max_n_images]
        compacted: list[LiteMessage] = []
        instruction_kept = False
        for i, turn in enumerate(turns):
            if i >= compact_before:
                compacted.extend(turn["observations"])
                if turn["assistant"] is not None:
                    compacted.append(turn["assistant"])
                continue

            if not instruction_kept:
                for observation in turn["observations"]:
                    if observation.get("role") != USER_ROLE:
                        continue
                    text_parts = [
                        part
                        for part in observation.get("content", [])
                        if isinstance(part, dict)
                        and part.get("type") == TEXT_PART
                        and part.get("text")
                    ]
                    if text_parts:
                        compacted.append({"role": USER_ROLE, "content": text_parts})
                        instruction_kept = True
                        break

            assistant = turn["assistant"]
            if assistant is None or assistant.get("role") != ASSISTANT_ROLE:
                continue
            text_parts = [
                part
                for part in assistant.get("content", [])
                if isinstance(part, dict) and part.get("type") == TEXT_PART and part.get("text")
            ]
            if text_parts:
                compacted.append({"role": ASSISTANT_ROLE, "content": text_parts})

        if system_message is not None:
            compacted.insert(0, system_message)
        return compacted


__all__ = ["Holo4HistoryProtocol"]
