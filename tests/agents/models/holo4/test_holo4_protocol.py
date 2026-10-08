"""Holo4 history compaction tests."""

from __future__ import annotations

from lite.agents.models.holo4.protocol import Holo4HistoryProtocol


def _user(text: str, image: int) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image", "index": image},
        ],
    }


def _assistant(note: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": note}],
        "reasoning_content": f"private {note}",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "computer", "arguments": {"actions": []}},
            }
        ],
    }


def test_compaction_keeps_instruction_and_old_notes_but_drops_old_tool_state() -> None:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        _user("task instruction", 0),
        _assistant("remember alpha", "call_0"),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": [{"type": "text", "text": "old tool output"}],
        },
        _user("old screenshot", 1),
        _assistant("remember beta", "call_1"),
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "recent tool output"}],
        },
        _user("latest screenshot", 2),
    ]
    compacted = Holo4HistoryProtocol(max_n_images=1).process_messages(messages)
    assert compacted[:4] == [
        messages[0],
        {"role": "user", "content": [{"type": "text", "text": "task instruction"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "remember alpha"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "remember beta"}]},
    ]
    assert compacted[-2:] == [messages[-2], messages[-1]]
    assert "old tool output" not in str(compacted)
    assert "private" not in str(compacted)
    assert "call_0" not in str(compacted)


def test_short_history_is_unchanged() -> None:
    messages = [_user("task", 0), _assistant("note", "call_0")]
    assert Holo4HistoryProtocol(max_n_images=4).process_messages(messages) == messages
