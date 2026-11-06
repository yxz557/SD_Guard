from __future__ import annotations

import unittest

from guard_training.common.trace_serialization import serialize_messages, tokenize_serialized


class CharacterTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=False,
    ):
        output = {"input_ids": list(range(len(text)))}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(text)
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


class TraceSerializationTest(unittest.TestCase):
    def test_tool_boundaries_and_metadata_are_preserved(self) -> None:
        messages = [
            {"role": "system", "content": [{"type": "text", "content": "system"}]},
            {"role": "user", "content": [{"type": "text", "content": "task"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call-1", "function": "search", "args": {"q": "a"}},
                    {"id": "call-2", "function": "read", "args": {"id": 2}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "tool_call": {"function": "search"},
                "content": [{"type": "text", "content": "result one"}],
                "error": None,
            },
            {
                "role": "tool",
                "tool_call_id": "call-2",
                "tool_call": {"function": "read"},
                "content": [{"type": "text", "content": "result two"}],
                "error": None,
            },
            {"role": "assistant", "content": [{"type": "text", "content": "done"}]},
        ]
        serialized = serialize_messages(messages)
        kinds = [boundary.kind for boundary in serialized.boundaries]
        self.assertEqual(kinds.count("tool_call_end"), 2)
        self.assertEqual(kinds.count("assistant_decision_end"), 1)
        self.assertEqual(kinds.count("tool_result_end"), 2)
        self.assertEqual(kinds.count("tool_round_end"), 1)
        self.assertEqual(kinds.count("trace_end"), 1)
        self.assertEqual(kinds.count("readout_end"), 1)
        tool_calls = [b for b in serialized.boundaries if b.kind == "tool_call_end"]
        self.assertEqual([b.round_index for b in tool_calls], [0, 0])
        self.assertEqual([b.tool_call_id for b in tool_calls], ["call-1", "call-2"])
        self.assertLess(serialized.trace_end_char, serialized.readout_end_char or 0)

        tokenized = tokenize_serialized(CharacterTokenizer(), serialized)
        self.assertEqual(tokenized.trace_end_token, serialized.trace_end_char - 1)
        self.assertEqual(tokenized.readout_end_token, len(serialized.text) - 1)
        self.assertEqual(len(tokenized.boundaries), len(serialized.boundaries))


if __name__ == "__main__":
    unittest.main()

