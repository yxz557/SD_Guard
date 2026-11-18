from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path


AGENTLAB_ROOT = Path(__file__).resolve().parents[1]
if str(AGENTLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTLAB_ROOT))

from trace_artifacts import (
    assistant_message_dict,
    extract_tool_trace,
    redact_secrets,
    write_raw_trace,
)


class TraceArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = AGENTLAB_ROOT / "tests" / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(
            prefix="agentlab-trace-test-", dir=temp_root,
        ))

    def tearDown(self) -> None:
        # Trace artifacts are intentionally read-only, so make them writable
        # before removing the temporary test tree on Windows.
        for path in sorted(self.temp_dir.rglob("*"), reverse=True):
            try:
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def write_example(self, task_text: str = "do the task") -> Path:
        return write_raw_trace(
            self.temp_dir,
            component="Tool Chaining",
            identity={"run_id": "r0", "sample": "s0"},
            task={"instruction": task_text},
            target={"model_id": "Qwen/Qwen3-8B"},
            trajectory={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "c0",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "c0",
                        "name": "lookup",
                        "content": "result",
                    },
                ]
            },
            provenance={"runner": "test"},
        )

    def test_write_raw_trace_and_hash(self) -> None:
        trace_path = self.write_example()
        trace = json.loads(trace_path.read_text(encoding="utf-8"))

        self.assertEqual(trace["schema_version"], "sd_guard.raw_trace.v1")
        self.assertEqual(trace["benchmark_component"], "Tool Chaining")
        self.assertEqual(trace["task"]["instruction"], "do the task")

        expected = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        sidecar = trace_path.with_suffix(".sha256").read_text(encoding="utf-8")
        self.assertEqual(sidecar.split()[0], expected)

    def test_identical_retry_is_idempotent(self) -> None:
        first = self.write_example()
        second = self.write_example()
        self.assertEqual(first, second)

    def test_changed_content_cannot_overwrite_same_identity(self) -> None:
        self.write_example()
        with self.assertRaises(FileExistsError):
            self.write_example(task_text="changed")

    def test_extract_tool_trace(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "c0",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "c0",
                "name": "lookup",
                "content": "ok",
            },
        ]
        events = extract_tool_trace(messages)
        self.assertEqual(
            [event["event"] for event in events],
            ["tool_call_proposed", "tool_result"],
        )

    def test_assistant_message_keeps_reasoning_and_tool_call(self) -> None:
        class Function:
            name = "lookup"
            arguments = '{"query":"x"}'

        class ToolCall:
            id = "c0"
            function = Function()

        class Message:
            content = None
            reasoning_content = "need to look this up"
            tool_calls = [ToolCall()]

        message = assistant_message_dict(Message())
        self.assertEqual(message["reasoning_content"], "need to look this up")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "lookup")

    def test_redact_secrets_preserves_non_secret_generation_config(self) -> None:
        config = {
            "attacker": {"api_key": "secret", "max_tokens": 2048},
            "auxiliary_api_key": "secret-2",
            "victim": {"model": "Qwen/Qwen3-8B", "top_k": 20},
        }
        redacted = redact_secrets(config)
        self.assertEqual(redacted["attacker"]["api_key"], "[REDACTED]")
        self.assertEqual(redacted["auxiliary_api_key"], "[REDACTED]")
        self.assertEqual(redacted["attacker"]["max_tokens"], 2048)
        self.assertEqual(redacted["victim"]["model"], "Qwen/Qwen3-8B")


if __name__ == "__main__":
    unittest.main()
