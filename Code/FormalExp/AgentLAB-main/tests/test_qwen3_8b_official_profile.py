import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


AGENTLAB_ROOT = Path(__file__).resolve().parents[1]
if str(AGENTLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTLAB_ROOT))

from local_env import resolve_auxiliary_api_key
from qwen3_8b_official_profile import (
    COMMON_ENV,
    GPT51_MODEL,
    OBJECTIVE_ATTACKER_MODEL,
    OPENAI_BASE_URL,
    build_run_specs,
)
from start_qwen3_8b_vllm import _service_command


def option_value(command, option):
    index = command.index(option)
    return command[index + 1]


class OfficialProfileTests(unittest.TestCase):
    def test_tool_chaining_uses_gpt51_for_planner_and_judge(self):
        specs = build_run_specs("tool-chaining", "full")
        self.assertEqual(len(specs), 1)
        command = specs[0].command
        self.assertEqual(option_value(command, "--attacker_model"), GPT51_MODEL)
        self.assertEqual(option_value(command, "--aux_base_url"), OPENAI_BASE_URL)
        self.assertNotIn("--aux_api_key", command)

    def test_tool_smoke_covers_both_backends(self):
        specs = build_run_specs("tool-chaining", "smoke")
        self.assertEqual(
            {option_value(spec.command, "--dataset_source") for spec in specs},
            {"SHADE_Arena", "Agent_SafetyBench"},
        )

    def test_intent_preserves_official_local_attacker(self):
        command = build_run_specs("intent-hijacking", "full")[0].command
        self.assertEqual(
            option_value(command, "--attacker_model"),
            "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
        )
        self.assertEqual(option_value(command, "--planner_judge_model"), GPT51_MODEL)

    def test_objective_preserves_role_specific_official_models(self):
        command = build_run_specs("objective-drifting", "full")[0].command
        self.assertEqual(option_value(command, "--attacker_model"), OBJECTIVE_ATTACKER_MODEL)
        self.assertEqual(option_value(command, "--judge_model"), GPT51_MODEL)
        self.assertEqual(option_value(command, "--extractor_model"), GPT51_MODEL)
        self.assertEqual(option_value(command, "--stealthiness_model"), GPT51_MODEL)

    def test_memory_restores_all_shared_auxiliary_roles_to_gpt51(self):
        command = build_run_specs("memory-poisoning", "full")[0].command
        self.assertEqual(option_value(command, "--aux_model"), GPT51_MODEL)
        self.assertEqual(option_value(command, "--mem0_llm_model"), GPT51_MODEL)
        self.assertEqual(option_value(command, "--memory_planner_prompt_variant"), "official")
        self.assertNotIn("deepseek", " ".join(command).lower())

    def test_runtime_profile_enforces_offline_huggingface(self):
        self.assertEqual(COMMON_ENV["HF_HUB_OFFLINE"], "1")
        self.assertEqual(COMMON_ENV["TRANSFORMERS_OFFLINE"], "1")

    def test_runtime_profile_accepts_configured_gpt51_gateway(self):
        gateway = "https://gateway.example/v1"
        command = build_run_specs(
            "tool-chaining", "full", auxiliary_base_url=gateway,
        )[0].command
        self.assertEqual(option_value(command, "--aux_base_url"), gateway)

    def test_intent_services_fit_the_remote_24gb_gpu_via_cpu_offload(self):
        victim = _service_command("victim-for-intent")
        attacker = _service_command("intent-attacker")
        self.assertEqual(option_value(victim, "--cpu-offload-gb"), "8")
        self.assertEqual(option_value(attacker, "--cpu-offload-gb"), "20")
        self.assertEqual(option_value(victim, "--max-model-len"), "8192")
        self.assertEqual(option_value(attacker, "--max-model-len"), "8192")


class AuxiliaryCredentialTests(unittest.TestCase):
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "openai-key",
        "DEEPSEEK_API_KEY": "deepseek-key",
    }, clear=False)
    def test_openai_endpoint_selects_openai_key(self):
        self.assertEqual(
            resolve_auxiliary_api_key(
                base_url=OPENAI_BASE_URL,
                model=GPT51_MODEL,
            ),
            "openai-key",
        )

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "openai-key",
        "DEEPSEEK_API_KEY": "deepseek-key",
    }, clear=False)
    def test_deepseek_endpoint_selects_deepseek_key(self):
        self.assertEqual(
            resolve_auxiliary_api_key(
                base_url="https://api.deepseek.com",
                model="deepseek-flash",
            ),
            "deepseek-key",
        )


if __name__ == "__main__":
    unittest.main()
