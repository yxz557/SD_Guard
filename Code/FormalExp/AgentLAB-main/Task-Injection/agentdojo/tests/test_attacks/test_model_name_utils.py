from agentdojo.attacks.model_name_utils import resolve_model_name_for_cache


def test_remote_model_name_is_unchanged():
    def unexpected_getter(_: str) -> str:
        raise AssertionError("local model lookup should not run")

    assert resolve_model_name_for_cache("gpt-5.1", unexpected_getter) == "gpt-5.1"


def test_vllm_parsed_resolves_to_model_id(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_PORT", "8123")
    requested_ports = []

    def get_model_id(port: str) -> str:
        requested_ports.append(port)
        return "/home/models/Qwen3-8B"

    resolved = resolve_model_name_for_cache("vllm_parsed", get_model_id)

    assert resolved == "_home_models_Qwen3-8B"
    assert requested_ports == ["8123"]


def test_local_pipeline_preserves_defense_suffix():
    resolved = resolve_model_name_for_cache(
        "local-repeat_user_prompt",
        lambda port: "/home/models/Qwen3-8B",
        port="8001",
    )

    assert resolved == "_home_models_Qwen3-8B-repeat_user_prompt"


def test_vllm_parsed_pipeline_preserves_defense_suffix():
    resolved = resolve_model_name_for_cache(
        "vllm_parsed-transformers_pi_detector",
        lambda port: "Qwen/Qwen3-8B",
        port="8002",
    )

    assert resolved == "Qwen_Qwen3-8B-transformers_pi_detector"
