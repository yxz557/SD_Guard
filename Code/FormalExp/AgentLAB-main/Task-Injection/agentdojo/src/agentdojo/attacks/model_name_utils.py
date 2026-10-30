import os
from collections.abc import Callable


LOCAL_PIPELINE_NAMES = ("local", "vllm_parsed")


def resolve_model_name_for_cache(
    pipeline_name: str,
    get_local_model_id: Callable[[str], str],
    port: str | None = None,
) -> str:
    """Resolve a pipeline name to the model-specific cache directory name.

    Local pipelines are named after their provider (``local`` or
    ``vllm_parsed``), while long-horizon artifacts are stored under the model
    ID returned by vLLM. Defense-enabled pipelines append the defense name to
    the provider with a hyphen; preserve that suffix after resolving the model
    ID so artifact writers and readers use the same directory.
    """
    name = str(pipeline_name)
    provider = next(
        (
            candidate
            for candidate in LOCAL_PIPELINE_NAMES
            if name == candidate or name.startswith(f"{candidate}-")
        ),
        None,
    )
    if provider is None:
        return name

    defense_suffix = name[len(provider) :]
    local_port = port or os.getenv("LOCAL_LLM_PORT", "8000")
    model_id = get_local_model_id(local_port)
    return model_id.replace("/", "_") + defense_suffix
