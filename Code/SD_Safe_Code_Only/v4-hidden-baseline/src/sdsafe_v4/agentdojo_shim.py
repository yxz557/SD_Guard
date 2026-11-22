"""Import shims (this study's own directory only) so the *frozen* AgentDojo tool schemas
can be rebuilt without installing anything.

Why: the accepted `nothink_ctx40k` round sequences are frozen-template renders, and the
frozen Qwen3 template needs the suite tool schema.  The only torch/transformers
environment on this host has no `agentdojo` runtime dependencies (cohere / tenacity /
google-genai / openai ...), and there is no network.  The tool schema itself lives in the
vendored `external/AgentLAB/Task-Injection/agentdojo/src` tree, which only depends on
`pydantic`; the LLM provider modules that pull the missing packages are never called.

These shims replace nothing at runtime: they are empty module objects for optional
provider SDKs, installed *before* the AgentDojo import.  The rebuilt tool list is then
verified against the archive's own `request_tools_sha256`, so a wrong schema cannot pass.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from . import SDSAFE

VENDOR = [
    str(SDSAFE / "external/AgentLAB/Task-Injection/agentdojo/src"),
    str(SDSAFE / "external/AgentLAB/src"),
]


class _Sentinel:
    """Permissive stand-in for any attribute of an optional SDK."""

    def __call__(self, *args: Any, **kwargs: Any) -> "_Sentinel":
        return self

    def __getattr__(self, name: str) -> "_Sentinel":
        return self

    def __iter__(self):
        return iter(())

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __getitem__(self, item):
        return self

    def __mro_entries__(self, bases):
        return (object,)

    def __getitem_class(cls):  # pragma: no cover
        return cls


_SENTINEL = _Sentinel()


class _StubModule(types.ModuleType):
    __path__: list[str] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _SENTINEL


def _ensure(name: str) -> None:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        mod_name = ".".join(parts[:i])
        if mod_name not in sys.modules:
            module = _StubModule(mod_name)
            module.__path__ = []
            sys.modules[mod_name] = module
            if i > 1:
                setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], module)


STUB_ROOTS = (
    "cohere", "tenacity", "openai", "anthropic", "mistralai", "ollama",
    "google.genai", "litellm", "groq", "reka", "vertexai", "together", "fireworks",
    "deepdiff", "docx", "pypdf", "PyPDF2", "yaml_extra", "iso639", "emoji",
)


class _Finder:
    """Meta-path finder: any missing submodule of an optional SDK becomes a stub."""

    def find_module(self, fullname, path=None):  # pragma: no cover - legacy API
        return self if self._wanted(fullname) else None

    def find_spec(self, fullname, path=None, target=None):
        if not self._wanted(fullname):
            return None
        import importlib.machinery

        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def _wanted(self, fullname: str) -> bool:
        if fullname in sys.modules:
            return False
        return any(fullname == r or fullname.startswith(r + ".") for r in STUB_ROOTS)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module) -> None:
        return None


def install() -> None:
    for path in reversed(VENDOR):
        if path not in sys.path:
            sys.path.insert(0, path)
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    for name in STUB_ROOTS:
        if name.split(".")[0] in ("google",):
            _ensure(name)
            continue
        try:
            __import__(name)
        except Exception:  # noqa: BLE001 - optional SDK, never used for tool schemas
            _ensure(name)


def tools_for_suite(benchmark_version: str, suite_name: str):
    """The frozen suite tool schema in the exact shape the producer hashed.

    ``openai``'s ``ChatCompletionToolParam`` / ``FunctionDefinition`` are ``TypedDict``s,
    i.e. plain ``dict``s at runtime, so the producer's
    ``_function_to_openai`` result is reproduced literally here.  The result is verified
    against the archive's ``request_tools_sha256`` by the caller.
    """
    install()
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(benchmark_version, suite_name)
    runtime = FunctionsRuntime(suite.tools)
    return tuple(
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_json_schema(),
            },
        }
        for tool in runtime.functions.values()
    )
