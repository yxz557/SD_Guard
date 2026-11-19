"""Load machine-local AgentLAB secrets without adding a dependency.

The default file is ``config/local_secrets.env``.  It is intentionally ignored
by Git.  Existing process environment variables take precedence unless
``override=True`` is requested explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple


AGENTLAB_ROOT = Path(__file__).resolve().parent
DEFAULT_SECRETS_FILE = AGENTLAB_ROOT / "config" / "local_secrets.env"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_local_env(
    path: Optional[str] = None,
    *,
    override: bool = False,
) -> Tuple[Path, Tuple[str, ...]]:
    """Load KEY=VALUE pairs and return ``(resolved_path, loaded_names)``.

    Blank lines and comments are ignored.  ``export KEY=VALUE`` is accepted so
    the same file can also be sourced from a shell when desired.  Values are
    never printed by this module.
    """

    configured_path = path or os.environ.get("AGENTLAB_SECRETS_FILE")
    env_path = Path(configured_path).expanduser() if configured_path else DEFAULT_SECRETS_FILE
    env_path = env_path.resolve()
    if not env_path.is_file():
        return env_path, ()

    loaded = []
    with env_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ValueError(
                    f"Invalid secrets entry at {env_path}:{line_number}; expected KEY=VALUE"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or not key.replace("_", "A").isalnum() or key[0].isdigit():
                raise ValueError(f"Invalid environment name at {env_path}:{line_number}")
            value = _strip_quotes(value)
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.append(key)

    return env_path, tuple(loaded)


def validate_secret_value(name: str, value: Optional[str]) -> str:
    """Reject missing, placeholder, or non-ASCII API keys before HTTP use."""

    value = (value or "").strip()
    if not value:
        raise ValueError(f"{name} is empty")
    lowered = value.lower()
    placeholder_tokens: Iterable[str] = (
        "replace",
        "changeme",
        "your_key",
        "your-key",
        "api_key_here",
        "替换",
        "你的",
        "密钥",
    )
    if any(token in lowered for token in placeholder_tokens):
        raise ValueError(f"{name} still contains a placeholder")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain ASCII characters only") from exc
    return value
