"""Research-level, fail-closed lock on every data-annotation API path.

The user's standing constraint for this study is **zero new annotation requests**: no new calls,
no retries, no cache-completion calls, no health probes, no second judge, no label repair.  Only
the three already-landed teacher caches may be read.

This module is deliberately not configurable by environment variables, CLI flags or provider
settings.  The only way to annotate is to edit the source constant, which is outside the study's
authorised action space.  The guard is called *before* the API key is read and *before* any
network object is constructed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/root/sdsafe-compute/v6-param")
ART = ROOT / "artifacts"
FROZEN_MANIFEST = ART / "teacher/FROZEN_MANIFEST.json"

# Frozen authority: never flip this in the cache-only study.
ANNOTATIONS_LOCKED = True
ANNOTATION_REQUESTS_ALLOWED_PER_RUN = 0


class AnnotationLocked(RuntimeError):
    """Raised before any credential read or network request."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_no_annotations(context: str = "") -> None:
    """Fail closed.  Called at every annotation entry point before credential/network use.

    There is intentionally no environment/CLI escape hatch.  ``ANNOTATION_REQUESTS_ALLOWED_PER_RUN``
    is frozen at 0 for the cache-only study; even a mistaken future change keeps the lock on.
    """
    if ANNOTATION_REQUESTS_ALLOWED_PER_RUN != 0 or ANNOTATIONS_LOCKED:
        raise AnnotationLocked(
            "NO_NEW_ANNOTATIONS: data-annotation API is frozen for this study; "
            "new/retried/health-probe/second-judge/label-repair calls are forbidden "
            f"(context={context or 'unspecified'})")
    # Unreachable with the frozen constants above; kept fail-closed if someone edits them.
    raise AnnotationLocked(
        f"NO_NEW_ANNOTATIONS: annotation budget is permanently zero (context={context or 'unspecified'})")


def frozen_manifest(create: bool = False) -> dict[str, Any]:
    """Describe the frozen teacher caches and the zero-request policy.

    ``create=True`` writes the manifest once.  Verification only reads it; it never enables a
    call.  A missing or changed file is reported, not repaired.
    """
    files = {}
    for path in sorted((ART / "teacher").glob("*.jsonl")):
        files[path.name] = {"bytes": path.stat().st_size, "lines": len(path.read_text().splitlines()),
                            "sha256": _sha(path)}
    payload: dict[str, Any] = {
        "policy": "NO_NEW_ANNOTATIONS",
        "annotation_requests_allowed_per_run": ANNOTATION_REQUESTS_ALLOWED_PER_RUN,
        "provider": "deepseek",
        "model": "deepseek-flash",
        "credentials_must_not_be_read": True,
        "network_requests_allowed": 0,
        "frozen_teacher_caches": files,
        "view_version": "v6param-view-3",
        "created_by": "cache-only-r1",
    }
    if create:
        FROZEN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        FROZEN_MANIFEST.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def verify_frozen_manifest() -> dict[str, Any]:
    """Compare the current cache bytes with the frozen manifest; report drift (never fix it)."""
    if not FROZEN_MANIFEST.is_file():
        return {"ok": False, "reason": "missing FROZEN_MANIFEST.json", "files": {}}
    frozen = json.loads(FROZEN_MANIFEST.read_text())
    current = frozen_manifest(create=False)
    drift = {}
    for name, meta in frozen.get("frozen_teacher_caches", {}).items():
        now = current["frozen_teacher_caches"].get(name)
        if now != meta:
            drift[name] = {"frozen": meta, "now": now}
    return {"ok": not drift and ANNOTATION_REQUESTS_ALLOWED_PER_RUN == 0,
            "drift": drift, "manifest_sha256": _sha(FROZEN_MANIFEST),
            "requests_allowed": ANNOTATION_REQUESTS_ALLOWED_PER_RUN}


if __name__ == "__main__":
    print(json.dumps(frozen_manifest(create=True), indent=1))
    print(json.dumps(verify_frozen_manifest(), indent=1))
