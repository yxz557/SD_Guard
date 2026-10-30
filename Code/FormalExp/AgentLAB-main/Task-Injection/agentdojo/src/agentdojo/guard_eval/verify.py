from __future__ import annotations

import argparse
from pathlib import Path

from agentdojo.guard_eval.artifacts import discover_files, verify_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable SD_Guard raw-trace hashes.")
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    traces = discover_files(args.input, "raw_trace.json")
    if not traces:
        raise SystemExit("No raw_trace.json files found.")
    for trace in traces:
        digest = verify_trace(trace)
        print(f"OK {digest}  {trace}")
    print(f"Verified {len(traces)} trace(s).")


if __name__ == "__main__":
    main()
