#!/usr/bin/env python3
"""Evaluate shared LA/GA game UI grounding prediction JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import aggregate_results, evaluate_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL using the shared sample contract")
    parser.add_argument("--output", help="Write aggregate JSON (default: stdout)")
    args = parser.parse_args()
    rows = []
    with Path(args.input).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(evaluate_sample(json.loads(line)))
            except (TypeError, ValueError, KeyError) as exc:
                parser.error(f"{args.input}:{line_number}: {exc}")
    payload = aggregate_results(rows)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
