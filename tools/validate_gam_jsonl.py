"""Validate GroundAnything JSONL rows and GAM answer grammar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grounding_anything.protocol import TaskType, validate_answer
from train.data.row_contract import RecordReject, validate_uniform_row


def validate_file(path: Path, *, task: TaskType = TaskType.BBOX) -> dict[str, int]:
    counts = {"rows": 0, "valid": 0, "invalid": 0}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            counts["rows"] += 1
            try:
                row = json.loads(raw)
                validate_uniform_row(row)
                for message in row["messages"]:
                    if message["role"] == "assistant":
                        validate_answer(message["content"], task)
                counts["valid"] += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, RecordReject) as error:
                counts["invalid"] += 1
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--task", choices=[item.value for item in TaskType if item is not TaskType.TRAJECTORY], default="bbox")
    args = parser.parse_args()
    print(json.dumps(validate_file(args.input, task=TaskType(args.task)), ensure_ascii=False))


if __name__ == "__main__":
    main()
