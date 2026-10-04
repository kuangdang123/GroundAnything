"""Build a GroundAnything Arrow cache from canonical GAM JSONL.

The cache is intentionally produced through ``datasets.Dataset.save_to_disk``
so its state/dataset metadata stay compatible with the existing DLM loader.
Text lengths are tokenizer lengths; an approximate character fallback is only
available when explicitly requested for dry runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from grounding_anything.protocol import TaskType, validate_answer
from train.data.row_contract import validate_uniform_row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "tokenizer length calculation requires transformers; install the training environment"
        ) from error
    return AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)


def _render_for_length(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _token_length(messages: list[dict[str, Any]], tokenizer: Any | None, allow_approximate: bool) -> int:
    if tokenizer is None:
        if not allow_approximate:
            raise RuntimeError(
                "a tokenizer is required for cache lengths; pass --allow-approximate-length only for dry runs"
            )
        return max(1, sum(len(str(message["content"])) for message in messages))
    rendered = _render_for_length(messages, tokenizer)
    encoded = tokenizer(rendered, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    length = len(ids)
    if length <= 0:
        raise ValueError("tokenizer produced an empty row")
    return length


def normalize_row(row: dict[str, Any], *, source_id: str, line_number: int, tokenizer: Any | None, allow_approximate_length: bool, task: TaskType) -> dict[str, Any]:
    validate_uniform_row(row)
    for message in row["messages"]:
        if message["role"] == "assistant":
            validate_answer(message["content"], task)
    images = [{"bytes": None, "path": path} for path in row["images"]]
    return {
        "id": row["id"],
        "messages": row["messages"],
        "images": images,
        "length": _token_length(row["messages"], tokenizer, allow_approximate_length),
    }


def read_rows(input_path: Path, *, source_id: str, tokenizer: Any | None, allow_approximate_length: bool, task: TaskType) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"input": 0, "written": 0, "rejected": 0}
    with input_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            counts["input"] += 1
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("row must be an object")
                rows.append(normalize_row(value, source_id=source_id, line_number=line_number, tokenizer=tokenizer, allow_approximate_length=allow_approximate_length, task=task))
                counts["written"] += 1
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                counts["rejected"] += 1
                raise ValueError(f"{input_path}:{line_number}: {error}") from error
    if not rows:
        raise ValueError(f"no rows found in {input_path}")
    return rows, counts


def write_cache(rows: list[dict[str, Any]], cache_path: Path) -> None:
    try:
        from datasets import Dataset, Features, Sequence, Value
    except ImportError as error:
        raise RuntimeError(
            "Arrow cache export requires the training environment package 'datasets'"
        ) from error
    features = Features({
        "id": Value("string"),
        "messages": Sequence({"role": Value("string"), "content": Value("string"), "loss": Value("float64")}),
        "images": Sequence({"bytes": Value("binary"), "path": Value("string")}),
        "length": Value("int64"),
    })
    dataset = Dataset.from_list(rows, features=features)
    if cache_path.exists():
        if any(cache_path.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty cache directory: {cache_path}")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))


def build_cache(input_path: Path, cache_path: Path, manifest_path: Path, *, source_id: str, tokenizer_path: Path | None = None, allow_approximate_length: bool = False, task: TaskType = TaskType.BBOX) -> dict[str, Any]:
    tokenizer = _load_tokenizer(tokenizer_path) if tokenizer_path else None
    rows, counts = read_rows(input_path, source_id=source_id, tokenizer=tokenizer, allow_approximate_length=allow_approximate_length, task=task)
    write_cache(rows, cache_path)
    manifest = {
        "schema_version": 1,
        "manifest_type": "gam_cache_manifest",
        "source_id": source_id,
        "cache_path": str(cache_path.resolve()),
        "num_rows": len(rows),
        "columns": ["id", "messages", "images", "length"],
        "length_mode": "tokenizer" if tokenizer_path else "approximate_characters",
        "tokenizer_path": str(tokenizer_path.resolve()) if tokenizer_path else None,
        "input_jsonl": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--allow-approximate-length", action="store_true")
    parser.add_argument("--task", choices=[item.value for item in TaskType if item is not TaskType.TRAJECTORY], default="bbox")
    args = parser.parse_args()
    print(json.dumps(build_cache(args.input, args.cache, args.manifest, source_id=args.source_id, tokenizer_path=args.tokenizer, allow_approximate_length=args.allow_approximate_length, task=TaskType(args.task)), ensure_ascii=False))


if __name__ == "__main__":
    main()
