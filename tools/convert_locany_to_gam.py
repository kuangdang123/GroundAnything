"""Convert LocateAnything ShareGPT annotations to GroundAnything GAM JSONL.

The converter is deliberately strict at the boundary: it never edits the input
file, rewrites LA's 0..1000 coordinates onto GAM's 0..999 grid, and validates
every emitted row against the canonical GAM row contract.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from grounding_anything.protocol import (
    GroundingEntry,
    TaskType,
    canonicalize_phrase,
    serialize_answer,
)
from train.data.row_contract import RecordReject, validate_uniform_row


_LA_BBOX_TOKEN_RE = re.compile(
    r"<ref>(?P<label>.*?)</ref>"
    r"|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
    r"|<box>none</box>",
    re.DOTALL,
)
_LA_POINT_TOKEN_RE = re.compile(
    r"<ref>(?P<label>.*?)</ref>"
    r"|<box><(?P<x1>\d+)><(?P<y1>\d+)></box>"
    r"|<box>none</box>",
    re.DOTALL,
)
_IMAGE_TOKEN_RE = re.compile(r"<image(?:-(\d+))?>")


def _grid_1000_to_999(value: int) -> int:
    if not 0 <= value <= 1000:
        raise ValueError(f"LocateAnything coordinate outside [0, 1000]: {value}")
    return min(999, round(value * 999 / 1000))


def _convert_box(values: Iterable[str]) -> tuple[int, int, int, int]:
    converted = tuple(_grid_1000_to_999(int(value)) for value in values)
    x1, y1, x2, y2 = converted
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate box after 0..999 conversion: {converted}")
    return converted


def _safe_phrase(value: str, fallback: str) -> str:
    candidate = " ".join(str(value).split()).strip(" .。")
    if not candidate or any(token in candidate for token in ("<", ">")):
        candidate = fallback
    return canonicalize_phrase(candidate)


def _infer_negative_phrase(prompt: str, fallback: str) -> str:
    """Extract the target portion of common LA prompts for a GAM None entry."""

    text = _IMAGE_TOKEN_RE.sub("", prompt).strip()
    for marker in (":", "："):
        if marker in text:
            text = text.rsplit(marker, 1)[1].strip()
    return _safe_phrase(text, fallback)


def _image_paths(record: dict[str, Any]) -> list[str]:
    values: Any
    if "image_list" in record:
        values = record["image_list"]
    elif "image" in record:
        values = [record["image"]]
    else:
        values = []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
        raise ValueError("LA record image/image_list must contain non-empty strings")
    return values


def _messages(record: dict[str, Any], images: list[str]) -> tuple[list[dict[str, Any]], str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("LA record conversations must be a non-empty list")
    messages: list[dict[str, Any]] = []
    first_user = None
    for item in conversations:
        if not isinstance(item, dict) or not isinstance(item.get("value"), str):
            raise ValueError("conversation turns need from and string value")
        speaker = item.get("from")
        role = {"human": "user", "gpt": "assistant", "system": "system"}.get(speaker)
        if role is None:
            raise ValueError(f"unsupported LA speaker: {speaker!r}")
        content = item["value"]
        content = _IMAGE_TOKEN_RE.sub("<image>", content)
        if role == "user" and first_user is None:
            first_user = content
        messages.append({"role": role, "content": content, "loss": None})
    if not images:
        return messages, first_user or "target", first_user or "target"
    used = sum(message["content"].count("<image>") for message in messages)
    if used == 0:
        for message in messages:
            if message["role"] == "user":
                message["content"] = "<image>\n" + message["content"]
                used = 1
                break
    if used != len(images):
        raise ValueError(f"image placeholder count {used} does not match images {len(images)}")
    return messages, first_user or "target", first_user or "target"


def _answer_to_gam(answer: str, prompt: str, *, task: TaskType, default_phrase: str) -> str:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("assistant answer must be a non-empty string")
    entries: list[GroundingEntry] = []
    active_label = default_phrase
    saw_box = False
    saw_none = False
    token_re = _LA_POINT_TOKEN_RE if task is TaskType.POINT or task == TaskType.POINT.value else _LA_BBOX_TOKEN_RE
    coordinate_names = ("x1", "y1") if task is TaskType.POINT or task == TaskType.POINT.value else ("x1", "y1", "x2", "y2")
    for match in token_re.finditer(answer):
        label = match.group("label")
        if label is not None:
            active_label = _safe_phrase(label, default_phrase)
            continue
        if match.group("x1") is None:
            saw_none = True
            continue
        saw_box = True
        converted = tuple(_grid_1000_to_999(int(match.group(name))) for name in coordinate_names)
        if task is TaskType.BBOX or task == TaskType.BBOX.value:
            x1, y1, x2, y2 = converted
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"degenerate box after 0..999 conversion: {converted}")
        box = converted
        if entries and entries[-1].phrase == active_label and entries[-1].coordinates is not None:
            entries[-1] = GroundingEntry(active_label, entries[-1].coordinates + (box,))
        else:
            entries.append(GroundingEntry(active_label, (box,)))
    if saw_box:
        return serialize_answer(entries, task)
    if saw_none:
        return serialize_answer(
            [GroundingEntry(_infer_negative_phrase(prompt, default_phrase), None)], task
        )
    raise ValueError("assistant answer contains no recognized LA box or none token")


def convert_record(record: dict[str, Any], *, source: str, line_number: int, task: TaskType, default_phrase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    images = _image_paths(record)
    messages, prompt, _ = _messages(record, images)
    assistant = next((message for message in reversed(messages) if message["role"] == "assistant"), None)
    if assistant is None:
        raise ValueError("record has no assistant turn")
    assistant["content"] = _answer_to_gam(
        assistant["content"], prompt, task=task, default_phrase=default_phrase
    )
    row_id = str(record.get("id") or f"locany-{Path(source).stem}-{line_number:08d}")
    row = {"id": row_id, "messages": messages, "images": images, "source": source}
    validate_uniform_row(row)
    metadata = {
        "id": row_id,
        "source": source,
        "input_line": line_number,
        "locany_image_fields": [key for key in ("image", "image_list") if key in record],
        "coordinate_grid": {"input": 1000, "output": 999},
    }
    return row, metadata


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    rejects_path: Path | None = None,
    source: str | None = None,
    task: TaskType = TaskType.BBOX,
    default_phrase: str = "target",
) -> dict[str, int]:
    source_name = source or input_path.as_posix()
    counts = {"input": 0, "written": 0, "rejected": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if rejects_path:
        rejects_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source_file, output_path.open("w", encoding="utf-8", newline="\n") as destination:
        metadata_file = metadata_path.open("w", encoding="utf-8", newline="\n") if metadata_path else None
        rejects_file = rejects_path.open("w", encoding="utf-8", newline="\n") if rejects_path else None
        try:
            for line_number, raw in enumerate(source_file, 1):
                if not raw.strip():
                    continue
                counts["input"] += 1
                try:
                    record = json.loads(raw)
                    row, metadata = convert_record(record, source=source_name, line_number=line_number, task=task, default_phrase=default_phrase)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, RecordReject) as error:
                    counts["rejected"] += 1
                    if rejects_file is None:
                        raise ValueError(f"{input_path}:{line_number}: {error}") from error
                    rejects_file.write(json.dumps({
                        "source": source_name,
                        "line": line_number,
                        "reason": type(error).__name__,
                        "error": str(error),
                        "raw": raw.rstrip("\r\n"),
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    continue
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if metadata_file:
                    metadata_file.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["written"] += 1
        finally:
            if metadata_file:
                metadata_file.close()
            if rejects_file:
                rejects_file.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--rejects-output", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--task", choices=[item.value for item in TaskType if item is not TaskType.TRAJECTORY], default="bbox")
    parser.add_argument("--default-phrase", default="target")
    args = parser.parse_args()
    print(json.dumps(convert_file(args.input, args.output, metadata_path=args.metadata_output, rejects_path=args.rejects_output, source=args.source, task=TaskType(args.task), default_phrase=args.default_phrase), ensure_ascii=False))


if __name__ == "__main__":
    main()
