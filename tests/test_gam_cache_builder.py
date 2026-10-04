import json
from pathlib import Path

import pytest

from tools.build_gam_cache import normalize_row, read_rows


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "|".join(message["role"] + ":" + message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def row():
    return {
        "id": "x",
        "messages": [
            {"role": "user", "content": "<image> find x", "loss": None},
            {"role": "assistant", "content": "<|object_ref_start|>x<|object_ref_end|><|box_start|><1><2><3><4><|box_end|>", "loss": None},
        ],
        "images": ["a.png"],
        "source": "test",
    }


def test_normalize_row_creates_collator_image_records_and_length():
    normalized = normalize_row(row(), source_id="test", line_number=1, tokenizer=FakeTokenizer(), allow_approximate_length=False, task="bbox")
    assert normalized["images"] == [{"bytes": None, "path": "a.png"}]
    assert normalized["length"] > 0
    assert set(normalized) == {"id", "messages", "images", "length"}


def test_approximate_length_requires_explicit_opt_in():
    with pytest.raises(RuntimeError, match="tokenizer is required"):
        normalize_row(row(), source_id="test", line_number=1, tokenizer=None, allow_approximate_length=False, task="bbox")
    normalized = normalize_row(row(), source_id="test", line_number=1, tokenizer=None, allow_approximate_length=True, task="bbox")
    assert normalized["length"] > 0


def test_read_rows_is_strict_about_invalid_json(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row()) + "\n" + "not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not-json"):
        read_rows(path, source_id="test", tokenizer=FakeTokenizer(), allow_approximate_length=False, task="bbox")
