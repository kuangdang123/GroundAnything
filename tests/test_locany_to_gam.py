import json
from pathlib import Path

import pytest

from tools.convert_locany_to_gam import convert_record, convert_file
from tools.validate_gam_jsonl import validate_file


def _record(answer="<ref>button</ref><box><0><10><1000><500></box>"):
    return {
        "conversations": [
            {"from": "human", "value": "<image-1> Locate the button."},
            {"from": "gpt", "value": answer},
        ],
        "image": "screens/a.png",
    }


def test_convert_maps_roles_grid_and_image_placeholder():
    row, metadata = convert_record(_record(), source="game", line_number=3, task="bbox", default_phrase="target")
    assert row["images"] == ["screens/a.png"]
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["content"] == (
        "<|object_ref_start|>button<|object_ref_end|>"
        "<|box_start|><0><10><999><500><|box_end|>"
    )
    assert metadata["coordinate_grid"] == {"input": 1000, "output": 999}


def test_convert_prepends_missing_image_placeholder():
    row, _ = convert_record(
        {"conversations": [{"from": "human", "value": "Find it"}, {"from": "gpt", "value": "<ref>x</ref><box><1><2><3><4></box>"}], "image": "a.png"},
        source="x", line_number=1, task="bbox", default_phrase="target",
    )
    assert row["messages"][0]["content"].startswith("<image>")


def test_negative_becomes_explicit_gam_none_entry():
    row, _ = convert_record(_record("<box>none</box>"), source="game", line_number=1, task="bbox", default_phrase="target")
    assert row["messages"][1]["content"].endswith("<|box_start|>None<|box_end|>")
    assert "button" in row["messages"][1]["content"] or "Locate the button" in row["messages"][1]["content"]


def test_converter_and_validator_round_trip(tmp_path: Path):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    assert convert_file(source, output)["written"] == 1
    assert validate_file(output)["valid"] == 1


def test_mismatched_image_placeholders_are_rejected():
    bad = _record()
    bad["conversations"][0]["value"] = "<image-1> <image-2> find"
    with pytest.raises(ValueError, match="placeholder count"):
        convert_record(bad, source="x", line_number=1, task="bbox", default_phrase="target")
