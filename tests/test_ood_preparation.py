"""Behavior tests for BFCL OOD conversion."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_ood_bfcl.py"


def load_prepare_ood_module():
    spec = importlib.util.spec_from_file_location("prepare_ood_bfcl", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_ood_bfcl"] = module
    spec.loader.exec_module(module)
    return module


prepare_ood_bfcl = load_prepare_ood_module()


def test_first_nonempty_alternative_skips_empty_strings():
    assert prepare_ood_bfcl.first_nonempty_alternative(["", "2023-2024"]) == "2023-2024"
    assert prepare_ood_bfcl.first_nonempty_alternative(["", ""]) is None


def test_gold_omits_arguments_whose_alternatives_are_all_empty():
    ground_truth = [{
        "demo_tool": {
            "required_argument": ["value"],
            "optional_argument": ["", ""],
        }
    }]
    assert prepare_ood_bfcl.gold_tool_calls_from_ground_truth(ground_truth) == [
        {"name": "demo_tool", "arguments": {"required_argument": "value"}}
    ]


def test_shallow_schema_normalization_maps_top_property_float_but_keeps_nested_float():
    parameters = {
        "type": "dict",
        "properties": {
            "gravity": {"type": "float", "description": "g"},
            "budget": {
                "type": "dict",
                "properties": {"min": {"type": "float"}},
            },
            "values": {"type": "array", "items": {"type": "float"}},
        },
    }
    normalized_schema = prepare_ood_bfcl.normalize_bfcl_parameter_schema(parameters)
    assert normalized_schema["type"] == "object"
    assert normalized_schema["properties"]["gravity"]["type"] == "number"
    assert normalized_schema["properties"]["budget"]["type"] == "object"
    assert normalized_schema["properties"]["budget"]["properties"]["min"]["type"] == "float"
    assert normalized_schema["properties"]["values"]["items"]["type"] == "float"


@pytest.mark.integration
def test_build_ood_examples_reproduces_prepared_ood_test_file():
    converted_examples, report = prepare_ood_bfcl.build_ood_examples(
        ROOT / "data" / "ood_raw",
        ROOT / "data" / "ood_conversion_notes.json",
    )
    assert report["skipped"] == []
    assert report["n"] == 200
    committed_text = (ROOT / "data" / "ood_test.jsonl").read_text(encoding="utf-8")
    rebuilt_text = "".join(
        json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n"
        for example in converted_examples
    )
    assert rebuilt_text == committed_text
