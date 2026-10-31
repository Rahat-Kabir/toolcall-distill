import json
from pathlib import Path
import subprocess
import sys

import pytest

from bfcl_scoring import score_bfcl_predictions


def score_arguments(arguments, alternatives, raw_output=None):
    prediction = {"id": "example", "predicted_tool_calls": [
        {"name": "distance", "arguments": arguments},
    ]}
    if raw_output is not None:
        prediction = {"id": "example", "raw_output": raw_output}
    return score_bfcl_predictions(
        [prediction],
        [{"id": "example", "bfcl_id": "simple_1"}],
        [{"id": "simple_1", "ground_truth": [{"distance": alternatives}]}],
    )


def test_accepts_every_listed_alternative():
    result = score_arguments({"unit": "miles"}, {"unit": ["mi", "miles", "mile"]})
    assert result["counts"]["are_arguments_correct"] == 1


def test_accepts_optional_omission_even_when_a_default_is_listed_first():
    result = score_arguments({}, {"speed": [299792458, ""]})
    assert result["counts"]["are_arguments_correct"] == 1


@pytest.mark.parametrize("arguments,alternatives", [
    ({"unit": "kilometres"}, {"unit": ["mi", "miles"]}),
    ({}, {"unit": ["mi"]}),
    ({"extra": 1}, {}),
    ({"speed": ""}, {"speed": [299792458, ""]}),
    ({"value": True}, {"value": [1]}),
    ({"values": [2, 1]}, {"values": [[1, 2]]}),
])
def test_rejects_wrong_values_missing_required_and_unlisted_arguments(arguments, alternatives):
    result = score_arguments(arguments, alternatives)
    assert result["counts"]["are_arguments_correct"] == 0


def test_compares_nested_alternatives_using_existing_argument_normalization():
    result = score_arguments(
        {"settings": {"values": [" 2 ", "true"]}},
        {"settings": [{"values": [2, True]}]},
    )
    assert result["counts"]["are_arguments_correct"] == 1


@pytest.mark.parametrize("raw_output", ["hello", "[]", '[wrong(unit="mi")]',
    '[distance(unit="mi"), distance(unit="mi")]',
])
def test_raw_outputs_require_the_expected_tool_name_and_call_count(raw_output):
    result = score_arguments({}, {"unit": ["mi"]}, raw_output=raw_output)
    assert result["counts"]["are_arguments_correct"] == 0


def test_matches_predictions_by_id_and_counts_missing_predictions_as_failures():
    result = score_bfcl_predictions(
        [{"id": "second", "raw_output": "[distance()]"}],
        [{"id": "first", "bfcl_id": "simple_1"}, {"id": "second", "bfcl_id": "simple_1"}],
        [{"id": "simple_1", "ground_truth": [{"distance": {}}]}],
    )
    assert result["example_count"] == 2
    assert result["missing_prediction_count"] == 1
    assert [row["are_arguments_correct"] for row in result["examples"]] == [False, True]


@pytest.mark.parametrize("predictions", [
    [{"id": "unknown", "raw_output": "[distance()]"}],
    [{"id": "example", "raw_output": "[distance()]"}] * 2,
])
def test_rejects_unknown_and_duplicate_prediction_ids(predictions):
    with pytest.raises(ValueError):
        score_bfcl_predictions(predictions, [{"id": "example", "bfcl_id": "simple_1"}],
                               [{"id": "simple_1", "ground_truth": [{"distance": {}}]}])


@pytest.mark.parametrize("answers", [[], [{"id": "simple_1", "ground_truth": []}],
    [{"id": "simple_1", "ground_truth": [{"distance": {"unit": []}}]}],
])
def test_rejects_missing_or_malformed_source_answers(answers):
    with pytest.raises(ValueError):
        score_bfcl_predictions([], [{"id": "example", "bfcl_id": "simple_1"}], answers)


def test_preserves_multi_call_order():
    result = score_bfcl_predictions(
        [{"id": "example", "raw_output": "[second(), first()]"}],
        [{"id": "example", "bfcl_id": "multiple_1"}],
        [{"id": "multiple_1", "ground_truth": [{"first": {}}, {"second": {}}]}],
    )
    assert result["counts"]["is_correct_tool_name"] == 0


def test_empty_alternative_allows_omission_but_not_an_arbitrary_value():
    assert score_arguments({}, {"optional": [""]})["counts"]["are_arguments_correct"] == 1
    assert score_arguments({"optional": "anything"}, {"optional": [""]})["counts"]["are_arguments_correct"] == 0


def test_accepts_multiple_calls_with_alternatives_in_the_expected_order():
    result = score_bfcl_predictions(
        [{"id": "example", "raw_output": '[first(unit="miles"), second()]'}],
        [{"id": "example", "bfcl_id": "multiple_1"}],
        [{"id": "multiple_1", "ground_truth": [
            {"first": {"unit": ["mi", "miles"]}}, {"second": {"optional": [10, ""]}},
        ]}],
    )
    assert result["counts"]["are_arguments_correct"] == 1


def test_cli_reports_strict_and_alternative_scores_without_changing_inputs(tmp_path):
    def write_rows(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    raw_directory = tmp_path / "raw"
    for category in ("simple", "multiple"):
        write_rows(raw_directory / "possible_answer" / f"BFCL_v3_{category}.json", [
            {"id": f"{category}_1", "ground_truth": [{"distance": {"unit": ["mi", "miles"]}}]},
        ])
    gold_path = tmp_path / "gold.jsonl"
    write_rows(gold_path, [{"id": "example", "bfcl_id": "simple_1", "gold_tool_calls": [
        {"name": "distance", "arguments": {"unit": "mi"}},
    ]}])
    base_path = tmp_path / "base.jsonl"
    other_path = tmp_path / "other.jsonl"
    write_rows(base_path, [{"id": "example", "raw_output": '[distance(unit="mi")]'}])
    write_rows(other_path, [{"id": "example", "raw_output": '[distance(unit="miles")]'}])
    original_gold = gold_path.read_bytes()
    output_path = tmp_path / "report.json"
    script_path = Path(__file__).resolve().parents[1] / "bfcl_scoring.py"
    completed = subprocess.run([
        sys.executable, str(script_path), "--gold", str(gold_path),
        "--raw-directory", str(raw_directory), "--prediction", f"base={base_path}",
        "--prediction", f"other={other_path}", "--output", str(output_path),
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["models"]["other"]["strict"]["rates"]["are_arguments_correct"] == 0
    assert report["models"]["other"]["alternative_aware"]["rates"]["are_arguments_correct"] == 1
    assert report["paired_comparisons"]["other_vs_base"]["strict"]["observed_delta"] == -1
    assert report["paired_comparisons"]["other_vs_base"]["alternative_aware"]["delta_ci95"] == [0, 0]
    assert gold_path.read_bytes() == original_gold
