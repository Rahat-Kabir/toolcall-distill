"""Chart data and CLI checks do not require optional plotting packages."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))


def write_summary(directory, relative_path, summary):
    path = directory / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary), encoding="utf-8")


def write_alternative_aware_scores(tmp_path):
    write_summary(tmp_path, "larger_eval/bfcl_alternative_scores.json", {
        "models": {
            "base": {"alternative_aware": {"rates": {"are_arguments_correct": 0.878}}},
            "control": {"alternative_aware": {"rates": {"are_arguments_correct": 0.904}}},
        },
        "paired_comparisons": {
            "control_vs_base": {"alternative_aware": {
                "base_ci95": [0.848, 0.906], "other_ci95": [0.878, 0.93],
            }},
        },
    })


def test_mixed_chart_uses_alternative_aware_bfcl_scores_and_intervals(tmp_path):
    from make_charts import load_chart_specifications

    write_summary(tmp_path, "larger_eval/summary.json", {"models": [
        {"label": "base", "toolace_args": 0.628, "ood_args": 0.596,
         "toolace_args_ci95": [0.584, 0.67], "ood_args_ci95": [0.552, 0.638]},
        {"label": "control", "toolace_args": 0.728, "ood_args": 0.542,
         "toolace_args_ci95": [0.69, 0.768], "ood_args_ci95": [0.498, 0.586]},
    ]})
    write_alternative_aware_scores(tmp_path)
    chart = load_chart_specifications(tmp_path, "mixed")[0]
    assert chart["filename"] == "mixed_vs_original.png"
    assert chart["labels"] == ["base", "control"]
    assert chart["series"][0]["label"] == "ToolACE strict arguments"
    assert chart["series"][0]["values"] == pytest.approx([62.8, 72.8])
    assert chart["series"][0]["intervals"][1] == pytest.approx([69.0, 76.8])
    assert chart["series"][1]["label"] == "BFCL alternative-aware arguments"
    assert chart["series"][1]["values"] == pytest.approx([87.8, 90.4])
    assert chart["series"][1]["intervals"][0] == pytest.approx([84.8, 90.6])
    assert chart["series"][1]["intervals"][1] == pytest.approx([87.8, 93.0])
    assert "alternative-aware" in chart["footnote"]


def test_models_chart_uses_mapped_alternative_aware_bfcl_scores(tmp_path):
    from make_charts import load_chart_specifications

    write_summary(tmp_path, "models/comparison.json", {"models": [
        {"label": "LFM2.5-1.2B base", "toolace_args": 0.63, "ood_args": 0.58},
        {"label": "Llama-3.1-8B-Instruct", "toolace_args": 0.485, "ood_args": 0.585},
    ]})
    write_summary(tmp_path, "bfcl_alternative_scores_200.json", {"models": {
        "base": {"alternative_aware": {"rates": {"are_arguments_correct": 0.89}}},
        "llama": {"alternative_aware": {"rates": {"are_arguments_correct": 0.77}}},
    }})
    chart = load_chart_specifications(tmp_path, "models")[0]
    assert chart["filename"] == "model_comparison.png"
    assert chart["series"][0]["values"] == pytest.approx([63.0, 48.5])
    assert chart["series"][1]["label"] == "BFCL alternative-aware arguments"
    assert chart["series"][1]["values"] == pytest.approx([89.0, 77.0])


def test_models_chart_rejects_unmapped_model_labels(tmp_path):
    from make_charts import load_chart_specifications

    write_summary(tmp_path, "models/comparison.json", {"models": [
        {"label": "LFM2.5-1.2B base", "toolace_args": 0.63, "ood_args": 0.58},
        {"label": "Unknown model", "toolace_args": 0.5, "ood_args": 0.5},
    ]})
    write_summary(tmp_path, "bfcl_alternative_scores_200.json", {"models": {
        "base": {"alternative_aware": {"rates": {"are_arguments_correct": 0.89}}},
    }})
    with pytest.raises(ValueError, match="Unknown model"):
        load_chart_specifications(tmp_path, "models")


def test_check_inputs_needs_only_the_selected_chart_and_writes_no_images(tmp_path):
    write_summary(tmp_path, "ablation/summary.json", {"points": [
        {"n_train": 0, "toolace_args": 0.63, "ood_args": 0.58},
        {"n_train": 1000, "toolace_args": 0.675, "ood_args": 0.505},
    ]})
    write_summary(tmp_path, "bfcl_alternative_scores_200.json", {"models": {
        "base": {"alternative_aware": {"rates": {"are_arguments_correct": 0.89}}},
        "lora_1k": {"alternative_aware": {"rates": {"are_arguments_correct": 0.9}}},
    }})
    output_directory = tmp_path / "images"
    completed = subprocess.run([
        sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts/make_charts.py"),
        "--chart", "ablation", "--results-directory", str(tmp_path),
        "--output-directory", str(output_directory), "--check-inputs",
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "INPUTS_OK 1 chart" in completed.stdout
    assert not output_directory.exists()


def test_ablation_chart_uses_alternative_aware_bfcl_scores(tmp_path):
    from make_charts import load_chart_specifications

    write_summary(tmp_path, "ablation/summary.json", {"points": [
        {"n_train": 0, "toolace_args": 0.63, "ood_args": 0.58},
        {"n_train": 1000, "toolace_args": 0.675, "ood_args": 0.505},
    ]})
    write_summary(tmp_path, "bfcl_alternative_scores_200.json", {"models": {
        "base": {"alternative_aware": {"rates": {"are_arguments_correct": 0.89}}},
        "lora_1k": {"alternative_aware": {"rates": {"are_arguments_correct": 0.9}}},
    }})
    chart = load_chart_specifications(tmp_path, "ablation")[0]
    assert chart["kind"] == "lines"
    assert chart["series"][0]["label"] == "ToolACE strict arguments"
    assert chart["series"][0]["values"] == pytest.approx([63.0, 67.5])
    assert chart["series"][1]["label"] == "BFCL alternative-aware arguments"
    assert chart["series"][1]["values"] == pytest.approx([89.0, 90.0])
    assert "alternative-aware" in chart["footnote"]


def test_all_charts_require_all_sources_before_rendering(tmp_path):
    completed = subprocess.run([
        sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts/make_charts.py"),
        "--results-directory", str(tmp_path), "--check-inputs",
    ], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "comparison.json" in completed.stderr
    assert "No module named" not in completed.stderr


def test_rejects_invalid_confidence_interval_instead_of_drawing_it(tmp_path):
    from make_charts import load_chart_specifications

    write_summary(tmp_path, "larger_eval/summary.json", {"models": [{
        "label": "base", "toolace_args": 0.63, "ood_args": 0.58,
        "toolace_args_ci95": [0.7, 0.8], "ood_args_ci95": [0.552, 0.638],
    }]})
    write_alternative_aware_scores(tmp_path)
    with pytest.raises(ValueError, match="interval"):
        load_chart_specifications(tmp_path, "mixed")
