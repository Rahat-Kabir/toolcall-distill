"""Render experiment charts from saved summaries, without running inference."""

import argparse
import json
import math
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHART_FILES = {
    "before-after": ("comparison.json", "before_after_toolcall.png"),
    "ablation": ("ablation/summary.json", "ablation_curve.png"),
    "models": ("models/comparison.json", "model_comparison.png"),
    "mixed": ("larger_eval/summary.json", "mixed_vs_original.png"),
}
ALTERNATIVE_AWARE_FILES = {
    "ablation": "bfcl_alternative_scores_200.json",
    "models": "bfcl_alternative_scores_200.json",
    "mixed": "larger_eval/bfcl_alternative_scores.json",
}
MODEL_CHART_LABELS_TO_SCORES_MODEL = {
    "LFM2.5-1.2B base": "base",
    "LFM2.5+LoRA ~9.6k": "lora",
    "Qwen3-4B-Instruct": "qwen",
    "Llama-3.1-8B-Instruct": "llama",
}
ABLATION_CHART_POSITIONS_TO_SCORES_MODEL = {
    0: "base",
    1000: "lora_1k",
    3000: "lora_3k",
    9600: "lora",
}
ALTERNATIVE_AWARE_FOOTNOTE = (
    "BFCL: local alternative-aware matching against source answer\n"
    "alternatives; not official BFCL AST or execution evaluation."
)


def percentage(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Chart rates must be finite numbers between 0 and 1")
    return 100 * value


def percentage_interval(interval, value):
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("Confidence interval must contain two endpoints")
    lower, upper = [percentage(endpoint) for endpoint in interval]
    if not lower <= value <= upper:
        raise ValueError("Confidence interval must contain its observed rate")
    return [lower, upper]


def alternative_aware_argument_rate(scores, model_name):
    models = scores.get("models")
    if not isinstance(models, dict) or model_name not in models:
        raise ValueError(f"alternative-aware scores lack model {model_name!r}")
    try:
        return models[model_name]["alternative_aware"]["rates"]["are_arguments_correct"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"model {model_name!r} lacks alternative-aware argument rates") from error


def alternative_aware_argument_intervals(scores):
    """One interval per model: base from any vs-base comparison, others from their own."""
    comparisons = scores.get("paired_comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        raise ValueError("alternative-aware scores lack paired comparisons")
    intervals = {}
    for comparison_name, comparison in comparisons.items():
        if not comparison_name.endswith("_vs_base"):
            continue
        try:
            intervals.setdefault("base", comparison["alternative_aware"]["base_ci95"])
            intervals[comparison_name[: -len("_vs_base")]] = comparison["alternative_aware"]["other_ci95"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"comparison {comparison_name!r} lacks alternative-aware intervals") from error
    return intervals


def alternative_aware_model_names(chart_name, rows):
    if chart_name == "models":
        try:
            return [MODEL_CHART_LABELS_TO_SCORES_MODEL[row["label"]] for row in rows]
        except KeyError as error:
            raise ValueError(f"no alternative-aware BFCL scores mapped for model label {error}") from error
    if chart_name == "ablation":
        try:
            return [ABLATION_CHART_POSITIONS_TO_SCORES_MODEL[row["n_train"]] for row in rows]
        except KeyError as error:
            raise ValueError(f"no alternative-aware BFCL scores mapped for training count {error}") from error
    return [row["label"] for row in rows]


def prepare_chart_specification(chart_name, summary, alternative_aware_scores=None):
    chart = {
        "filename": CHART_FILES[chart_name][1], "kind": "bars",
        "ylabel": "Examples passing (%)", "footnote": "", "rotation": 0,
    }
    if chart_name == "before-after":
        metrics = ["is_parseable_tool_call", "is_correct_tool_name", "are_arguments_correct"]
        chart.update(title="Tool calling: base vs ToolACE LoRA (greedy)", value_labels=True,
                     labels=["Parseable", "Correct names/count", "Strict arguments"], series=[
            {"label": label, "values": [percentage(summary[key]["rates"][metric]) for metric in metrics]}
            for key, label in [("baseline", "Base LFM2.5-1.2B"), ("posttrain", "+ ToolACE LoRA")]
        ])
        return chart

    rows = summary["points" if chart_name == "ablation" else "models"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Chart needs a nonempty list of result rows")
    if not isinstance(alternative_aware_scores, dict):
        raise ValueError(f"{chart_name} chart requires alternative-aware BFCL scores")
    scores_model_names = alternative_aware_model_names(chart_name, rows)
    series = [
        {"label": "ToolACE strict arguments",
         "values": [percentage(row["toolace_args"]) for row in rows]},
        {"label": "BFCL alternative-aware arguments",
         "values": [percentage(alternative_aware_argument_rate(alternative_aware_scores, model_name))
                    for model_name in scores_model_names]},
    ]
    footnote = ALTERNATIVE_AWARE_FOOTNOTE
    chart.update(
        labels=[str(row["n_train"]) if chart_name == "ablation" else row["label"] for row in rows],
        series=series,
        footnote=footnote,
    )
    if not all(isinstance(label, str) and label.strip() for label in chart["labels"]):
        raise ValueError("Chart labels must be nonempty strings")
    if chart_name == "ablation":
        positions = [row["n_train"] for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in positions):
            raise ValueError("Ablation example counts must be finite nonnegative numbers")
        chart.update(kind="lines", positions=positions,
                     title="Ordered-prefix data ablation (single run)",
                     xlabel="Training examples (full endpoint: examples seen)")
        chart["footnote"] += "\nFull endpoint approximates steps x batch; prefix subsets are not random samples."
    elif chart_name == "models":
        chart.update(title="Model comparison (greedy)", rotation=15)
    else:
        chart.update(title="Mixed vs ToolACE-only vs base (greedy, 95% CI)", rotation=15)
        score_intervals = alternative_aware_argument_intervals(alternative_aware_scores)
        chart["series"][0]["intervals"] = [
            percentage_interval(row["toolace_args_ci95"], value)
            for row, value in zip(rows, chart["series"][0]["values"])
        ]
        chart["series"][1]["intervals"] = [
            percentage_interval(score_intervals[row["label"]], value)
            for row, value in zip(rows, chart["series"][1]["values"])
        ]
    return chart


def load_chart_specifications(results_directory, chart_name="all"):
    """Read and validate all selected inputs before any image is written."""
    names = list(CHART_FILES) if chart_name == "all" else [chart_name]
    charts = []
    for name in names:
        source_path = Path(results_directory) / CHART_FILES[name][0]
        summary = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"{source_path}: expected a JSON object")
        alternative_aware_scores = None
        if name in ALTERNATIVE_AWARE_FILES:
            scores_path = Path(results_directory) / ALTERNATIVE_AWARE_FILES[name]
            alternative_aware_scores = json.loads(scores_path.read_text(encoding="utf-8"))
            if not isinstance(alternative_aware_scores, dict):
                raise ValueError(f"{scores_path}: expected a JSON object")
        try:
            charts.append(prepare_chart_specification(name, summary, alternative_aware_scores))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source_path}: invalid chart data: {error}") from error
    return charts


def render_chart(chart, output_directory):
    # Plotting remains optional: validation and the default tests need only Python.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plotting

    figure, axes = plotting.subplots(figsize=(10, 5.5))
    colors = ["#219ebc", "#fb8500"]
    positions = chart.get("positions", list(range(len(chart["labels"]))))
    try:
        for index, series in enumerate(chart["series"]):
            if chart["kind"] == "lines":
                axes.plot(positions, series["values"], "o-" if index == 0 else "s--",
                          label=series["label"], color=colors[index])
                if index == 0:
                    for position, value in zip(positions, series["values"]):
                        axes.text(position, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
            else:
                bar_positions = [position + (index - 0.5) * 0.35 for position in positions]
                error_bars = None
                if "intervals" in series:
                    error_bars = [
                        [value - interval[0] for value, interval in zip(series["values"], series["intervals"])],
                        [interval[1] - value for value, interval in zip(series["values"], series["intervals"])],
                    ]
                axes.bar(bar_positions, series["values"], 0.35, label=series["label"],
                         color=colors[index], yerr=error_bars, capsize=4)
                if chart.get("value_labels"):
                    for position, value in zip(bar_positions, series["values"]):
                        axes.text(position, value + 1.5, f"{value:.1f}", ha="center", fontsize=9)
        axes.set_xticks(positions)
        axes.set_xticklabels(chart["labels"], rotation=chart["rotation"],
                            ha="right" if chart["rotation"] else "center")
        axes.set_ylim(0, 105)
        axes.set_ylabel(chart["ylabel"])
        axes.set_xlabel(chart.get("xlabel", ""))
        axes.set_title(chart["title"])
        axes.grid(axis="y", alpha=0.3)
        axes.legend(loc="lower right")
        bottom_margin = 0.18 if chart["footnote"] else 0
        figure.tight_layout(rect=(0, bottom_margin, 1, 1))
        if chart["footnote"]:
            figure.text(0.5, 0.025, chart["footnote"], ha="center", fontsize=8)
        output_path = Path(output_directory) / chart["filename"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=140)
    finally:
        plotting.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=PROJECT_DIRECTORY / "results")
    parser.add_argument("--output-directory", type=Path, help="Default: same as --results-directory")
    parser.add_argument("--chart", choices=["all", *CHART_FILES], default="all")
    parser.add_argument("--check-inputs", action="store_true", help="Validate selected summaries without plotting")
    arguments = parser.parse_args()
    try:
        charts = load_chart_specifications(arguments.results_directory, arguments.chart)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if arguments.check_inputs:
        print(f"INPUTS_OK {len(charts)} chart(s)")
        return
    output_directory = arguments.output_directory or arguments.results_directory
    for chart in charts:
        print(f"Wrote {render_chart(chart, output_directory)}")


if __name__ == "__main__":
    main()
