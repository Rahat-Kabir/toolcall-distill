"""Score BFCL source alternatives, alongside the unchanged strict metric.

Each argument may equal any listed nonempty-string alternative. The empty
string in source answers means omission is allowed, not a literal empty value.
Argument equality uses eval_toolcalls normalization. Names, call order/count,
and argument keys remain exact. This is not official BFCL AST evaluation:
no AST, language-specific transformations, or function execution is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

from bootstrap_metrics import paired_bootstrap_rate_ci
from eval_toolcalls import (
    aggregate_scores,
    are_argument_values_equal,
    index_examples_by_id,
    parse_tool_calls_from_model_output,
    read_jsonl,
    score_predictions,
    validate_tool_calls,
)


def validate_source_ground_truth(ground_truth):
    if not isinstance(ground_truth, list) or not ground_truth:
        raise ValueError("BFCL ground_truth must be a nonempty list")
    for entry in ground_truth:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("BFCL ground_truth entry must contain one tool")
        name, arguments = next(iter(entry.items()))
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise ValueError("BFCL ground_truth requires a tool name and arguments object")
        for argument_name, alternatives in arguments.items():
            if not isinstance(argument_name, str) or not isinstance(alternatives, list) or not alternatives:
                raise ValueError("BFCL arguments require names and nonempty alternative lists")


def arguments_match_alternatives(arguments, argument_alternatives):
    if arguments.keys() - argument_alternatives.keys():
        return False
    for argument_name, alternatives in argument_alternatives.items():
        if argument_name not in arguments:
            if "" not in alternatives:
                return False
        elif not any(
            alternative != "" and are_argument_values_equal(arguments[argument_name], alternative)
            for alternative in alternatives
        ):
            return False
    return True


def score_bfcl_predictions(predictions, gold_examples, source_answers):
    """Join converted BFCL examples to raw source answers; score predictions by ID."""
    predictions_by_id = index_examples_by_id(predictions)
    gold_by_id = index_examples_by_id(gold_examples)
    answers_by_id = index_examples_by_id(source_answers)
    if not gold_by_id:
        raise ValueError("BFCL gold examples must be nonempty")
    if predictions_by_id.keys() - gold_by_id.keys():
        raise ValueError("Prediction ids absent from BFCL gold")
    scored_examples = []
    for example_id, example in gold_by_id.items():
        source_id = example.get("bfcl_id")
        if not isinstance(source_id, str) or source_id not in answers_by_id:
            raise ValueError(f"Missing BFCL source answer for example {example_id}")
        ground_truth = answers_by_id[source_id].get("ground_truth")
        validate_source_ground_truth(ground_truth)
        expected_calls = [next(iter(entry.items())) for entry in ground_truth]
        prediction = predictions_by_id.get(example_id)
        calls = []
        if prediction is not None:
            if "predicted_tool_calls" in prediction:
                calls = validate_tool_calls(prediction["predicted_tool_calls"])
            elif isinstance(prediction.get("raw_output"), str):
                calls = parse_tool_calls_from_model_output(prediction["raw_output"])
            else:
                raise ValueError(f"Prediction {example_id} requires raw_output or predicted_tool_calls")
        names_match = [call["name"] for call in calls] == [name for name, _ in expected_calls]
        scored_examples.append({
            "id": example_id,
            "is_parseable_tool_call": bool(calls),
            "is_correct_tool_name": names_match,
            "are_arguments_correct": names_match and all(
                arguments_match_alternatives(call["arguments"], alternatives)
                for call, (_, alternatives) in zip(calls, expected_calls)
            ),
        })
    return {
        **aggregate_scores(scored_examples),
        "missing_prediction_count": len(gold_by_id.keys() - predictions_by_id.keys()),
        "examples": scored_examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--prediction", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--raw-directory", type=Path, default=Path("data/ood_raw"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prediction_paths = {}
    for specification in arguments.prediction:
        label, separator, path = specification.partition("=")
        if not separator or not label.strip() or not path or label in prediction_paths:
            parser.error("Each --prediction requires a unique LABEL=PATH")
        prediction_paths[label] = Path(path)
    if "base" not in prediction_paths:
        parser.error("Include --prediction base=PATH for paired comparisons")
    source_paths = [arguments.raw_directory / "possible_answer" / f"BFCL_v3_{category}.json"
                    for category in ("simple", "multiple")]
    input_paths = [arguments.gold, *source_paths, *prediction_paths.values()]
    if arguments.output.resolve() in {path.resolve() for path in input_paths}:
        parser.error("Output must not overwrite an input")
    gold_examples = read_jsonl(arguments.gold)
    source_answers = [answer for path in source_paths for answer in read_jsonl(path)]
    models = {}
    for label, path in prediction_paths.items():
        predictions = read_jsonl(path)
        models[label] = {
            "strict": score_predictions(predictions, gold_examples),
            "alternative_aware": score_bfcl_predictions(predictions, gold_examples, source_answers),
        }
    paired_comparisons = {}
    comparison_pairs = [("base", label) for label in models if label != "base"]
    if "control" in models and "mixed" in models:
        comparison_pairs.append(("control", "mixed"))
    for reference, label in comparison_pairs:
        paired_comparisons[f"{label}_vs_{reference}"] = {}
        for metric in ("strict", "alternative_aware"):
            reference_flags = [int(row["are_arguments_correct"])
                               for row in models[reference][metric]["examples"]]
            model_flags = [int(row["are_arguments_correct"])
                           for row in models[label][metric]["examples"]]
            paired_comparisons[f"{label}_vs_{reference}"][metric] = paired_bootstrap_rate_ci(
                reference_flags, model_flags, seed=43,
            )
    report = {
        "metric_note": __doc__,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths},
        "models": models,
        "paired_comparisons": paired_comparisons,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: {metric: result["rates"]["are_arguments_correct"]
                             for metric, result in bundle.items()}
                      for label, bundle in models.items()}, indent=2))
    print(f"Report: {arguments.output}")


if __name__ == "__main__":
    main()
