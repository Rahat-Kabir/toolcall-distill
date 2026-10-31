"""Parse and score tool calls, and run local Transformers generation.

Arguments compare recursively as dictionaries (key order ignored) and ordered
lists. Strings are stripped. Numeric strings equal the same int/float value;
boolean strings equal booleans case-insensitively. Booleans never equal numbers.
Names and call order/count must match exactly. Argument success also requires
name success. Empty or malformed model output produces no calls.
"""

import argparse
import ast
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re


METRIC_NAMES = ("is_parseable_tool_call", "is_correct_tool_name", "are_arguments_correct")


def find_argument_boundaries(argument_text):
    """Locate outer commas and the closing parenthesis, ignoring quoted literals."""
    opening_brackets = ["("]
    matching_brackets = {")": "(", "]": "[", "}": "{"}
    quoted_character = None
    is_escaped = False
    boundaries = []
    for character_index, character in enumerate(argument_text):
        if quoted_character:
            if is_escaped:
                is_escaped = False
            elif character == "\\":
                is_escaped = True
            elif character == quoted_character:
                quoted_character = None
        elif character in "\"'":
            quoted_character = character
        elif character in "([{":
            opening_brackets.append(character)
        elif character in ")]}":
            if not opening_brackets or opening_brackets.pop() != matching_brackets[character]:
                return []
            if not opening_brackets:
                return [*boundaries, character_index]
        elif character == "," and opening_brackets == ["("]:
            boundaries.append(character_index)
    return []


def parse_pythonic_tool_call_list(raw_text):
    """Read literal keyword arguments safely, preserving non-Python tool names."""
    raw_text = raw_text.strip()
    if not (raw_text.startswith("[") and raw_text.endswith("]")):
        return []
    remaining_text = raw_text[1:-1].strip()
    parsed_tool_calls = []
    while remaining_text:
        argument_start = re.search(r"\(\s*(?=(?:[^=()[\]{},*\n\r]+\s*=|\)))", remaining_text)
        if argument_start is None:
            return []
        tool_name = remaining_text[:argument_start.start()].strip()
        if not tool_name or re.search(r"[\[\]=\n\r]", tool_name):
            return []
        argument_text = remaining_text[argument_start.start() + 1:]
        boundaries = find_argument_boundaries(argument_text)
        if not boundaries:
            return []
        arguments = {}
        segment_start = 0
        for boundary in boundaries:
            keyword_text = argument_text[segment_start:boundary].strip()
            segment_start = boundary + 1
            if not keyword_text and boundary == boundaries[-1]:
                continue
            argument_name, separator, value_text = keyword_text.partition("=")
            argument_name = argument_name.strip()
            if not separator or not re.fullmatch(r"[^=()[\]{},*\n\r]+", argument_name) or argument_name in arguments:
                return []
            try:
                try:
                    argument_value = json.loads(value_text.strip())
                except json.JSONDecodeError:
                    argument_value = ast.literal_eval(value_text.strip())
                arguments[argument_name] = json.loads(json.dumps(argument_value, allow_nan=False))
            except (SyntaxError, ValueError, TypeError, OverflowError, RecursionError):
                return []
        parsed_tool_calls.append({"name": tool_name, "arguments": arguments})
        remaining_text = argument_text[boundaries[-1] + 1:].strip()
        if remaining_text:
            if not remaining_text.startswith(","):
                return []
            remaining_text = remaining_text[1:].strip()
    return parsed_tool_calls


def validate_tool_calls(tool_calls):
    """Normalize JSON/OpenAI call objects or raise ValueError at the input edge."""
    if not isinstance(tool_calls, list):
        raise ValueError("Tool calls must be a list")
    normalized_tool_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError("Each tool call must be an object")
        function_call = tool_call.get("function", tool_call)
        if not isinstance(function_call, dict):
            raise ValueError("Tool function must be an object")
        tool_name = function_call.get("name")
        arguments = function_call.get("arguments")
        if arguments is None and "parameters" in function_call:
            arguments = function_call.get("parameters")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ValueError("Tool arguments contain invalid JSON") from error
        if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(arguments, dict):
            raise ValueError("Tool calls require a nonempty name and argument dictionary")
        normalized_tool_calls.append({"name": tool_name, "arguments": arguments})
    return normalized_tool_calls


def parse_consecutive_json_tool_calls(segment_text):
    """Read one or more JSON tool-call objects from a segment, skipping semicolons."""
    if not isinstance(segment_text, str):
        return []
    json_decoder = json.JSONDecoder()
    character_index = 0
    parsed_tool_calls = []
    while character_index < len(segment_text):
        while character_index < len(segment_text) and segment_text[character_index] in ' \t\n\r;':
            character_index += 1
        if character_index >= len(segment_text):
            break
        try:
            decoded_value, next_index = json_decoder.raw_decode(segment_text, character_index)
        except json.JSONDecodeError:
            break
        character_index = next_index
        if isinstance(decoded_value, dict):
            decoded_value = decoded_value.get("tool_calls", [decoded_value])
        try:
            parsed_tool_calls.extend(validate_tool_calls(decoded_value))
        except ValueError:
            break
    return parsed_tool_calls


def parse_tool_calls_from_model_output(raw_text):
    """Accept LFM markers, Llama python_tag JSON, Qwen tool_call tags, bare lists, or JSON."""
    if not isinstance(raw_text, str):
        return []
    if "<|tool_call_start|>" in raw_text:
        marked_outputs = re.findall(
            r"<\|tool_call_start\|>(.*?)(?:<\|tool_call_end\|>|$)", raw_text, re.DOTALL,
        )
        return [tool_call for marked_output in marked_outputs
                for tool_call in parse_tool_calls_from_model_output(marked_output)]
    if "<|python_tag|>" in raw_text:
        marked_outputs = re.findall(
            r"<\|python_tag\|>(.*?)(?:<\|eom_id\|>|$)", raw_text, re.DOTALL,
        )
        parsed_tool_calls = []
        for marked_output in marked_outputs:
            parsed_tool_calls.extend(parse_consecutive_json_tool_calls(marked_output))
        if parsed_tool_calls:
            return parsed_tool_calls
    if "<tool_call>" in raw_text:
        marked_outputs = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>", raw_text, re.DOTALL,
        )
        parsed = []
        for marked_output in marked_outputs:
            parsed.extend(parse_tool_calls_from_model_output(marked_output))
        if parsed:
            return parsed
    raw_text = raw_text.strip()
    pythonic_tool_calls = parse_pythonic_tool_call_list(raw_text)
    if pythonic_tool_calls:
        return pythonic_tool_calls
    for list_match in re.finditer(r"\[", raw_text):
        list_text = raw_text[list_match.start():]
        for end_match in re.finditer(r"\]", list_text):
            pythonic_tool_calls = parse_pythonic_tool_call_list(list_text[:end_match.end()])
            if pythonic_tool_calls:
                return pythonic_tool_calls
    json_decoder = json.JSONDecoder()
    for object_match in re.finditer(r"[\[{]", raw_text):
        try:
            decoded_value, _ = json_decoder.raw_decode(raw_text[object_match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded_value, dict):
            decoded_value = decoded_value.get("tool_calls", [decoded_value])
        try:
            parsed_tool_calls = validate_tool_calls(decoded_value)
        except ValueError:
            continue
        if parsed_tool_calls:
            return parsed_tool_calls
    return []


def are_argument_values_equal(predicted_value, gold_value):
    if isinstance(predicted_value, str):
        predicted_value = predicted_value.strip()
    if isinstance(gold_value, str):
        gold_value = gold_value.strip()
    if isinstance(predicted_value, dict) and isinstance(gold_value, dict):
        return predicted_value.keys() == gold_value.keys() and all(
            are_argument_values_equal(predicted_value[key], gold_value[key]) for key in gold_value
        )
    if isinstance(predicted_value, list) and isinstance(gold_value, list):
        return len(predicted_value) == len(gold_value) and all(
            are_argument_values_equal(predicted_item, gold_item)
            for predicted_item, gold_item in zip(predicted_value, gold_value)
        )
    if isinstance(predicted_value, bool) or isinstance(gold_value, bool):
        if isinstance(predicted_value, str):
            return predicted_value.lower() == str(gold_value).lower()
        if isinstance(gold_value, str):
            return gold_value.lower() == str(predicted_value).lower()
        return type(predicted_value) is type(gold_value) and predicted_value == gold_value
    if isinstance(predicted_value, (int, float)) or isinstance(gold_value, (int, float)):
        if not isinstance(predicted_value, (int, float, str)) or not isinstance(gold_value, (int, float, str)):
            return False
        try:
            return Decimal(str(predicted_value)) == Decimal(str(gold_value))
        except InvalidOperation:
            return False
    return type(predicted_value) is type(gold_value) and predicted_value == gold_value


def score_tool_calls(predicted_tool_calls, gold_tool_calls):
    """Score validated calls; name and argument levels require exact call count."""
    is_correct_tool_name = bool(gold_tool_calls) and (
        [tool_call["name"] for tool_call in predicted_tool_calls]
        == [tool_call["name"] for tool_call in gold_tool_calls]
    )
    return {
        "is_parseable_tool_call": bool(predicted_tool_calls),
        "is_correct_tool_name": is_correct_tool_name,
        "are_arguments_correct": is_correct_tool_name and all(
            are_argument_values_equal(predicted_call["arguments"], gold_call["arguments"])
            for predicted_call, gold_call in zip(predicted_tool_calls, gold_tool_calls)
        ),
    }


def aggregate_scores(example_scores):
    counts = Counter(metric_name for example_score in example_scores
                     for metric_name in METRIC_NAMES if example_score[metric_name])
    return {
        "example_count": len(example_scores),
        "counts": {metric_name: counts[metric_name] for metric_name in METRIC_NAMES},
        "rates": {metric_name: counts[metric_name] / len(example_scores) if example_scores else 0.0
                  for metric_name in METRIC_NAMES},
    }


def read_jsonl(input_path):
    examples = []
    with Path(input_path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{input_path}:{line_number}: invalid JSON") from error
            if not isinstance(example, dict):
                raise ValueError(f"{input_path}:{line_number}: expected a JSON object")
            examples.append(example)
    if not examples:
        raise ValueError(f"{input_path}: file is empty")
    return examples


def write_jsonl(output_path, examples):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n")


def index_examples_by_id(examples):
    indexed_examples = {}
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, (str, int)) or isinstance(example_id, bool):
            raise ValueError("Each example requires a string or integer id")
        if example_id in indexed_examples:
            raise ValueError(f"Duplicate example id: {example_id}")
        indexed_examples[example_id] = example
    return indexed_examples


def score_prediction_files(predictions_path, gold_path):
    """Score JSONL prediction and gold files by id."""
    return score_predictions(read_jsonl(predictions_path), read_jsonl(gold_path))


def score_predictions(predictions, gold_examples):
    """Match by id; missing predictions fail all levels and unknown ids are errors."""
    if isinstance(predictions, (str, Path)) or isinstance(gold_examples, (str, Path)):
        raise TypeError(
            "score_predictions expects lists of dicts; use score_prediction_files for paths"
        )
    predictions_by_id = index_examples_by_id(predictions)
    gold_examples_by_id = index_examples_by_id(gold_examples)
    unknown_ids = predictions_by_id.keys() - gold_examples_by_id.keys()
    if unknown_ids:
        raise ValueError(f"Prediction ids absent from gold: {sorted(map(str, unknown_ids))}")
    scored_examples = []
    for example_id, gold_example in gold_examples_by_id.items():
        gold_tool_calls = validate_tool_calls(gold_example.get("gold_tool_calls"))
        if not gold_tool_calls:
            raise ValueError(f"Gold example {example_id} has no tool calls")
        prediction = predictions_by_id.get(example_id)
        predicted_tool_calls = []
        if prediction is not None:
            if "predicted_tool_calls" in prediction:
                predicted_tool_calls = validate_tool_calls(prediction["predicted_tool_calls"])
            elif isinstance(prediction.get("raw_output"), str):
                predicted_tool_calls = parse_tool_calls_from_model_output(prediction["raw_output"])
            else:
                raise ValueError(f"Prediction {example_id} requires raw_output or predicted_tool_calls")
        scored_examples.append({"id": example_id, **score_tool_calls(predicted_tool_calls, gold_tool_calls)})
    return {**aggregate_scores(scored_examples),
            "missing_prediction_count": len(gold_examples_by_id.keys() - predictions_by_id.keys()),
            "examples": scored_examples}


def generate_predictions(model_name, input_path, output_path, limit=None, device="cpu", max_new_tokens=128):
    """Generate locally; loading from the Hub only downloads free model files."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Generation output must differ from input")
    examples = read_jsonl(input_path)
    index_examples_by_id(examples)
    examples = examples[:limit]
    for example in examples:
        if not isinstance(example.get("tools"), list) or not isinstance(example.get("prompt_messages"), list):
            raise ValueError("Generation requires tools and prompt_messages lists")
        if not example["prompt_messages"] or example["prompt_messages"][-1].get("role") != "user":
            raise ValueError("Generation prompts must end with a user message")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32 if device == "cpu" else "auto", low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    predictions = []
    for example in examples:
        encoded_prompt = tokenizer.apply_chat_template(
            example["prompt_messages"], tools=example["tools"], add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        ).to(device)
        with torch.inference_mode():
            generated_tokens = model.generate(
                **encoded_prompt, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        raw_output = tokenizer.decode(
            generated_tokens[0, encoded_prompt["input_ids"].shape[1]:], skip_special_tokens=False,
        )
        predictions.append({"id": example["id"], "raw_output": raw_output})
        print(f"Generated {len(predictions)}/{len(examples)}: {example['id']}", flush=True)
    write_jsonl(output_path, predictions)
    return predictions


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    subcommands = argument_parser.add_subparsers(dest="command", required=True)
    score_parser = subcommands.add_parser("score")
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--gold", type=Path, default=Path("data/test.jsonl"))
    score_parser.add_argument("--summary", type=Path)
    generation_parser = subcommands.add_parser("generate")
    generation_parser.add_argument("--model", required=True)
    generation_parser.add_argument("--input", type=Path, required=True)
    generation_parser.add_argument("--output", type=Path, required=True)
    generation_parser.add_argument("--limit", type=int)
    generation_parser.add_argument("--device", default="cpu")
    generation_parser.add_argument("--max-new-tokens", type=int, default=128)
    arguments = argument_parser.parse_args()
    if arguments.command == "generate":
        generate_predictions(arguments.model, arguments.input, arguments.output, arguments.limit,
                             arguments.device, arguments.max_new_tokens)
    else:
        if arguments.predictions.resolve() == arguments.gold.resolve():
            argument_parser.error("Predictions and gold must be different files")
        summary = score_predictions(read_jsonl(arguments.predictions), read_jsonl(arguments.gold))
        summary_path = arguments.summary or arguments.predictions.with_suffix(".summary.json")
        if summary_path.resolve() in {arguments.gold.resolve(), arguments.predictions.resolve()}:
            argument_parser.error("Summary must not overwrite predictions or gold")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "examples"}, indent=2))
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
