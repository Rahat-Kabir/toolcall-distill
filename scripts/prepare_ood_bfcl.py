"""Convert BFCL v3 simple+multiple rows into the ToolACE-shaped OOD JSONL.

Reads raw BFCL files under data/ood_raw/, applies the pinned ID lists in
data/ood_conversion_notes.json, and writes data/ood_test.jsonl.

Gold rule: for each argument, take the first non-empty BFCL alternative.
If every alternative is the empty string, omit that argument (optional default).
Schema aliases dict/list/int/bool/str become object/array/integer/boolean/string.
Nested "float" under items/ deeper properties is left unchanged (shallow map).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = "You are a helpful assistant with access to tools. Call tools when needed."
GOLD_NOTES = (
    "First non-empty BFCL alternative per arg; optional empty alternatives omitted. "
    "Stricter than BFCL AST multi-answer match."
)
SOURCE_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Shallow alias map used when the OOD set was built: root parameters.type plus
# one level of properties[*].type. Nested items/properties keep raw BFCL types
# (so nested "float" stays "float").
SCHEMA_TYPE_ALIASES = {
    "dict": "object",
    "list": "array",
    "int": "integer",
    "bool": "boolean",
    "str": "string",
    "float": "number",
}


def normalize_bfcl_parameter_schema(parameter_schema):
    if not isinstance(parameter_schema, dict):
        raise ValueError("BFCL parameters must be an object")
    normalized_schema = json.loads(json.dumps(parameter_schema))
    root_type = normalized_schema.get("type")
    if isinstance(root_type, str) and root_type in SCHEMA_TYPE_ALIASES:
        normalized_schema["type"] = SCHEMA_TYPE_ALIASES[root_type]
    properties = normalized_schema.get("properties")
    if isinstance(properties, dict):
        for property_schema in properties.values():
            if not isinstance(property_schema, dict):
                continue
            property_type = property_schema.get("type")
            if isinstance(property_type, str) and property_type in SCHEMA_TYPE_ALIASES:
                property_schema["type"] = SCHEMA_TYPE_ALIASES[property_type]
    return normalized_schema


def read_jsonl_objects(input_path):
    examples = []
    with Path(input_path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
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


def index_by_id(examples):
    indexed = {}
    for example in examples:
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError("BFCL row requires a string id")
        if example_id in indexed:
            raise ValueError(f"Duplicate BFCL id: {example_id}")
        indexed[example_id] = example
    return indexed


def extract_user_text(question):
    if not isinstance(question, list):
        raise ValueError("BFCL question must be a list")
    for turn_group in question:
        if not isinstance(turn_group, list):
            raise ValueError("BFCL question turn group must be a list")
        for message in turn_group:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("BFCL user content must be a string")
                return content
    raise ValueError("BFCL question has no user message")


def convert_tools(function_definitions):
    if not isinstance(function_definitions, list) or not function_definitions:
        raise ValueError("BFCL function list must be nonempty")
    tools = []
    for function_definition in function_definitions:
        if not isinstance(function_definition, dict):
            raise ValueError("BFCL function must be an object")
        tool_name = function_definition.get("name")
        parameters = function_definition.get("parameters")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("BFCL function requires a name")
        if not isinstance(parameters, dict):
            raise ValueError("BFCL function requires a parameters object")
        tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": function_definition.get("description", ""),
                "parameters": normalize_bfcl_parameter_schema(parameters),
            },
        })
    return tools


def first_nonempty_alternative(alternatives):
    if not isinstance(alternatives, list) or not alternatives:
        raise ValueError("BFCL argument alternatives must be a nonempty list")
    for alternative in alternatives:
        if alternative != "":
            return alternative
    return None


def gold_tool_calls_from_ground_truth(ground_truth):
    if not isinstance(ground_truth, list) or not ground_truth:
        raise ValueError("BFCL ground_truth must be a nonempty list")
    gold_tool_calls = []
    for entry in ground_truth:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each ground_truth entry must map one tool name to arguments")
        tool_name, argument_alternatives = next(iter(entry.items()))
        if not isinstance(tool_name, str) or not isinstance(argument_alternatives, dict):
            raise ValueError("ground_truth tool name/arguments are invalid")
        arguments = {}
        for argument_name, alternatives in argument_alternatives.items():
            selected_value = first_nonempty_alternative(alternatives)
            if selected_value is None:
                continue
            arguments[argument_name] = selected_value
        gold_tool_calls.append({"name": tool_name, "arguments": arguments})
    if not gold_tool_calls:
        raise ValueError("ground_truth produced no gold tool calls")
    return gold_tool_calls


def convert_bfcl_example(category, question_row, answer_row):
    if question_row.get("id") != answer_row.get("id"):
        raise ValueError("Question and answer ids do not match")
    bfcl_id = question_row["id"]
    user_text = extract_user_text(question_row["question"])
    tools = convert_tools(question_row["function"])
    gold_tool_calls = gold_tool_calls_from_ground_truth(answer_row["ground_truth"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "function": tool_call}
                for tool_call in gold_tool_calls
            ],
        },
    ]
    return {
        "id": f"bfcl-{category}-{bfcl_id}",
        "source": SOURCE_DATASET,
        "bfcl_category": category,
        "bfcl_id": bfcl_id,
        "tools": tools,
        "messages": messages,
        "prompt_messages": messages[:2],
        "gold_tool_calls": gold_tool_calls,
        "gold_notes": GOLD_NOTES,
    }


def load_conversion_notes(notes_path):
    notes = json.loads(Path(notes_path).read_text(encoding="utf-8"))
    if not isinstance(notes, dict):
        raise ValueError("Conversion notes must be a JSON object")
    selected_simple_ids = notes.get("selected_simple_ids")
    selected_multiple_ids = notes.get("selected_multiple_ids")
    if not isinstance(selected_simple_ids, list) or not isinstance(selected_multiple_ids, list):
        raise ValueError("Conversion notes require selected_simple_ids and selected_multiple_ids lists")
    if not selected_simple_ids or not selected_multiple_ids:
        raise ValueError("Selected BFCL id lists must be nonempty")
    if len(selected_simple_ids) != len(set(selected_simple_ids)):
        raise ValueError("selected_simple_ids contains duplicates")
    if len(selected_multiple_ids) != len(set(selected_multiple_ids)):
        raise ValueError("selected_multiple_ids contains duplicates")
    return notes, selected_simple_ids, selected_multiple_ids


def build_ood_examples(raw_directory, notes_path):
    notes, selected_simple_ids, selected_multiple_ids = load_conversion_notes(notes_path)
    raw_directory = Path(raw_directory)
    questions_by_category = {
        "simple": index_by_id(read_jsonl_objects(raw_directory / "BFCL_v3_simple.json")),
        "multiple": index_by_id(read_jsonl_objects(raw_directory / "BFCL_v3_multiple.json")),
    }
    answers_by_category = {
        "simple": index_by_id(read_jsonl_objects(raw_directory / "possible_answer" / "BFCL_v3_simple.json")),
        "multiple": index_by_id(read_jsonl_objects(raw_directory / "possible_answer" / "BFCL_v3_multiple.json")),
    }
    selected_by_category = {
        "simple": selected_simple_ids,
        "multiple": selected_multiple_ids,
    }
    converted_examples = []
    skipped = []
    for category, selected_ids in selected_by_category.items():
        for bfcl_id in selected_ids:
            try:
                question_row = questions_by_category[category][bfcl_id]
                answer_row = answers_by_category[category][bfcl_id]
                converted_examples.append(convert_bfcl_example(category, question_row, answer_row))
            except (KeyError, ValueError) as error:
                skipped.append({"category": category, "bfcl_id": bfcl_id, "reason": str(error)})
    report = {
        "n": len(converted_examples),
        "simple": sum(1 for example in converted_examples if example["bfcl_category"] == "simple"),
        "multiple": sum(1 for example in converted_examples if example["bfcl_category"] == "multiple"),
        "skipped": skipped,
        "seed": notes.get("seed"),
        "source_files": notes.get("source_files"),
        "limits": notes.get("limits"),
        "selected_simple_ids": selected_simple_ids,
        "selected_multiple_ids": selected_multiple_ids,
        "selection_note": notes.get("selection_note"),
    }
    return converted_examples, report


def write_jsonl(output_path, examples):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--raw-directory", type=Path, default=Path("data/ood_raw"))
    argument_parser.add_argument("--notes-path", type=Path, default=Path("data/ood_conversion_notes.json"))
    argument_parser.add_argument("--output-path", type=Path, default=Path("data/ood_test.jsonl"))
    arguments = argument_parser.parse_args()
    converted_examples, report = build_ood_examples(arguments.raw_directory, arguments.notes_path)
    if report["skipped"]:
        raise SystemExit(f"Skipped {len(report['skipped'])} rows: {report['skipped'][:3]}")
    write_jsonl(arguments.output_path, converted_examples)
    print(json.dumps({key: report[key] for key in ("n", "simple", "multiple", "skipped", "seed")}, indent=2))
    print(f"Wrote {arguments.output_path}")


if __name__ == "__main__":
    main()
