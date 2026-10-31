"""Convert free ToolACE downloads to LFM-compatible train and held-out JSONL."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from eval_toolcalls import parse_pythonic_tool_call_list, write_jsonl


TOOLS_MARKER = "Here is a list of functions in JSON format that you can invoke:"


def normalize_parameter_schema(parameter_schema):
    if isinstance(parameter_schema, list):
        return [normalize_parameter_schema(item) for item in parameter_schema]
    if not isinstance(parameter_schema, dict):
        return parameter_schema
    normalized_schema = {}
    for field_name, field_value in parameter_schema.items():
        if field_name == "type" and isinstance(field_value, str):
            normalized_schema[field_name] = {
                "dict": "object", "int": "integer", "float": "number",
                "bool": "boolean", "str": "string", "list": "array",
            }.get(field_value, field_value)
        else:
            normalized_schema[field_name] = normalize_parameter_schema(field_value)
    return normalized_schema


def parse_tools_from_system(system_text):
    """Decode the first JSON list after the ToolACE marker, ignoring its suffix."""
    if not isinstance(system_text, str) or TOOLS_MARKER not in system_text:
        raise ValueError("System prompt is missing the ToolACE tools marker")
    tools_text = system_text.split(TOOLS_MARKER, 1)[1].lstrip()
    try:
        raw_tools, _ = json.JSONDecoder().raw_decode(tools_text)
    except json.JSONDecodeError as error:
        raise ValueError("System tools are not valid JSON") from error
    if not isinstance(raw_tools, list):
        raise ValueError("System tools must be a list")
    tools = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            raise ValueError("Tool definition must be an object")
        function_definition = raw_tool.get("function", raw_tool)
        if not isinstance(function_definition, dict):
            raise ValueError("Tool function must be an object")
        tool_name = function_definition.get("name")
        parameters = function_definition.get("parameters")
        if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(parameters, dict):
            raise ValueError("Tool requires a name and parameter schema")
        tools.append({"type": "function", "function": {
            "name": tool_name,
            "description": function_definition.get("description", ""),
            "parameters": normalize_parameter_schema(parameters),
        }})
    return tools


def convert_row(row, row_index):
    """Validate one Hub row and retain every conversation turn for SFT."""
    tools = parse_tools_from_system(row.get("system"))
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Conversations must be a nonempty list")
    system_instruction = row["system"].split(TOOLS_MARKER, 1)[0].strip()
    messages = [{"role": "system", "content": system_instruction}]
    for conversation in conversations:
        if not isinstance(conversation, dict):
            raise ValueError("Conversation turn must be an object")
        role = conversation.get("from")
        content = conversation.get("value")
        if role not in {"user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError("Conversation requires user/assistant/tool role and string value")
        parsed_tool_calls = parse_pythonic_tool_call_list(content) if role == "assistant" else []
        if parsed_tool_calls:
            messages.append({"role": "assistant", "tool_calls": [
                {"type": "function", "function": tool_call} for tool_call in parsed_tool_calls
            ]})
        else:
            if role == "assistant" and content.strip().startswith("[") and "(" in content:
                raise ValueError("Assistant contains an unparseable Pythonic call list")
            messages.append({"role": role, "content": content})
    if len(messages) < 3 or messages[1]["role"] != "user":
        raise ValueError("Conversation must start with a user followed by a response")
    return {"id": f"toolace-{row_index:05d}", "tools": tools, "messages": messages}


def calculate_prompt_hash(converted_example):
    normalized_prompt = {
        "tools": converted_example["tools"],
        "user": " ".join(converted_example["messages"][1]["content"].split()),
    }
    return hashlib.sha256(json.dumps(normalized_prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_converted_examples(converted_examples, test_count=200, seed=42):
    """Deduplicate before sampling so an entire first-prompt group stays isolated."""
    unique_examples = {}
    for converted_example in converted_examples:
        prompt_hash = calculate_prompt_hash(converted_example)
        if prompt_hash not in unique_examples:
            unique_examples[prompt_hash] = converted_example
    eligible_prompt_hashes = [prompt_hash for prompt_hash, example in unique_examples.items()
                             if example["messages"][2].get("tool_calls")]
    if len(eligible_prompt_hashes) < test_count:
        raise ValueError(f"Only {len(eligible_prompt_hashes)} single-turn targets; requested {test_count}")
    selected_prompt_hashes = set(random.Random(seed).sample(eligible_prompt_hashes, test_count))
    train_examples = []
    test_examples = []
    for prompt_hash, example in unique_examples.items():
        if prompt_hash in selected_prompt_hashes:
            test_example = deepcopy(example)
            test_example["messages"] = test_example["messages"][:3]
            test_example["prompt_messages"] = deepcopy(test_example["messages"][:2])
            test_example["gold_tool_calls"] = [deepcopy(tool_call["function"])
                                               for tool_call in test_example["messages"][2]["tool_calls"]]
            test_examples.append(test_example)
        else:
            train_examples.append(example)
    return train_examples, test_examples


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--output-directory", type=Path, default=Path("data"))
    argument_parser.add_argument("--test-count", type=int, default=200)
    argument_parser.add_argument("--seed", type=int, default=42)
    arguments = argument_parser.parse_args()
    if arguments.test_count <= 0:
        argument_parser.error("test-count must be positive")
    from datasets import load_dataset

    dataset = load_dataset("Team-ACE/ToolACE", split="train")
    print(f"Dataset: Team-ACE/ToolACE; rows: {len(dataset)}", flush=True)
    print(f"Columns: {dataset.column_names}", flush=True)
    print(f"Features: {dataset.features}", flush=True)
    first_row = dataset[0]
    print("Example shape:", json.dumps({
        "system": type(first_row["system"]).__name__,
        "conversations": {"type": type(first_row["conversations"]).__name__,
                          "length": len(first_row["conversations"]),
                          "first_turn": {key: type(value).__name__
                                         for key, value in first_row["conversations"][0].items()}},
    }), flush=True)
    if set(dataset.column_names) != {"system", "conversations"}:
        raise ValueError("Unexpected ToolACE columns")
    converted_examples = []
    rejected_reasons = Counter()
    for row_index, row in enumerate(dataset):
        try:
            converted_examples.append(convert_row(row, row_index))
        except ValueError as error:
            rejected_reasons[str(error)] += 1
    train_examples, test_examples = split_converted_examples(
        converted_examples, arguments.test_count, arguments.seed,
    )
    write_jsonl(arguments.output_directory / "train.jsonl", train_examples)
    write_jsonl(arguments.output_directory / "test.jsonl", test_examples)
    report = {
        "source_count": len(dataset), "converted_count": len(converted_examples),
        "rejected_count": sum(rejected_reasons.values()), "rejected_reasons": dict(rejected_reasons),
        "duplicate_prompt_count": len(converted_examples) - len(train_examples) - len(test_examples),
        "train_count": len(train_examples), "test_count": len(test_examples), "seed": arguments.seed,
    }
    (arguments.output_directory / "preparation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
