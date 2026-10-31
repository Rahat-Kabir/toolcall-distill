"""Locate caller-supplied evaluation data without downloading or loading models."""

from pathlib import Path
import sys


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def read_evaluation_examples(input_path, limit=None):
    sys.path.insert(0, str(PROJECT_DIRECTORY))
    from eval_toolcalls import index_examples_by_id, read_jsonl

    examples = read_jsonl(input_path)
    index_examples_by_id(examples)
    for example in examples:
        if not isinstance(example.get("tools"), list) or not all(
            isinstance(tool, dict) for tool in example["tools"]
        ):
            raise ValueError(f"Example {example['id']} requires a tools list of objects")
        messages = example.get("prompt_messages")
        if not isinstance(messages, list) or not messages or not all(
            isinstance(message, dict) for message in messages
        ) or messages[-1].get("role") != "user":
            raise ValueError(f"Example {example['id']} prompt_messages must end with a user message")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Example {example['id']} has an unsupported prompt role")
            if not isinstance(message.get("content"), str) and not (
                message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list)
            ):
                raise ValueError(f"Example {example['id']} prompt messages require text content")
    return examples[:limit]
