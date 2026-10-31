"""Build mixed ToolACE+Hermes train sets and larger held-out evals without leakage."""

from __future__ import annotations

import hashlib
import json
import random
import re
from copy import deepcopy
from pathlib import Path

from prepare_data import (
    calculate_prompt_hash,
    normalize_parameter_schema,
    split_converted_examples,
)
# BFCL conversion helpers are imported lazily by the build CLI via importlib.


HERMES_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ROLE_ALIASES = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "tool": "tool"}


def normalize_user_text(user_text):
    if not isinstance(user_text, str):
        raise ValueError("user text must be a string")
    return " ".join(user_text.split())


def calculate_near_duplicate_key(tools, user_text):
    """Stable key for prompt+tools; used to block train/test leakage."""
    normalized = {
        "tools": tools,
        "user": normalize_user_text(user_text),
    }
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def parse_hermes_tool_calls(assistant_text):
    if not isinstance(assistant_text, str):
        raise ValueError("assistant text must be a string")
    tool_calls = []
    for block in HERMES_TOOL_CALL_PATTERN.findall(assistant_text):
        try:
            decoded = json.loads(block)
        except json.JSONDecodeError as error:
            raise ValueError("Hermes tool_call is not valid JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("Hermes tool_call must be an object")
        name = decoded.get("name")
        arguments = decoded.get("arguments", decoded.get("parameters"))
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Hermes tool_call requires a name")
        if not isinstance(arguments, dict):
            raise ValueError("Hermes tool_call arguments must be an object")
        tool_calls.append({"name": name, "arguments": arguments})
    return tool_calls


def normalize_hermes_tools(tools_field):
    if isinstance(tools_field, str):
        tools_field = json.loads(tools_field)
    if not isinstance(tools_field, list):
        raise ValueError("Hermes tools must be a list")
    tools = []
    for raw_tool in tools_field:
        if not isinstance(raw_tool, dict):
            raise ValueError("Tool definition must be an object")
        function_definition = raw_tool.get("function", raw_tool)
        if not isinstance(function_definition, dict):
            raise ValueError("Tool function must be an object")
        tool_name = function_definition.get("name")
        parameters = function_definition.get("parameters")
        if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(parameters, dict):
            raise ValueError("Tool requires a name and parameter schema")
        tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": function_definition.get("description", ""),
                "parameters": normalize_parameter_schema(parameters),
            },
        })
    return tools


def convert_hermes_row(row, example_id):
    """Convert Hermes/Glaive shareGPT-style row into ToolACE train JSONL shape."""
    if not isinstance(row, dict):
        raise ValueError("Hermes row must be an object")
    tools = normalize_hermes_tools(row.get("tools"))
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("Conversations must be a nonempty list")
    messages = []
    for turn in conversations:
        if not isinstance(turn, dict):
            raise ValueError("Conversation turn must be an object")
        raw_role = turn.get("from") or turn.get("role")
        content = turn.get("value") if "value" in turn else turn.get("content")
        if raw_role not in ROLE_ALIASES:
            raise ValueError(f"Unsupported Hermes role: {raw_role}")
        role = ROLE_ALIASES[raw_role]
        if role == "system":
            # Drop embedded <tools> XML; tools are passed separately at render time.
            if isinstance(content, str) and "<tools>" in content:
                content = content.split("<tools>")[0].strip() or (
                    "You are a helpful assistant with access to tools. Call tools when needed."
                )
            messages.append({"role": "system", "content": content})
            continue
        if role == "assistant":
            if not isinstance(content, str):
                raise ValueError("Assistant content must be a string")
            parsed_tool_calls = parse_hermes_tool_calls(content)
            if parsed_tool_calls:
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "function": tool_call}
                        for tool_call in parsed_tool_calls
                    ],
                })
            else:
                messages.append({"role": "assistant", "content": content})
            continue
        if not isinstance(content, str):
            raise ValueError(f"{role} content must be a string")
        messages.append({"role": role, "content": content})
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant with access to tools. Call tools when needed.",
        })
    # Require first user then assistant with tool_calls (same as ToolACE train usefulness).
    user_index = next((i for i, message in enumerate(messages) if message["role"] == "user"), None)
    if user_index is None:
        raise ValueError("Hermes row has no user turn")
    assistant_index = next(
        (i for i, message in enumerate(messages) if i > user_index and message["role"] == "assistant"),
        None,
    )
    if assistant_index is None or "tool_calls" not in messages[assistant_index]:
        raise ValueError("Hermes row has no assistant tool call after the user")
    # Keep through first assistant tool call for SFT parity with ToolACE truncation.
    truncated_messages = messages[: assistant_index + 1]
    return {"id": example_id, "tools": tools, "messages": truncated_messages, "source": "hermes"}


def blocked_keys_from_examples(examples):
    keys = set()
    for example in examples:
        messages = example.get("messages") or example.get("prompt_messages")
        if not messages:
            raise ValueError(f"Example {example.get('id')} missing messages")
        user_message = next((m for m in messages if m.get("role") == "user"), None)
        if user_message is None:
            raise ValueError(f"Example {example.get('id')} missing user message")
        keys.add(calculate_near_duplicate_key(example["tools"], user_message["content"]))
        keys.add(calculate_prompt_hash({"tools": example["tools"], "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_message["content"]},
        ]}))
    return keys


def filter_examples_not_in_blocked_keys(examples, blocked_keys):
    kept = []
    rejected = 0
    for example in examples:
        user_message = next((m for m in example["messages"] if m.get("role") == "user"), None)
        if user_message is None:
            raise ValueError(f"Example {example.get('id')} missing user message")
        key = calculate_near_duplicate_key(example["tools"], user_message["content"])
        prompt_key = calculate_prompt_hash({"tools": example["tools"], "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_message["content"]},
        ]})
        if key in blocked_keys or prompt_key in blocked_keys:
            rejected += 1
            continue
        kept.append(example)
    return kept, rejected


def sample_examples(examples, sample_count, seed):
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if sample_count > len(examples):
        raise ValueError(f"Requested {sample_count} examples but only {len(examples)} available")
    if sample_count == len(examples):
        return list(examples)
    return random.Random(seed).sample(list(examples), sample_count)


def build_mixed_training_rows(toolace_train_rows, external_rows, external_sample_count, seed):
    sampled_external = sample_examples(external_rows, external_sample_count, seed)
    mixed = list(toolace_train_rows) + sampled_external
    random.Random(seed + 1).shuffle(mixed)
    return mixed, {
        "toolace_train_count": len(toolace_train_rows),
        "external_available": len(external_rows),
        "external_sampled": len(sampled_external),
        "mixed_total": len(mixed),
        "seed": seed,
    }


def select_bfcl_ids(all_ids, selected_count, seed, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    candidates = [example_id for example_id in all_ids if example_id not in exclude_ids]
    if len(candidates) < selected_count:
        # Allow filling from excluded pool only after exhausting unused ids.
        remaining_needed = selected_count - len(candidates)
        excluded_pool = [example_id for example_id in all_ids if example_id in exclude_ids]
        if len(excluded_pool) < remaining_needed:
            raise ValueError(
                f"Need {selected_count} BFCL ids but only {len(all_ids)} total"
            )
        filler = random.Random(seed).sample(excluded_pool, remaining_needed)
        return candidates + filler
    return random.Random(seed).sample(candidates, selected_count)
