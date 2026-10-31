"""Behavior tests for mixed-data conversion and leakage guards."""

import json

import pytest

from mixed_data_prep import (
    build_mixed_training_rows,
    calculate_near_duplicate_key,
    convert_hermes_row,
    filter_examples_not_in_blocked_keys,
    parse_hermes_tool_calls,
    sample_examples,
    select_bfcl_ids,
)


def test_parse_hermes_tool_calls_reads_semicolon_free_json_blocks():
    assistant_text = (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Dhaka"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "get_time", "arguments": {"tz": "Asia/Dhaka"}}\n</tool_call>'
    )
    assert parse_hermes_tool_calls(assistant_text) == [
        {"name": "get_weather", "arguments": {"city": "Dhaka"}},
        {"name": "get_time", "arguments": {"tz": "Asia/Dhaka"}},
    ]


def test_parse_hermes_tool_calls_rejects_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_hermes_tool_calls('<tool_call>{not json}</tool_call>')


def test_filter_examples_without_user_message_raises_descriptive_error():
    examples = [{"id": "no-user", "tools": [], "messages": [
        {"role": "assistant", "content": "hello"},
    ]}]
    with pytest.raises(ValueError, match="missing user message"):
        filter_examples_not_in_blocked_keys(examples, set())


def test_convert_hermes_row_builds_tool_calls_and_strips_tools_xml():
    row = {
        "tools": json.dumps([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }]),
        "conversations": [
            {
                "from": "system",
                "value": "You are helpful.\n<tools>\n[]\n</tools>",
            },
            {"from": "human", "value": "Weather in Dhaka?"},
            {
                "from": "gpt",
                "value": '<tool_call>{"name":"get_weather","arguments":{"city":"Dhaka"}}</tool_call>',
            },
        ],
    }
    converted = convert_hermes_row(row, "hermes-0001")
    assert converted["id"] == "hermes-0001"
    assert converted["tools"][0]["function"]["name"] == "get_weather"
    assert "<tools>" not in converted["messages"][0]["content"]
    assert converted["messages"][1]["role"] == "user"
    assert converted["messages"][2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert converted["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"city": "Dhaka"}


def test_filter_examples_not_in_blocked_keys_drops_matching_prompt():
    tools = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {"type": "object", "properties": {}}}}]
    blocked_example = {
        "id": "blocked",
        "tools": tools,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "same prompt"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "a", "arguments": {}}}]},
        ],
    }
    other_example = {
        "id": "ok",
        "tools": tools,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "different prompt"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "a", "arguments": {}}}]},
        ],
    }
    blocked_keys = {calculate_near_duplicate_key(tools, "same prompt")}
    kept, rejected = filter_examples_not_in_blocked_keys(
        [blocked_example, other_example], blocked_keys
    )
    assert rejected == 1
    assert [example["id"] for example in kept] == ["ok"]


def test_sample_examples_is_deterministic_for_seed():
    examples = [{"id": str(i)} for i in range(10)]
    assert sample_examples(examples, 3, seed=7) == sample_examples(examples, 3, seed=7)
    assert sample_examples(examples, 3, seed=7) != sample_examples(examples, 3, seed=8)


def test_build_mixed_training_rows_preserves_all_toolace_and_samples_external():
    toolace = [{"id": f"t{i}", "messages": [], "tools": []} for i in range(4)]
    external = [{"id": f"e{i}", "messages": [], "tools": []} for i in range(10)]
    mixed, report = build_mixed_training_rows(toolace, external, external_sample_count=3, seed=3)
    assert report["toolace_train_count"] == 4
    assert report["external_sampled"] == 3
    assert report["mixed_total"] == 7
    assert len(mixed) == 7
    assert {row["id"] for row in mixed if row["id"].startswith("t")} == {"t0", "t1", "t2", "t3"}


def test_select_bfcl_ids_prefers_ids_outside_exclude_set():
    all_ids = [f"simple_{i}" for i in range(10)]
    exclude = {f"simple_{i}" for i in range(7)}
    selected = select_bfcl_ids(all_ids, selected_count=5, seed=1, exclude_ids=exclude)
    assert len(selected) == 5
    unused = [example_id for example_id in selected if example_id not in exclude]
    assert len(unused) == 3  # only 3 unused available; 2 filled from excluded
