from copy import deepcopy

import pytest

from prepare_data import calculate_prompt_hash, convert_row, parse_tools_from_system, split_converted_examples


def make_toolace_row(user_text="Get the weather", first_response='[Weather(city="Dhaka")]'):
    return {
        "system": 'Choose a tool.\nHere is a list of functions in JSON format that you can invoke:\n'
                  '[{"name":"Weather","parameters":{"type":"dict","properties":{"days":{"type":"int"}}}}]. Extra instructions.',
        "conversations": [
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": first_response},
            {"from": "tool", "value": '{"temperature": 29}'},
            {"from": "assistant", "value": "It is warm."},
            {"from": "user", "value": "And tomorrow?"},
            {"from": "assistant", "value": '[Weather(city="Dhaka", days=1)]'},
        ],
    }


def test_conversion_preserves_full_conversation_and_dictionary_arguments():
    row = make_toolace_row()
    original_row = deepcopy(row)
    converted_example = convert_row(row, 9)
    assert row == original_row
    assert converted_example["id"] == "toolace-00009"
    assert [message["role"] for message in converted_example["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "user", "assistant"
    ]
    assert converted_example["messages"][2] == {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": "Weather", "arguments": {"city": "Dhaka"}}}
    ]}
    assert converted_example["tools"][0]["function"]["parameters"] == {
        "type": "object", "properties": {"days": {"type": "integer"}}
    }


def test_split_is_repeatable_deduplicated_isolated_and_uses_only_the_first_target():
    converted_examples = [convert_row(make_toolace_row(f"Weather in city {number}"), number)
                          for number in range(8)]
    converted_examples.append(convert_row(make_toolace_row("  Weather  in city 0  "), 8))
    original_examples = deepcopy(converted_examples)
    train_examples, test_examples = split_converted_examples(converted_examples, test_count=3)
    assert (train_examples, test_examples) == split_converted_examples(converted_examples, test_count=3)
    assert converted_examples == original_examples
    assert len(train_examples) == 5
    assert len(test_examples) == 3
    assert not {calculate_prompt_hash(example) for example in train_examples} & {
        calculate_prompt_hash(example) for example in test_examples
    }
    for test_example in test_examples:
        assert len(test_example["messages"]) == 3
        assert test_example["prompt_messages"] == test_example["messages"][:2]
        assert test_example["gold_tool_calls"] == [{"name": "Weather", "arguments": {"city": "Dhaka"}}]
    assert all(len(example["messages"]) == 7 for example in train_examples)


def test_later_tool_calls_do_not_make_a_row_eligible_for_the_single_turn_test_split():
    clarification_example = convert_row(make_toolace_row(first_response="Which city?"), 1)
    with pytest.raises(ValueError, match="Only 0 single-turn targets"):
        split_converted_examples([clarification_example], test_count=1)


def test_empty_tool_lists_are_retained_for_training():
    row = make_toolace_row(first_response="No matching tools.")
    row["system"] = 'Here is a list of functions in JSON format that you can invoke:\n[]'
    row["conversations"] = row["conversations"][:2]
    assert convert_row(row, 0)["tools"] == []


def test_missing_tool_marker_is_reported_at_the_conversion_edge():
    with pytest.raises(ValueError, match="missing the ToolACE tools marker"):
        parse_tools_from_system("No tool definitions here")
