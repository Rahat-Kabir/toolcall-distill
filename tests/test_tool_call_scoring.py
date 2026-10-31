import json

import pytest

from eval_toolcalls import (
    parse_tool_calls_from_model_output,
    score_prediction_files,
    score_predictions,
    score_tool_calls,
)


def test_right_call_passes_all_three_levels():
    predicted_tool_calls = parse_tool_calls_from_model_output(
        '<|tool_call_start|>[Market Trends API(country="us")]<|tool_call_end|>'
    )
    assert predicted_tool_calls == [{"name": "Market Trends API", "arguments": {"country": "us"}}]
    assert score_tool_calls(predicted_tool_calls, predicted_tool_calls) == {
        "is_parseable_tool_call": True,
        "is_correct_tool_name": True,
        "are_arguments_correct": True,
    }


def test_wrong_name_fails_name_and_argument_levels():
    assert score_tool_calls(
        [{"name": "wrong", "arguments": {}}], [{"name": "right", "arguments": {}}]
    ) == {
        "is_parseable_tool_call": True,
        "is_correct_tool_name": False,
        "are_arguments_correct": False,
    }


def test_wrong_arguments_fail_only_argument_level():
    assert score_tool_calls(
        [{"name": "weather", "arguments": {"city": "Paris"}}],
        [{"name": "weather", "arguments": {"city": "Dhaka"}}],
    ) == {
        "is_parseable_tool_call": True,
        "is_correct_tool_name": True,
        "are_arguments_correct": False,
    }


@pytest.mark.parametrize("raw_text", ["hello", '{"answer": 42}', '[1, 2]', '[]',
    '[weather(city=unknown)]', '[weather(**{"city": "Paris"})]',
    '[weather(city=__import__("os").getcwd())]', '[weather(city="Paris", city="Rome")]'])
def test_invalid_or_non_tool_text_returns_no_calls(raw_text):
    assert parse_tool_calls_from_model_output(raw_text) == []
    assert not any(score_tool_calls([], [{"name": "weather", "arguments": {}}]).values())


def test_argument_key_order_does_not_affect_equality():
    assert score_tool_calls(
        [{"name": "weather", "arguments": {"city": "Dhaka", "days": 3}}],
        [{"name": "weather", "arguments": {"days": 3, "city": "Dhaka"}}],
    )["are_arguments_correct"]


def test_multiple_calls_must_match_names_arguments_order_and_count():
    gold_tool_calls = [{"name": "first", "arguments": {"number": 1}},
                       {"name": "second", "arguments": {}}]
    assert parse_tool_calls_from_model_output('[first(number=1), second()]') == gold_tool_calls
    assert score_tool_calls(gold_tool_calls, gold_tool_calls)["are_arguments_correct"]
    for predicted_tool_calls in [gold_tool_calls[::-1], gold_tool_calls[:1], gold_tool_calls * 2]:
        assert not score_tool_calls(predicted_tool_calls, gold_tool_calls)["is_correct_tool_name"]
        assert not score_tool_calls(predicted_tool_calls, gold_tool_calls)["are_arguments_correct"]


@pytest.mark.parametrize("predicted_value,gold_value", [
    (" 3 ", 3), (3.5, "3.5"), ("-2", -2), (True, " TRUE "),
    ("false", False), ({"items": ["  text  ", "2", "false"]}, {"items": ["text", 2, False]}),
])
def test_number_boolean_and_nested_string_tolerance(predicted_value, gold_value):
    assert score_tool_calls(
        [{"name": "check", "arguments": {"value": predicted_value}}],
        [{"name": "check", "arguments": {"value": gold_value}}],
    )["are_arguments_correct"]


@pytest.mark.parametrize("predicted_value,gold_value", [(True, 1), (False, 0), ("3x", 3), ([1, 2], [2, 1])])
def test_different_types_and_list_order_are_not_silently_equal(predicted_value, gold_value):
    assert not score_tool_calls(
        [{"name": "check", "arguments": {"value": predicted_value}}],
        [{"name": "check", "arguments": {"value": gold_value}}],
    )["are_arguments_correct"]


@pytest.mark.parametrize("raw_text", [
    '{"name":"weather","arguments":{"city":"Dhaka"}}',
    '{"tool_calls":[{"type":"function","function":{"name":"weather","arguments":"{\\"city\\":\\"Dhaka\\"}"}}]}',
    'Here is the call: [{"name":"weather","arguments":{"city":"Dhaka"}}]',
    '```python\n[weather(city="Dhaka")]\n```',
])
def test_json_and_fenced_output_return_the_same_call(raw_text):
    assert parse_tool_calls_from_model_output(raw_text) == [
        {"name": "weather", "arguments": {"city": "Dhaka"}}
    ]


@pytest.mark.parametrize("raw_text,tool_name,arguments", [
    ('[User Feed (Video Posts) V2(username="sunnydays")]', "User Feed (Video Posts) V2", {"username": "sunnydays"}),
    ('[Financial API(from="2025-01-01", to="2025-12-31")]', "Financial API", {"from": "2025-01-01", "to": "2025-12-31"}),
    ("[Get Agent's Listings(zuid='ZXY123')]", "Get Agent's Listings", {"zuid": "ZXY123"}),
    ('[check(value={"enabled": true, "missing": null})]', "check", {"value": {"enabled": True, "missing": None}}),
])
def test_dataset_names_reserved_keywords_and_template_json_literals_are_preserved(raw_text, tool_name, arguments):
    assert parse_tool_calls_from_model_output(raw_text) == [{"name": tool_name, "arguments": arguments}]


@pytest.mark.parametrize("raw_text,tool_name,arguments", [
    ('[Get Scores($top=10)]', "Get Scores", {"$top": 10}),
    ('[Random Meme API(number=5, media-type="image")]', "Random Meme API", {"number": 5, "media-type": "image"}),
    ('[Search Title, Text, or URL(query="Lincoln")]', "Search Title, Text, or URL", {"query": "Lincoln"}),
    ('[GET /categories/{categoryId}/v1(categoryid="PLA89")]', "GET /categories/{categoryId}/v1", {"categoryid": "PLA89"}),
])
def test_tool_and_argument_names_keep_dataset_punctuation(raw_text, tool_name, arguments):
    assert parse_tool_calls_from_model_output(raw_text) == [{"name": tool_name, "arguments": arguments}]


def test_separate_lfm_blocks_preserve_all_predicted_calls():
    predicted_tool_calls = parse_tool_calls_from_model_output(
        '<|tool_call_start|>[first()]<|tool_call_end|>\n'
        '<|tool_call_start|>[second()]<|tool_call_end|>'
    )
    assert predicted_tool_calls == [
        {"name": "first", "arguments": {}}, {"name": "second", "arguments": {}}
    ]
    assert not score_tool_calls(predicted_tool_calls, predicted_tool_calls[:1])["are_arguments_correct"]


def test_aggregate_matches_ids_and_counts_missing_predictions_as_failures():
    from eval_toolcalls import score_predictions

    summary = score_predictions(
        [{"id": "second", "raw_output": '[Weather(city=" Dhaka ")]'}],
        [{"id": "first", "gold_tool_calls": [{"name": "Weather", "arguments": {"city": "Dhaka"}}]},
         {"id": "second", "gold_tool_calls": [{"name": "Weather", "arguments": {"city": "Dhaka"}}]}],
    )
    assert summary["example_count"] == 2
    assert summary["missing_prediction_count"] == 1
    assert summary["counts"] == {
        "is_parseable_tool_call": 1, "is_correct_tool_name": 1, "are_arguments_correct": 1
    }
    assert summary["rates"] == {
        "is_parseable_tool_call": 0.5, "is_correct_tool_name": 0.5, "are_arguments_correct": 0.5
    }


def test_preparsed_prediction_calls_are_scored_through_the_same_public_api():
    from eval_toolcalls import score_predictions

    summary = score_predictions(
        [{"id": 1, "predicted_tool_calls": [{"name": "Weather", "arguments": {}}]}],
        [{"id": 1, "gold_tool_calls": [{"name": "Weather", "arguments": {}}]}],
    )
    assert summary["rates"]["are_arguments_correct"] == 1.0


@pytest.mark.parametrize("predictions,error_message", [
    ([{"id": 2, "raw_output": "hello"}], "absent from gold"),
    ([{"id": 1, "raw_output": "hello"}, {"id": 1, "raw_output": "hello"}], "Duplicate"),
    ([{"id": 1}], "requires raw_output"),
])
def test_invalid_prediction_records_fail_clearly(predictions, error_message):
    from eval_toolcalls import score_predictions

    with pytest.raises(ValueError, match=error_message):
        score_predictions(predictions, [{"id": 1, "gold_tool_calls": [{"name": "Weather", "arguments": {}}]}])


def test_argument_names_with_spaces_are_preserved_as_dictionary_keys():
    assert parse_tool_calls_from_model_output('[Battle Statistics(Server="Alpha", Battle ID="98765")]') == [
        {"name": "Battle Statistics", "arguments": {"Server": "Alpha", "Battle ID": "98765"}}
    ]


def test_parse_qwen_tool_call_xml_tags():
    raw = """<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"""
    assert parse_tool_calls_from_model_output(raw) == [
        {"name": "get_weather", "arguments": {"city": "Paris"}}
    ]


def test_parse_llama_parameters_json_object():
    raw = '{"name": "get_weather", "parameters": {"city": "Paris"}}'
    assert parse_tool_calls_from_model_output(raw) == [
        {"name": "get_weather", "arguments": {"city": "Paris"}}
    ]


def test_parse_multiple_qwen_tool_call_blocks():
    raw = """<tool_call>
{"name": "a", "arguments": {"x": 1}}
</tool_call>
<tool_call>
{"name": "b", "arguments": {"y": 2}}
</tool_call>"""
    assert parse_tool_calls_from_model_output(raw) == [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]


def test_score_predictions_with_path_strings_is_rejected(tmp_path):
    """Jobs once passed path strings into score_predictions and crashed in index_examples_by_id."""
    predictions_path = tmp_path / "predictions.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "ex-1", "raw_output": "<|tool_call_start|>[demo(x=1)]<|tool_call_end|>"}) + "\n"
    )
    gold_path.write_text(
        json.dumps({"id": "ex-1", "gold_tool_calls": [{"name": "demo", "arguments": {"x": 1}}]}) + "\n"
    )
    with pytest.raises(TypeError, match="lists of dicts"):
        score_predictions(str(predictions_path), str(gold_path))


def test_score_prediction_files_scores_from_paths(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    predictions_path.write_text(
        json.dumps({"id": "ex-1", "raw_output": "<|tool_call_start|>[demo(x=1)]<|tool_call_end|>"}) + "\n"
    )
    gold_path.write_text(
        json.dumps({"id": "ex-1", "gold_tool_calls": [{"name": "demo", "arguments": {"x": 1}}]}) + "\n"
    )
    summary = score_prediction_files(predictions_path, gold_path)
    assert summary["rates"]["are_arguments_correct"] == 1.0
    assert summary["example_count"] == 1


def test_llama_python_tag_semicolon_separated_calls_are_all_parsed():
    """Llama 3.1 emits multiple JSON objects after one <|python_tag|>, separated by semicolons."""
    raw_text = (
        '<|python_tag|>{"name": "Emoji API", "parameters": {"name": "sunglasses"}}; '
        '{"name": "GET Hashtag", "parameters": {"tag": "ThrowbackThursday"}}; '
        '{"name": "Search Top Music", "parameters": {"country": "MX", "language": "es"}}'
        '<|eom_id|>'
    )
    assert parse_tool_calls_from_model_output(raw_text) == [
        {"name": "Emoji API", "arguments": {"name": "sunglasses"}},
        {"name": "GET Hashtag", "arguments": {"tag": "ThrowbackThursday"}},
        {"name": "Search Top Music", "arguments": {"country": "MX", "language": "es"}},
    ]


def test_llama_python_tag_single_call_still_parses():
    raw_text = (
        '<|python_tag|>{"name": "Get Geocoding", "parameters": '
        '{"version": "v1", "lat": 34.052235, "lng": -118.243683, "lang": "en"}}'
        '<|eom_id|>'
    )
    assert parse_tool_calls_from_model_output(raw_text) == [
        {
            "name": "Get Geocoding",
            "arguments": {
                "version": "v1",
                "lat": 34.052235,
                "lng": -118.243683,
                "lang": "en",
            },
        }
    ]
