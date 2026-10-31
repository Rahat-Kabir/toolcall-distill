"""Shared runner behavior, with inference replaced only at the model boundary."""

import base64
from contextlib import nullcontext
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))


def test_completed_predictions_survive_a_later_inference_failure(tmp_path, capsys):
    from run_eval_job import run_evaluation

    def generate_output(example):
        if example["id"] == "second":
            raise RuntimeError("inference failed")
        return '[weather(city="Dhaka")]'

    output_path = tmp_path / "predictions.jsonl"
    with pytest.raises(RuntimeError, match="inference failed"):
        run_evaluation(
            [{"id": "first"}, {"id": "second"}], output_path, generate_output,
            {"model": "example", "seed": 11}, log_marker="PREDICTIONS_B64_GZ",
        )
    predictions = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert predictions == [{"id": "first", "raw_output": '[weather(city="Dhaka")]',
                            "model": "example", "seed": 11}]
    logged_payload = capsys.readouterr().out.split("PREDICTIONS_B64_GZ_BEGIN\n")[1].splitlines()[0]
    assert gzip.decompress(base64.b64decode(logged_payload)) == output_path.read_bytes()


def test_existing_predictions_are_not_overwritten_without_explicit_permission(tmp_path):
    from run_eval_job import run_evaluation

    output_path = tmp_path / "predictions.jsonl"
    output_path.write_text("existing output\n")
    with pytest.raises(FileExistsError):
        run_evaluation([{"id": "first"}], output_path, lambda example: "unused", {})
    assert output_path.read_text() == "existing output\n"


@pytest.mark.parametrize("decoding_options", [
    {"max_new_tokens": 16, "do_sample": False},
    {"max_new_tokens": 64, "do_sample": True, "temperature": 0.7, "top_p": 0.9},
])
def test_uses_native_tool_template_and_passes_decoding_settings(decoding_options):
    from run_eval_job import generate_model_output

    class EncodedPrompt(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class InputTokens:
        shape = (1, 3)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, messages, **options):
            assert messages == [{"role": "user", "content": "weather?"}]
            assert options == {"tools": [{"type": "function"}], "tokenize": True,
                               "return_tensors": "pt", "return_dict": True,
                               "add_generation_prompt": True}
            return EncodedPrompt(input_ids=InputTokens())

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [44, 45]
            assert skip_special_tokens is False
            return "<|tool_call_start|>[weather()]<|tool_call_end|>"

    class Model:
        def generate(self, **options):
            assert options["pad_token_id"] == 0
            for key, value in decoding_options.items():
                assert options[key] == value
            if not decoding_options["do_sample"]:
                assert "temperature" not in options and "top_p" not in options
            return [[1, 2, 3, 44, 45]]

    output = generate_model_output(
        Model(), Tokenizer(), {"tools": [{"type": "function"}],
                              "prompt_messages": [{"role": "user", "content": "weather?"}]},
        "cpu", decoding_options, nullcontext,
    )
    assert output == "<|tool_call_start|>[weather()]<|tool_call_end|>"


@pytest.mark.parametrize("changes,extra_arguments,error_text", [
    ({"prompt_messages": []}, [], "end with a user"),
    ({"tools": "bad"}, [], "tools"),
    ({}, ["--max-new-tokens", "0"], "positive"),
    ({}, ["--sample", "--temperature", "0"], "temperature"),
    ({}, ["--sample", "--top-p", "1.5"], "top-p"),
    ({}, ["--temperature", "0.7"], "--sample"),
])
def test_cli_rejects_invalid_requests_before_model_loading(tmp_path, changes, extra_arguments, error_text):
    example = {"id": "example", "tools": [],
               "prompt_messages": [{"role": "user", "content": "hello"}], **changes}
    input_path = tmp_path / "test.jsonl"
    input_path.write_text(json.dumps(example) + "\n")
    completed = subprocess.run(
        [sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts" / "run_eval_job.py"),
         "--model", "unused", "--input", str(input_path), "--output", str(tmp_path / "output.jsonl"),
         "--check-inputs", *extra_arguments], capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert error_text in completed.stderr
    assert "No module named" not in completed.stderr


def test_saved_log_blocks_work_with_the_larger_eval_collector(tmp_path, capsys):
    from run_eval_job import run_evaluation

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps({
        "id": "example", "gold_tool_calls": [{"name": "weather", "arguments": {}}],
    }) + "\n")
    for split in ("TOOLACE", "OOD"):
        run_evaluation(
            [{"id": "example"}], tmp_path / f"{split}.jsonl", lambda example: "[weather()]", {},
            log_marker=f"PREDS_BASE_{split}_B64_GZ",
        )
    log_path = tmp_path / "job.log"
    log_path.write_text(capsys.readouterr().out, encoding="utf-8")
    report_directory = tmp_path / "collected"
    completed = subprocess.run([
        sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts" / "collect_larger_eval.py"),
        "--log-path", str(log_path), "--toolace-gold", str(gold_path), "--ood-gold", str(gold_path),
        "--output-directory", str(report_directory), "--labels", "base",
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((report_directory / "summary.json").read_text())
    assert summary["models"][0]["toolace_args"] == 1
    assert summary["models"][0]["ood_args"] == 1
