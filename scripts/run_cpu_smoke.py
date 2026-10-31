"""Run three real CPU generations through the evaluation CLI; save all evidence."""

import json
from pathlib import Path
import subprocess
import sys


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from eval_toolcalls import (parse_tool_calls_from_model_output, read_jsonl,
                            score_predictions, write_jsonl)


MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"


def run_generation(model_name, report_file):
    report_file.write(f"\nReal generation: {model_name}; device=cpu; limit=3; max_new_tokens=128\n")
    report_file.flush()
    return subprocess.run(
        [sys.executable, str(PROJECT_DIRECTORY / "eval_toolcalls.py"), "generate",
         "--model", model_name, "--input", "data/test.jsonl", "--output", "data/smoke_predictions.jsonl",
         "--limit", "3", "--device", "cpu", "--max-new-tokens", "128"],
        cwd=PROJECT_DIRECTORY, stdout=report_file, stderr=subprocess.STDOUT, check=False,
    )


def main():
    report_path = PROJECT_DIRECTORY / "data/smoke_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected_model = MODEL_NAME
    with report_path.open("w", encoding="utf-8") as report_file:
        report_file.write("Free local CPU smoke. This is a three-example pipeline check.\n")
        completed_generation = run_generation(selected_model, report_file)
        if completed_generation.returncode:
            report_file.flush()
            generation_log = report_path.read_text(encoding="utf-8").lower()
            is_out_of_memory = completed_generation.returncode in {-9, 137} or any(
                memory_message in generation_log
                for memory_message in ["out of memory", "cannot allocate memory", "memoryerror"]
            )
            if not is_out_of_memory:
                raise RuntimeError(f"CPU generation failed; see {report_path}")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            example = read_jsonl(PROJECT_DIRECTORY / "data/test.jsonl")[0]
            rendered_conversation = tokenizer.apply_chat_template(
                example["messages"], tools=example["tools"], tokenize=False,
            )
            forced_output = rendered_conversation.rsplit("<|im_start|>assistant\n", 1)[1]
            forced_tool_calls = parse_tool_calls_from_model_output(forced_output)
            if forced_tool_calls != example["gold_tool_calls"]:
                raise RuntimeError("Tokenizer-only round trip failed")
            report_file.write("\nLFM memory fallback: tokenizer-only FORCED output (not model generation):\n")
            report_file.write(forced_output + "\n")
            selected_model = "HuggingFaceTB/SmolLM2-135M-Instruct"
            completed_generation = run_generation(selected_model, report_file)
            completed_generation.check_returncode()
        predictions = read_jsonl(PROJECT_DIRECTORY / "data/smoke_predictions.jsonl")
        predicted_ids = {prediction["id"] for prediction in predictions}
        gold_examples = [example for example in read_jsonl(PROJECT_DIRECTORY / "data/test.jsonl")
                         if example["id"] in predicted_ids]
        write_jsonl(PROJECT_DIRECTORY / "data/smoke_gold.jsonl", gold_examples)
        summary = score_predictions(predictions, gold_examples)
        (PROJECT_DIRECTORY / "data/smoke_summary.json").write_text(
            json.dumps({"model": selected_model, **summary}, indent=2) + "\n", encoding="utf-8",
        )
        report_file.write(f"\nScored actual model: {selected_model}\n")
        for prediction, example_score in zip(predictions[:3], summary["examples"][:3]):
            report_file.write(f"\nID: {prediction['id']}\nRAW OUTPUT:\n{prediction['raw_output']}\n")
            report_file.write("SCORES: " + json.dumps(example_score) + "\n")
        report_file.write("\nSUMMARY:\n" + json.dumps(summary, indent=2) + "\n")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
