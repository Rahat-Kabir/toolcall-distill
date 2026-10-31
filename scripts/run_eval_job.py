# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.10.0",
#   "transformers==4.57.6",
#   "accelerate==1.13.0",
#   "peft==0.18.1",
# ]
# ///
"""Generate tool-call predictions from one model/adapter and one JSONL dataset.

Scoring is a separate local step. Each completed prediction is flushed to disk;
an optional gzip/base64 log block allows recovery from hosted-job logs.
"""

import argparse
import base64
import gzip
import json
import math
from pathlib import Path
import re
import sys

from job_inputs import PROJECT_DIRECTORY, read_evaluation_examples

sys.path.insert(0, str(PROJECT_DIRECTORY))


def generate_model_output(model, tokenizer, example, device, decoding_options, inference_context):
    # Let the native template tokenize once, avoiding duplicate special tokens.
    encoded_prompt = tokenizer.apply_chat_template(
        example["prompt_messages"], tools=example["tools"], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)
    padding_token_id = tokenizer.pad_token_id
    if padding_token_id is None:
        padding_token_id = tokenizer.eos_token_id
    with inference_context():
        generated_tokens = model.generate(
            **encoded_prompt, **decoding_options, pad_token_id=padding_token_id,
        )
    new_tokens = generated_tokens[0][encoded_prompt["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def load_output_generator(model_name, adapter_repo, device, dtype, seed, decoding_options):
    """Load inference dependencies only after input/configuration checks pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if dtype == "auto":
        if device == "cpu":
            dtype = "float32"
        else:
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=getattr(torch, dtype), low_cpu_mem_usage=True,
    ).to(device)
    if adapter_repo:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_repo)
    model.eval()
    print(f"Loaded model={model_name} adapter={adapter_repo} device={device} dtype={dtype}", flush=True)

    def generate_output(example):
        # Historical sampling runs reset the seed for every example.
        if decoding_options["do_sample"]:
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
        return generate_model_output(
            model, tokenizer, example, device, decoding_options, torch.inference_mode,
        )

    return generate_output, {"device": device, "dtype": dtype}


def run_evaluation(examples, output_path, generate_output, metadata, log_marker=None, overwrite=False):
    """Persist completed inference results, including when a later example fails."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_count = 0
    # Exclusive creation protects prior GPU work unless replacement is requested.
    with output_path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as output_file:
        try:
            for example in examples:
                raw_output = generate_output(example)
                if not isinstance(raw_output, str):
                    raise ValueError("Model output must be a string")
                prediction = {**metadata, "id": example["id"], "raw_output": raw_output}
                output_file.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                output_file.flush()
                completed_count += 1
                print(f"Generated {completed_count}/{len(examples)}: {example['id']}", flush=True)
        finally:
            # Recover partial outputs from logs before an inference error propagates.
            if log_marker:
                payload = base64.b64encode(gzip.compress(output_path.read_bytes())).decode("ascii")
                print(f"{log_marker}_BEGIN\n{payload}\n{log_marker}_END", flush=True)
    return completed_count


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base Hub model ID or local model directory")
    parser.add_argument("--adapter", help="Optional Hub adapter ID or local adapter directory")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with id, tools, prompt_messages")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL destination")
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, cuda:N, or auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Generate only the first N examples")
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (default with --sample: 0.7)")
    parser.add_argument("--top-p", type=float, help="Sampling nucleus probability (default with --sample: 0.9)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed, reset per example")
    parser.add_argument("--log-marker", help="Emit MARKER_BEGIN/END gzip/base64 predictions for log recovery")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace existing predictions")
    parser.add_argument("--check-inputs", action="store_true", help="Validate without loading a model or writing output")
    arguments = parser.parse_args()
    if not arguments.model.strip() or (arguments.adapter is not None and not arguments.adapter.strip()):
        parser.error("Model and adapter identifiers must be nonempty")
    if not re.fullmatch(r"cpu|auto|cuda(?::[0-9]+)?", arguments.device):
        parser.error("--device must be cpu, auto, cuda, or cuda:N")
    if arguments.max_new_tokens <= 0 or (arguments.limit is not None and arguments.limit <= 0):
        parser.error("Token and example limits must be positive")
    if not 0 <= arguments.seed < 2**63:
        parser.error("--seed must be between 0 and 2**63 - 1")
    if not arguments.sample and (arguments.temperature is not None or arguments.top_p is not None):
        parser.error("--temperature and --top-p require --sample")
    if arguments.sample:
        arguments.temperature = 0.7 if arguments.temperature is None else arguments.temperature
        arguments.top_p = 0.9 if arguments.top_p is None else arguments.top_p
        if not math.isfinite(arguments.temperature) or arguments.temperature <= 0:
            parser.error("--temperature must be finite and positive")
        if not math.isfinite(arguments.top_p) or not 0 < arguments.top_p <= 1:
            parser.error("--top-p must be in (0, 1]")
    if arguments.log_marker and not re.fullmatch(r"[A-Z][A-Z0-9_]*", arguments.log_marker):
        parser.error("--log-marker must use uppercase letters, digits, and underscores")
    if arguments.input.resolve() == arguments.output.resolve():
        parser.error("Output must differ from input")
    if arguments.output.is_dir():
        parser.error("Output must be a file path")
    if arguments.output.exists() and not arguments.overwrite and not arguments.check_inputs:
        parser.error("Output already exists; use a new path or --overwrite")
    return arguments


def main():
    arguments = parse_arguments()
    examples = read_evaluation_examples(arguments.input, arguments.limit)
    if arguments.check_inputs:
        print(f"INPUTS_OK {len(examples)} examples from {arguments.input}")
        return
    decoding_options = {"max_new_tokens": arguments.max_new_tokens, "do_sample": arguments.sample}
    if arguments.sample:
        decoding_options.update(temperature=arguments.temperature, top_p=arguments.top_p)
    generate_output, resolved_runtime = load_output_generator(
        arguments.model, arguments.adapter, arguments.device, arguments.dtype,
        arguments.seed, decoding_options,
    )
    metadata = {
        "model": arguments.model, "adapter": arguments.adapter, "seed": arguments.seed,
        **resolved_runtime, **decoding_options,
    }
    count = run_evaluation(
        examples, arguments.output, generate_output, metadata,
        log_marker=arguments.log_marker, overwrite=arguments.overwrite,
    )
    print(f"Saved {count} predictions to {arguments.output}; score them locally.", flush=True)


if __name__ == "__main__":
    main()
