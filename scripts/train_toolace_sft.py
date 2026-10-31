# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "unsloth",
#     "datasets",
#     "trl==0.22.2",
#     "huggingface_hub[hf_transfer]",
#     "trackio",
#     "tensorboard",
#     "transformers==4.57.3",
# ]
# ///
"""Fine-tune LiquidAI/LFM2.5-1.2B-Instruct for ToolACE tool calling with Unsloth LoRA.

Adapted from the official Unsloth HF Jobs script (unsloth/jobs sft-lfm2.5.py):
same LoRA targets, lr, rank defaults, and Unsloth trainer path. Data is our
caller-supplied ToolACE train split, rendered with apply_chat_template(..., tools=...)
exactly as eval, truncated through the first assistant turn, with labels on
assistant tokens only. Packing stays off.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"
LOSS_LOG_MARKER = "TRAIN_LOSS_LOG_JSON"



def require_training_file(input_path: Path) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"Training file not found: {input_path}. Run prepare_data.py first.")
    if input_path.stat().st_size == 0:
        raise ValueError(f"Training file is empty: {input_path}")
    return input_path


def truncate_through_first_assistant(messages):
    truncated = []
    for message in messages:
        truncated.append(message)
        if message.get("role") == "assistant":
            break
    if not truncated or truncated[-1].get("role") != "assistant":
        raise ValueError("Training example requires an assistant turn")
    return truncated


def build_assistant_only_labels(tokenizer, rendered_text, maximum_sequence_length):
    if tokenizer.bos_token and rendered_text.startswith(tokenizer.bos_token):
        rendered_text = rendered_text[len(tokenizer.bos_token) :]
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offset_mapping = list(encoded["offset_mapping"])
    # First-assistant targets sit at the end; drop the left side if over length.
    if len(input_ids) > maximum_sequence_length:
        overflow = len(input_ids) - maximum_sequence_length
        input_ids = input_ids[overflow:]
        offset_mapping = offset_mapping[overflow:]
    labels = [-100] * len(input_ids)
    search_start = 0
    found_span = False
    while True:
        header_at = rendered_text.find(ASSISTANT_HEADER, search_start)
        if header_at < 0:
            break
        content_at = header_at + len(ASSISTANT_HEADER)
        end_at = rendered_text.find(IM_END, content_at)
        if end_at < 0:
            end_at = len(rendered_text)
        else:
            end_at += len(IM_END)
        found_span = True
        for token_index, (start_offset, end_offset) in enumerate(offset_mapping):
            if start_offset == end_offset:
                continue
            if start_offset >= content_at and end_offset <= end_at:
                labels[token_index] = input_ids[token_index]
        search_start = end_at
    if not found_span:
        raise ValueError("Rendered text has no assistant span")
    if all(label == -100 for label in labels):
        raise ValueError("Assistant mask produced no supervised tokens")
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def parse_args():
    parser = argparse.ArgumentParser(description="ToolACE LoRA SFT for LFM2.5 with Unsloth")
    parser.add_argument("--base-model", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--output-repo", help="Destination required only with --push-to-hub")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--num-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-local", default="results/training/adapter")
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--public-repo", action="store_true", default=False,
                        help="Upload to a public repo; the default upload is private")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        logger.error("CUDA is required")
        sys.exit(1)
    logger.info("CUDA available: %s", torch.cuda.get_device_name(0))


def push_adapter_to_hub(model, tokenizer, output_repo, public_repo, hub_api):
    """Both the repo creation and the upload must agree on the visibility."""
    repo_visibility = "public" if public_repo else "private"
    hub_api.create_repo(output_repo, repo_type="model", private=not public_repo, exist_ok=True)
    print(f"Pushing {repo_visibility} adapter to {output_repo}...")
    model.push_to_hub(output_repo, tokenizer=tokenizer, private=not public_repo)
    print(f"Adapter at https://huggingface.co/{output_repo}")


def main():
    args = parse_args()
    train_path = require_training_file(args.train_jsonl)
    if args.push_to_hub and not args.output_repo:
        raise ValueError("--push-to-hub requires --output-repo")
    if not args.num_epochs and not args.max_steps:
        args.num_epochs = 1
        logger.info("Defaulting to --num-epochs=1")

    print("=" * 70)
    print("LFM2.5 ToolACE LoRA SFT (Unsloth)")
    print("=" * 70)
    print(f"  base_model={args.base_model}")
    print(f"  output_repo={args.output_repo}")
    print(f"  num_epochs={args.num_epochs} max_steps={args.max_steps}")
    print(f"  batch={args.batch_size}x{args.gradient_accumulation} lr={args.learning_rate}")
    print(f"  lora_r={args.lora_r} max_seq_length={args.max_seq_length}")
    print(f"  num_samples={args.num_samples or 'all'} push_to_hub={args.push_to_hub}")
    print(f"  packing=False assistant_tokens_only=True")
    print()

    check_cuda()
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from huggingface_hub import HfApi, login
    import torch

    token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if token:
        login(token=token)
        logger.info("Logged in to Hugging Face Hub")
    else:
        logger.warning("HF_TOKEN not set")

    rows = []
    with train_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row.get("tools"), list) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Invalid train row: {row.get('id')}")
            row = {
                "id": row["id"],
                "tools": row["tools"],
                "messages": truncate_through_first_assistant(row["messages"]),
            }
            rows.append(row)
    if args.num_samples:
        rows = rows[: min(args.num_samples, len(rows))]
    if len(rows) < 1:
        raise RuntimeError("No training rows loaded")
    print(f"Loaded {len(rows)} training examples from {train_path}")

    print("\n[1/5] Loading model...")
    started = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    print(f"Model ready in {time.time() - started:.1f}s")

    print("\n[2/5] Tokenizing with assistant-only labels...")
    started = time.time()
    # Unsloth's wrapped tokenizer can break tools= in apply_chat_template (Jinja Undefined).
    # Stock Transformers tokenizer matches base-model vocab and matches eval rendering.
    from transformers import AutoTokenizer as StockAutoTokenizer
    rendering_tokenizer = StockAutoTokenizer.from_pretrained(args.base_model)

    def sanitize_for_template(value):
        return json.loads(json.dumps(value, ensure_ascii=False))

    tokenized_rows = []
    skipped_rows = []
    for index, row in enumerate(rows):
        messages = sanitize_for_template(row["messages"])
        tools = sanitize_for_template(row["tools"])
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls") and "content" not in message:
                message["content"] = ""
        rendered = rendering_tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        try:
            encoded = build_assistant_only_labels(
                rendering_tokenizer, rendered, args.max_seq_length,
            )
        except ValueError as error:
            skipped_rows.append({"id": row["id"], "reason": str(error)})
            continue
        encoded["id"] = row["id"]
        tokenized_rows.append(encoded)
        if (index + 1) % 500 == 0 or index + 1 == len(rows):
            print(f"  tokenized {index + 1}/{len(rows)} skipped={len(skipped_rows)}", flush=True)
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} rows without supervised labels (e.g. {skipped_rows[:3]})")
    if len(tokenized_rows) < 8:
        raise RuntimeError(f"Too few tokenized rows: {len(tokenized_rows)}")
    train_dataset = Dataset.from_list(tokenized_rows)
    print(f"Tokenized in {time.time() - started:.1f}s; kept {len(tokenized_rows)}")

    print("\n[3/5] Configuring trainer...")
    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    run_name = args.run_name or (
        f"lfm25-toolace-{args.num_epochs}ep" if args.num_epochs else f"lfm25-toolace-{args.max_steps}steps"
    )
    training_config = SFTConfig(
        output_dir=args.save_local,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=5 if (args.max_steps or 100) > 5 else 0,
        num_train_epochs=args.num_epochs if args.num_epochs else 1,
        max_steps=args.max_steps if args.max_steps else -1,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=False,
        report_to=["tensorboard"],
        run_name=run_name,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=max(args.logging_steps * 5, 20),
        save_total_limit=2,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    from transformers import TrainerCallback

    class LossRecorder(TrainerCallback):
        def __init__(self):
            self.points = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return control
            point = {
                "step": int(state.global_step),
                "loss": float(logs["loss"]),
                "learning_rate": float(logs.get("learning_rate", -1)),
            }
            self.points.append(point)
            print(f"LOSS_POINT {json.dumps(point)}", flush=True)
            return control

    loss_recorder = LossRecorder()

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_config,
        callbacks=[loss_recorder],
    )

    print(f"\n[4/5] Training (~{steps_per_epoch} steps/epoch, effective_batch={effective_batch})...")
    started = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - started
    print(f"Training wall seconds: {train_seconds:.1f}")
    print(f"Metrics: {json.dumps(train_result.metrics, default=str)}")

    print("\n[5/5] Saving adapter...")
    Path(args.save_local).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save_local)
    tokenizer.save_pretrained(args.save_local)
    adapter_files = sorted(p.name for p in Path(args.save_local).glob("*") if p.is_file())
    print(f"Saved adapter files: {adapter_files}")
    if not any(name.startswith("adapter") or name == "adapter_config.json" or name.endswith(".safetensors") for name in adapter_files):
        raise RuntimeError(f"Adapter save missing expected files: {adapter_files}")

    if args.push_to_hub:
        push_adapter_to_hub(model, tokenizer, args.output_repo, args.public_repo, HfApi(token=token))

    summary = {
        "train_seconds": train_seconds,
        "metrics": train_result.metrics,
        "loss_points": loss_recorder.points,
        "num_examples": len(train_dataset),
        "steps_per_epoch_estimate": steps_per_epoch,
        "output_repo": args.output_repo if args.push_to_hub else None,
        "save_local": args.save_local,
        "adapter_files": adapter_files,
    }
    print(f"{LOSS_LOG_MARKER}_BEGIN", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"{LOSS_LOG_MARKER}_END", flush=True)
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
