"""Free local check: chat-template render + assistant-only label mask on 2 train rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def truncate_through_first_assistant(messages):
    truncated = []
    for message in messages:
        truncated.append(message)
        if message.get("role") == "assistant":
            break
    if not truncated or truncated[-1].get("role") != "assistant":
        raise ValueError("Example has no assistant turn to train on")
    return truncated


def build_assistant_only_labels(tokenizer, rendered_text, maximum_sequence_length):
    """Mask every token except spans that belong to assistant responses."""
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
    # Match training: first-assistant targets sit at the end, so drop the left side.
    if len(input_ids) > maximum_sequence_length:
        overflow = len(input_ids) - maximum_sequence_length
        input_ids = input_ids[overflow:]
        offset_mapping = offset_mapping[overflow:]
    labels = [-100] * len(input_ids)

    search_start = 0
    assistant_spans = []
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
        assistant_spans.append((content_at, end_at))
        search_start = end_at

    if not assistant_spans:
        raise ValueError("Rendered text contains no assistant spans")

    for token_index, (start_offset, end_offset) in enumerate(offset_mapping):
        if start_offset == end_offset:
            continue
        for span_start, span_end in assistant_spans:
            if start_offset >= span_start and end_offset <= span_end:
                labels[token_index] = input_ids[token_index]
                break

    return input_ids, labels, assistant_spans, rendered_text


def describe_mask(tokenizer, input_ids, labels, rendered_text, assistant_spans):
    supervised_token_count = sum(label != -100 for label in labels)
    total_token_count = len(input_ids)
    supervised_text = tokenizer.decode(
        [token for token, label in zip(input_ids, labels) if label != -100],
        skip_special_tokens=False,
    )
    return {
        "total_tokens": total_token_count,
        "supervised_tokens": supervised_token_count,
        "supervised_ratio": supervised_token_count / max(total_token_count, 1),
        "assistant_span_count": len(assistant_spans),
        "assistant_span_chars": assistant_spans,
        "supervised_decode_preview": supervised_text[:500],
        "contains_tool_call_markers": "<|tool_call_start|>" in supervised_text,
        "prompt_leaked_into_labels": any(
            marker in supervised_text
            for marker in ("List of tools:", "<|im_start|>user", "<|im_start|>system")
        ),
    }


def write_label_mask_report(reports, output_path):
    """results/ is gitignored, so its subdirectories may not exist yet on a fresh clone."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")


def main():
    from transformers import AutoTokenizer

    train_path = ROOT / "data" / "train.jsonl"
    rows = []
    with train_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            truncated = truncate_through_first_assistant(row["messages"])
            if truncated[-1].get("tool_calls"):
                rows.append({**row, "messages": truncated})
            if len(rows) >= 2:
                break
    if len(rows) < 2:
        raise RuntimeError("Need two train rows whose first assistant turn is a tool call")

    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Instruct")
    reports = []
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids, labels, spans, rendered_text = build_assistant_only_labels(
            tokenizer, rendered, maximum_sequence_length=4096,
        )
        report = describe_mask(tokenizer, input_ids, labels, rendered_text, spans)
        report["id"] = row["id"]
        reports.append(report)
        print(f"=== {row['id']} ===")
        print(json.dumps({k: report[k] for k in report if k != "assistant_span_chars"}, indent=2))
        print("supervised_preview:", report["supervised_decode_preview"])
        print()

    if any(report["prompt_leaked_into_labels"] for report in reports):
        raise SystemExit("Label mask leaked prompt tokens")
    if not all(report["contains_tool_call_markers"] for report in reports):
        raise SystemExit("Supervised spans missing tool-call markers")
    if not all(report["supervised_tokens"] > 0 for report in reports):
        raise SystemExit("No supervised tokens")
    print("LABEL_MASK_OK")
    out = ROOT / "results" / "training" / "label_mask_check.json"
    write_label_mask_report(reports, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
