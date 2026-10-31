#!/usr/bin/env python3
"""Build new ToolACE split, larger BFCL OOD, and mixed Hermes train (no leakage)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_toolcalls import write_jsonl
from mixed_data_prep import (
    blocked_keys_from_examples,
    build_mixed_training_rows,
    convert_hermes_row,
    filter_examples_not_in_blocked_keys,
    select_bfcl_ids,
)
from prepare_data import convert_row, split_converted_examples


def load_prepare_ood_module():
    module_path = ROOT / "scripts" / "prepare_ood_bfcl.py"
    spec = importlib.util.spec_from_file_location("prepare_ood_bfcl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_toolace_split(test_count, seed, output_directory):
    dataset = load_dataset("Team-ACE/ToolACE", split="train")
    converted_examples = []
    rejected_reasons = Counter()
    for row_index, row in enumerate(dataset):
        try:
            converted_examples.append(convert_row(row, row_index))
        except ValueError as error:
            rejected_reasons[str(error)] += 1
    train_examples, test_examples = split_converted_examples(
        converted_examples, test_count=test_count, seed=seed
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    train_path = output_directory / "toolace_train.jsonl"
    test_path = output_directory / "toolace_test.jsonl"
    write_jsonl(train_path, train_examples)
    write_jsonl(test_path, test_examples)
    summary = {
        "seed": seed,
        "test_count": test_count,
        "converted": len(converted_examples),
        "rejected": dict(rejected_reasons),
        "train_count": len(train_examples),
        "test_count_actual": len(test_examples),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    (output_directory / "toolace_split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return train_examples, test_examples, summary


def build_larger_bfcl_ood(raw_directory, notes_path, selected_count, seed, output_path, prefer_unused_from_notes=True):
    prepare_ood = load_prepare_ood_module()
    notes, _selected_simple_unused, _selected_multiple_unused = prepare_ood.load_conversion_notes(notes_path)
    raw_directory = Path(raw_directory)
    simple_questions = prepare_ood.index_by_id(
        prepare_ood.read_jsonl_objects(raw_directory / "BFCL_v3_simple.json")
    )
    multiple_questions = prepare_ood.index_by_id(
        prepare_ood.read_jsonl_objects(raw_directory / "BFCL_v3_multiple.json")
    )
    simple_answers = prepare_ood.index_by_id(
        prepare_ood.read_jsonl_objects(raw_directory / "possible_answer" / "BFCL_v3_simple.json")
    )
    multiple_answers = prepare_ood.index_by_id(
        prepare_ood.read_jsonl_objects(raw_directory / "possible_answer" / "BFCL_v3_multiple.json")
    )
    all_simple_ids = sorted(simple_questions)
    all_multiple_ids = sorted(multiple_questions)
    # Aim ~2:1 simple:multiple like the original 100/100 but scaled; fall back to available.
    target_simple = min(len(all_simple_ids), int(round(selected_count * 2 / 3)))
    target_multiple = selected_count - target_simple
    if target_multiple > len(all_multiple_ids):
        target_multiple = len(all_multiple_ids)
        target_simple = selected_count - target_multiple
    exclude_simple = set(notes.get("selected_simple_ids", [])) if prefer_unused_from_notes else set()
    exclude_multiple = set(notes.get("selected_multiple_ids", [])) if prefer_unused_from_notes else set()
    selected_simple = select_bfcl_ids(all_simple_ids, target_simple, seed, exclude_simple)
    selected_multiple = select_bfcl_ids(all_multiple_ids, target_multiple, seed + 1, exclude_multiple)
    examples = []
    skipped = []
    for bfcl_id in selected_simple:
        try:
            examples.append(
                prepare_ood.convert_bfcl_example("simple", simple_questions[bfcl_id], simple_answers[bfcl_id])
            )
        except Exception as error:  # noqa: BLE001 - record and continue for build report
            skipped.append({"id": bfcl_id, "error": str(error)})
    for bfcl_id in selected_multiple:
        try:
            examples.append(
                prepare_ood.convert_bfcl_example(
                    "multiple", multiple_questions[bfcl_id], multiple_answers[bfcl_id]
                )
            )
        except Exception as error:  # noqa: BLE001
            skipped.append({"id": bfcl_id, "error": str(error)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_ood.write_jsonl(output_path, examples)
    report = {
        "n": len(examples),
        "simple": len(selected_simple),
        "multiple": len(selected_multiple),
        "selected_simple_ids": selected_simple,
        "selected_multiple_ids": selected_multiple,
        "skipped": skipped,
        "seed": seed,
        "prefer_unused_from_notes": prefer_unused_from_notes,
        "overlap_with_original_simple": len(set(selected_simple) & exclude_simple),
        "overlap_with_original_multiple": len(set(selected_multiple) & exclude_multiple),
        "output_path": str(output_path),
    }
    notes_out = output_path.with_name(output_path.stem + "_notes.json")
    notes_out.write_text(json.dumps(report, indent=2) + "\n")
    return examples, report


def load_and_convert_hermes_sources(max_rows_per_file=None):
    data_files = [
        "func-calling-singleturn.json",
        "glaive-function-calling-5k.json",
    ]
    converted = []
    rejected = Counter()
    for data_file in data_files:
        dataset = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            data_files=data_file,
            split="train",
        )
        for row_index, row in enumerate(dataset):
            if max_rows_per_file is not None and row_index >= max_rows_per_file:
                break
            example_id = f"hermes-{data_file.split('.')[0]}-{row_index:05d}"
            try:
                converted.append(convert_hermes_row(row, example_id))
            except (ValueError, json.JSONDecodeError) as error:
                rejected[str(error)] += 1
    return converted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolace-test-count", type=int, default=500)
    parser.add_argument("--toolace-seed", type=int, default=7)
    parser.add_argument("--ood-count", type=int, default=500)
    parser.add_argument("--ood-seed", type=int, default=7)
    parser.add_argument("--external-sample-count", type=int, default=3000)
    parser.add_argument("--mix-seed", type=int, default=7)
    parser.add_argument("--mixed-output-directory", type=Path, default=ROOT / "data" / "mixed_data")
    parser.add_argument("--larger-eval-directory", type=Path, default=ROOT / "data" / "larger_eval")
    args = parser.parse_args()

    print("Building ToolACE split...", flush=True)
    toolace_train, toolace_test, toolace_summary = build_toolace_split(
        args.toolace_test_count, args.toolace_seed, args.larger_eval_directory
    )
    print(json.dumps(toolace_summary, indent=2), flush=True)

    print("Building larger BFCL OOD...", flush=True)
    ood_examples, ood_report = build_larger_bfcl_ood(
        raw_directory=ROOT / "data" / "ood_raw",
        notes_path=ROOT / "data" / "ood_conversion_notes.json",
        selected_count=args.ood_count,
        seed=args.ood_seed,
        output_path=args.larger_eval_directory / "ood_test_500.jsonl",
        prefer_unused_from_notes=True,
    )
    print(json.dumps({k: ood_report[k] for k in ood_report if k.endswith("_ids") is False}, indent=2), flush=True)

    print("Converting Hermes/Glaive external tool-calling data...", flush=True)
    external_rows, rejected = load_and_convert_hermes_sources()
    print({"external_converted": len(external_rows), "rejected": dict(rejected)}, flush=True)

    # Block leakage against BOTH new ToolACE test and larger BFCL OOD (and original 200 tests).
    from eval_toolcalls import read_jsonl

    original_toolace_test = read_jsonl(ROOT / "data" / "test.jsonl")
    original_ood = read_jsonl(ROOT / "data" / "ood_test.jsonl")
    blocked = set()
    for collection in (toolace_test, ood_examples, original_toolace_test, original_ood):
        blocked |= blocked_keys_from_examples(collection)
    filtered_external, n_rejected_leak = filter_examples_not_in_blocked_keys(external_rows, blocked)
    print({"external_after_leak_filter": len(filtered_external), "leak_rejected": n_rejected_leak}, flush=True)

    mixed_rows, mix_report = build_mixed_training_rows(
        toolace_train,
        filtered_external,
        external_sample_count=min(args.external_sample_count, len(filtered_external)),
        seed=args.mix_seed,
    )
    args.mixed_output_directory.mkdir(parents=True, exist_ok=True)
    control_path = args.mixed_output_directory / "control_toolace_train.jsonl"
    mixed_path = args.mixed_output_directory / "mixed_train.jsonl"
    external_pool_path = args.mixed_output_directory / "external_pool.jsonl"
    write_jsonl(control_path, toolace_train)
    write_jsonl(mixed_path, mixed_rows)
    write_jsonl(external_pool_path, filtered_external)
    mix_report.update({
        "leak_rejected": n_rejected_leak,
        "external_rejected_reasons": dict(rejected),
        "control_path": str(control_path),
        "mixed_path": str(mixed_path),
        "blocked_key_count": len(blocked),
        "note": (
            "Mixed = all new-split ToolACE train + sampled Hermes/Glaive after "
            "dedupe against ToolACE/BFCL test sets (prompt+tools hash)."
        ),
    })
    (args.mixed_output_directory / "build_summary.json").write_text(
        json.dumps(mix_report, indent=2) + "\n"
    )
    print(json.dumps(mix_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
