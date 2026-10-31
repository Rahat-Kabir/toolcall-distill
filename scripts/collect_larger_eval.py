#!/usr/bin/env python3
"""Extract prediction markers from a larger-eval job log and score + bootstrap."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bootstrap_metrics import paired_bootstrap_rate_ci
from eval_toolcalls import read_jsonl, score_predictions


MARKER_PATTERN = re.compile(
    "BEGIN_(PREDS_[A-Z0-9_]+_B64_GZ)" + "\n" + "(.*?)" + "\n" + r"END_\1",
    re.DOTALL,
)
SUFFIX_MARKER_PATTERN = re.compile(
    r"(PREDS_[A-Z0-9_]+_B64_GZ)_BEGIN\n(.*?)\n\1_END", re.DOTALL,
)


def flags_for_metric(scored_summary, metric_key):
    return [1 if example[metric_key] else 0 for example in scored_summary["examples"]]


def absolute_rate_ci(flags, seed):
    return paired_bootstrap_rate_ci(flags, flags, n_bootstrap=5000, seed=seed)["base_ci95"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--toolace-gold", type=Path, required=True)
    parser.add_argument("--ood-gold", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--labels", default="base,control,mixed")
    args = parser.parse_args()

    log_text = args.log_path.read_text(encoding="utf-8", errors="replace")
    pred_dir = args.output_directory / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    # Accept historical job logs and the shared runner's MARKER_BEGIN/END format.
    blocks = MARKER_PATTERN.findall(log_text) + SUFFIX_MARKER_PATTERN.findall(log_text)
    for marker, body in blocks:
        payload = gzip.decompress(base64.b64decode(body.strip())).decode().replace("\r\n", "\n")
        path = pred_dir / (marker.lower().replace("_b64_gz", "") + ".jsonl")
        if not payload.endswith(chr(10)):
            payload += chr(10)
        path.write_text(payload, encoding="utf-8", newline="\n")
        written[marker] = path
    if not written:
        raise SystemExit("No PREDS_* markers found in log")

    toolace_gold = read_jsonl(args.toolace_gold)
    ood_gold = read_jsonl(args.ood_gold)
    models = []
    per_model = {}
    for label in [part.strip() for part in args.labels.split(",") if part.strip()]:
        toolace_key = f"PREDS_{label.upper()}_TOOLACE_B64_GZ"
        ood_key = f"PREDS_{label.upper()}_OOD_B64_GZ"
        if toolace_key not in written or ood_key not in written:
            raise SystemExit(f"Missing preds for label={label}: have {sorted(written)}")
        toolace_scores = score_predictions(read_jsonl(written[toolace_key]), toolace_gold)
        ood_scores = score_predictions(read_jsonl(written[ood_key]), ood_gold)
        per_model[label] = {"toolace": toolace_scores, "ood": ood_scores}
        toolace_flags = flags_for_metric(toolace_scores, "are_arguments_correct")
        ood_flags = flags_for_metric(ood_scores, "are_arguments_correct")
        models.append({
            "label": label,
            "toolace_args": toolace_scores["rates"]["are_arguments_correct"],
            "toolace_name": toolace_scores["rates"]["is_correct_tool_name"],
            "ood_args": ood_scores["rates"]["are_arguments_correct"],
            "ood_name": ood_scores["rates"]["is_correct_tool_name"],
            "toolace_counts": toolace_scores["counts"],
            "ood_counts": ood_scores["counts"],
            "toolace_args_ci95": absolute_rate_ci(toolace_flags, seed=10 + len(models)),
            "ood_args_ci95": absolute_rate_ci(ood_flags, seed=20 + len(models)),
        })

    deltas = {}
    if "base" in per_model:
        base_toolace = flags_for_metric(per_model["base"]["toolace"], "are_arguments_correct")
        base_ood = flags_for_metric(per_model["base"]["ood"], "are_arguments_correct")
        for label, bundle in per_model.items():
            if label == "base":
                continue
            deltas[label] = {
                "toolace_vs_base": paired_bootstrap_rate_ci(
                    base_toolace,
                    flags_for_metric(bundle["toolace"], "are_arguments_correct"),
                    n_bootstrap=5000,
                    seed=42,
                ),
                "ood_vs_base": paired_bootstrap_rate_ci(
                    base_ood,
                    flags_for_metric(bundle["ood"], "are_arguments_correct"),
                    n_bootstrap=5000,
                    seed=43,
                ),
            }
        if "control" in per_model and "mixed" in per_model:
            deltas["mixed_vs_control"] = {
                "toolace": paired_bootstrap_rate_ci(
                    flags_for_metric(per_model["control"]["toolace"], "are_arguments_correct"),
                    flags_for_metric(per_model["mixed"]["toolace"], "are_arguments_correct"),
                    n_bootstrap=5000,
                    seed=44,
                ),
                "ood": paired_bootstrap_rate_ci(
                    flags_for_metric(per_model["control"]["ood"], "are_arguments_correct"),
                    flags_for_metric(per_model["mixed"]["ood"], "are_arguments_correct"),
                    n_bootstrap=5000,
                    seed=45,
                ),
            }

    summary = {
        "models": models,
        "deltas": deltas,
        "n_toolace": len(toolace_gold),
        "n_ood": len(ood_gold),
        "prediction_files": {key: str(path) for key, path in written.items()},
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "summary.json").write_text(json.dumps(summary, indent=2) + chr(10))
    print(json.dumps({"models": models, "delta_keys": list(deltas)}, indent=2))


if __name__ == "__main__":
    main()
