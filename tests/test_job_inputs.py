"""File-based runner checks must work without model packages or network access."""

from pathlib import Path
import subprocess
import sys

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def test_runner_checks_explicit_input_files_without_loading_models(tmp_path):
    input_path = tmp_path / "test.jsonl"
    input_path.write_text(
        '{"id": "sample", "tools": [], "prompt_messages": [{"role": "user", "content": "hello"}]}\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts" / "run_eval_job.py"),
         "--model", "unused-model", "--input", str(input_path),
         "--output", str(tmp_path / "predictions.jsonl"), "--check-inputs"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "INPUTS_OK" in completed.stdout
    assert not (tmp_path / "predictions.jsonl").exists()


def test_training_rejects_missing_data_before_loading_gpu_packages(tmp_path):
    missing_path = tmp_path / "missing.jsonl"
    completed = subprocess.run(
        [sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts" / "train_toolace_sft.py"),
         "--train-jsonl", str(missing_path)],
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "Training file not found" in completed.stderr
    assert not missing_path.exists()
