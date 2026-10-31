"""Training-CLI flag behavior is checkable without Unsloth or CUDA."""

import subprocess
import sys
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))


class RecordingHubApi:
    def __init__(self):
        self.created_repos = []

    def create_repo(self, output_repo, repo_type, private, exist_ok):
        self.created_repos.append(
            {"repo": output_repo, "repo_type": repo_type, "private": private}
        )


class RecordingModel:
    def __init__(self):
        self.uploads = []

    def push_to_hub(self, output_repo, tokenizer, private):
        self.uploads.append({"repo": output_repo, "private": private})


def test_removed_private_repo_flag_is_rejected_by_the_cli():
    completed = subprocess.run(
        [
            sys.executable, "-S", str(PROJECT_DIRECTORY / "scripts/train_toolace_sft.py"),
            "--train-jsonl", "unused.jsonl", "--private-repo",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--private-repo" in completed.stderr


def test_public_repo_flag_defaults_to_private_and_can_be_enabled(monkeypatch):
    import train_toolace_sft

    monkeypatch.setattr(sys, "argv", ["train_toolace_sft.py", "--train-jsonl", "unused.jsonl"])
    assert train_toolace_sft.parse_args().public_repo is False

    monkeypatch.setattr(
        sys, "argv",
        ["train_toolace_sft.py", "--train-jsonl", "unused.jsonl", "--public-repo"],
    )
    assert train_toolace_sft.parse_args().public_repo is True


def test_push_adapter_passes_the_requested_visibility_to_both_upload_calls():
    import train_toolace_sft

    for public_repo in (False, True):
        hub_api = RecordingHubApi()
        model = RecordingModel()
        train_toolace_sft.push_adapter_to_hub(
            model, tokenizer=None, output_repo="acct/adapter",
            public_repo=public_repo, hub_api=hub_api,
        )
        expected_private = not public_repo
        assert hub_api.created_repos == [
            {"repo": "acct/adapter", "repo_type": "model", "private": expected_private}
        ]
        assert model.uploads == [{"repo": "acct/adapter", "private": expected_private}]
