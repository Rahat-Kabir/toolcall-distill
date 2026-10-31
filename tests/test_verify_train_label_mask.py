"""The verifier must mask exactly what training masks, including overflow."""

import json
import sys
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY / "scripts"))


class CharacterTokenizer:
    """One token per character, mirroring an offset-mapping tokenizer call."""

    bos_token = None

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def rendered_example_text():
    return 'tools and user prompt <|im_start|>assistant\n[weather(city="Dhaka")]<|im_end|>'


def test_truncation_matches_training_by_dropping_the_left_side():
    import verify_train_label_mask

    tokenizer = CharacterTokenizer()
    maximum_sequence_length = 30
    input_ids, labels, _spans, rendered_text = verify_train_label_mask.build_assistant_only_labels(
        tokenizer, rendered_example_text(), maximum_sequence_length,
    )
    assert tokenizer.calls[0]["truncation"] is False
    overflow = len(rendered_text) - maximum_sequence_length
    assert input_ids == list(range(overflow, len(rendered_text)))
    assert len(labels) == maximum_sequence_length
    supervised = [token for token, label in zip(input_ids, labels) if label != -100]
    content_start = rendered_text.index("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assert supervised
    assert all(token >= content_start for token in supervised)


def test_write_label_mask_report_creates_missing_parent_directories(tmp_path):
    import verify_train_label_mask

    output_path = tmp_path / "results" / "training" / "label_mask_check.json"
    reports = [{"id": "row-1", "supervised_tokens": 3}]
    verify_train_label_mask.write_label_mask_report(reports, output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == reports
