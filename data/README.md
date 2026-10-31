# Local datasets

Downloaded source data and generated JSONL files are ignored by Git.
The repository includes only this guide and ood_conversion_notes.json,
which pins the original 200 BFCL example IDs and documents conversion rules.

From the repository root:

```sh
python prepare_data.py
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='gorilla-llm/Berkeley-Function-Calling-Leaderboard', repo_type='dataset', local_dir='data/ood_raw', allow_patterns=['BFCL_v3_simple.json', 'BFCL_v3_multiple.json', 'possible_answer/BFCL_v3_simple.json', 'possible_answer/BFCL_v3_multiple.json'])"
python scripts/prepare_ood_bfcl.py
python -m pytest -q -m integration
```

These commands download public datasets and write local files. They do not
launch hosted inference or training. Review the upstream dataset terms in
[THIRD_PARTY.md](../THIRD_PARTY.md).

For the larger experiment, prepare the original files above, then run:

```sh
python scripts/build_mixed_and_larger_eval.py
```

That writes ToolACE-500 and BFCL-500 evaluation sets under data/larger_eval/
and control/mixed training files under data/mixed_data/. Historical source
commit revisions were not recorded, so downloading today's upstream revisions
is not a guarantee of byte-identical historical data.
