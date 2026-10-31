# Agent guidance

## Project

This repository prepares tool-calling training data, evaluates model outputs,
and provides LoRA training runners for LiquidAI/LFM2.5-1.2B-Instruct.
See README.md for setup and docs/experiment.md for methods and results.
No external agent skills or personal configuration are required.

## Working rules

- Keep changes focused. Read the surrounding code before editing.
- Use clear, full-word snake_case names and pure functions for conversion/scoring.
- Validate files, CLI arguments, and external payloads at their boundaries.
- Test observable behavior through public functions or the CLI. Mock only external edges.
- For regressions, observe a failing test before fixing the behavior.
- Run relevant tests and describe exactly what was verified.
- Update documentation when behavior or setup changes.
- Do not expose credentials or include downloaded data, model weights, raw job logs,
  or personal working notes in source changes.
- Paid jobs, remote uploads, deployments, commits, and pushes require explicit
  authorization. Local tests and public downloads do not require paid services.

## Commands

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
# Requires prepared BFCL files; see README.md:
python -m pytest -q -m integration
```

The default suite is offline and requires only pytest. Model generation and
training have separate dependencies and are not part of the default tests.

## Code map

- prepare_data.py: ToolACE conversion, prompt deduplication, deterministic splits.
- mixed_data_prep.py: external-data conversion and held-out overlap filtering.
- eval_toolcalls.py: safe parsing, strict scoring, local generation CLI.
- bfcl_scoring.py: source-alternative scoring alongside the strict metric.
- bootstrap_metrics.py: paired bootstrap confidence intervals.
- scripts/run_eval_job.py: shared model/adapter generation CLI; scoring stays local.
- scripts/make_charts.py: chart selection and rendering from saved summaries.
- scripts/: data preparation, training, input validation, and result collection.
- tests/: unit/CLI tests; integration marker for downloaded-data checks.

## Invariants

- Keep train/test prompts disjoint under the project's prompt hash.
- Test prompts end before the target assistant response.
- LFM tool calls use <|tool_call_start|>[Name(...)]<|tool_call_end|>.
- Tool-name scoring requires the entire ordered call sequence and count to match.
- Strict argument scoring uses eval_toolcalls.py normalization.
- BFCL source alternatives are a separate local metric, not official BFCL AST scores.
- Runners read supplied files and use the current scorer; do not embed datasets or
  copies of source modules in Python strings.
