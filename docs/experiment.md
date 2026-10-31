# Tool-calling LoRA experiment

## Question and setup

Can LoRA SFT improve LiquidAI/LFM2.5-1.2B-Instruct tool calling, and does adding
external function-calling data help on a separate evaluation distribution?

The main source is ToolACE. Training uses Unsloth LoRA with rank/alpha 16,
learning rate 2e-4, effective batch size 8, and maximum sequence length 2048.
Packing is off. Supervision ends after the first assistant turn; later dialogue
turns are not trained. Long sequences are left-truncated. Evaluation is greedy
with at most 256 new tokens, using native chat templates and separate tool schemas.

## Data

The original ToolACE conversion accepted 10,782 of 11,300 rows; 518 lacked the
supported JSON-tools marker. Removing 36 duplicate first prompts left 10,546
training rows and 200 held-out first-call examples (seed 42). Deduplication hashes
the tools and whitespace-normalized first user text. Empty-tool conversations
remain in training. Of 10,546 training conversations, 786 have later turns that
are excluded from supervision; six exceed 2048 tokens after first-turn truncation.

The larger experiment re-splits the converted pool (seed 7): 10,246 train / 500
test. The mixed set adds 3,000 converted Hermes/Glaive examples, making 13,246
rows. Prompt+tools overlap checks found no held-out overlap in either training
set. These checks do not rule out paraphrases or upstream/pretraining contamination.
BFCL-500 comprises 333 simple and 167 multiple-function-selection examples;
'multiple' here still requires one call. It overlaps the earlier BFCL-200 set.

## Results

### Measurement note: extra BOS token

The sampling, BFCL, external-model (Qwen/Llama), and 500-row evaluation runs
used a render-then-tokenize path that added one extra BOS token to every
prompt. The original ToolACE-200 baseline and post-training runs did not.
Base and trained models received the same prompt in each comparison, so the
comparisons are paired, but absolute numbers in those runs may differ slightly
from a single-BOS run. `scripts/run_eval_job.py` uses native tokenization and
no longer adds the extra token.

### Original ToolACE-200

| Metric | Base | Full ToolACE LoRA |
| --- | ---: | ---: |
| Parseable | 98.5% | 98.5% |
| Ordered names/count | 80.0% | 89.0% |
| Strict arguments | 63.0% | 71.0% |

Argument improvement: +8.0 percentage points, paired bootstrap 95% CI
[+3.0, +13.0]. Single-call accuracy rose 71.9% to 78.9%; multi-call 51.2% to
60.5%. Sampling at temperature 0.7/top-p 0.9 across three generation seeds gave
61.7% +/- 1.0% base and 68.2% +/- 1.6% LoRA (sample standard deviations).

### Larger evaluation (500 examples per dataset)

| Model | ToolACE strict args | BFCL strict selected-answer args | BFCL alternative-aware args |
| --- | ---: | ---: | ---: |
| Base | 62.8% | 59.6% | 87.8% |
| ToolACE-only control | 72.8% | 54.2% | 90.4% |
| Mixed Hermes/Glaive | 72.6% | 55.0% | 90.0% |

ToolACE paired deltas vs base: control +10.0 pp [+6.6, +13.6], mixed +9.8 pp
[+6.4, +13.2]. Alternative-aware BFCL deltas vs base: control +2.6 pp
[+0.2, +5.0], mixed +2.2 pp [0.0, +4.2]. Mixed vs control on BFCL: -0.4 pp
[-1.6, +0.8]. Intervals use 5,000 paired bootstrap resamples over examples.

**ToolACE improvement is supported. Mixing has no demonstrated advantage over
control in this run. The strict BFCL drop alone does not establish worse generalization.**

The alternative-aware BFCL gains are small and not firmly established. The mixed
interval [0.0, +4.2] touches zero, so the mixed BFCL gain is not shown. The control
interval [+0.2, +5.0] is only just above zero, so treat that gain as tentative.

### Why the BFCL interpretation changed

The original converter selects one nonempty answer per argument. BFCL may
accept both 'mi' and 'miles', or allow omission of an optional default. Comparing
against one selected answer rejects such valid alternatives. The separate
alternative-aware scorer uses the raw source alternatives; an empty-string
source marker permits omission, not a supplied empty value.

This is a local metric with the project's argument normalization, not official
BFCL AST or execution evaluation. Original predictions and strict results were
not changed during re-scoring.

### Original BFCL-200 and model comparisons

| Model | ToolACE-200 strict args | BFCL-200 strict args | BFCL-200 alternative-aware args |
| --- | ---: | ---: | ---: |
| Base LFM2.5-1.2B | 63.0% | 58.0% | 89.0% |
| ToolACE LoRA, 1k prefix | 67.5% | 50.5% | 90.0% |
| ToolACE LoRA, 3k prefix | 70.0% | 50.5% | 90.5% |
| Full ToolACE LoRA | 71.0% | 52.5% | 89.5% |
| Qwen3-4B-Instruct-2507 | 65.0% | 56.5% | 90.5% |
| Llama-3.1-8B-Instruct | 48.5% | 58.5% | 77.0% |

Full LoRA vs base BFCL alternative-aware delta: +0.5 pp [-2.5, +4.0].
The Llama strict results include a parser correction for semicolon-separated
JSON calls; no new generation was required. Models differ in size, pretraining,
and native format, so this is not a controlled model ranking.

## Limits and reproducibility

- Full original training ran 1,200 steps, approximately 0.91 epoch. The 1k/3k
  ablations ran one epoch on ordered prefixes, not random subsets.
- Control and mixed training each ran 1,200 steps, so the larger mixed set sees
  fewer epochs. Only one training run per configuration was evaluated.
- Bootstrap intervals describe sampling variation over these examples, not
  variation across training seeds or all tool-calling workloads.
- Evaluation covers first-call turns with tools, not multi-turn task completion,
  refusal, or the decision to avoid tools.
- Model/dataset source revisions and a full historical environment lock were
  not recorded. Current scripts and selection IDs support reconstruction, not
  a guarantee of bit-identical historical runs.
- Historical adapters and raw outputs are not distributed with this code.
  GPU training/inference after the public cleanup has not been rerun.
- Total cloud-compute cost was estimated at approximately $4.27 from recorded
  job durations and rates, including failed attempts; this is not a billing receipt.

[larger_eval_summary.json](larger_eval_summary.json) preserves compact aggregate results for the 500-example evaluation.
The committed charts under `docs/images/` use the alternative-aware BFCL scores.
See [README.md](../README.md) for runnable commands and
[THIRD_PARTY.md](../THIRD_PARTY.md) for attribution and source terms.

## Evaluation implementation

The seven experiment-specific evaluation entrypoints have been replaced by
`scripts/run_eval_job.py`. Model, adapter, dataset, decoding settings, and output
path are explicit CLI arguments. Native templates tokenize the prompt once;
completed predictions are flushed to disk before moving to the next example.
Scoring and BFCL re-scoring run locally against those saved files. Optional log
blocks preserve compatibility with result collection, including partial runs.

This refactor preserves the published aggregate results; it does not replace
historical predictions or claim a new model-quality measurement. See the
[README evaluation recipes](../README.md#evaluation-recipes) for base, adapter,
OOD, sampling, external-model, and larger-evaluation commands.
