# Third-party sources

This repository contains conversion/evaluation code and aggregate experiment
results. It does not distribute downloaded datasets, model weights, adapters,
or the upstream training-script snapshot. Source terms were checked on 2026-10-31.

| Source | Use | Upstream terms |
| --- | --- | --- |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Main training and held-out evaluation data | Dataset card declares Apache-2.0. |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Mixed-training pool, including its Glaive-derived file | Dataset card declares Apache-2.0; consult the upstream card for component provenance. |
| [Berkeley Function Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) | BFCL v3 simple/multiple evaluation inputs and answer alternatives | Dataset card declares Apache-2.0. |
| [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) | Base model | Model card declares the Liquid AI LFM Open License v1.0; see the license notes below. |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | External-model comparison evaluation | Gated on Hugging Face: access requires accepting the "LLAMA 3.1 COMMUNITY LICENSE AGREEMENT" on the model page; see the license notes below. |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | External-model comparison evaluation | Repository declares Apache-2.0 and is not gated. |
| [Unsloth HF Jobs scripts](https://huggingface.co/datasets/unsloth/jobs/blob/main/sft-lfm2.5.py) | Starting point for scripts/train_toolace_sft.py | Adapted training configuration and runner structure; attribution retained in the file. No explicit license was confirmed in the inspected upstream script/card. The file is excluded from the project MIT license and included for reproducibility; resolve upstream terms before redistributing it separately. |

ToolACE paper: [ToolACE: Winning the Points of LLM Function Calling](https://arxiv.org/abs/2409.00920).
BFCL project: [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).

## Model license notes

Quoted fragments come from the official license texts linked above (checked
2026-10-31), not from memory.

**LFM Open License v1.0** (LiquidAI/LFM2.5-1.2B-Instruct):

- Commercial-use threshold: the license defines "Threshold" as "annual revenue
  of 10 million United States dollars ($10,000,000) or more"; rights for
  Commercial Use are conditioned on not exceeding that Threshold, and
  Commercial Use by a Legal Entity that "exceeds the Threshold is not
  licensed" under the agreement.
- Redistribution: "You must give any other recipients of the Work or
  Derivative Works a copy of this License"; "You must cause any modified
  files to carry prominent notices stating that You changed the files"; and
  Derivative Works in source form must retain the copyright, patent,
  trademark, and attribution notices from the Work.
- NOTICE: "If the Work includes a 'NOTICE' text file as part of its
  distribution, then any Derivative Works that You distribute" must include
  a readable copy of the attribution notices contained within that file.

**Llama 3.1 Community License Agreement** (meta-llama/Llama-3.1-8B-Instruct,
gated): the commercial trigger is "greater than 700 million monthly active
users in the preceding calendar month", past which "you must request a
license from Meta". Redistribution of Llama Materials requires that you
"(A) provide a copy of this Agreement with any such Llama Materials; and
(B) prominently display 'Built with Llama'" on related documentation.

**Apache-2.0** (Qwen/Qwen3-4B-Instruct-2507): declared on the repository;
not gated.

This repository distributes no model weights; the terms above govern use of
the upstream models, not this project's code.

## Selection metadata

The selection metadata under data/ records example IDs and conversion rules,
not a copy of the evaluation questions. Project code is MIT (see LICENSE),
except scripts/train_toolace_sft.py as noted above. No claim is made that a
project license overrides upstream model, dataset, or adapted-code terms.
