# Alignment by Stereotyping

Code and data for the paper **"Alignment by Stereotyping: Demographic Profiles Raise LLM Accuracy Through Individual Homogenization"**.

## Overview

We study a paradox in LLM sociodemographic adaptation: providing a demographic profile improves group-level value alignment accuracy, yet simultaneously causes models to homogenize individuals toward their group centroid — erasing the very individual variation the profile was meant to serve.

We evaluate seven LLMs across three conditions:

| Condition | Description |
|---|---|
| **None** | No user context; model answers without demographic information |
| **Profile** | Explicit demographic attributes prepended to each query |
| **Dialogue** | Multi-turn conversation history replacing the profile label |

Key findings:
1. Profile conditioning raises group-level alignment accuracy but homogenizes ~80–93% of individuals toward their demographic group centroid
2. Scaling amplifies this trade-off within model families — larger models homogenize more
3. Dialogue history partially reverses homogenization, suggesting richer individual context reduces demographic stereotyping

---

## Repository Structure

```
├── datasets/
│   ├── wvs_benchmarks/               # Seed demographic profiles and WVS questions
│   ├── wvs_generated_dialogues/      # Synthetic dialogues (career & investment advice)
│   └── prism_validation/             # PRISM real-conversation validation data (CC BY 4.0)
├── llm_behavior_adaptation/
│   ├── dialogue_dataset_creation/    # Dialogue generation pipeline
│   └── value_measurement/            # Values prediction, metrics, and configs
│       ├── prompts/                  # Prompt templates (no-profile, anchored)
│       └── values_prediction_configs/  # Per-model YAML run configs
├── scripts/
│   ├── permutation_test_homogenization.py  # Permutation test for homogenization rate
│   ├── analyze_anchored_gpt51.py           # Anchor experiment analysis (GPT-5.1)
│   ├── compute_ba_none_vaa.py              # BA_none VAA computation across models
│   ├── generate_emnlp_figures.py           # Figure generation (all paper figures)
│   ├── convert_prism_to_pipeline.py        # PRISM → pipeline format conversion
│   ├── run_permutation_tests_all.sh        # Run permutation tests for all models
│   └── human_validation/                   # PRISM validation study materials
├── wvs_values_results/
│   ├── <ModelName>/
│   │   ├── BA_none_values_results/         # None condition outputs
│   │   ├── BA_user_values_results/         # Profile condition outputs (LFS)
│   │   ├── BA_anchored_values_results/     # Anchor experiment outputs (LFS)
│   │   ├── career/ & investment/           # Dialogue condition outputs (LFS)
│   │   ├── experiments_results.json        # Pre-computed alignment metrics
│   │   └── permutation_test_results.json   # Permutation test z-scores and p-values
│   └── ba_none_vaa_comparison.json         # None-condition VAA across all models
├── requirements.txt
└── setup.py
```

> **Large result files** (BA_user, BA_dialogue, BA_anchored `total_1000.jsonl` and GPT-5.1 `experiments_results.json`) are tracked with Git LFS.

---

## Environment Setup

```bash
conda create -n llm_alignment python=3.10 -y
conda activate llm_alignment
pip install -e .
```

Set your API key (scripts read from the environment):

```bash
export OPENAI_API_KEY=your_key_here   # or OPENROUTER_API_KEY for other models
```

---

## Reproducing Results

### Step 1: Run model value predictions

Query a model for WVS value predictions under a given condition using a YAML config:

```bash
python -m llm_behavior_adaptation.value_measurement.wvs_values_prediction \
    --config llm_behavior_adaptation/value_measurement/values_prediction_configs/<model>/<config>.yaml
```

Configs are provided for all seven models across all conditions. Each YAML specifies the model identifier, input dialogue file, run mode (`profiles` | `dialogue` | `no_profile`), and output path.

### Step 2: Compute alignment metrics

```bash
python -m llm_behavior_adaptation.value_measurement.wvs_values_comparison \
    --user-profile-dataset datasets/wvs_benchmarks/sampled_demographic_features.csv \
    --user-value-dataset datasets/wvs_benchmarks/sampled_values_df.csv \
    --ba-user-results <path/to/BA_user_values_results/total_1000.jsonl> \
    --ba-dialogue-career-results <path/to/career/BA_dialogue_values_results/total_1000.jsonl> \
    --ba-dialogue-investment-results <path/to/investment/BA_dialogue_values_results/total_1000.jsonl> \
    --results-output-path <path/to/experiments_results.json>
```

### Step 3: Run permutation tests for homogenization

```bash
# All models at once
bash scripts/run_permutation_tests_all.sh

# Single model
python scripts/permutation_test_homogenization.py \
    --model GPT-5.1 \
    --results-dir wvs_values_results/gpt-5.1/
```

Results are written to `wvs_values_results/<model>/permutation_test_results.json`.

### Step 4: Generate paper figures

```bash
python scripts/generate_emnlp_figures.py
```

Outputs all figures to `figures/`.

---

## Pre-computed Results

The `wvs_values_results/` directory contains pre-computed outputs for all seven models:

| Model | None VAA | Homog. Rate | Permutation Results |
|---|---|---|---|
| GPT-5.1 | 0.584 | 80.8% | `gpt-5.1/permutation_test_results.json` |
| QwQ-32B | 0.400 | 66.9% | `QwQ-32B/permutation_test_results.json` |
| Qwen2.5-72B | 0.558 | 83.2% | `Qwen2.5-72B-Instruct/permutation_test_results.json` |
| Llama-3.1-70B | 0.336 | 55.3% | `Llama-3.1-70B-Instruct/permutation_test_results.json` |
| DeepSeek-V3 | 0.547 | 79.8% | `DeepSeek-V3/permutation_test_results.json` |
| Qwen2.5-7B | 0.599 | 69.4% | `Qwen2.5-7B-Instruct/permutation_test_results.json` |
| Llama-3.1-8B | 0.510 | 28.0% | `Llama-3.1-8B-Instruct/permutation_test_results.json` |

---

## PRISM Validation

Real-conversation validation uses 80 human–chatbot conversations from the [PRISM dataset](https://arxiv.org/abs/2404.16019) (Kirk et al., 2024, CC BY 4.0).

To convert PRISM to pipeline format:

```bash
python scripts/convert_prism_to_pipeline.py
# outputs: datasets/prism_validation/prism_dialogues.jsonl
#          datasets/prism_validation/prism_demographics.csv
```

Validation configs are in `values_prediction_configs/prism_validation/`.
