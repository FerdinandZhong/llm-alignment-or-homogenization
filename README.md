# Alignment by Stereotyping

Code and data for the paper **"Alignment by Stereotyping: How LLMs Sacrifice Individual Distinctiveness for Cultural Adaptation"**.

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
4. A dialogue-length ablation (K=1/3/5 turns) across three models reveals model-dependent de-homogenization profiles: a threshold (GPT-5.1), a gradient (DeepSeek-V3), and persistence even at five turns (Qwen2.5-72B)

---

## Repository Structure

```
├── datasets/
│   ├── wvs_benchmarks/               # WVS question metadata (respondent-level WVS data not redistributed — see License)
│   ├── wvs_generated_dialogues/      # Synthetic dialogues (career & investment advice)
│   └── prism_validation/             # PRISM real-conversation validation data (CC BY 4.0)
├── llm_behavior_adaptation/
│   ├── dialogue_dataset_creation/    # Dialogue generation pipeline
│   │   ├── prompts/                  # Dialogue generation prompt templates
│   │   └── dialogues_validation/     # Dialogue quality validation (LLM + human)
│   └── value_measurement/            # Values prediction, metrics, and configs
│       ├── prompts/                  # Prompt templates (no-profile, anchored)
│       └── values_prediction_configs/  # Per-model YAML run configs
├── scripts/
│   ├── permutation_test_homogenization.py  # Permutation test for homogenization rate
│   ├── dialogue_permtest_per_domain.py     # Per-domain dialogue permutation tests
│   ├── prism_density_correlation.py        # PRISM signal-density correlation analysis
│   ├── analyze_anchored_gpt51.py           # Anchor experiment analysis (GPT-5.1)
│   ├── compute_ba_none_vaa.py              # BA_none VAA computation across models
│   ├── values_measures_compute_all_results.py  # Compute all alignment metrics
│   ├── generate_emnlp_figures.py           # Figure generation (all paper figures)
│   ├── convert_prism_to_pipeline.py        # PRISM → pipeline format conversion
│   ├── run_permutation_tests_all.sh        # Run permutation tests for all models
│   ├── l2_baseline_robustness.py           # Human baseline robustness (canonical: 50.0%)
│   ├── vaa_effect_size.py                  # Per-model Cohen's d and 95% bootstrap CIs
│   ├── profile_run_to_run_gpt.py           # GPT-5.1 run-to-run variability measurement
│   ├── question_subset_robustness.py       # Item-selection robustness (subsample + LOCO)
│   ├── homog_rate_robustness.py            # Homogenization rate robustness checks
│   ├── dialogue_length_ablation_setup.py   # Build K=1/3 dialogue-truncation configs
│   ├── dialogue_length_ablation_analyze.py # Per-K homogenization rate + bootstrap CIs
│   ├── dialogue_length_density_control.py  # Demographic-density control for the ablation
│   ├── run_with_resume.sh                  # Auto-resume wrapper for long inference runs
│   └── human_validation/                   # PRISM validation study materials
├── wvs_values_results/
│   ├── <ModelName>/
│   │   ├── none_values_results/            # None condition outputs (total_20.jsonl)
│   │   ├── profile_values_results/         # Profile condition outputs (total_1000.jsonl, LFS)
│   │   ├── career/ & investment/           # Dialogue outputs (total_1000.jsonl, LFS)
│   │   │   └── dialogue_turns{1,3}_values_results/  # K=1/3 truncations for the ablation
│   │   ├── experiments_results.json        # Pre-computed alignment metrics
│   │   ├── permutation_test_results.json   # Permutation test z-scores and p-values
│   │   └── dialogue_length_ablation_career.json    # Dialogue-length ablation (K=1,3,5)
│   ├── prism_validation/                   # PRISM model outputs (LFS)
│   └── ba_none_vaa_comparison.json         # None-condition VAA across all models
├── requirements.txt
└── setup.py
```

> **Large result files** (profile and dialogue `total_1000.jsonl` outputs, and GPT-5.1 `experiments_results.json`) are tracked with Git LFS.

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

> **Note:** `sampled_demographic_features.csv` and `sampled_values_df.csv` contain individual-level WVS Wave 7 responses, which the WVS conditions of use prohibit redistributing, so they are **not** included in this repository. WVS Wave 7 is available directly from the [WVS website](https://www.worldvaluessurvey.org/) (free for academic use after registration); the specific 1,000-respondent sample used in our experiments is available from the authors on request.

```bash
python -m llm_behavior_adaptation.value_measurement.wvs_values_comparison \
    --user-profile-dataset datasets/wvs_benchmarks/sampled_demographic_features.csv \
    --user-value-dataset datasets/wvs_benchmarks/sampled_values_df.csv \
    --ba-user-results <path/to/profile_values_results/total_1000.jsonl> \
    --ba-dialogue-career-results <path/to/career/dialogue_values_results/total_1000.jsonl> \
    --ba-dialogue-investment-results <path/to/investment/dialogue_values_results/total_1000.jsonl> \
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

---

## License

Code is released under the [MIT License](LICENSE). The synthetic dialogue dataset and evaluation outputs are derived from [WVS Wave 7](https://www.worldvaluessurvey.org/) and [WorldValuesBench](https://aclanthology.org/2024.lrec-main.1539/), both made available for academic research. Accordingly, all data artifacts in this repository are intended for **non-commercial research use only**.

Individual-level WVS respondent data (`sampled_demographic_features.csv`, `sampled_values_df.csv`) is **not redistributed** in this repository, in keeping with the WVS conditions of use. Obtain WVS Wave 7 directly from the [WVS website](https://www.worldvaluessurvey.org/); the exact 1,000-respondent sample used in our experiments is available from the authors on request.
