"""
Permutation test for dialogue condition: run per-domain (career, investment),
then average HR and z-scores across both domains.

This is the correct methodology: merging option_ids across domains is wrong
because career and investment dialogues answer questions in different contexts.

Output: wvs_values_results/{model}/dialogue_permutation_test_results.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.permutation_test_homogenization import (
    _process_model_outputs,
    _get_human_responses,
    _uid_group_map,
    _compute_group_centroids,
    _compute_homogenization_rate,
    ATTRIBUTES,
)
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (
    ValuesComparison,
    load_jsonl_file,
)
import numpy as np
import pandas as pd

DATASET_DIR = "datasets/wvs_benchmarks"

MODELS = {
    "Llama-3.1-8B-Instruct":  "wvs_values_results/Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct": "wvs_values_results/Llama-3.1-70B-Instruct",
    "Qwen2.5-7B-Instruct":    "wvs_values_results/Qwen2.5-7B-Instruct",
    "QwQ-32B":                "wvs_values_results/QwQ-32B",
    "DeepSeek-V3":            "wvs_values_results/DeepSeek-V3",
    "gpt-5.1":                "wvs_values_results/gpt-5.1",
    "Qwen2.5-72B-Instruct":   "wvs_values_results/Qwen2.5-72B-Instruct",
}


def load_domain_responses(model_dir, domain):
    path = os.path.join(model_dir, domain, "dialogue_values_results", "total_1000.jsonl")
    if not os.path.exists(path):
        return None
    raw = _process_model_outputs(load_jsonl_file(path))
    model_resp = {}
    for uid, qdict in raw.items():
        model_resp[uid] = {}
        for qid, val in qdict.items():
            if isinstance(val, dict):
                model_resp[uid][qid] = val.get("option_id", val)
            else:
                model_resp[uid][qid] = val
    return model_resp


def run_permtest_for_domain(model_name, domain, model_resp, vc, human_resp, num_permutations=10000, seed=42):
    common_uids = sorted(set(human_resp.keys()) & set(model_resp.keys()))
    print(f"    {domain}: {len(common_uids)} common users")

    rng = np.random.default_rng(seed)
    model_response_list = [model_resp[uid] for uid in common_uids]

    results = {}
    for attr in ATTRIBUTES:
        uid_to_group = _uid_group_map(vc, attr, common_uids)
        valid_uids = [u for u in common_uids if u in uid_to_group]
        if len(valid_uids) < 10:
            continue

        centroids = _compute_group_centroids(uid_to_group, human_resp, vc.all_questions)
        real_rate = _compute_homogenization_rate(
            valid_uids, uid_to_group, human_resp, model_resp, centroids, vc.all_questions
        )

        permuted_rates = np.empty(num_permutations)
        for k in range(num_permutations):
            idx = rng.permutation(len(common_uids))
            shuffled = {common_uids[i]: model_response_list[idx[i]] for i in range(len(common_uids))}
            permuted_rates[k] = _compute_homogenization_rate(
                valid_uids, uid_to_group, human_resp, shuffled, centroids, vc.all_questions
            )

        null_mean = float(np.mean(permuted_rates))
        null_std = float(np.std(permuted_rates))
        z = (real_rate - null_mean) / null_std if null_std > 0 else float("inf")
        p = (np.sum(permuted_rates >= real_rate) + 1) / (num_permutations + 1)

        results[attr] = {
            "real_homogenization_rate": real_rate,
            "null_mean": null_mean,
            "null_std": null_std,
            "p_value": p,
            "effect_size_z": z,
            "n_valid_users": len(valid_uids),
        }
        print(f"      {attr}: HR={real_rate:.1%} z={z:.2f} p={p:.4f}")

    return results


def merge_domain_results(career_results, invest_results):
    """Average HR and z-scores across both domains for each attribute."""
    all_attrs = set(career_results or {}) | set(invest_results or {})
    merged = {}
    for attr in all_attrs:
        parts = []
        if career_results and attr in career_results:
            parts.append(career_results[attr])
        if invest_results and attr in invest_results:
            parts.append(invest_results[attr])
        if not parts:
            continue
        merged[attr] = {
            "real_homogenization_rate": float(np.mean([p["real_homogenization_rate"] for p in parts])),
            "effect_size_z": float(np.mean([p["effect_size_z"] for p in parts])),
            "p_value": float(np.mean([p["p_value"] for p in parts])),
            "n_valid_users": int(np.mean([p["n_valid_users"] for p in parts])),
            "domains": [d for d, r in [("career", career_results), ("investment", invest_results)] if r and attr in r],
        }
    return merged


def run_permutation_test_for_model(model_name, model_dir, vc, human_resp, num_permutations=10000, seed=42):
    print(f"  {model_name}:")
    career_resp = load_domain_responses(model_dir, "career")
    invest_resp = load_domain_responses(model_dir, "investment")

    if not career_resp and not invest_resp:
        print(f"    no dialogue data, skipping")
        return None

    career_results = None
    invest_results = None

    if career_resp:
        career_results = run_permtest_for_domain(model_name, "career", career_resp, vc, human_resp, num_permutations, seed)
    if invest_resp:
        invest_results = run_permtest_for_domain(model_name, "investment", invest_resp, vc, human_resp, num_permutations, seed + 1)

    merged = merge_domain_results(career_results, invest_results)

    mean_z = float(np.mean([v["effect_size_z"] for v in merged.values()]))
    mean_hr = float(np.mean([v["real_homogenization_rate"] for v in merged.values()]))
    sig6 = sum(1 for v in merged.values() if v["p_value"] < 0.05)

    print(f"  {model_name}: mean HR={mean_hr:.1%} mean_z={mean_z:.2f} sig={sig6}/6")

    return {
        "model": model_name,
        "results": merged,
        "summary": {"mean_homog_rate": mean_hr, "mean_z": mean_z, "sig_6": sig6},
        "by_domain": {
            "career": {
                attr: {"real_homogenization_rate": v["real_homogenization_rate"], "effect_size_z": v["effect_size_z"]}
                for attr, v in (career_results or {}).items()
            },
            "investment": {
                attr: {"real_homogenization_rate": v["real_homogenization_rate"], "effect_size_z": v["effect_size_z"]}
                for attr, v in (invest_results or {}).items()
            },
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()

    user_profile = pd.read_csv(f"{DATASET_DIR}/sampled_demographic_features.csv")
    user_value = pd.read_csv(f"{DATASET_DIR}/sampled_values_df.csv")
    with open(f"{DATASET_DIR}/picked_questions.json") as f:
        picked_questions = json.load(f)

    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results={},
        ba_dialogue_career_results={},
        ba_dialogue_investment_results={},
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )
    human_resp = _get_human_responses(vc)

    all_results = {}
    for model_name, model_dir in MODELS.items():
        print(f"\n=== {model_name} ===")
        res = run_permutation_test_for_model(model_name, model_dir, vc, human_resp, num_permutations=args.permutations)
        if res:
            out_path = os.path.join(model_dir, "dialogue_permutation_test_results.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            all_results[model_name] = res["summary"]

    print("\n\n=== SUMMARY ===")
    print("%-30s %6s %8s %6s" % ("Model", "HR%", "Mean z", "Sig/6"))
    hrs = []
    for m, s in all_results.items():
        print("%-30s %6.1f%% %8.2f %6d/6" % (m, s["mean_homog_rate"] * 100, s["mean_z"], s["sig_6"]))
        hrs.append(s["mean_homog_rate"])
    if hrs:
        print("Overall mean HR: %.1f%%, range: %.1f%%--%.1f%%" % (
            sum(hrs) / len(hrs) * 100, min(hrs) * 100, max(hrs) * 100))


if __name__ == "__main__":
    main()
