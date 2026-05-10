"""
Permutation test for demographic homogenization in LLM value alignment.

Tests whether matched demographic profiles specifically drive homogenization
beyond what would occur under random user-response pairings.

Methodology follows Dror et al. (ACL 2018) for permutation testing in NLP
and Wang et al. (EMNLP 2024, "JobFair") for LLM demographic bias evaluation.

Usage:
    python scripts/permutation_test_homogenization.py \
        --ba-user-results wvs_values_results/Model/BA_user_values_results/total_1000.jsonl \
        --output-path wvs_values_results/Model/permutation_test_results.json \
        --num-permutations 10000 \
        --seed 42
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_behavior_adaptation.value_measurement.formulas import componentwise_centroid
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (
    ATTRIBUTES,
    ValuesComparison,
    load_jsonl_file,
)

logger = logging.getLogger(__name__)

DATASET_DIR = "datasets/wvs_benchmarks"


def _l2_distance(
    a: Mapping,
    b: Mapping,
    question_metadata: Mapping,
) -> float:
    """L2 (Euclidean) distance between two answer vectors, normalized per question scale.
    Matches the paper's ||·||₂ formulation for homogenization rate."""
    d = 0.0
    for q, info in question_metadata.items():
        if q not in a or q not in b:
            continue
        lo = int(info["answer_scale_min"])
        hi = int(info["answer_scale_max"])
        try:
            av = float(a[q])
            bv = float(b[q])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(av) and np.isfinite(bv)):
            continue
        scale = hi - lo
        if scale > 0:
            d += ((av - bv) / scale) ** 2
    return np.sqrt(d)


def _process_model_outputs(original_answers_list: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    processed: Dict[str, Dict[str, Dict]] = {}
    for answer_details in original_answers_list:
        for user_id, answers in answer_details.items():
            per_user: Dict[str, Dict] = {}
            for cat_answers in list(answers.values()):
                for answer in cat_answers:
                    per_user.update(answer)
            processed[user_id] = per_user
    return processed


def _get_human_responses(vc: ValuesComparison) -> Dict[str, Dict[str, int]]:
    return (
        vc.user_value_dataset.astype({"D_INTERVIEW": str})
        .groupby("D_INTERVIEW", as_index=True)
        .last()[list(vc.all_questions.keys())]
        .to_dict(orient="index")
    )


def _get_model_responses(vc: ValuesComparison, model_results: Dict) -> Dict[str, Dict[str, int]]:
    return vc._pick_model_results_option_id(model_results)


def _uid_group_map(vc: ValuesComparison, attribute: str, valid_uids) -> Dict[str, str]:
    df = vc.user_profile_dataset.copy()
    df["D_INTERVIEW"] = df["D_INTERVIEW"].astype(str)
    df = df.set_index("D_INTERVIEW")
    label_series = vc._group_label_series(df.reset_index(), attribute)
    label_series.index = df.index
    label_series = label_series.dropna()
    uid_set = set(valid_uids)
    return {uid: str(label_series.loc[uid]) for uid in label_series.index if uid in uid_set}


def _compute_group_centroids(
    uid_to_group: Dict[str, str],
    human_responses: Dict[str, Dict[str, int]],
    all_questions: Mapping,
) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, int]]] = {}
    for uid, group in uid_to_group.items():
        if uid in human_responses:
            grouped.setdefault(group, []).append(human_responses[uid])

    centroids = {}
    for group_name, answers_list in grouped.items():
        if answers_list:
            centroids[group_name] = componentwise_centroid(answers_list, all_questions)
    return centroids


def _compute_homogenization_rate(
    uids: List[str],
    uid_to_group: Dict[str, str],
    human_responses: Dict[str, Dict[str, int]],
    model_responses: Dict[str, Dict[str, int]],
    group_centroids: Dict[str, Dict[str, int]],
    all_questions: Mapping,
) -> float:
    """Paper formula: I(||S_u^human - S_g^human||₂ > ||S_u^model - S_g^human||₂)"""
    toward_centroid = 0
    total = 0

    for uid in uids:
        group = uid_to_group.get(uid)
        if group is None or group not in group_centroids:
            continue
        if uid not in human_responses or uid not in model_responses:
            continue

        centroid = group_centroids[group]
        d_human = _l2_distance(human_responses[uid], centroid, all_questions)
        d_model = _l2_distance(model_responses[uid], centroid, all_questions)

        if d_model < d_human:
            toward_centroid += 1
        total += 1

    return toward_centroid / total if total > 0 else 0.0


def run_permutation_test(
    vc: ValuesComparison,
    model_results: Dict,
    num_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)

    human_resp = _get_human_responses(vc)
    model_resp = _get_model_responses(vc, model_results)

    common_uids = sorted(set(human_resp.keys()) & set(model_resp.keys()))
    if not common_uids:
        raise ValueError("No common user IDs between human and model results")

    logger.info("Running permutation test: %d users, %d permutations", len(common_uids), num_permutations)

    model_response_list = [model_resp[uid] for uid in common_uids]

    results = {}
    for attribute in tqdm(ATTRIBUTES, desc="Attributes"):
        uid_to_group = _uid_group_map(vc, attribute, common_uids)
        valid_uids = [uid for uid in common_uids if uid in uid_to_group]

        if len(valid_uids) < 10:
            logger.warning("Skipping attribute %s: only %d valid users", attribute, len(valid_uids))
            continue

        group_centroids = _compute_group_centroids(uid_to_group, human_resp, vc.all_questions)

        real_rate = _compute_homogenization_rate(
            valid_uids, uid_to_group, human_resp, model_resp, group_centroids, vc.all_questions
        )

        permuted_rates = np.empty(num_permutations)
        for k in tqdm(range(num_permutations), desc=f"  {attribute}", leave=False):
            shuffled_indices = rng.permutation(len(common_uids))
            shuffled_model_resp = {common_uids[i]: model_response_list[shuffled_indices[i]] for i in range(len(common_uids))}

            permuted_rates[k] = _compute_homogenization_rate(
                valid_uids, uid_to_group, human_resp, shuffled_model_resp, group_centroids, vc.all_questions
            )

        p_value = (np.sum(permuted_rates >= real_rate) + 1) / (num_permutations + 1)
        null_mean = float(np.mean(permuted_rates))
        null_std = float(np.std(permuted_rates))
        effect_size_z = (real_rate - null_mean) / null_std if null_std > 0 else float("inf")

        results[attribute] = {
            "real_homogenization_rate": round(real_rate, 4),
            "null_mean": round(null_mean, 4),
            "null_std": round(null_std, 4),
            "null_min": round(float(np.min(permuted_rates)), 4),
            "null_max": round(float(np.max(permuted_rates)), 4),
            "null_percentile_95": round(float(np.percentile(permuted_rates, 95)), 4),
            "null_percentile_99": round(float(np.percentile(permuted_rates, 99)), 4),
            "p_value": round(float(p_value), 6),
            "effect_size_z": round(effect_size_z, 2),
            "demographic_effect": round(real_rate - null_mean, 4),
            "n_valid_users": len(valid_uids),
            "n_groups": len(group_centroids),
            "interpretation": "significant" if p_value < 0.05 else "not_significant",
        }

        logger.info(
            "%s: real=%.3f, null=%.3f±%.3f, p=%.4f, z=%.1f",
            attribute, real_rate, null_mean, null_std, p_value, effect_size_z,
        )

    all_significant = all(r["interpretation"] == "significant" for r in results.values())
    effects = [r["demographic_effect"] for r in results.values()]

    return {
        "config": {
            "num_permutations": num_permutations,
            "seed": seed,
            "n_users": len(common_uids),
            "n_questions": len(vc.all_questions),
        },
        "results": results,
        "summary": {
            "all_significant": all_significant,
            "mean_demographic_effect": round(float(np.mean(effects)), 4),
            "max_demographic_effect": round(float(np.max(effects)), 4),
            "min_demographic_effect": round(float(np.min(effects)), 4),
            "min_p_value": round(min(r["p_value"] for r in results.values()), 6),
            "max_p_value": round(max(r["p_value"] for r in results.values()), 6),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Permutation test for demographic homogenization in LLM value alignment"
    )
    parser.add_argument("--ba-user-results", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--user-profile-dataset", type=str, default=f"{DATASET_DIR}/sampled_demographic_features.csv"
    )
    parser.add_argument(
        "--user-value-dataset", type=str, default=f"{DATASET_DIR}/sampled_values_df.csv"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)

    with open(f"{DATASET_DIR}/picked_questions.json", "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    model_results = _process_model_outputs(load_jsonl_file(args.ba_user_results))

    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results=model_results,
        ba_dialogue_career_results={},
        ba_dialogue_investment_results={},
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )

    output = run_permutation_test(
        vc=vc,
        model_results=model_results,
        num_permutations=args.num_permutations,
        seed=args.seed,
    )

    output["config"]["ba_user_path"] = args.ba_user_results

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults written to {args.output_path}")
    print(f"\nSummary:")
    print(f"  All attributes significant: {output['summary']['all_significant']}")
    print(f"  Mean demographic effect: {output['summary']['mean_demographic_effect']:.4f}")
    print(f"  P-value range: {output['summary']['min_p_value']:.6f} - {output['summary']['max_p_value']:.6f}")


if __name__ == "__main__":
    main()
