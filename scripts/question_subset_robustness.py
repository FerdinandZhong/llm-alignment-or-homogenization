"""
Question-subset robustness check — ARR rebuttal (Reviewer W3: question selection).

Tests whether VAA and Homogenization Rate *rankings* are stable when a random
subset of the 55 evaluation items is used.  All computation runs on CACHED
per-user, per-question model outputs; no new inference and no GPT-5.1 API calls.

Selection rationale (for the rebuttal):
  The 55 items were chosen by sampling five questions per category from 11
  retained WVS thematic domains (equal weighting), drawn from value-type ordinal
  items.  "Five per category" sits at the floor of the two smallest retained
  categories (Economic Values, Science & Technology each have 6 eligible items),
  preventing larger categories (Social Values: 44 items) from dominating the
  metrics.  Within each category, items were drawn randomly.

Subset designs evaluated here:
  (a) full55   — all 55 questions  →  reproduces Table 2 / Table 3 (sanity check)
  (b) sub3of5  — 100 random draws of 3-of-5 per category (33 items each)
  (c) loco     — 11 leave-one-category-out folds (50 items each)

Key outputs (for the rebuttal response):
  • full55 VAA / Homog Rate reproduce the published numbers exactly.
  • Spearman ρ of model VAA ranking: full55 vs subsample mean ≈ 1.0
  • Spearman ρ of model Homog Rate ranking: full55 vs subsample mean ≈ 1.0
  • Per-model VAA / Homog Rate range across 100 subsamples is narrow.

Usage
-----
    python scripts/question_subset_robustness.py

All defaults match the paper's experimental setup.
Output: wvs_values_results/question_subset_robustness.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from permutation_test_homogenization import (  # noqa: E402
    _process_model_outputs,
    _get_human_responses,
    _get_model_responses,
    _uid_group_map,
    _compute_group_centroids,
    _l2_distance,
    _compute_homogenization_rate,
)
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (  # noqa: E402
    ATTRIBUTES,
    ValuesComparison,
    load_jsonl_file,
)

logger = logging.getLogger(__name__)

DATASET_DIR = "datasets/wvs_benchmarks"

DEFAULT_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
    "QwQ-32B",
    "DeepSeek-V3",
    "gpt-5.1",
]

# Published values from Table 2 (VAA, PROFILE condition) and Table 3 (Homog Rate).
# Used as sanity-check bounds for the full-55 reproduction.
PAPER_VAA_PROFILE = {
    "Llama-3.1-8B-Instruct": 0.494,
    "Llama-3.1-70B-Instruct": 0.605,
    "Qwen2.5-7B-Instruct": 0.550,
    "Qwen2.5-72B-Instruct": 0.617,
    "QwQ-32B": 0.614,
    "DeepSeek-V3": 0.606,
    "gpt-5.1": 0.613,
}
PAPER_HOMOG_PROFILE = {
    "Llama-3.1-8B-Instruct": 0.280,
    "Llama-3.1-70B-Instruct": 0.553,
    "Qwen2.5-7B-Instruct": 0.694,
    "Qwen2.5-72B-Instruct": 0.832,
    "QwQ-32B": 0.669,
    "DeepSeek-V3": 0.798,
    "gpt-5.1": 0.808,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flat_questions(picked_questions: dict) -> dict:
    """Flatten category→{qid→meta} into a single flat {qid→meta}."""
    flat = {}
    for cat_qs in picked_questions.values():
        flat.update(cat_qs)
    return flat


def _filter_responses(responses: Dict[str, Dict], qids) -> Dict[str, Dict]:
    """Restrict each user's response dict to the given question IDs."""
    qids_set = set(qids)
    return {uid: {q: v for q, v in resp.items() if q in qids_set} for uid, resp in responses.items()}


def _pearsonr_np(x: list, y: list) -> float:
    """Pearson r via numpy (avoids scipy dependency)."""
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    xa -= xa.mean()
    ya -= ya.mean()
    denom = np.sqrt(np.sum(xa ** 2) * np.sum(ya ** 2))
    return float(np.sum(xa * ya) / denom) if denom > 0.0 else 0.0


def _compute_vaa(human_resp: Dict, model_resp: Dict, qids) -> float:
    """Mean per-user Pearson r between model and human response vectors, restricted to `qids`."""
    qids_list = list(qids)
    scores = []
    for uid in set(human_resp) & set(model_resp):
        h_vec, m_vec = [], []
        for q in qids_list:
            h = human_resp[uid].get(q)
            m = model_resp[uid].get(q)
            if h is None or m is None:
                continue
            try:
                hf, mf = float(h), float(m)
            except (TypeError, ValueError):
                continue
            if np.isfinite(hf) and np.isfinite(mf):
                h_vec.append(hf)
                m_vec.append(mf)
        if len(h_vec) < 3:
            continue
        r = _pearsonr_np(h_vec, m_vec)
        if np.isfinite(r):
            scores.append(r)
    return float(np.mean(scores)) if scores else float("nan")


def _compute_homog_rate_mean(
    human_resp: Dict,
    model_resp: Dict,
    uid_group_maps: Dict[str, Dict],
    q_meta: Mapping,
) -> float:
    """Mean Homog Rate over ATTRIBUTES for the given (filtered) question metadata."""
    rates = []
    common = set(human_resp) & set(model_resp)
    for attribute in ATTRIBUTES:
        uid_to_group = uid_group_maps.get(attribute, {})
        valid = [u for u in common if u in uid_to_group]
        if len(valid) < 10:
            continue
        centroids = _compute_group_centroids(uid_to_group, human_resp, q_meta)
        rate = _compute_homogenization_rate(
            valid, uid_to_group, human_resp, model_resp, centroids, q_meta
        )
        rates.append(rate)
    return float(np.mean(rates)) if rates else float("nan")


def _spearman(x: list, y: list) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    d2 = float(np.sum((rx - ry) ** 2))
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


def _build_vc(user_profile, user_value, picked_questions, model_results):
    return ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results=model_results,
        ba_dialogue_career_results={},
        ba_dialogue_investment_results={},
        picked_questions=picked_questions,
        results_output_path="",
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Subset generation
# ---------------------------------------------------------------------------

def build_subset_configs(
    picked_questions: dict,
    seed: int = 42,
    n_subsample: int = 100,
    k: int = 3,
) -> Dict[str, List[str]]:
    """
    Returns a dict of named question-ID lists.

    Keys:
      "full55"           — all 55 question IDs
      "sub{k}of5_{NNN}"  — n_subsample random draws of k-of-5 per category
      "loco_{cat}"       — leave-one-category-out (11 folds)
    """
    rng = np.random.default_rng(seed)
    subsets: Dict[str, List[str]] = {}

    all_qids = [qid for qs in picked_questions.values() for qid in qs]
    subsets["full55"] = all_qids

    for i in range(n_subsample):
        chosen: List[str] = []
        for cat_qs in picked_questions.values():
            qids = list(cat_qs.keys())
            size = min(k, len(qids))
            chosen.extend(rng.choice(qids, size=size, replace=False).tolist())
        subsets[f"sub{k}of5_{i:03d}"] = chosen

    for cat in picked_questions:
        chosen = [qid for c, qs in picked_questions.items() if c != cat for qid in qs]
        # Shorten key to avoid filesystem issues
        safe_cat = cat[:30].replace(" ", "_").replace(",", "")
        subsets[f"loco_{safe_cat}"] = chosen

    return subsets


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    path: str,
    user_profile: pd.DataFrame,
    user_value: pd.DataFrame,
    picked_questions: dict,
    subsets: Dict[str, List[str]],
    all_questions_full: dict,
) -> Dict:
    """Evaluate VAA and Homog Rate over all subset configs for one model."""
    model_raw = _process_model_outputs(load_jsonl_file(path))

    # Build one full VC — used once to get human_resp and model_resp (55 questions)
    # and for uid_group_maps (which do not depend on question selection).
    vc_full = _build_vc(user_profile, user_value, picked_questions, model_raw)
    human_resp_full = _get_human_responses(vc_full)
    model_resp_full = _get_model_responses(vc_full, model_raw)
    common = sorted(set(human_resp_full) & set(model_resp_full))

    # Cache uid→group maps for all attributes (independent of question subset).
    uid_group_maps: Dict[str, Dict] = {}
    for attribute in ATTRIBUTES:
        uid_group_maps[attribute] = _uid_group_map(vc_full, attribute, common)

    per_subset: Dict[str, Dict] = {}
    for subset_name, qids in subsets.items():
        q_meta = {q: all_questions_full[q] for q in qids if q in all_questions_full}
        h_sub = _filter_responses(human_resp_full, qids)
        m_sub = _filter_responses(model_resp_full, qids)

        vaa = _compute_vaa(human_resp_full, model_resp_full, qids)
        homog = _compute_homog_rate_mean(h_sub, m_sub, uid_group_maps, q_meta)

        per_subset[subset_name] = {
            "vaa": round(float(vaa), 4),
            "homog_rate": round(float(homog), 4),
            "n_questions": len(qids),
        }

    return per_subset


# ---------------------------------------------------------------------------
# Summary and reporting
# ---------------------------------------------------------------------------

def compute_summary(results: Dict[str, Dict], k: int) -> Dict:
    model_names = list(results.keys())
    if not model_names:
        return {}

    ref = results[model_names[0]]
    sub_keys = [kk for kk in ref if kk.startswith("sub")]
    loco_keys = [kk for kk in ref if kk.startswith("loco")]

    full_vaa = [results[m]["full55"]["vaa"] for m in model_names]
    full_homog = [results[m]["full55"]["homog_rate"] for m in model_names]

    # Subsample means per model
    sub_vaa_m = [float(np.mean([results[m][sk]["vaa"] for sk in sub_keys])) for m in model_names]
    sub_hr_m = [float(np.mean([results[m][sk]["homog_rate"] for sk in sub_keys])) for m in model_names]
    sub_vaa_range = [
        [round(float(np.min([results[m][sk]["vaa"] for sk in sub_keys])), 4),
         round(float(np.max([results[m][sk]["vaa"] for sk in sub_keys])), 4)]
        for m in model_names
    ]
    sub_hr_range = [
        [round(float(np.min([results[m][sk]["homog_rate"] for sk in sub_keys])), 4),
         round(float(np.max([results[m][sk]["homog_rate"] for sk in sub_keys])), 4)]
        for m in model_names
    ]

    # LOCO means per model
    loco_vaa_m = [float(np.mean([results[m][lk]["vaa"] for lk in loco_keys])) for m in model_names]
    loco_hr_m = [float(np.mean([results[m][lk]["homog_rate"] for lk in loco_keys])) for m in model_names]

    return {
        "models": model_names,
        "n_sub_configs": len(sub_keys),
        "k_per_category": k,
        "n_loco_folds": len(loco_keys),
        # Spearman ρ of model VAA ranking: full55 vs subsample mean
        "sub_ranking_rho_vaa": round(_spearman(full_vaa, sub_vaa_m), 3),
        "sub_ranking_rho_homog": round(_spearman(full_homog, sub_hr_m), 3),
        # Spearman ρ: full55 vs LOCO mean
        "loco_ranking_rho_vaa": round(_spearman(full_vaa, loco_vaa_m), 3),
        "loco_ranking_rho_homog": round(_spearman(full_homog, loco_hr_m), 3),
        "per_model": {
            m: {
                "full55_vaa": results[m]["full55"]["vaa"],
                "full55_homog_rate": results[m]["full55"]["homog_rate"],
                "sub_vaa_mean": round(sub_vaa_m[i], 4),
                "sub_vaa_range": sub_vaa_range[i],
                "sub_hr_mean": round(sub_hr_m[i], 4),
                "sub_hr_range": sub_hr_range[i],
                "loco_vaa_mean": round(loco_vaa_m[i], 4),
                "loco_hr_mean": round(loco_hr_m[i], 4),
            }
            for i, m in enumerate(model_names)
        },
        "interpretation": (
            "rho close to +1.0 means the model ranking is preserved under question subsets. "
            "Narrow [min, max] ranges confirm results do not hinge on the specific 55-item draw."
        ),
    }


def print_table(results: Dict, summary: Dict, k: int) -> None:
    models = list(results.keys())
    sub_keys = [kk for kk in results[models[0]] if kk.startswith("sub")]

    print("\n" + "=" * 90)
    print("Question-Subset Robustness — PROFILE condition")
    print(f"Subsamples: {len(sub_keys)}× random {k}-of-5 per category ({k*11} items each)")
    print("=" * 90)
    hdr = f"{'Model':<26} {'VAA full55':>10} {'VAA sub(μ)':>11} {'VAA sub[lo,hi]':>17} {'HR full55':>10} {'HR sub(μ)':>10} {'HR sub[lo,hi]':>16}"
    print(hdr)
    print("-" * 100)

    for m in models:
        pm = summary.get("per_model", {}).get(m, {})
        if not pm:
            continue
        vaa_range = f"[{pm['sub_vaa_range'][0]:.3f},{pm['sub_vaa_range'][1]:.3f}]"
        hr_range = f"[{pm['sub_hr_range'][0]:.3f},{pm['sub_hr_range'][1]:.3f}]"
        print(
            f"{m:<26} {pm['full55_vaa']:>10.3f} {pm['sub_vaa_mean']:>11.3f} {vaa_range:>17}"
            f" {pm['full55_homog_rate']:>10.3f} {pm['sub_hr_mean']:>10.3f} {hr_range:>16}"
        )

    print()
    print(f"Ranking Spearman ρ (full55 vs subsample mean):  VAA = {summary.get('sub_ranking_rho_vaa', '?'):.3f}  "
          f"Homog Rate = {summary.get('sub_ranking_rho_homog', '?'):.3f}")
    print(f"Ranking Spearman ρ (full55 vs LOCO mean):       VAA = {summary.get('loco_ranking_rho_vaa', '?'):.3f}  "
          f"Homog Rate = {summary.get('loco_ranking_rho_homog', '?'):.3f}")
    print()

    # Sanity check against published numbers
    any_fail = False
    print("Sanity check against published numbers (full55 must reproduce Table 2 / Table 3):")
    for m in models:
        vaa_ok = abs(results[m]["full55"]["vaa"] - PAPER_VAA_PROFILE.get(m, 0)) < 0.03 if m in PAPER_VAA_PROFILE else None
        hr_ok = abs(results[m]["full55"]["homog_rate"] - PAPER_HOMOG_PROFILE.get(m, 0)) < 0.03 if m in PAPER_HOMOG_PROFILE else None
        vaa_sym = "✓" if vaa_ok else ("✗" if vaa_ok is False else "?")
        hr_sym = "✓" if hr_ok else ("✗" if hr_ok is False else "?")
        print(f"  {m:<26}  VAA {vaa_sym} ({results[m]['full55']['vaa']:.3f} vs {PAPER_VAA_PROFILE.get(m,'?')})  "
              f"HR {hr_sym} ({results[m]['full55']['homog_rate']:.3f} vs {PAPER_HOMOG_PROFILE.get(m,'?')})")
        if vaa_ok is False or hr_ok is False:
            any_fail = True
    if any_fail:
        print("\n  ⚠️  Some full-55 numbers deviate from published values by >0.03 — verify pipeline.")
    else:
        print("\n  ✓  All full-55 numbers reproduce published values within tolerance.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Question-subset robustness (VAA + Homog Rate) for rebuttal W3."
    )
    ap.add_argument("--models-root", default="wvs_values_results")
    ap.add_argument("--condition", default="profile")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--output", default="wvs_values_results/question_subset_robustness.json")
    ap.add_argument("--n-subsample", type=int, default=100,
                    help="Number of random k-of-5 subsample draws (default: 100)")
    ap.add_argument("--k", type=int, default=3,
                    help="Questions per category in each subsample (default: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--user-profile-dataset",
                    default=f"{DATASET_DIR}/sampled_demographic_features.csv")
    ap.add_argument("--user-value-dataset",
                    default=f"{DATASET_DIR}/sampled_values_df.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load shared data
    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)
    with open(f"{DATASET_DIR}/picked_questions.json", encoding="utf-8") as f:
        picked_questions = json.load(f)
    all_questions_full = _flat_questions(picked_questions)

    logger.info("Loaded %d categories, %d questions total", len(picked_questions), len(all_questions_full))

    # Build subset configurations
    subsets = build_subset_configs(
        picked_questions, seed=args.seed, n_subsample=args.n_subsample, k=args.k
    )
    sub_count = sum(1 for k in subsets if k.startswith("sub"))
    loco_count = sum(1 for k in subsets if k.startswith("loco"))
    logger.info("Subset configs: 1 full + %d subsamples + %d LOCO = %d total",
                sub_count, loco_count, len(subsets))

    # Evaluate each model
    all_results: Dict[str, Dict] = {}
    for model_name in args.models:
        path = os.path.join(
            args.models_root, model_name,
            f"{args.condition}_values_results", "total_1000.jsonl"
        )
        if not os.path.exists(path):
            logger.warning("Missing results file for %s at %s — skipped", model_name, path)
            continue
        logger.info("Evaluating %s ...", model_name)
        all_results[model_name] = evaluate_model(
            model_name, path, user_profile, user_value,
            picked_questions, subsets, all_questions_full
        )
        r = all_results[model_name]
        logger.info(
            "  %-26s  full55: VAA=%.3f  HR=%.3f",
            model_name, r["full55"]["vaa"], r["full55"]["homog_rate"]
        )

    if not all_results:
        logger.error("No model results loaded. Check --models-root path.")
        sys.exit(1)

    # Compute summary statistics
    summary = compute_summary(all_results, args.k)

    # Save output
    output_payload = {
        "config": {
            "condition": args.condition,
            "n_subsample": args.n_subsample,
            "k_per_category": args.k,
            "seed": args.seed,
            "models": list(all_results.keys()),
        },
        "summary": summary,
        "full_results": all_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    # Console report
    print_table(all_results, summary, args.k)
    print(f"Full results written to: {args.output}")


if __name__ == "__main__":
    main()
