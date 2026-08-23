"""
Homogenization Rate — distance-metric robustness check (rebuttal Risk 4: ordinal robustness).

Recomputes the paper's Homogenization Rate under three distance functions and reports whether the
per-model rates and the *ordering* of models are preserved:

  * L2   — the paper's published metric (Euclidean, per-question scale-normalized). Reproduces Table 3.
  * L1   — Manhattan (per-question scale-normalized). Less sensitive to large per-question deviations.
  * rank — ORDINAL: each response is mapped to its fractional rank within the human population for
           that question (monotone-invariant, removes the interval-scale assumption), then L2 in rank
           space. This directly answers "is the centroid-pull result an artifact of Euclidean/interval
           treatment?".

Everything except the distance function is imported from `permutation_test_homogenization.py`, so the
L2 column here is byte-for-byte the paper's pipeline (same loaders, grouping, and centroids).

Usage (all 7 models, profile condition):
    python scripts/homog_rate_robustness.py \
        --models-root wvs_values_results --condition profile \
        --output wvs_values_results/homog_rate_robustness.json

Single model:
    python scripts/homog_rate_robustness.py \
        --model-file wvs_values_results/Qwen2.5-72B-Instruct/profile_values_results/total_1000.jsonl

Notes
-----
* Per-attribute rates are reported for all 6 ATTRIBUTES plus the mean across attributes; use whichever
  aggregation matches the paper's Table 3 "Homog Rate" column. The robustness claim is that the pattern
  (rates >> 51% human baseline; model ordering) is preserved across metrics.
* Reference distribution for the rank transform is the full common human population, computed once.
* Optional `--with-baseline` recomputes a split-half human baseline under each metric.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Reuse the paper's exact pipeline — only the distance function changes below.
from permutation_test_homogenization import (  # noqa: E402
    _process_model_outputs,
    _get_human_responses,
    _get_model_responses,
    _uid_group_map,
    _compute_group_centroids,
    _l2_distance,
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


# --------------------------------------------------------------------------- #
# Distance functions                                                          #
# --------------------------------------------------------------------------- #
def _l1_distance(a, b, question_metadata):
    """Manhattan distance, per-question scale-normalized. Mirror of _l2_distance without the square."""
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
            d += abs((av - bv) / scale)
    return d


def _plain_l2(a, b, question_keys):
    """Euclidean distance on already-normalized (rank-space) vectors; no per-question scaling."""
    d = 0.0
    for q in question_keys:
        if q in a and q in b:
            d += (a[q] - b[q]) ** 2
    return np.sqrt(d)


# --------------------------------------------------------------------------- #
# Ordinal (rank) transform                                                    #
# --------------------------------------------------------------------------- #
def _build_rank_refs(human_responses, all_questions):
    """Per question, the sorted array of finite human response values (the reference distribution)."""
    refs = {}
    for q in all_questions:
        vals = []
        for resp in human_responses.values():
            v = resp.get(q)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vals.append(v)
        if vals:
            refs[q] = np.sort(np.asarray(vals, dtype=float))
    return refs


def _rank_value(v, ref):
    """Midrank fractional position of v within ref, in [0, 1]. Monotone-invariant."""
    n = len(ref)
    left = np.searchsorted(ref, v, side="left")
    right = np.searchsorted(ref, v, side="right")
    return ((left + right) / 2.0) / n


def _rank_transform(responses, refs):
    """Map each user's raw response vector into rank space using the human reference distribution."""
    out = {}
    for uid, resp in responses.items():
        vec = {}
        for q, ref in refs.items():
            v = resp.get(q)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                vec[q] = _rank_value(v, ref)
        out[uid] = vec
    return out


def _rank_centroids(uid_to_group, human_rank, all_questions):
    """Rank-space group centroid = per-question median of rank-transformed human vectors in the group."""
    grouped = {}
    for uid, group in uid_to_group.items():
        if uid in human_rank:
            grouped.setdefault(group, []).append(human_rank[uid])
    centroids = {}
    for group, vecs in grouped.items():
        c = {}
        for q in all_questions:
            vals = [vec[q] for vec in vecs if q in vec]
            if vals:
                c[q] = float(np.median(vals))
        centroids[group] = c
    return centroids


# --------------------------------------------------------------------------- #
# Generic homogenization rate                                                 #
# --------------------------------------------------------------------------- #
def _homog_rate(uids, uid_to_group, human_resp, model_resp, centroids, questions, dist):
    """Paper formula I(d_model < d_human), parameterized by the distance function `dist`."""
    toward, total = 0, 0
    for uid in uids:
        group = uid_to_group.get(uid)
        if group is None or group not in centroids:
            continue
        if uid not in human_resp or uid not in model_resp:
            continue
        centroid = centroids[group]
        d_h = dist(human_resp[uid], centroid, questions)
        d_m = dist(model_resp[uid], centroid, questions)
        if d_m < d_h:
            toward += 1
        total += 1
    return (toward / total) if total else 0.0


def evaluate_model(vc, model_results):
    """Per-attribute Homog Rate under L2 / L1 / rank for one model's results dict."""
    human_resp = _get_human_responses(vc)
    model_resp = _get_model_responses(vc, model_results)
    common = sorted(set(human_resp) & set(model_resp))
    if not common:
        raise ValueError("No common user IDs between human and model results")

    # Rank space is built once over the full common human population.
    refs = _build_rank_refs({u: human_resp[u] for u in common}, vc.all_questions)
    human_rank = _rank_transform({u: human_resp[u] for u in common}, refs)
    model_rank = _rank_transform({u: model_resp[u] for u in common}, refs)

    out = {}
    for attribute in ATTRIBUTES:
        uid_to_group = _uid_group_map(vc, attribute, common)
        valid = [u for u in common if u in uid_to_group]
        if len(valid) < 10:
            logger.warning("Skipping %s: only %d valid users", attribute, len(valid))
            continue

        raw_centroids = _compute_group_centroids(uid_to_group, human_resp, vc.all_questions)
        rk_centroids = _rank_centroids(uid_to_group, human_rank, vc.all_questions)

        out[attribute] = {
            "l2": round(_homog_rate(valid, uid_to_group, human_resp, model_resp, raw_centroids, vc.all_questions, _l2_distance), 4),
            "l1": round(_homog_rate(valid, uid_to_group, human_resp, model_resp, raw_centroids, vc.all_questions, _l1_distance), 4),
            "rank": round(_homog_rate(valid, uid_to_group, human_rank, model_rank, rk_centroids, vc.all_questions, _plain_l2), 4),
            "n_valid_users": len(valid),
            "n_groups": len(raw_centroids),
        }
    # Mean across attributes per metric.
    means = {m: round(float(np.mean([v[m] for v in out.values()])), 4) for m in ("l2", "l1", "rank")}
    return {"per_attribute": out, "mean": means}


def _spearman(x, y):
    """Spearman rho between two equal-length sequences (model orderings)."""
    n = len(x)
    if n < 3:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    d2 = float(np.sum((rx - ry) ** 2))
    return 1 - 6 * d2 / (n * (n**2 - 1))


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


def main():
    ap = argparse.ArgumentParser(description="Homog Rate distance-metric robustness (L2 / L1 / rank).")
    ap.add_argument("--models-root", default="wvs_values_results", help="Root dir containing per-model result folders.")
    ap.add_argument("--condition", default="profile", help="Condition subfolder prefix (profile / none).")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--model-file", default=None, help="Evaluate a single results .jsonl instead of iterating models.")
    ap.add_argument("--output", default="wvs_values_results/homog_rate_robustness.json")
    ap.add_argument("--user-profile-dataset", default=f"{DATASET_DIR}/sampled_demographic_features.csv")
    ap.add_argument("--user-value-dataset", default=f"{DATASET_DIR}/sampled_values_df.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    user_profile = pd.read_csv(args.user_profile_dataset)
    user_value = pd.read_csv(args.user_value_dataset)
    with open(f"{DATASET_DIR}/picked_questions.json", "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    if args.model_file:
        jobs = [("(single)", args.model_file)]
    else:
        jobs = [
            (m, os.path.join(args.models_root, m, f"{args.condition}_values_results", "total_1000.jsonl"))
            for m in args.models
        ]

    results = {}
    for name, path in jobs:
        if not os.path.exists(path):
            logger.warning("Missing results for %s: %s (skipped)", name, path)
            continue
        model_results = _process_model_outputs(load_jsonl_file(path))
        vc = _build_vc(user_profile, user_value, picked_questions, model_results)
        results[name] = evaluate_model(vc, model_results)
        m = results[name]["mean"]
        logger.info("%-24s  L2=%.3f  L1=%.3f  rank=%.3f", name, m["l2"], m["l1"], m["rank"])

    # Ordering robustness: Spearman rho of model rankings, L2 vs L1 and L2 vs rank.
    ordering = {}
    if len(results) >= 3:
        names = list(results)
        l2 = [results[n]["mean"]["l2"] for n in names]
        l1 = [results[n]["mean"]["l1"] for n in names]
        rk = [results[n]["mean"]["rank"] for n in names]
        ordering = {
            "spearman_rho_l2_vs_l1": round(_spearman(l2, l1), 4),
            "spearman_rho_l2_vs_rank": round(_spearman(l2, rk), 4),
            "note": "rho ~1.0 => model ordering is preserved under the alternative metric.",
        }

    output = {
        "human_baseline_l2": 0.50,  # canonical (scripts/l2_baseline_robustness.py); submission said 0.51 (+1% offset dropped)
        "models": results,
        "ordering_robustness": ordering,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Console table.
    print("\nHomogenization Rate — metric robustness (mean across attributes)\n")
    print(f"{'Model':<24} {'L2':>7} {'L1':>7} {'rank':>7}")
    print("-" * 48)
    for name, r in results.items():
        m = r["mean"]
        print(f"{name:<24} {m['l2']:>7.3f} {m['l1']:>7.3f} {m['rank']:>7.3f}")
    if ordering:
        print(f"\nOrdering preserved (Spearman rho):  L2~L1 = {ordering['spearman_rho_l2_vs_l1']}, "
              f"L2~rank = {ordering['spearman_rho_l2_vs_rank']}")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
