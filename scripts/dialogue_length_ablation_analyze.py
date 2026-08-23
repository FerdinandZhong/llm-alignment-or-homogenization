"""Dialogue-length ablation analysis (R1 W1 camera-ready addition).

Computes the career-domain Homogenization Rate at dialogue lengths K in {1,3,5}
(plus the PROFILE compact-label anchor) for a single model, on a MATCHED user set,
reusing the paper's canonical Table-3 metric (L2 to componentwise-median group
centroid). Centroids are built over the full human population (paper definition);
HR is scored only on users present in every K so the comparison is apples-to-apples.

K=5 = cached full dialogue (career/dialogue_values_results/total_1000.jsonl).
K=1,3 = truncated runs (career/dialogue_turns{K}_values_results/first_200.jsonl).
PROFILE = cached profile_values_results/total_1000.jsonl (compact label; upper anchor).

Runs on whatever K-files exist, so it can validate on K=5+PROFILE before K=1/3 land.

Usage: python scripts/dialogue_length_ablation_analyze.py --model gpt-5.1
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

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

DATASET_DIR = "datasets/wvs_benchmarks"


def _load_model_resp(path):
    """Load per-user {qid: option_id} from a values-results jsonl, or None if absent."""
    if not os.path.exists(path):
        return None
    raw = _process_model_outputs(load_jsonl_file(path))
    out = {}
    for uid, qdict in raw.items():
        out[uid] = {qid: (v.get("option_id", v) if isinstance(v, dict) else v) for qid, v in qdict.items()}
    return out


def _mean_hr(model_resp, scored_uids, human_resp, vc, centroids_by_attr, group_by_attr):
    """Mean career HR over the 6 attributes, on the matched scored_uids."""
    rates = []
    for attr in ATTRIBUTES:
        if attr not in centroids_by_attr:
            continue
        rates.append(_compute_homogenization_rate(
            scored_uids, group_by_attr[attr], human_resp, model_resp,
            centroids_by_attr[attr], vc.all_questions))
    return float(np.mean(rates)) if rates else float("nan"), rates


def _bootstrap_ci(model_resp, scored_uids, human_resp, vc, centroids_by_attr, group_by_attr, n=1000, seed=42):
    """95% bootstrap CI on the mean-over-attributes HR, resampling users."""
    rng = np.random.default_rng(seed)
    uids = list(scored_uids)
    boots = []
    for _ in range(n):
        samp = [uids[i] for i in rng.integers(0, len(uids), len(uids))]
        m, _r = _mean_hr(model_resp, samp, human_resp, vc, centroids_by_attr, group_by_attr)
        boots.append(m)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--domain", default="career")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    mdir = f"wvs_values_results/{args.model}"

    user_profile = pd.read_csv(f"{DATASET_DIR}/sampled_demographic_features.csv")
    user_value = pd.read_csv(f"{DATASET_DIR}/sampled_values_df.csv")
    with open(f"{DATASET_DIR}/picked_questions.json") as f:
        picked_questions = json.load(f)
    vc = ValuesComparison(
        user_profile_dataset=user_profile, user_value_dataset=user_value,
        ba_user_results={}, ba_dialogue_career_results={}, ba_dialogue_investment_results={},
        picked_questions=picked_questions, results_output_path="", verbose=0,
    )
    human_resp = _get_human_responses(vc)

    # Candidate response sets: PROFILE (compact label) + K in {1,3,5}
    sources = {
        "PROFILE": f"{mdir}/profile_values_results/total_1000.jsonl",
        "K=1": f"{mdir}/{args.domain}/dialogue_turns1_values_results/first_200.jsonl",
        "K=3": f"{mdir}/{args.domain}/dialogue_turns3_values_results/first_200.jsonl",
        "K=5": f"{mdir}/{args.domain}/dialogue_values_results/total_1000.jsonl",
    }
    resp = {k: _load_model_resp(p) for k, p in sources.items()}
    have = {k: v for k, v in resp.items() if v is not None}
    missing = [k for k, v in resp.items() if v is None]
    if missing:
        print(f"[note] not yet available: {missing} (runs still in progress?)")
    if "K=5" not in have:
        sys.exit("K=5 cached dialogue results missing — cannot anchor the curve.")

    # Matched user set: users present in EVERY available dialogue/profile source (+ human).
    matched = set(human_resp)
    for v in have.values():
        matched &= set(v)
    matched = sorted(matched)
    print(f"model={args.model} domain={args.domain} | levels={list(have)} | matched users={len(matched)}")

    # Centroids from FULL human population per attribute (paper definition); score on matched.
    centroids_by_attr, group_by_attr, scored_by_attr = {}, {}, {}
    all_uids = sorted(human_resp)
    for attr in ATTRIBUTES:
        g_full = _uid_group_map(vc, attr, all_uids)
        centroids_by_attr[attr] = _compute_group_centroids(g_full, human_resp, vc.all_questions)
        group_by_attr[attr] = g_full
        scored_by_attr[attr] = [u for u in matched if u in g_full]
    # scored_uids common across attrs is handled inside _compute_homogenization_rate via group membership.

    print(f"\n{'level':8} {'meanHR':>7}  {'95% CI':>16}   per-attr HR")
    out = {"model": args.model, "domain": args.domain, "n_matched": len(matched), "levels": {}}
    for level in ["PROFILE", "K=1", "K=3", "K=5"]:
        if level not in have:
            continue
        mean_hr, per = _mean_hr(have[level], matched, human_resp, vc, centroids_by_attr, group_by_attr)
        lo, hi = _bootstrap_ci(have[level], matched, human_resp, vc, centroids_by_attr, group_by_attr, n=args.bootstrap)
        print(f"{level:8} {mean_hr:6.1%}  [{lo:5.1%}, {hi:5.1%}]   " + " ".join(f"{r:.0%}" for r in per))
        out["levels"][level] = {"mean_hr": mean_hr, "ci95": [lo, hi], "per_attr_hr": per}

    outpath = f"{mdir}/dialogue_length_ablation_{args.domain}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {outpath}")
    print("Expected trend (dose-response): PROFILE >= K=1 >= K=3 >= K=5 (more turns -> less homogenization).")


if __name__ == "__main__":
    main()
