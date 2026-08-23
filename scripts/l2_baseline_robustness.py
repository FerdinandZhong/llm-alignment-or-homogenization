"""
CANONICAL human-baseline computation (Table 3 metric) — camera-ready.

This is the authoritative baseline script for the paper. It computes the natural
human clustering rate using the ACTUAL Table 3 metric (L2 distance, componentwise
median group centroid, HR formula I(d_model < d_human)) and reports it with a
bootstrap confidence interval, together with robustness to (a) sampling strategy,
(b) demographic partition, and (c) centroid definition (median vs mean).

Headline result: the natural human baseline is 50.0% (essentially chance), stable
across all three variations; per-attribute 95% CIs are ~[45, 56]. The value 51.0%
that appears in the submission is this 50.0% estimate plus a +1% offset that has no
methodological basis and is dropped in the camera-ready (the point estimate 50.0%
lies well below the 55-83% model rates, so no conclusion changes).

NOTE: `llm_behavior_adaptation/value_measurement/human_resampling_baseline.py` uses
a DIFFERENT metric (EMD distance, group-vs-global centroid) and must NOT be used for
the Table 3 baseline; it is retained only for the group-vs-global diagnostic.

The paper's baseline (main.tex line 177): "split-half resampling of the 1,000
WVS human vectors, one half treated as 'model' predictions, averaged over 100
random splits." We operationalize this exactly with the L2 HR formula:

  For each of 100 random splits of the users in a demographic attribute:
    - designate one half as the "model" pool, the other half as "humans"
    - group centroids are the componentwise median over ALL users (as the
      models' centroids are), or the MEAN in the mean-centroid variant
    - for each human-half user u in group g, draw a same-group user u' from the
      model pool as the pseudo-"model" prediction, and score
          I( ||S_{u'} - S_g||_2  <  ||S_u - S_g||_2 )
  HR_baseline = mean indicator over users, averaged over splits.

Leave-one-out variant: the pseudo-"model" prediction for u is a random OTHER
same-group user (no split), scored the same way.

All human data only; no model outputs, no inference.

Usage:
    python scripts/l2_baseline_robustness.py
Output: wvs_values_results/l2_baseline_robustness.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from permutation_test_homogenization import (  # noqa: E402
    _get_human_responses,
    _uid_group_map,
    _l2_distance,
)
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import (  # noqa: E402
    ATTRIBUTES,
    ValuesComparison,
)

DATASET_DIR = "datasets/wvs_benchmarks"


def _median_centroids(uid_to_group, human_resp, all_questions):
    """Componentwise median group centroids (paper's definition), clamped to scale."""
    from llm_behavior_adaptation.value_measurement.formulas import componentwise_centroid
    grouped = {}
    for uid, g in uid_to_group.items():
        if uid in human_resp:
            grouped.setdefault(g, []).append(human_resp[uid])
    return {g: componentwise_centroid(v, all_questions) for g, v in grouped.items()}


def _mean_centroids(uid_to_group, human_resp, all_questions):
    """Componentwise MEAN group centroids (alternative centroid definition)."""
    grouped = {}
    for uid, g in uid_to_group.items():
        if uid in human_resp:
            grouped.setdefault(g, []).append(human_resp[uid])
    centroids = {}
    for g, vecs in grouped.items():
        c = {}
        for q in all_questions:
            vals = []
            for v in vecs:
                x = v.get(q)
                try:
                    xf = float(x)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(xf):
                    vals.append(xf)
            if vals:
                c[q] = float(np.mean(vals))
        centroids[g] = c
    return centroids


def _split_half_baseline(valid, uid_to_group, human_resp, centroids, all_questions, rng, n_splits=100):
    """Split-half: one half = 'model' pool; score same-group partner vs own answer."""
    by_group = {}
    for u in valid:
        by_group.setdefault(uid_to_group[u], []).append(u)

    rates = []
    for _ in range(n_splits):
        shuffled = list(rng.permutation(valid))
        half = len(shuffled) // 2
        model_pool = set(shuffled[:half])
        human_half = shuffled[half:]

        model_by_group = {}
        for u in model_pool:
            model_by_group.setdefault(uid_to_group[u], []).append(u)

        toward, total = 0, 0
        for u in human_half:
            g = uid_to_group[u]
            if g not in centroids or g not in model_by_group:
                continue
            u_model = model_by_group[g][rng.integers(0, len(model_by_group[g]))]
            d_model = _l2_distance(human_resp[u_model], centroids[g], all_questions)
            d_human = _l2_distance(human_resp[u], centroids[g], all_questions)
            if d_model < d_human:
                toward += 1
            total += 1
        if total:
            rates.append(toward / total)
    return rates


def _loo_baseline(valid, uid_to_group, human_resp, centroids, all_questions, rng):
    """Leave-one-out flavor: pseudo-'model' = a random OTHER same-group user."""
    by_group = {}
    for u in valid:
        by_group.setdefault(uid_to_group[u], []).append(u)

    toward, total = 0, 0
    for u in valid:
        g = uid_to_group[u]
        if g not in centroids:
            continue
        others = [x for x in by_group[g] if x != u]
        if not others:
            continue
        u_model = others[rng.integers(0, len(others))]
        d_model = _l2_distance(human_resp[u_model], centroids[g], all_questions)
        d_human = _l2_distance(human_resp[u], centroids[g], all_questions)
        if d_model < d_human:
            toward += 1
        total += 1
    return (toward / total) if total else float("nan")


def main():
    user_profile = pd.read_csv(f"{DATASET_DIR}/sampled_demographic_features.csv")
    user_value = pd.read_csv(f"{DATASET_DIR}/sampled_values_df.csv")
    with open(f"{DATASET_DIR}/picked_questions.json", encoding="utf-8") as f:
        picked = json.load(f)

    vc = ValuesComparison(
        user_profile_dataset=user_profile,
        user_value_dataset=user_value,
        ba_user_results={},
        ba_dialogue_career_results={},
        ba_dialogue_investment_results={},
        picked_questions=picked,
        results_output_path="",
        verbose=0,
    )
    human_resp = _get_human_responses(vc)
    all_q = vc.all_questions
    common = sorted(human_resp.keys())

    rng = np.random.default_rng(42)
    per_attr = {}
    for attr in ATTRIBUTES:
        uid_to_group = _uid_group_map(vc, attr, common)
        valid = [u for u in common if u in uid_to_group]
        if len(valid) < 20:
            continue
        med_c = _median_centroids(uid_to_group, human_resp, all_q)
        mean_c = _mean_centroids(uid_to_group, human_resp, all_q)

        sh_med = _split_half_baseline(valid, uid_to_group, human_resp, med_c, all_q, rng)
        loo_med = _loo_baseline(valid, uid_to_group, human_resp, med_c, all_q, rng)
        sh_mean = _split_half_baseline(valid, uid_to_group, human_resp, mean_c, all_q, rng)

        per_attr[attr] = {
            "split_half_median_mean": round(float(np.mean(sh_med)) * 100, 1),
            "split_half_median_ci": [round(float(np.percentile(sh_med, 2.5)) * 100, 1),
                                     round(float(np.percentile(sh_med, 97.5)) * 100, 1)],
            "loo_median": round(loo_med * 100, 1),
            "split_half_mean_centroid": round(float(np.mean(sh_mean)) * 100, 1),
            "n_valid": len(valid),
        }

    def _avg(key):
        vals = [v[key] for v in per_attr.values() if key in v]
        return round(float(np.mean(vals)), 1) if vals else None

    ci_lo = round(float(np.mean([v["split_half_median_ci"][0] for v in per_attr.values()])), 1)
    ci_hi = round(float(np.mean([v["split_half_median_ci"][1] for v in per_attr.values()])), 1)

    summary = {
        "headline_baseline": _avg("split_half_median_mean"),
        "headline_ci_95": [ci_lo, ci_hi],
        "split_half_median_across_attrs": _avg("split_half_median_mean"),
        "loo_median_across_attrs": _avg("loo_median"),
        "split_half_mean_centroid_across_attrs": _avg("split_half_mean_centroid"),
        "paper_submission_value": 51.0,
        "paper_submission_note": "50.0 estimate + a +1% offset with no methodological basis; drop for camera-ready",
        "metric": "L2 distance, HR formula I(d_model<d_human), same as Table 3",
    }

    out = {"summary": summary, "per_attribute": per_attr}
    outp = Path("wvs_values_results/l2_baseline_robustness.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nL2 human-baseline robustness (Table 3 metric)\n" + "=" * 60)
    print(f"{'Attribute':<26} {'SH/med':>7} {'95% CI':>14} {'LOO':>6} {'SH/mean':>8}")
    print("-" * 66)
    for a, v in per_attr.items():
        ci = f"[{v['split_half_median_ci'][0]:.1f},{v['split_half_median_ci'][1]:.1f}]"
        print(f"{a:<26} {v['split_half_median_mean']:>6.1f}% {ci:>14} {v['loo_median']:>5.1f}% {v['split_half_mean_centroid']:>7.1f}%")
    print("-" * 66)
    print(f"{'MEAN across attributes':<26} {summary['split_half_median_across_attrs']:>6.1f}% "
          f"{'':>14} {summary['loo_median_across_attrs']:>5.1f}% {summary['split_half_mean_centroid_across_attrs']:>7.1f}%")
    print(f"\nCANONICAL baseline (camera-ready): {summary['headline_baseline']:.1f}% "
          f"(95% CI ~[{ci_lo:.1f}, {ci_hi:.1f}])   metric: {summary['metric']}")
    print(f"Submission value 51.0% = 50.0% estimate + a +1% offset (no methodological basis; drop for camera-ready).")
    print(f"Written to {outp}")


if __name__ == "__main__":
    main()
