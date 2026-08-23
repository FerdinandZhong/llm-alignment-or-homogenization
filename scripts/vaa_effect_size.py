"""
VAA effect sizes and confidence intervals (ARR rebuttal, Reviewer 3 W1).

Reviewer 3 notes that some PROFILE VAA gains are small and that, with n=1000,
tiny effects can yield very small p-values; they ask for effect sizes or
confidence intervals to support the claimed improvement.

This script computes, per model, the per-user change in Value Alignment Accuracy
from NONE to PROFILE:

    delta_u = r(profile_u, human_u) - r(none_default, human_u)

and reports:
  * mean delta (matches the published Delta column)
  * 95% bootstrap CI on the mean delta (5,000 resamples)
  * Cohen's d = mean(delta) / std(delta)  (paired effect size)
  * n users

All computation uses cached outputs; no new inference. The NONE default vector
is the consensus (modal option per question) over the 20 recorded NONE draws,
exactly as in scripts/compute_ba_none_vaa.py.

Usage:
    python scripts/vaa_effect_size.py
Output: wvs_values_results/vaa_effect_size.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compute_ba_none_vaa import (  # noqa: E402
    load_human_data,
    load_picked_qids,
    parse_ba_none_responses,
    consensus_vector,
    pearson,
)

ROOT = Path(__file__).parent.parent

MODELS = {
    "Llama-3.1-8B": "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B": "Llama-3.1-70B-Instruct",
    "Qwen2.5-7B": "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B": "Qwen2.5-72B-Instruct",
    "QwQ-32B": "QwQ-32B",
    "DeepSeek-V3": "DeepSeek-V3",
    "GPT-5.1": "gpt-5.1",
}


def load_profile_per_user(jsonl_path: Path, picked_qids: set) -> dict:
    """Parse PROFILE total_1000.jsonl into {uid: {qid: option_id}}."""
    per_user = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = list(row.keys())[0]
            q_map = {}
            for cat_questions in row[uid].values():
                for q_entry in cat_questions:
                    for qid, ans in q_entry.items():
                        if qid in picked_qids and isinstance(ans, dict):
                            oid = ans.get("option_id", -1)
                            if oid != -1:
                                q_map[qid] = oid
            per_user[str(uid)] = q_map
    return per_user


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple:
    """95% bootstrap CI on the mean of `deltas`."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = deltas[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_d_paired(deltas: np.ndarray) -> float:
    """Paired Cohen's d = mean(delta) / std(delta)."""
    sd = float(np.std(deltas, ddof=1))
    return float(np.mean(deltas) / sd) if sd > 0 else float("nan")


def main() -> None:
    human_data = load_human_data()
    picked_qids = load_picked_qids()
    print(f"Loaded {len(human_data)} human users, {len(picked_qids)} picked questions\n")

    results = {}
    for label, model_dir in MODELS.items():
        none_dir = ROOT / "wvs_values_results" / model_dir / "none_values_results"
        prof_path = ROOT / "wvs_values_results" / model_dir / "profile_values_results" / "total_1000.jsonl"
        none_files = list(none_dir.glob("*.jsonl"))
        if not none_files or not prof_path.exists():
            print(f"[SKIP] {label}: missing NONE or PROFILE results")
            continue

        none_file = max(none_files, key=lambda p: sum(1 for _ in open(p)))
        none_runs = parse_ba_none_responses(none_file)
        none_vec = consensus_vector(none_runs, picked_qids)
        prof_per_user = load_profile_per_user(prof_path, picked_qids)

        deltas = []
        for uid, human_vec in human_data.items():
            if uid not in prof_per_user:
                continue
            r_none = pearson(none_vec, human_vec)
            r_prof = pearson(prof_per_user[uid], human_vec)
            if np.isfinite(r_none) and np.isfinite(r_prof):
                deltas.append(r_prof - r_none)

        deltas = np.asarray(deltas, dtype=float)
        if len(deltas) < 10:
            print(f"[SKIP] {label}: too few paired users ({len(deltas)})")
            continue

        ci_lo, ci_hi = bootstrap_ci(deltas)
        d = cohens_d_paired(deltas)
        results[label] = {
            "mean_delta": round(float(np.mean(deltas)), 4),
            "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "cohens_d": round(d, 4),
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
            "n_users": int(len(deltas)),
        }

    # Report
    print(f"{'Model':<16} {'ΔVAA':>8} {'95% CI':>20} {'Cohen d':>9} {'CI≠0':>6}")
    print("-" * 64)
    for label, r in results.items():
        ci = f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
        flag = "yes" if r["ci_excludes_zero"] else "no"
        print(f"{label:<16} {r['mean_delta']:>+8.3f} {ci:>20} {r['cohens_d']:>9.3f} {flag:>6}")

    print("\nInterpretation: Cohen's d magnitude bands (Cohen 1988): 0.2 small, 0.5 medium, 0.8 large.")
    print("Small ΔVAA with a small d is the paper's point: the alignment gain is modest, yet")
    print("homogenization is large even when ΔVAA is near zero or negative (see Qwen2.5-7B).")

    out = ROOT / "wvs_values_results" / "vaa_effect_size.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
