"""
Compute per-user BA_none VAA (Value Alignment Accuracy) for all models.

The BA_none condition uses a fixed prompt with no user profile. Because all
responses are deterministic (same prompt → same answer), the 20 recorded runs
are treated as repeated samples of a single default-culture vector. We take the
modal/average option_id per question, then compute Pearson(default_vector,
user_i_WVS_ground_truth) for every user in the 1000-user dataset.

This produces a per-user Pearson r that is directly comparable to the existing
BA_user per-user VAA (mean of per-user Pearson correlations).
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent

MODELS = {
    "GPT-5.1":        "gpt-5.1",
    "Qwen2.5-7B":     "Qwen2.5-7B-Instruct",
    "Llama-3.1-8B":   "Llama-3.1-8B-Instruct",
    "Qwen2.5-72B":    "Qwen2.5-72B-Instruct",
    "DeepSeek-V3":    "DeepSeek-V3",
    "QwQ-32B":        "QwQ-32B",
    "Llama-3.1-70B":  "Llama-3.1-70B-Instruct",
}

HUMAN_CSV   = ROOT / "datasets/wvs_benchmarks/sampled_values_df.csv"
PICKED_QS   = ROOT / "datasets/wvs_benchmarks/picked_questions.json"


def load_human_data():
    """Return {str(D_INTERVIEW): {Q_id: float}} for all 1000 users."""
    import csv
    users = {}
    with open(HUMAN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = str(row["D_INTERVIEW"])
            q_map = {}
            for k, v in row.items():
                if k.startswith("Q") and v.strip():
                    try:
                        q_map[k] = float(v)
                    except ValueError:
                        pass
            users[uid] = q_map
    return users


def load_picked_qids():
    with open(PICKED_QS) as f:
        picked = json.load(f)
    qids = set()
    for cat_qs in picked.values():
        qids.update(cat_qs.keys())
    return qids


def parse_ba_none_responses(jsonl_path):
    """
    Parse BA_none JSONL → list of {Q_id: option_id} dicts (one per run).
    Structure: {user_idx: {category: [{Q_id: {option_id, reason}}]}}
    """
    responses = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = list(row.keys())[0]
            cats = row[uid]
            q_map = {}
            for cat_questions in cats.values():
                for q_entry in cat_questions:
                    for qid, ans in q_entry.items():
                        if isinstance(ans, dict) and "option_id" in ans and ans["option_id"] != -1:
                            q_map[qid] = ans["option_id"]
                        # else: parse failure [-1, reason] or missing — skip
            responses.append(q_map)
    return responses


def consensus_vector(responses, picked_qids):
    """
    Build consensus {Q_id: modal_option_id} from repeated BA_none runs.
    Uses mode per question; falls back to mean if all values distinct.
    """
    all_qids = picked_qids & set(responses[0].keys())
    result = {}
    for qid in all_qids:
        vals = [r[qid] for r in responses if qid in r]
        if not vals:
            continue
        counter = Counter(vals)
        result[qid] = counter.most_common(1)[0][0]
    return result


def pearson(x_dict, y_dict):
    """
    Pearson r between two {Q_id: value} dicts on their common keys.
    Returns nan if fewer than 2 valid pairs.
    """
    keys = sorted(x_dict.keys() & y_dict.keys())
    if len(keys) < 2:
        return float("nan")
    xs = [float(x_dict[k]) for k in keys]
    ys = [float(y_dict[k]) for k in keys]
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 2:
        return float("nan")
    r, _ = stats.pearsonr(xs, ys)
    return float(r)


def load_ba_user_vaa(model_dir):
    """
    Extract BA_user per-user Pearson VAA from experiments_results.json.
    Returns float or None.
    """
    results_path = ROOT / "wvs_values_results" / model_dir / "experiments_results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        d = json.load(f)
    try:
        return d["against_human"]["ba_user_against_human"][
            "correlation_with_human_pearson"
        ]["avg_correlation"]
    except (KeyError, TypeError):
        return None


def main():
    human_data  = load_human_data()
    picked_qids = load_picked_qids()

    print(f"Loaded {len(human_data)} human users, {len(picked_qids)} picked questions\n")

    rows = []
    for label, model_dir in MODELS.items():
        none_dir = ROOT / "wvs_values_results" / model_dir / "none_values_results"
        jsonl_files = list(none_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"[SKIP] {label}: no BA_none JSONL found in {none_dir}")
            continue

        # Use the file with the most rows (handles partial runs)
        jsonl_path = max(jsonl_files, key=lambda p: p.stat().st_size)
        responses = parse_ba_none_responses(jsonl_path)
        if not responses:
            print(f"[SKIP] {label}: empty JSONL")
            continue

        default_vec = consensus_vector(responses, picked_qids)

        # Check consistency across runs
        consistency = None
        if len(responses) > 1:
            same = sum(
                1 for r in responses[1:]
                if all(r.get(q) == responses[0].get(q) for q in default_vec)
            )
            consistency = f"{same}/{len(responses)-1} runs identical to first"

        # Per-user Pearson against all 1000 human users
        per_user_r = []
        for uid, human_q_map in human_data.items():
            r = pearson(default_vec, human_q_map)
            if not np.isnan(r):
                per_user_r.append(r)

        ba_none_vaa     = float(np.mean(per_user_r)) if per_user_r else float("nan")
        ba_none_median  = float(np.median(per_user_r)) if per_user_r else float("nan")
        ba_user_vaa     = load_ba_user_vaa(model_dir)
        delta           = (ba_user_vaa - ba_none_vaa) if ba_user_vaa is not None else None

        rows.append({
            "label":        label,
            "ba_none_vaa":  ba_none_vaa,
            "ba_none_med":  ba_none_median,
            "ba_user_vaa":  ba_user_vaa,
            "delta":        delta,
            "n_users":      len(per_user_r),
            "n_ba_none_runs": len(responses),
            "consistency":  consistency,
        })

        print(f"{label}:")
        print(f"  BA_none runs: {len(responses)} ({consistency or 'single run'})")
        print(f"  Default vec Qs: {len(default_vec)}")
        print(f"  BA_none per-user VAA: mean={ba_none_vaa:.3f}  median={ba_none_median:.3f}  (n={len(per_user_r)})")
        if ba_user_vaa is not None:
            direction = "Profile helps" if delta > 0 else "Profile hurts"
            print(f"  BA_user per-user VAA: {ba_user_vaa:.3f}  delta={delta:+.3f}  → {direction}")
        print()

    # Summary table
    print("=" * 75)
    print(f"{'Model':<18} {'BA_none VAA':>12} {'BA_user VAA':>12} {'Delta':>8}  Direction")
    print("-" * 75)
    for r in rows:
        bu   = f"{r['ba_user_vaa']:.3f}" if r["ba_user_vaa"] is not None else "N/A"
        d    = f"{r['delta']:+.3f}"       if r["delta"] is not None       else "N/A"
        dirn = ""
        if r["delta"] is not None:
            dirn = "Profile helps" if r["delta"] > 0 else "Profile hurts"
        print(f"{r['label']:<18} {r['ba_none_vaa']:>12.3f} {bu:>12} {d:>8}  {dirn}")
    print("=" * 75)

    # Save results
    out_path = ROOT / "wvs_values_results" / "ba_none_vaa_comparison.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
