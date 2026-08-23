"""
Experiment B: PRISM demographic signal density vs. centroid distance correlation.

Tests whether de-homogenization in PRISM real conversations is explained by
signal quantity (dilution account) or by format (label-compactness account).

- Demographic density: fraction of words in each conversation matching
  demographic attribute keywords (age, education, SES, occupation, origin, immigration).
- Centroid distance: L2 distance between model's WVS response vector and the
  per-question median of WVS users sharing the same continent_of_residence.
- Correlation: Spearman rho between density and distance across 80 users.

If rho > 0 (and significant): more demographic signal → more homogenization → dilution account.
If rho ≈ 0 (and non-significant): format suppresses retrieval independently of signal quantity.
"""

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── paths ────────────────────────────────────────────────────────────────────
PRISM_DIALOGUES   = ROOT / "datasets/prism_validation/prism_dialogues.jsonl"
PRISM_DEMOGRAPHICS = ROOT / "datasets/prism_validation/prism_demographics.csv"
WVS_VALUES        = ROOT / "datasets/wvs_benchmarks/sampled_values_df.csv"
WVS_DEMOGRAPHICS  = ROOT / "datasets/wvs_benchmarks/sampled_demographic_features.csv"

MODELS = {
    "gpt-5.1":                  "gpt-5.1",
    "Llama-3.1-70B-Instruct":   "Llama-3.1-70B",
    "QwQ-32B":                  "QwQ-32B",
    "Qwen2.5-72B-Instruct":     "Qwen2.5-72B",
    "DeepSeek-V3":              "DeepSeek-V3",
}

# ── demographic keyword sets ─────────────────────────────────────────────────
DEMO_KEYWORDS = {
    "age":        r"\b(year[s]? old|age[d]?|young|middle.aged|retired|elder|senior|teen)\b",
    "education":  r"\b(degree|university|college|bachelor|master|phd|graduate|school|diploma|educated)\b",
    "ses":        r"\b(income|salary|afford|wealth|poor|rich|middle class|working class|socioeconomic)\b",
    "occupation": r"\b(my job|my work|unemployed|employed|profession|career|my role|my position)\b",
    "origin":     r"\b(born in|from [a-z]+|nationality|country of origin|native|heritage)\b",
    "immigration":r"\b(immigrant|visa|citizen|citizenship|moved (to|from)|expat|foreigner)\b",
}

DEMO_PATTERN = re.compile(
    "|".join(f"(?P<{k}>{v})" for k, v in DEMO_KEYWORDS.items()),
    re.IGNORECASE,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def demographic_density(turns: list) -> float:
    """Keyword match count / total word count across all turns."""
    full_text = " ".join(t["content"] for t in turns if t.get("content"))
    words = len(full_text.split())
    if words == 0:
        return 0.0
    matches = len(DEMO_PATTERN.findall(full_text))
    return matches / words


def load_csv_as_dict(path: Path, key_col: str) -> dict:
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row[key_col]] = row
    return rows


def load_wvs_values(path: Path) -> dict:
    """Returns {D_INTERVIEW: {Q1: int, Q2: int, ...}}"""
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["D_INTERVIEW"]
            result[uid] = {k: round(float(v)) for k, v in row.items()
                           if k != "D_INTERVIEW" and v.strip() != ""}
    return result


def group_centroid(vectors: list[dict]) -> dict:
    """Per-question median across a list of {Qx: int} dicts."""
    from statistics import median
    all_qs = set(q for v in vectors for q in v)
    centroid = {}
    for q in all_qs:
        vals = [v[q] for v in vectors if q in v]
        if vals:
            centroid[q] = round(median(vals))
    return centroid


def l2_distance(a: dict, b: dict) -> float:
    """L2 distance between two response vectors on shared questions."""
    shared = set(a) & set(b)
    if not shared:
        return float("nan")
    return math.sqrt(sum((a[q] - b[q]) ** 2 for q in shared))


def load_model_responses(model_dir: Path) -> dict:
    """Load total_80.jsonl → {user_id: {Qx: int}}"""
    result = {}
    path = model_dir / "BA_dialogue_values_results/total_80.jsonl"
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            for uid, categories in entry.items():
                responses = {}
                for cat_questions in categories.values():
                    for q_entry in cat_questions:
                        for qid, val in q_entry.items():
                            if isinstance(val, dict) and "option_id" in val:
                                responses[qid] = int(val["option_id"])
                result[uid] = responses
    return result


def spearman_rho(x: list, y: list) -> tuple[float, float]:
    """Spearman rho and approximate p-value (t approximation)."""
    n = len(x)
    assert n == len(y) and n > 2
    rank_x = _ranks(x)
    rank_y = _ranks(y)
    d2 = sum((rx - ry) ** 2 for rx, ry in zip(rank_x, rank_y))
    rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
    # t-approximation for p-value
    import math
    t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    # two-tailed p from normal approx (large n ok; n=80 is borderline)
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return rho, p


def _ranks(vals: list) -> list:
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")

    # PRISM conversations
    prism_convos = {}
    with open(PRISM_DIALOGUES) as f:
        for line in f:
            prism_convos.update(json.loads(line))

    prism_demo = load_csv_as_dict(PRISM_DEMOGRAPHICS, "D_INTERVIEW")
    wvs_values = load_wvs_values(WVS_VALUES)
    wvs_demo   = load_csv_as_dict(WVS_DEMOGRAPHICS, "D_INTERVIEW")

    # Group WVS users by continent
    continent_groups: dict[str, list[dict]] = defaultdict(list)
    for uid, demo in wvs_demo.items():
        cont = demo.get("continent_of_residence", "").strip()
        if cont and uid in wvs_values:
            continent_groups[cont].append(wvs_values[uid])

    # Precompute continent centroids
    continent_centroids = {
        cont: group_centroid(vecs)
        for cont, vecs in continent_groups.items()
    }

    # Compute density per PRISM user
    densities = {}
    for uid, turns in prism_convos.items():
        densities[uid] = demographic_density(turns)

    print(f"\n{'Model':<30} {'rho':>8} {'p':>10} {'n':>5}  interpretation")
    print("-" * 65)

    for model_dir_name, label in MODELS.items():
        model_dir = ROOT / "wvs_values_results/prism_validation" / model_dir_name
        if not model_dir.exists():
            continue
        model_resp = load_model_responses(model_dir)

        pairs = []
        for uid in prism_convos:
            if uid not in model_resp or uid not in prism_demo:
                continue
            cont = prism_demo[uid].get("continent_of_residence", "").strip()
            centroid = continent_centroids.get(cont)
            if not centroid:
                continue
            dist = l2_distance(model_resp[uid], centroid)
            if math.isnan(dist):
                continue
            pairs.append((densities[uid], dist))

        if len(pairs) < 10:
            print(f"{label:<30} {'N/A':>8}  (n={len(pairs)})")
            continue

        dens_vals, dist_vals = zip(*pairs)
        rho, p = spearman_rho(list(dens_vals), list(dist_vals))

        if p < 0.05 and rho > 0:
            interp = "dilution account supported"
        elif p < 0.05 and rho < 0:
            interp = "unexpected: more signal → less homogenization"
        else:
            interp = "no significant correlation → compactness account"

        print(f"{label:<30} {rho:>8.3f} {p:>10.4f} {len(pairs):>5}  {interp}")

    print("\nInterpretation guide:")
    print("  rho > 0, p < 0.05 : more demographic mentions → more homogenization (dilution)")
    print("  rho ≈ 0, p > 0.05 : signal quantity does not predict homogenization (compactness)")


if __name__ == "__main__":
    main()
