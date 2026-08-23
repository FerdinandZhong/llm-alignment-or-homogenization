"""Confound control for the dialogue-length ablation.

Question: does truncating a synthetic dialogue to its first K turns mainly reduce
DEMOGRAPHIC signal, or INDIVIDUATING context? If demographic content is front-loaded
(present already at K=1), then the length ablation varies individuating context while
holding the demographic prototype trigger roughly constant — so a change in
homogenization across K reflects individuation, not demographic quantity.

Reuses the App-G demographic keyword set (scripts/prism_density_correlation.py).
Measures, over the first-K user turns: demographic keyword count, user word count,
density (KW/word), and what fraction of the FULL-dialogue demographic keywords are
already present at each K.

Usage: python scripts/dialogue_length_density_control.py [--domain career]
"""
import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prism_density_correlation import DEMO_PATTERN  # App-G demographic keyword regex

DOMAIN_FILE = {
    "career": "datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl",
    "investment": "datasets/wvs_generated_dialogues/investment_advice/all_samples.jsonl",
}
TURNS = [1, 3, 5]


def _first_k(msgs, k):
    sub = msgs[: 2 * k]
    user_txt = " ".join(m["content"] for m in sub if m.get("role") == "user" and m.get("content"))
    kw = len(DEMO_PATTERN.findall(user_txt))
    return kw, max(len(user_txt.split()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="career")
    args = ap.parse_args()

    per = {k: {"kw": [], "words": []} for k in TURNS}
    with open(DOMAIN_FILE[args.domain]) as f:
        for line in f:
            (_uid, msgs), = json.loads(line).items()
            for k in TURNS:
                kw, w = _first_k(msgs, k)
                per[k]["kw"].append(kw)
                per[k]["words"].append(w)

    full = per[5]["kw"]
    out = {"domain": args.domain, "n_dialogues": len(full), "by_turns": {}}
    print(f"{'K':>2} {'demoKW':>7} {'userWords':>10} {'density':>8} {'%offull':>8}")
    for k in TURNS:
        kw = st.mean(per[k]["kw"])
        w = st.mean(per[k]["words"])
        ratios = [c / f for c, f in zip(per[k]["kw"], full) if f > 0]  # ratio undefined when full has 0 KW
        frac = st.mean(ratios) if ratios else float("nan")
        out["by_turns"][k] = {"mean_demo_kw": kw, "mean_user_words": w,
                              "density": kw / w, "frac_of_full_demo": frac}
        print(f"{k:>2} {kw:>7.2f} {w:>10.0f} {kw / w:>8.4f} {frac:>7.0%}")

    outpath = f"wvs_values_results/gpt-5.1/dialogue_length_density_control_{args.domain}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {outpath}")
    print("Interpretation: demographics front-loaded (high %offull at K=1) + density falling with K")
    print("=> truncation varies individuating context, not demographic quantity.")


if __name__ == "__main__":
    main()
