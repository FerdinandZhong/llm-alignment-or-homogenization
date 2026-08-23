"""
Direct PROFILE run-to-run variability for GPT-5.1 (ARR rebuttal, Reviewer xdGu W1).

The submission stored a single PROFILE generation per user, so run-to-run (decoding)
variability of the PROFILE VAA could not be reported from the cached outputs. This
script re-generates PROFILE answers for a SLICE of users K times, using the SAME
prompt (prompts/direct_question.json), the SAME per-user profile rendering
(retrieve_user_profile_wvs + render_json), and the SAME JSON parsing as the production
pipeline (wvs_values_prediction.py). It then computes the aggregate VAA of each pass
and reports the run-to-run standard deviation.

Because PROFILE aggregates INDEPENDENT per-user generations, the aggregate VAA SD
scales as 1/sqrt(n_users). We therefore also extrapolate the measured slice SD to the
full n=1000 evaluation:  SD_1000 = SD_slice * sqrt(n_slice / 1000).

Decoding matches production: response_format=json_object, no temperature override
(server default), consistent with how the NONE/PROFILE passes were generated.

Usage:
    python scripts/profile_run_to_run_gpt.py --n-users 25 --k 5 --concurrency 20
Env/inputs:
    --token-file ~/tokens/litellm_key   (LiteLLM gateway key)
    --base-url   https://ai-gateway.cloudops.cloudera.com/v1
    --model      gpt-5.1
Outputs:
    wvs_values_results/gpt51_runtorun/pass_{k}.jsonl   (raw, one line per user)
    wvs_values_results/gpt51_runtorun/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from openai import AsyncOpenAI  # noqa: E402

from llm_behavior_adaptation.dialogue_dataset_creation.generation_utils import (  # noqa: E402
    render_json,
    retrieve_user_profile_wvs,
)
from compute_ba_none_vaa import load_human_data, load_picked_qids, pearson  # noqa: E402

DATASET_DIR = ROOT / "datasets/wvs_benchmarks"
PROMPT_PATH = ROOT / "llm_behavior_adaptation/value_measurement/prompts/direct_question.json"


def build_question_index() -> list[tuple[str, str]]:
    """Return [(qid, question_text), ...] over the picked questions."""
    with open(DATASET_DIR / "picked_questions.json", encoding="utf-8") as f:
        picked = json.load(f)
    out = []
    for cat in picked.values():
        for qid, qdetails in cat.items():
            out.append((qid, qdetails["question"]))
    return out


def parse_option_id(content: str) -> int:
    """Mirror wvs_values_prediction._llm_output_processing JSON parsing."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", content)
        try:
            obj = json.loads(sanitized)
        except json.JSONDecodeError:
            return -1
    try:
        return int(obj["option_id"])
    except (KeyError, TypeError, ValueError):
        return -1


async def query_one(client, model, sem, prompt_template, user_profile, question_text):
    msgs = [dict(m) for m in prompt_template]
    msgs[1]["content"] = msgs[1]["content"].format(user_details=user_profile)
    msgs[2]["content"] = msgs[2]["content"].format(values_question=question_text)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=msgs, response_format={"type": "json_object"}
                )
                return parse_option_id(resp.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"  [warn] call failed after retries: {e}", file=sys.stderr)
                    return -1
                await asyncio.sleep(1.5 * (attempt + 1))
    return -1


async def run_pass(client, model, sem, prompt_template, users, questions):
    """One full PROFILE pass -> {uid: {qid: option_id}}. All pairs run concurrently."""
    jobs = []  # (uid, qid, coro)
    for uid, profile in users:
        for qid, qtext in questions:
            jobs.append((uid, qid, query_one(client, model, sem, prompt_template, profile, qtext)))
    oids = await asyncio.gather(*[c for _, _, c in jobs])
    per_user: dict = {uid: {} for uid, _ in users}
    for (uid, qid, _), oid in zip(jobs, oids):
        if oid != -1:
            per_user[uid][qid] = oid
    return per_user


def pass_vaa(per_user, human_data, picked_qids):
    """Aggregate VAA (mean per-user Pearson) for one pass."""
    rs = []
    for uid, vec in per_user.items():
        v = {q: o for q, o in vec.items() if q in picked_qids}
        h = human_data.get(str(uid))
        if h is None or len(v) < 2:
            continue
        r = pearson(v, h)
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan"), len(rs)


async def main_async(args):
    key = Path(os.path.expanduser(args.token_file)).read_text().strip()
    client = AsyncOpenAI(api_key=key, base_url=args.base_url)

    with open(PROMPT_PATH, encoding="utf-8") as f:
        prompt_template = json.load(f)

    profiles = pd.read_csv(DATASET_DIR / "sampled_demographic_features.csv").head(args.n_users)
    users = [(str(row["D_INTERVIEW"]), render_json(retrieve_user_profile_wvs(row.to_dict())))
             for _, row in profiles.iterrows()]

    questions = build_question_index()
    human_data = load_human_data()
    picked_qids = load_picked_qids()
    sem = asyncio.Semaphore(args.concurrency)

    outdir = ROOT / "wvs_values_results" / "gpt51_runtorun"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Users={len(users)}  questions={len(questions)}  passes={args.k}  "
          f"(~{len(users) * len(questions) * args.k} calls)\n", flush=True)

    vaas = []
    for k in range(args.k):
        out_path = outdir / f"pass_{k}.jsonl"
        if out_path.exists():
            # resume: reload from disk
            per_user = {}
            with open(out_path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    uid = list(obj.keys())[0]
                    per_user[uid] = obj[uid]
            vaa, n = pass_vaa(per_user, human_data, picked_qids)
            print(f"pass {k}: VAA={vaa:.4f}  (n_users_scored={n})  [resumed from disk]", flush=True)
        else:
            per_user = await run_pass(client, args.model, sem, prompt_template, users, questions)
            with open(out_path, "w", encoding="utf-8") as f:
                for uid, vec in per_user.items():
                    f.write(json.dumps({uid: vec}) + "\n")
            vaa, n = pass_vaa(per_user, human_data, picked_qids)
            print(f"pass {k}: VAA={vaa:.4f}  (n_users_scored={n})", flush=True)
            if k < args.k - 1:
                await asyncio.sleep(5)  # brief cooldown between passes
        vaas.append(vaa)

    arr = np.asarray([v for v in vaas if np.isfinite(v)])
    sd_slice = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    sd_1000 = sd_slice * (args.n_users / 1000.0) ** 0.5

    summary = {
        "model": args.model,
        "n_users": args.n_users,
        "k_passes": args.k,
        "pass_vaas": [round(v, 4) for v in vaas],
        "vaa_mean": round(float(arr.mean()), 4),
        "vaa_sd_slice": round(sd_slice, 4),
        "vaa_range_slice": round(float(arr.max() - arr.min()), 4),
        "vaa_sd_extrapolated_n1000": round(sd_1000, 4),
        "note": "SD_1000 = SD_slice * sqrt(n_slice/1000); independent per-user decoding noise averages out.",
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"PROFILE run-to-run VAA (GPT-5.1, {args.n_users} users, {args.k} passes)")
    print(f"  mean VAA           : {summary['vaa_mean']:.4f}")
    print(f"  run-to-run SD      : {sd_slice:.4f}  (slice of {args.n_users})")
    print(f"  extrapolated n=1000: {sd_1000:.4f}")
    print("=" * 60)
    print(f"Written to {outdir/'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=25)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--base-url", default="https://ai-gateway.cloudops.cloudera.com/v1")
    ap.add_argument("--token-file", default="~/tokens/litellm_key")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
