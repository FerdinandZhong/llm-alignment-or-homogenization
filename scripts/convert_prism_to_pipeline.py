"""
Convert PRISM conversations to the pipeline's dialogue + demographics format.

Outputs:
  datasets/prism_validation/prism_dialogues.jsonl  -- dialogue input for BA_dialogue mode
  datasets/prism_validation/prism_demographics.csv -- demographics for BA_user comparison
"""

import json
import csv
import os
from pathlib import Path
from datasets import Dataset

# ── paths ──────────────────────────────────────────────────────────────────
PRISM_ARROW = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "datasets",
    "HannahRoseKirk___prism-alignment", "conversations", "0.0.0",
    "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6", "prism-alignment-train.arrow",
)
PRISM_FINAL = "literature_review/prism_final_100.json"
OUT_DIR = Path("datasets/prism_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIALOGUES = OUT_DIR / "prism_dialogues.jsonl"
OUT_DEMOGRAPHICS = OUT_DIR / "prism_demographics.csv"

# ── demographic mappings ────────────────────────────────────────────────────

AGE_MIDPOINTS = {
    "18-24 years old": 21,
    "25-34 years old": 29,
    "35-44 years old": 39,
    "45-54 years old": 49,
    "55-64 years old": 59,
    "65 and over": 70,
    "65+ years old": 70,
}

EDUCATION_MAP = {
    "No formal education": "Primary education",
    "Some primary school": "Primary education",
    "Primary school graduate": "Primary education",
    "Some high school": "Lower secondary education",
    "High school graduate": "Upper secondary education",
    "Some college": "Post-secondary non-tertiary education",
    "Vocational training": "Post-secondary non-tertiary education",
    "Technical/vocational training": "Post-secondary non-tertiary education",
    "University Bachelors Degree": "University undergraduate level",
    "Bachelors degree": "University undergraduate level",
    "Some graduate school": "University postgraduate level",
    "Graduate degree": "University postgraduate level",
    "Masters degree": "University postgraduate level",
    "Doctoral degree": "University postgraduate level",
}

EMPLOYMENT_TO_OCCUPATION = {
    "Employed full time": "Skilled worker",
    "Employed part-time": "Skilled worker",
    "Employed part time": "Skilled worker",
    "Self-employed": "Self-employed",
    "Unemployed, seeking work": "Unemployed",
    "Unemployed, not seeking work": "Unemployed",
    "Student": "Student",
    "Retired": "Retired",
    "Stay at home parent/caregiver": "Housewife",
    "Disabled": "Unemployed",
    "Other": "Other",
}

SOCIOECONOMIC_DEFAULT = "Middle class"


def map_age(age_str: str) -> int:
    for key, val in AGE_MIDPOINTS.items():
        if key.lower() in age_str.lower():
            return val
    # try to extract a number
    import re
    nums = re.findall(r"\d+", age_str)
    return int(nums[0]) if nums else 35


def map_education(edu_str: str) -> str:
    for key, val in EDUCATION_MAP.items():
        if key.lower() in edu_str.lower():
            return val
    return "Upper secondary education"


def map_occupation(emp_str: str) -> str:
    for key, val in EMPLOYMENT_TO_OCCUPATION.items():
        if key.lower() in emp_str.lower():
            return val
    return "Skilled worker"


def map_immigration(status_str: str) -> str:
    if status_str.lower() in ("no", "false", "not immigrant", "native"):
        return "not immigrant"
    return "immigrant"


def extract_dialogue(conv_history: list) -> list:
    """
    Build alternating user/chatbot turns from PRISM conversation_history.
    Each turn has multiple model candidates; pick if_chosen=True, else first model response.
    """
    turns_by_idx: dict = {}
    for msg in conv_history:
        t = msg.get("turn", 0)
        role = msg.get("role", "")
        if role == "user":
            turns_by_idx.setdefault(t, {"user": None, "model": None})
            turns_by_idx[t]["user"] = msg["content"]
        elif role == "model":
            turns_by_idx.setdefault(t, {"user": None, "model": None})
            chosen = msg.get("if_chosen", False)
            # prefer chosen; keep first as fallback
            if chosen or turns_by_idx[t]["model"] is None:
                turns_by_idx[t]["model"] = msg["content"]

    dialogue = []
    for t in sorted(turns_by_idx.keys()):
        entry = turns_by_idx[t]
        if entry["user"]:
            dialogue.append({"role": "user", "content": entry["user"]})
        if entry["model"]:
            dialogue.append({"role": "chatbot", "content": entry["model"]})
    return dialogue


def main():
    # load selected 80 conversations
    with open(PRISM_FINAL) as f:
        selected = json.load(f)
    selected_ids = {e["conversation_id"]: e for e in selected}
    print(f"Loaded {len(selected_ids)} selected PRISM conversations")

    # load full PRISM dataset
    ds = Dataset.from_file(PRISM_ARROW)
    conv_map = {row["conversation_id"]: row for row in ds}
    print(f"Loaded {len(conv_map)} PRISM conversations from Arrow")

    demographics_rows = []
    dialogue_records = {}

    for prism_id_str, meta in selected_ids.items():
        if prism_id_str not in conv_map:
            print(f"  WARNING: {prism_id_str} not found in PRISM dataset, skipping")
            continue

        # create synthetic D_INTERVIEW id (9-digit, starting at 900000001)
        idx = list(selected_ids.keys()).index(prism_id_str)
        d_interview = 900000001 + idx

        demo = meta["demographics"]
        age_raw = demo.get("age", "35")
        gender = demo.get("gender", "Male")
        country = demo.get("reside_country") or demo.get("birth_country", "Unknown")
        continent = demo.get("continent", "Unknown")
        immigration = map_immigration(demo.get("immigration_status", "No"))
        education = map_education(demo.get("education", "Upper secondary education"))
        occupation = map_occupation(demo.get("employment", "Employed full time"))
        age = map_age(age_raw)

        demographics_rows.append({
            "D_INTERVIEW": d_interview,
            "gender": gender,
            "age": age,
            "place_of_residence": country,
            "continent_of_residence": continent,
            "immigration_status": immigration,
            "highest_level_of_education": education,
            "socioeconomic_status": SOCIOECONOMIC_DEFAULT,
            "occupation_group": occupation,
            # keep original PRISM fields for reference
            "_prism_conversation_id": prism_id_str,
            "_prism_user_id": meta["user_id"],
            "_prism_topic": meta["topic"],
        })

        # build dialogue
        full_conv = conv_map[prism_id_str]
        conv_history = full_conv.get("conversation_history", [])
        dialogue = extract_dialogue(conv_history)
        dialogue_records[d_interview] = dialogue

    print(f"Converted {len(demographics_rows)} entries")

    # write demographics CSV
    fieldnames = [
        "D_INTERVIEW", "gender", "age", "place_of_residence",
        "continent_of_residence", "immigration_status",
        "highest_level_of_education", "socioeconomic_status", "occupation_group",
        "_prism_conversation_id", "_prism_user_id", "_prism_topic",
    ]
    with open(OUT_DEMOGRAPHICS, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(demographics_rows)
    print(f"Wrote demographics: {OUT_DEMOGRAPHICS}")

    # write dialogue JSONL
    with open(OUT_DIALOGUES, "w") as f:
        for d_id, turns in dialogue_records.items():
            f.write(json.dumps({d_id: turns}) + "\n")
    print(f"Wrote dialogues: {OUT_DIALOGUES}")

    # quick sanity check
    print("\n--- Sample ---")
    sample_id = list(dialogue_records.keys())[0]
    print(f"D_INTERVIEW: {sample_id}")
    print(f"Turns: {len(dialogue_records[sample_id])}")
    for t in dialogue_records[sample_id][:4]:
        print(f"  [{t['role']}] {t['content'][:100]}...")


if __name__ == "__main__":
    main()
