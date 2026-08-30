"""
Export collected human validation data into formats compatible with the
existing evaluation pipeline.

Reads:
  - Participant JSON files from data/participants/
  - Qualtrics CSV export (demographics + WVS responses)

Produces:
  - human_validation_demographics.csv  (matches sampled_demographic_features.csv)
  - human_validation_values.csv        (matches sampled_values_df.csv)
  - career_advice/all_samples.jsonl    (matches dialogue JSONL format)
  - investment_advice/all_samples.jsonl

Usage:
    python scripts/human_validation/export_for_pipeline.py \
        --qualtrics-csv path/to/qualtrics_export.csv \
        --output-dir datasets/human_validation/
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PICKED_QUESTIONS_PATH = "datasets/wvs_benchmarks/picked_questions.json"

CONTINENT_MAP = {
    "1": "Africa", "2": "Asia", "3": "Europe",
    "4": "North America", "5": "South America", "6": "Oceania",
    "Africa": "Africa", "Asia": "Asia", "Europe": "Europe",
    "North America": "North America", "South America": "South America",
    "Oceania": "Oceania",
}

GENDER_MAP = {"1": "Male", "2": "Female", "3": "Other", "Male": "Male", "Female": "Female", "Other": "Other"}

IMMIGRATION_MAP = {
    "1": "immigrant",
    "2": "not immigrant",
    "Yes, I am an immigrant": "immigrant",
    "No, I am not an immigrant": "not immigrant",
    "immigrant": "immigrant",
    "not immigrant": "not immigrant",
}

EDUCATION_MAP = {
    "1": "Lower secondary education",
    "2": "Upper secondary education",
    "3": "Post-secondary non-tertiary education",
    "4": "Short-cycle tertiary education",
    "5": "Bachelor or equivalent",
    "6": "Master or equivalent",
    "7": "Doctoral or equivalent",
}

SES_MAP = {
    "1": "Working class",
    "2": "Lower middle class",
    "3": "Middle class",
    "4": "Upper middle class",
    "5": "Upper class",
}

OCCUPATION_MAP = {
    "1": "Manager", "2": "Professional", "3": "Technician",
    "4": "Clerical", "5": "Service / Sales", "6": "Skilled worker",
    "7": "Machine operator", "8": "Elementary occupation",
    "9": "Higher administrative", "10": "Retired", "11": "Student",
    "12": "not sure",
}


def load_qualtrics_csv(csv_path: str) -> dict[str, dict]:
    """Parse Qualtrics CSV export into per-participant records."""
    participants = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Qualtrics exports have 2 header rows (labels + import IDs); skip row 2
        header_row_2 = next(reader, None)
        if header_row_2 and all(v.startswith("{") or v == "" for v in header_row_2.values()):
            pass  # skipped the import ID row
        else:
            # wasn't a Qualtrics header row; this is real data
            pid = header_row_2.get("participant_id", "")
            if pid:
                participants[pid] = header_row_2

        for row in reader:
            pid = row.get("participant_id", "").strip()
            if not pid:
                continue
            consent = row.get("consent", "")
            if consent in ("2", "No, I do not consent"):
                continue
            participants[pid] = row

    return participants


def load_dialogue_files(data_dir: str) -> dict[str, dict]:
    """Load participant JSON files from the Gradio app."""
    dialogues = {}
    p_dir = Path(data_dir) / "data" / "participants"
    if not p_dir.exists():
        return dialogues
    for f in p_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        dialogues[data["participant_id"]] = data
    return dialogues


def normalize_demographic(row: dict) -> dict:
    """Convert Qualtrics response to pipeline-compatible demographic record."""
    return {
        "D_INTERVIEW": row.get("participant_id", ""),
        "gender": GENDER_MAP.get(row.get("gender", ""), row.get("gender", "")),
        "age": int(float(row.get("age", 0))) if row.get("age", "").strip() else 0,
        "place_of_residence": row.get("place_of_residence", ""),
        "continent_of_residence": CONTINENT_MAP.get(
            row.get("continent_of_residence", ""),
            row.get("continent_of_residence", ""),
        ),
        "immigration_status": IMMIGRATION_MAP.get(
            row.get("immigration_status", ""),
            row.get("immigration_status", ""),
        ),
        "highest_level_of_education": EDUCATION_MAP.get(
            row.get("highest_level_of_education", ""),
            row.get("highest_level_of_education", ""),
        ),
        "socioeconomic_status": SES_MAP.get(
            row.get("socioeconomic_status", ""),
            row.get("socioeconomic_status", ""),
        ),
        "occupation_group": OCCUPATION_MAP.get(
            row.get("occupation_group", ""),
            row.get("occupation_group", ""),
        ),
    }


def extract_wvs_responses(row: dict, question_ids: list[str]) -> dict:
    """Extract WVS question responses from Qualtrics row."""
    responses = {"D_INTERVIEW": row.get("participant_id", "")}
    for qid in question_ids:
        val = row.get(qid, "")
        try:
            responses[qid] = float(val) if val.strip() else ""
        except (ValueError, AttributeError):
            responses[qid] = ""
    return responses


def main():
    parser = argparse.ArgumentParser(description="Export human validation data for pipeline")
    parser.add_argument("--qualtrics-csv", required=True, help="Path to Qualtrics CSV export")
    parser.add_argument("--gradio-data-dir", default=str(Path(__file__).parent),
                        help="Path to human_validation/ directory with data/participants/")
    parser.add_argument("--output-dir", default="datasets/human_validation/",
                        help="Output directory for pipeline-compatible files")
    args = parser.parse_args()

    with open(PICKED_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        picked_questions = json.load(f)
    question_ids = [qid for cat in picked_questions.values() for qid in cat.keys()]

    qualtrics_data = load_qualtrics_csv(args.qualtrics_csv)
    dialogue_data = load_dialogue_files(args.gradio_data_dir)

    # Find participants with both survey AND dialogue data
    common_pids = sorted(set(qualtrics_data.keys()) & set(dialogue_data.keys()))
    print(f"Qualtrics participants: {len(qualtrics_data)}")
    print(f"Dialogue participants:  {len(dialogue_data)}")
    print(f"Matched participants:   {len(common_pids)}")

    if not common_pids:
        print("ERROR: No matched participants found. Check participant IDs.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "career_advice").mkdir(exist_ok=True)
    (out_dir / "investment_advice").mkdir(exist_ok=True)

    # --- Demographics CSV ---
    demo_path = out_dir / "human_validation_demographics.csv"
    demo_fields = [
        "D_INTERVIEW", "gender", "age", "place_of_residence",
        "continent_of_residence", "immigration_status",
        "highest_level_of_education", "socioeconomic_status", "occupation_group",
    ]
    with open(demo_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=demo_fields)
        writer.writeheader()
        for pid in common_pids:
            row = qualtrics_data[pid]
            row["participant_id"] = pid
            writer.writerow(normalize_demographic(row))
    print(f"Written: {demo_path}")

    # --- Values CSV ---
    values_path = out_dir / "human_validation_values.csv"
    values_fields = ["D_INTERVIEW"] + question_ids
    with open(values_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=values_fields)
        writer.writeheader()
        for pid in common_pids:
            row = qualtrics_data[pid]
            row["participant_id"] = pid
            writer.writerow(extract_wvs_responses(row, question_ids))
    print(f"Written: {values_path}")

    # --- Dialogue JSONL files ---
    career_path = out_dir / "career_advice" / "all_samples.jsonl"
    invest_path = out_dir / "investment_advice" / "all_samples.jsonl"

    with open(career_path, "w", encoding="utf-8") as fc, \
         open(invest_path, "w", encoding="utf-8") as fi:
        for pid in common_pids:
            d = dialogue_data[pid]
            fc.write(json.dumps({pid: d.get("career_dialogue", [])}, ensure_ascii=False) + "\n")
            fi.write(json.dumps({pid: d.get("investment_dialogue", [])}, ensure_ascii=False) + "\n")
    print(f"Written: {career_path}")
    print(f"Written: {invest_path}")

    print(f"\nExport complete. {len(common_pids)} participants ready for evaluation pipeline.")
    print(f"\nNext steps:")
    print(f"  1. Create YAML configs pointing to {out_dir}/")
    print(f"  2. Run wvs_values_prediction.py with these configs")
    print(f"  3. Run permutation_test_homogenization.py on results")


if __name__ == "__main__":
    main()
