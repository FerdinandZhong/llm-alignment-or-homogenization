"""
Generate a Qualtrics QSF survey file for the human validation study.

Reads WVS questions from picked_questions.json and produces a complete .qsf
file that can be imported directly into Qualtrics.

Survey structure:
  Block 1: Consent
  Block 2: Demographics (7 fields)
  Block 3: Chat task instructions + redirect placeholder
  Block 4: WVS Values Survey (55 questions, grouped by category)
  Block 5: Exit survey

Usage:
    python scripts/human_validation/generate_qualtrics_survey.py
    # produces scripts/human_validation/human_validation_survey.qsf
"""

import json
import os
from pathlib import Path

QUESTIONS_PATH = Path("datasets/wvs_benchmarks/picked_questions.json")
OUTPUT_PATH = Path(__file__).parent / "human_validation_survey.qsf"

SURVEY_NAME = "LLM Value Alignment - Human Validation Study"
SURVEY_ID = "SV_3KlnRzHdmEaV4bk"
OWNER_ID = "UR_3KlnRzHdmEaV4bk"
RS_ID = "RS_3KlnRzHdmEaV4bk"


def make_question_id(index: int) -> str:
    return f"QID{index}"


def make_mc_question(qid: str, text: str, choices: dict, **kwargs) -> dict:
    return {
        "SurveyID": SURVEY_ID,
        "Element": "SQ",
        "PrimaryAttribute": qid,
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": text,
            "QuestionType": "MC",
            "Selector": "SAVR",
            "SubSelector": "TX",
            "DataExportTag": kwargs.get("export_tag", qid),
            "QuestionDescription": text[:100],
            "Choices": choices,
            "ChoiceOrder": list(choices.keys()),
            "Validation": {
                "Settings": {
                    "ForceResponse": "ON",
                    "ForceResponseType": "ON",
                    "Type": "None",
                }
            },
            "Language": [],
            "QuestionID": qid,
        },
    }


def make_text_entry(qid: str, text: str, **kwargs) -> dict:
    return {
        "SurveyID": SURVEY_ID,
        "Element": "SQ",
        "PrimaryAttribute": qid,
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": text,
            "QuestionType": "TE",
            "Selector": "SL",
            "DataExportTag": kwargs.get("export_tag", qid),
            "QuestionDescription": text[:100],
            "Validation": {
                "Settings": {
                    "ForceResponse": kwargs.get("force", "ON"),
                    "ForceResponseType": kwargs.get("force", "ON"),
                    "Type": "None",
                }
            },
            "Language": [],
            "QuestionID": qid,
        },
    }


def make_descriptive(qid: str, text: str) -> dict:
    return {
        "SurveyID": SURVEY_ID,
        "Element": "SQ",
        "PrimaryAttribute": qid,
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "QuestionText": text,
            "QuestionType": "DB",
            "Selector": "TB",
            "QuestionDescription": "Descriptive text",
            "Language": [],
            "QuestionID": qid,
        },
    }


def build_survey() -> dict:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    question_elements = []
    qid_counter = 1

    # ===================================================================
    # BLOCK 1: CONSENT
    # ===================================================================
    consent_qids = []

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_descriptive(
        qid,
        "<h2>Information Sheet &amp; Consent</h2>"
        "<p>You are invited to participate in a research study about how AI "
        "chatbots interact with different users. This study is conducted by "
        "researchers at [Anonymous Institution].</p>"
        "<h3>What you will do:</h3>"
        "<ol>"
        "<li>Answer a few questions about yourself (demographics)</li>"
        "<li>Have two short conversations (~10 messages each) with an AI chatbot "
        "about career advice and investment advice</li>"
        "<li>Answer a values questionnaire (55 multiple-choice questions)</li>"
        "<li>Answer one brief exit question</li>"
        "</ol>"
        "<h3>Time &amp; Compensation:</h3>"
        "<p>The study takes approximately <b>30-35 minutes</b>. You will be "
        "compensated at the advertised rate.</p>"
        "<h3>Data &amp; Privacy:</h3>"
        "<p>Your responses are anonymous. We collect demographic information "
        "and conversation logs for research purposes only. No personally "
        "identifiable information is stored. Data will be used in aggregate "
        "for academic publication.</p>"
        "<h3>Voluntary Participation:</h3>"
        "<p>Participation is voluntary. You may withdraw at any time without penalty.</p>",
    ))
    consent_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid,
        "Do you consent to participate in this study?",
        {"1": {"Display": "Yes, I consent to participate"},
         "2": {"Display": "No, I do not consent"}},
        export_tag="consent",
    ))
    consent_qids.append(qid)

    # ===================================================================
    # BLOCK 2: DEMOGRAPHICS
    # ===================================================================
    demo_qids = []

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid, "What is your gender?",
        {"1": {"Display": "Male"}, "2": {"Display": "Female"}, "3": {"Display": "Other"}},
        export_tag="gender",
    ))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_text_entry(qid, "What is your age?", export_tag="age"))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_text_entry(
        qid, "What is your country of residence?", export_tag="place_of_residence",
    ))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid, "What is your continent of residence?",
        {"1": {"Display": "Africa"}, "2": {"Display": "Asia"},
         "3": {"Display": "Europe"}, "4": {"Display": "North America"},
         "5": {"Display": "South America"}, "6": {"Display": "Oceania"}},
        export_tag="continent_of_residence",
    ))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid, "Are you an immigrant in your current country of residence?",
        {"1": {"Display": "Yes, I am an immigrant"},
         "2": {"Display": "No, I am not an immigrant"}},
        export_tag="immigration_status",
    ))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid, "What is your highest level of education?",
        {"1": {"Display": "Lower secondary education"},
         "2": {"Display": "Upper secondary education"},
         "3": {"Display": "Post-secondary non-tertiary education"},
         "4": {"Display": "Short-cycle tertiary education"},
         "5": {"Display": "Bachelor or equivalent"},
         "6": {"Display": "Master or equivalent"},
         "7": {"Display": "Doctoral or equivalent"}},
        export_tag="highest_level_of_education",
    ))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid, "How would you describe your socioeconomic status?",
        {"1": {"Display": "Working class"}, "2": {"Display": "Lower middle class"},
         "3": {"Display": "Middle class"}, "4": {"Display": "Upper middle class"},
         "5": {"Display": "Upper class"}},
        export_tag="socioeconomic_status",
    ))
    demo_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid, "What best describes your current occupation?",
        {"1": {"Display": "Manager"}, "2": {"Display": "Professional"},
         "3": {"Display": "Technician"}, "4": {"Display": "Clerical"},
         "5": {"Display": "Service / Sales"}, "6": {"Display": "Skilled worker"},
         "7": {"Display": "Machine operator"}, "8": {"Display": "Elementary occupation"},
         "9": {"Display": "Higher administrative"}, "10": {"Display": "Retired"},
         "11": {"Display": "Student"}, "12": {"Display": "Not sure / Other"}},
        export_tag="occupation_group",
    ))
    demo_qids.append(qid)

    # ===================================================================
    # BLOCK 3: CHAT TASK
    # ===================================================================
    chat_qids = []

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_descriptive(
        qid,
        "<h2>Chat Task</h2>"
        "<p>You will now have <b>two short conversations</b> with an AI chatbot:</p>"
        "<ol>"
        "<li><b>Career advice</b> — chat about your career goals and challenges</li>"
        "<li><b>Investment advice</b> — chat about your financial goals and strategies</li>"
        "</ol>"
        "<p>Each conversation should be about <b>10 messages</b> (8 minimum). "
        "Please chat naturally — there are no right or wrong things to say.</p>"
        "<p><b>Important:</b> Click the link below to open the chat interface in a "
        "new tab. Your Participant ID will be passed automatically. After completing "
        "both conversations, you will receive a <b>completion token</b>. Return here "
        "and enter it below to continue.</p>"
        "<p style='font-size:18px'><b>Chat link:</b> "
        "<a href='${e://Field/chat_url}' target='_blank'>"
        "Open Chat Interface &rarr;</a></p>"
        "<p><i>If the link does not work, open this URL manually and enter your "
        "Participant ID: ${e://Field/chat_base_url}</i></p>",
    ))
    chat_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_text_entry(
        qid,
        "Please enter the <b>completion token</b> you received after finishing both conversations:",
        export_tag="completion_token",
    ))
    chat_qids.append(qid)

    # ===================================================================
    # BLOCK 4: WVS VALUES SURVEY
    # ===================================================================
    wvs_qids = []

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_descriptive(
        qid,
        "<h2>Values Questionnaire</h2>"
        "<p>Please answer the following questions about your personal values and "
        "opinions. There are no right or wrong answers — we are interested in "
        "your genuine views.</p>"
        "<p>This section contains <b>55 questions</b> grouped by topic. "
        "Each question uses a numbered scale described in the question text.</p>",
    ))
    wvs_qids.append(qid)

    for category_name, questions in picked_questions.items():
        qid = make_question_id(qid_counter); qid_counter += 1
        question_elements.append(make_descriptive(qid, f"<h3>{category_name}</h3>"))
        wvs_qids.append(qid)

        for q_key, q_data in questions.items():
            qid = make_question_id(qid_counter); qid_counter += 1
            scale_min = q_data["answer_scale_min"]
            scale_max = q_data["answer_scale_max"]
            choices = {str(v): {"Display": str(v)} for v in range(scale_min, scale_max + 1)}
            question_elements.append(make_mc_question(
                qid, q_data["question"], choices, export_tag=q_key,
            ))
            wvs_qids.append(qid)

    # ===================================================================
    # BLOCK 5: EXIT SURVEY
    # ===================================================================
    exit_qids = []

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_mc_question(
        qid,
        "During the conversations, did you intentionally share personal details "
        "about yourself (e.g., your age, occupation, education, country)?",
        {"1": {"Display": "Yes, I intentionally shared personal details"},
         "2": {"Display": "No, I did not share personal details"},
         "3": {"Display": "I am not sure / some came up naturally"}},
        export_tag="shared_personal_details",
    ))
    exit_qids.append(qid)

    qid = make_question_id(qid_counter); qid_counter += 1
    question_elements.append(make_text_entry(
        qid, "(Optional) Any comments about your experience?",
        export_tag="exit_comments", force="OFF",
    ))
    exit_qids.append(qid)

    # ===================================================================
    # BLOCKS ELEMENT — single element, all blocks in Payload dict
    # ===================================================================
    block_defs = [
        ("BL_1", "Consent", consent_qids),
        ("BL_2", "Demographics", demo_qids),
        ("BL_3", "Chat Task", chat_qids),
        ("BL_4", "WVS Values Survey", wvs_qids),
        ("BL_5", "Exit Survey", exit_qids),
    ]
    blocks_payload = []
    for blk_id, desc, q_ids in block_defs:
        blocks_payload.append({
            "Type": "Standard",
            "SubType": "",
            "Description": desc,
            "ID": blk_id,
            "BlockElements": [{"Type": "Question", "QuestionID": q} for q in q_ids],
            "Options": {"BlockLocking": "false", "RandomizeQuestions": "false"},
        })

    blocks_element = {
        "SurveyID": SURVEY_ID,
        "Element": "BL",
        "PrimaryAttribute": "Survey Blocks",
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": blocks_payload,
    }

    # ===================================================================
    # SURVEY FLOW — embedded data + blocks + end
    # ===================================================================
    flow_items = [
        {
            "Type": "EmbeddedData",
            "FlowID": "FL_1",
            "EmbeddedData": [
                {"Description": "participant_id", "Type": "Custom",
                 "Field": "participant_id", "VariableType": "String",
                 "DataVisibility": []},
                {"Description": "chat_base_url", "Type": "Custom",
                 "Field": "chat_base_url", "VariableType": "String",
                 "DataVisibility": []},
                {"Description": "chat_url", "Type": "Custom",
                 "Field": "chat_url", "VariableType": "String",
                 "DataVisibility": []},
            ],
        },
    ]
    for i, (blk_id, _, _) in enumerate(block_defs):
        flow_items.append({"Type": "Block", "ID": blk_id, "FlowID": f"FL_{i+2}"})
    flow_items.append({
        "Type": "EndSurvey", "FlowID": f"FL_{len(block_defs)+2}",
    })

    flow_element = {
        "SurveyID": SURVEY_ID,
        "Element": "FL",
        "PrimaryAttribute": "Survey Flow",
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "Type": "Root",
            "FlowID": "FL_0",
            "Flow": flow_items,
            "Properties": {"Count": len(flow_items)},
        },
    }

    # ===================================================================
    # SURVEY OPTIONS
    # ===================================================================
    options_element = {
        "SurveyID": SURVEY_ID,
        "Element": "SO",
        "PrimaryAttribute": "Survey Options",
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "BackButton": "false",
            "SaveAndContinue": "true",
            "SurveyProtection": "PublicSurvey",
            "BallotBoxStuffingPrevention": "false",
            "NoIndex": "Yes",
            "SecureResponseFiles": "true",
            "SurveyExpiration": "None",
            "SurveyTermination": "DefaultMessage",
            "Header": "",
            "Footer": "",
            "ProgressBarDisplay": "Text",
            "PartialData": "+1 week",
            "PreviousButton": " ← ",
            "NextButton": " → ",
            "SurveyTitle": SURVEY_NAME,
        },
    }

    # ===================================================================
    # RESPONSE SET
    # ===================================================================
    rs_element = {
        "SurveyID": SURVEY_ID,
        "Element": "RS",
        "PrimaryAttribute": RS_ID,
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": None,
    }

    # ===================================================================
    # ASSEMBLE
    # ===================================================================
    all_elements = [flow_element, blocks_element, options_element, rs_element] + question_elements

    qsf = {
        "SurveyEntry": {
            "SurveyID": SURVEY_ID,
            "SurveyName": SURVEY_NAME,
            "SurveyDescription": "Human validation study for LLM value alignment research",
            "SurveyOwnerID": OWNER_ID,
            "SurveyBrandID": OWNER_ID,
            "DivisionID": None,
            "SurveyLanguage": "EN",
            "SurveyActiveResponseSet": RS_ID,
            "SurveyStatus": "Inactive",
            "SurveyStartDate": "0000-00-00 00:00:00",
            "SurveyExpirationDate": "0000-00-00 00:00:00",
            "SurveyCreationDate": "2026-04-25 00:00:00",
            "CreatorID": OWNER_ID,
            "LastModified": "2026-04-25 00:00:00",
            "LastAccessed": "0000-00-00 00:00:00",
            "LastActivated": "0000-00-00 00:00:00",
            "Deleted": None,
        },
        "SurveyElements": all_elements,
    }

    return qsf


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[2])

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        picked_questions = json.load(f)

    survey = build_survey()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(survey, f, ensure_ascii=False, indent=2)

    n_questions = sum(len(qs) for qs in picked_questions.values())
    n_elements = len(survey["SurveyElements"])
    print(f"Generated QSF with {n_questions} WVS questions + demographics + consent + exit")
    print(f"Total elements: {n_elements}")
    print(f"Output: {OUTPUT_PATH}")
