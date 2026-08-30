"""
Gradio chat interface for human validation study.

Participants have two sequential chat sessions (career advice, investment advice)
with GPT-4o-mini. Conversations are saved in the same JSONL format as the
synthetic dialogues for direct compatibility with the evaluation pipeline.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/human_validation/app.py --share

The app accepts a participant ID via URL query parameter (?pid=XXXX) when
redirected from Qualtrics, or via manual text input.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "participants"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
MIN_TURNS = 8
MAX_TURNS = 15

CAREER_SYSTEM_PROMPT = (
    "You are a helpful career advisor. Provide thoughtful, practical career "
    "guidance based on what the user shares. Listen carefully and give "
    "personalized advice. Keep responses concise (2-3 paragraphs max). "
    "Do not proactively ask about age, gender, education, or other demographic "
    "details unless the user brings them up naturally."
)

INVESTMENT_SYSTEM_PROMPT = (
    "You are a helpful investment advisor. Provide thoughtful, practical "
    "investment guidance based on what the user shares. Listen carefully and "
    "give personalized advice. Keep responses concise (2-3 paragraphs max). "
    "Do not proactively ask about age, gender, education, or other demographic "
    "details unless the user brings them up naturally."
)

CAREER_OPENER = (
    "Welcome! I'm a career advisor here to help you think through your "
    "career goals and next steps. What's on your mind regarding your career?"
)

INVESTMENT_OPENER = (
    "Welcome! I'm an investment advisor here to help you think through your "
    "financial goals and investment strategies. What's on your mind regarding "
    "your investments or financial planning?"
)

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def call_llm(message: str, history: list[dict], system_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_participant_data(pid: str, career_history: list, invest_history: list):
    """Save both dialogues for one participant."""
    career_dialogue = []
    for turn in career_history:
        career_dialogue.append({"role": "user", "content": turn["user"]})
        career_dialogue.append({"role": "chatbot", "content": turn["bot"]})

    invest_dialogue = []
    for turn in invest_history:
        invest_dialogue.append({"role": "user", "content": turn["user"]})
        invest_dialogue.append({"role": "chatbot", "content": turn["bot"]})

    data = {
        "participant_id": pid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "career_dialogue": career_dialogue,
        "investment_dialogue": invest_dialogue,
    }

    out_path = DATA_DIR / f"{pid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path)


def generate_completion_token(pid: str) -> str:
    """Deterministic 8-char token for Qualtrics verification."""
    return hashlib.sha256(f"validation-{pid}".encode()).hexdigest()[:8].upper()


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------


def build_app():
    with gr.Blocks(
        title="Conversation Study",
        theme=gr.themes.Soft(),
        css=".turn-counter { font-size: 14px; color: #666; margin-bottom: 8px; }",
    ) as app:

        # ----- state -----
        participant_id = gr.State("")
        phase = gr.State("start")  # start → career → investment → done
        career_history = gr.State([])  # list of {"user": ..., "bot": ...}
        invest_history = gr.State([])

        # ============================================================
        # PAGE 1 — Start
        # ============================================================
        with gr.Column(visible=True) as start_page:
            gr.Markdown(
                "## Conversation Study\n\n"
                "Thank you for participating! In this study you will have **two "
                "short conversations** with an AI assistant:\n\n"
                "1. **Career advice** (~10 messages)\n"
                "2. **Investment advice** (~10 messages)\n\n"
                "Please chat naturally — there are no right or wrong things to "
                "say. The conversations typically take **15-20 minutes** total.\n\n"
                "Enter your **Participant ID** (provided by Qualtrics) below to begin."
            )
            pid_input = gr.Textbox(
                label="Participant ID",
                placeholder="Paste your participant ID here",
            )
            start_btn = gr.Button("Begin Career Chat", variant="primary", size="lg")
            start_error = gr.Markdown("", visible=False)

        # ============================================================
        # PAGE 2 — Career Chat
        # ============================================================
        with gr.Column(visible=False) as career_page:
            gr.Markdown("## Part 1: Career Advice Chat")
            career_turn_display = gr.Markdown(
                "Turn 0 / ~10", elem_classes=["turn-counter"]
            )
            career_chatbot = gr.Chatbot(
                value=[[None, CAREER_OPENER]],
                height=450,
                label="Career Advisor",
            )
            with gr.Row():
                career_input = gr.Textbox(
                    placeholder="Type your message...",
                    label="Your message",
                    scale=4,
                    show_label=False,
                )
                career_send = gr.Button("Send", variant="primary", scale=1)
            career_end_btn = gr.Button(
                "End Career Chat & Continue →",
                variant="secondary",
                visible=False,
                size="lg",
            )
            career_min_notice = gr.Markdown("", visible=False)

        # ============================================================
        # PAGE 3 — Investment Chat
        # ============================================================
        with gr.Column(visible=False) as invest_page:
            gr.Markdown("## Part 2: Investment Advice Chat")
            invest_turn_display = gr.Markdown(
                "Turn 0 / ~10", elem_classes=["turn-counter"]
            )
            invest_chatbot = gr.Chatbot(
                value=[[None, INVESTMENT_OPENER]],
                height=450,
                label="Investment Advisor",
            )
            with gr.Row():
                invest_input = gr.Textbox(
                    placeholder="Type your message...",
                    label="Your message",
                    scale=4,
                    show_label=False,
                )
                invest_send = gr.Button("Send", variant="primary", scale=1)
            invest_end_btn = gr.Button(
                "End Investment Chat & Finish →",
                variant="secondary",
                visible=False,
                size="lg",
            )
            invest_min_notice = gr.Markdown("", visible=False)

        # ============================================================
        # PAGE 4 — Done
        # ============================================================
        with gr.Column(visible=False) as done_page:
            gr.Markdown("## Study Complete!")
            completion_display = gr.Markdown("")
            gr.Markdown(
                "Please **copy the completion token** above and paste it into "
                "the Qualtrics survey to continue with the values questionnaire.\n\n"
                "Thank you for your participation!"
            )

        # ============================================================
        # Event handlers
        # ============================================================

        def on_start(pid_text, request: gr.Request):
            # Try URL param first, fall back to text input
            pid = pid_text.strip()
            if not pid and request:
                params = dict(request.query_params) if request.query_params else {}
                pid = params.get("pid", "")
            if not pid:
                return (
                    gr.update(visible=True),   # start_page
                    gr.update(visible=False),  # career_page
                    "",                        # participant_id
                    "start",                   # phase
                    gr.update(value="**Please enter a Participant ID.**", visible=True),
                )
            return (
                gr.update(visible=False),  # start_page
                gr.update(visible=True),   # career_page
                pid,                       # participant_id
                "career",                  # phase
                gr.update(visible=False),  # start_error
            )

        start_btn.click(
            on_start,
            inputs=[pid_input],
            outputs=[start_page, career_page, participant_id, phase, start_error],
        )

        # ----- Career chat -----

        def career_respond(message, chat_display, history):
            if not message.strip():
                return "", chat_display, history, gr.update(), gr.update(), gr.update()

            bot_reply = call_llm(message, history, CAREER_SYSTEM_PROMPT)
            history = history + [{"user": message, "bot": bot_reply}]
            chat_display = chat_display + [[message, bot_reply]]
            n = len(history)

            turn_text = f"**Turn {n} / ~10**"
            if n >= MAX_TURNS:
                turn_text += "  — Maximum reached. Please end the conversation."
            show_end = n >= MIN_TURNS
            input_interactive = n < MAX_TURNS
            notice = (
                gr.update(
                    value=f"*Please send at least {MIN_TURNS - n} more message(s) "
                    f"before ending.*",
                    visible=(n < MIN_TURNS),
                )
            )

            return (
                "" if input_interactive else gr.update(interactive=False),
                chat_display,
                history,
                gr.update(value=turn_text),
                gr.update(visible=show_end),
                notice,
            )

        career_send.click(
            career_respond,
            inputs=[career_input, career_chatbot, career_history],
            outputs=[
                career_input,
                career_chatbot,
                career_history,
                career_turn_display,
                career_end_btn,
                career_min_notice,
            ],
        )
        career_input.submit(
            career_respond,
            inputs=[career_input, career_chatbot, career_history],
            outputs=[
                career_input,
                career_chatbot,
                career_history,
                career_turn_display,
                career_end_btn,
                career_min_notice,
            ],
        )

        def on_career_end():
            return (
                gr.update(visible=False),  # career_page
                gr.update(visible=True),   # invest_page
                "investment",              # phase
            )

        career_end_btn.click(
            on_career_end,
            outputs=[career_page, invest_page, phase],
        )

        # ----- Investment chat -----

        def invest_respond(message, chat_display, history):
            if not message.strip():
                return "", chat_display, history, gr.update(), gr.update(), gr.update()

            bot_reply = call_llm(message, history, INVESTMENT_SYSTEM_PROMPT)
            history = history + [{"user": message, "bot": bot_reply}]
            chat_display = chat_display + [[message, bot_reply]]
            n = len(history)

            turn_text = f"**Turn {n} / ~10**"
            if n >= MAX_TURNS:
                turn_text += "  — Maximum reached. Please end the conversation."
            show_end = n >= MIN_TURNS
            input_interactive = n < MAX_TURNS
            notice = (
                gr.update(
                    value=f"*Please send at least {MIN_TURNS - n} more message(s) "
                    f"before ending.*",
                    visible=(n < MIN_TURNS),
                )
            )

            return (
                "" if input_interactive else gr.update(interactive=False),
                chat_display,
                history,
                gr.update(value=turn_text),
                gr.update(visible=show_end),
                notice,
            )

        invest_send.click(
            invest_respond,
            inputs=[invest_input, invest_chatbot, invest_history],
            outputs=[
                invest_input,
                invest_chatbot,
                invest_history,
                invest_turn_display,
                invest_end_btn,
                invest_min_notice,
            ],
        )
        invest_input.submit(
            invest_respond,
            inputs=[invest_input, invest_chatbot, invest_history],
            outputs=[
                invest_input,
                invest_chatbot,
                invest_history,
                invest_turn_display,
                invest_end_btn,
                invest_min_notice,
            ],
        )

        def on_invest_end(pid, c_hist, i_hist):
            save_participant_data(pid, c_hist, i_hist)
            token = generate_completion_token(pid)
            return (
                gr.update(visible=False),  # invest_page
                gr.update(visible=True),   # done_page
                "done",                    # phase
                gr.update(
                    value=f"### Your Completion Token\n\n"
                    f"# `{token}`\n\n"
                    f"Career chat: {len(c_hist)} turns | "
                    f"Investment chat: {len(i_hist)} turns\n\n"
                    f"Data saved successfully."
                ),
            )

        invest_end_btn.click(
            on_invest_end,
            inputs=[participant_id, career_history, invest_history],
            outputs=[invest_page, done_page, phase, completion_display],
        )

        # Pre-fill participant ID from URL on page load
        def prefill_pid(request: gr.Request):
            if request and request.query_params:
                params = dict(request.query_params)
                return params.get("pid", "")
            return ""

        app.load(prefill_pid, outputs=[pid_input])

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human validation chat interface")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", type=str, default=None, help="Override chat model")
    args = parser.parse_args()

    if args.model:
        MODEL = args.model

    app = build_app()
    app.launch(share=args.share, server_port=args.port)
