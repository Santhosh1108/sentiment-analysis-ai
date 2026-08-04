"""
Sentiment Analysis — a clean, professional text sentiment tool
built on a HuggingFace transformer pipeline.

Run with:
    streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st
from transformers import pipeline

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# DESIGN TOKENS / CSS — quiet, professional, data-forward
# ----------------------------------------------------------------------------
ACCENT = "#3B5BDB"      # indigo — primary accent
POSITIVE = "#1E824C"    # muted green
NEGATIVE = "#B3261E"    # muted red
INK = "#1B1F27"         # near-black text
MUTED = "#5B6472"       # secondary text
BG = "#F6F7F9"          # page background
CARD = "#FFFFFF"        # card background
BORDER = "#E4E7EC"      # hairline border

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}

    .stApp {{
        background: {BG};
    }}

    /* Kill default streamlit chrome noise */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    /* ---- Header ---- */
    .app-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 0.25rem;
    }}
    .app-title {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {INK};
        margin-bottom: 0.15rem;
    }}
    .app-subtitle {{
        font-size: 0.98rem;
        color: {MUTED};
        margin-bottom: 1.6rem;
    }}

    /* ---- Card ---- */
    .card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        margin-bottom: 1.1rem;
    }}
    .card h4 {{
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 0.9rem;
    }}

    /* ---- Text area ---- */
    .stTextArea textarea {{
        background-color: {CARD} !important;
        color: {INK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        font-size: 0.98rem !important;
    }}
    .stTextArea textarea:focus {{
        border: 1px solid {ACCENT} !important;
        box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
    }}

    /* ---- Primary button ---- */
    div.stButton > button {{
        background: {ACCENT};
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0.5rem 1.2rem;
        transition: background 0.15s ease;
    }}
    div.stButton > button:hover {{
        background: #2F49B8;
        color: #FFFFFF;
    }}
    div.stButton > button:focus:not(:active) {{
        outline: 2px solid rgba(59,91,219,0.35);
    }}

    /* ---- Secondary / ghost buttons (target the 2nd+ button columns) ---- */
    .ghost-btn > button {{
        background: transparent;
        color: {MUTED};
        border: 1px solid {BORDER};
    }}
    .ghost-btn > button:hover {{
        background: #F0F1F3;
        color: {INK};
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.6rem;
    }}

    /* ---- Metrics ---- */
    div[data-testid="stMetric"] {{
        background: {BG};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
    }}

    /* ---- Verdict pill ---- */
    .verdict {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        border-radius: 999px;
        padding: 0.35rem 0.95rem;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
    }}
    .verdict-positive {{
        background: rgba(30,130,76,0.10);
        color: {POSITIVE};
        border: 1px solid rgba(30,130,76,0.25);
    }}
    .verdict-negative {{
        background: rgba(179,38,30,0.10);
        color: {NEGATIVE};
        border: 1px solid rgba(179,38,30,0.25);
    }}
    .verdict-dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: currentColor;
    }}

    /* ---- Sentiment gauge (signature element) ---- */
    .gauge-track {{
        position: relative;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(90deg, {NEGATIVE} 0%, #E8B923 50%, {POSITIVE} 100%);
        margin: 0.6rem 0 0.3rem 0;
    }}
    .gauge-marker {{
        position: absolute;
        top: -5px;
        width: 3px;
        height: 18px;
        background: {INK};
        border-radius: 2px;
        transform: translateX(-50%);
    }}
    .gauge-labels {{
        display: flex;
        justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: {MUTED};
    }}

    hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 1.1rem 0;
    }}

    .caption {{
        font-size: 0.8rem;
        color: {MUTED};
    }}

    .history-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid {BORDER};
        font-size: 0.85rem;
    }}
    .history-row:last-child {{ border-bottom: none; }}
    .history-text {{
        color: {INK};
        max-width: 68%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .history-score {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: {MUTED};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("<div class='app-eyebrow'>NLP · TEXT CLASSIFICATION</div>", unsafe_allow_html=True)
st.markdown("<div class='app-title'>Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Classify the sentiment of any piece of text using a fine-tuned transformer model.</div>",
    unsafe_allow_html=True,
)

with st.spinner("Loading model..."):
    classifier = load_model()

# ----------------------------------------------------------------------------
# SIDEBAR — session log
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Session Log")

    total = len(st.session_state.history)
    positives = sum(1 for h in st.session_state.history if h["label"] == "POSITIVE")
    negatives = total - positives

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Positive", positives)
    c3.metric("Negative", negatives)

    st.markdown("<hr/>", unsafe_allow_html=True)

    if st.session_state.history:
        for h in reversed(st.session_state.history[-10:]):
            color = POSITIVE if h["label"] == "POSITIVE" else NEGATIVE
            snippet = (h["text"][:38] + "…") if len(h["text"]) > 38 else h["text"]
            st.markdown(
                f"""
                <div class='history-row'>
                    <span class='history-text'>{snippet}</span>
                    <span class='history-score' style='color:{color}'>{h['score']*100:.0f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<hr/>", unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export CSV",
            data=csv,
            file_name="sentiment_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear log", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("<span class='caption'>No entries yet.</span>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>Input Text</h4>", unsafe_allow_html=True)

    text = st.text_area(
        "Text to analyze",
        key="input_text",
        height=170,
        placeholder="Paste or type a sentence, review, comment, or feedback to analyze...",
        label_visibility="collapsed",
    )

    word_count = len(text.split())
    char_count = len(text)
    st.markdown(
        f"<span class='caption'>{word_count} words · {char_count} characters</span>",
        unsafe_allow_html=True,
    )

    st.write("")
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_clicked = st.button("Analyze", use_container_width=True)
    with btn_col2:
        st.markdown("<div class='ghost-btn'>", unsafe_allow_html=True)
        if st.button("Clear", use_container_width=True):
            st.session_state.input_text = ""
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Results ----
    if analyze_clicked:
        if not text.strip():
            st.warning("Enter some text before running the analysis.")
        else:
            with st.spinner("Analyzing..."):
                result = classifier(text[:512])[0]

            label = result["label"]
            score = result["score"]

            st.session_state.history.append(
                {
                    "text": text,
                    "label": label,
                    "score": score,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
            )

            # gauge position: 0 = fully negative, 100 = fully positive
            gauge_pos = (score * 100) if label == "POSITIVE" else (100 - score * 100)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h4>Result</h4>", unsafe_allow_html=True)

            verdict_class = "verdict-positive" if label == "POSITIVE" else "verdict-negative"
            verdict_text = "Positive" if label == "POSITIVE" else "Negative"
            st.markdown(
                f"""
                <div class='verdict {verdict_class}'>
                    <span class='verdict-dot'></span>{verdict_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{score*100:.1f}%")
            m2.metric("Predicted class", label.title())

            st.markdown(
                f"""
                <div class='gauge-track'>
                    <div class='gauge-marker' style='left:{gauge_pos}%;'></div>
                </div>
                <div class='gauge-labels'>
                    <span>Negative</span>
                    <span>Neutral</span>
                    <span>Positive</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Raw model output"):
                st.json(result)

            st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4>About this tool</h4>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <span class='caption'>
        This app runs binary sentiment classification (positive / negative) on
        input text using a DistilBERT model fine-tuned on the SST-2 dataset.
        </span>
        <br/><br/>
        <span class='caption'><b>Model</b><br/>distilbert-base-uncased-finetuned-sst-2-english</span>
        <br/><br/>
        <span class='caption'><b>Limitations</b><br/>
        Binary output only (no neutral class). Truncates input beyond 512 tokens.
        Accuracy may drop on sarcasm, mixed sentiment, or domain-specific jargon.
        </span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
