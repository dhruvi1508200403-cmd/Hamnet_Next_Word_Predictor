"""
🎭 THE ELSINORE ORACLE 🎭
A haunted little Streamlit app that asks two ghosts — one built of LSTM bones,
one built of GRU sinew — to finish your sentence the way Hamlet might have.

Run with:
    streamlit run app.py

Expected files in the same folder:
    next_word_lestm.h5        -> the LSTM model
    next_word_GRU_RNN_1.h5    -> the GRU model
    tokenizer.pickle          -> the fitted Keras tokenizer
"""

import time
import random
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence


# ══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="The Elsinore Oracle",
    page_icon="💀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════
#  THEME / CSS / DOODLES
# ══════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&family=IM+Fell+English+SC&family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"]  {
    font-family: 'EB Garamond', serif;
}

.stApp {
    background: radial-gradient(circle at 20% 10%, #2b2320 0%, #17120f 45%, #0c0a08 100%);
    color: #e8dcc4;
}

/* Flickering candle title */
.oracle-title {
    font-family: 'UnifrakturCook', cursive;
    font-size: 4rem;
    text-align: center;
    color: #d8c48c;
    text-shadow: 0 0 8px rgba(255, 200, 100, 0.55), 0 0 22px rgba(255, 140, 40, 0.25);
    animation: flicker 3.2s infinite ease-in-out;
    margin-bottom: -0.4rem;
}
@keyframes flicker {
    0%, 19%, 21%, 23%, 54%, 56%, 100% { opacity: 1; }
    20%, 22%, 55% { opacity: 0.72; }
}

.oracle-subtitle {
    font-family: 'IM Fell English SC', serif;
    text-align: center;
    color: #a89a78;
    letter-spacing: 3px;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

/* Drifting ghost across the top */
.ghost-drift {
    position: relative;
    height: 40px;
    overflow: hidden;
}
.ghost-drift span {
    position: absolute;
    left: -10%;
    font-size: 1.6rem;
    animation: drift 14s linear infinite;
    opacity: 0.55;
}
@keyframes drift {
    from { left: -5%; transform: translateY(0px); }
    50%  { transform: translateY(-8px); }
    to   { left: 105%; transform: translateY(0px); }
}

/* Parchment scroll containers for results */
.scroll-box {
    background: linear-gradient(180deg, #efe3c0 0%, #e6d7ab 100%);
    color: #241a0e;
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 0 0 1px #7a5a2e inset, 0 8px 22px rgba(0,0,0,0.55);
    font-family: 'IM Fell English SC', serif;
    min-height: 190px;
    position: relative;
}
.scroll-box::before {
    content: "";
    position: absolute;
    inset: 6px;
    border: 1px dashed #7a5a2e88;
    border-radius: 4px;
    pointer-events: none;
}
.scroll-title {
    font-family: 'UnifrakturCook', cursive;
    font-size: 1.6rem;
    margin-bottom: 0.3rem;
    color: #3a2410;
}
.predicted-word {
    display: inline-block;
    font-size: 1.5rem;
    font-weight: 700;
    padding: 0.15rem 0.6rem;
    margin: 0.15rem 0.15rem;
    border-radius: 4px;
    background: #241a0e;
    color: #f2e6c5;
    animation: reveal 0.35s ease-out;
}
@keyframes reveal {
    from { opacity: 0; transform: translateY(6px) scale(0.9); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.verdict-box {
    text-align: center;
    font-family: 'IM Fell English SC', serif;
    font-size: 1.25rem;
    padding: 0.9rem;
    border-radius: 8px;
    margin-top: 1rem;
    border: 1px solid #7a5a2e;
    background: rgba(60, 45, 25, 0.35);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a140f 0%, #0c0a08 100%);
    border-right: 1px solid #4a3720;
}

.stButton>button {
    font-family: 'IM Fell English SC', serif;
    background: linear-gradient(180deg, #5c3b1e, #3a2410);
    color: #f2e6c5;
    border: 1px solid #a8895a;
    border-radius: 20px;
    padding: 0.5rem 1.2rem;
    letter-spacing: 1px;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(180deg, #7a5028, #4a2f14);
    border-color: #e8c88c;
    color: #fff3d6;
    box-shadow: 0 0 14px rgba(230, 180, 100, 0.35);
}

hr {
    border-color: #4a3720 !important;
}
</style>

<div class="ghost-drift"><span>👻</span></div>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _center(svg: str, margin: str = "0.4rem 0") -> str:
    """Wraps an inline SVG doodle in a centered div so it doesn't hug the left edge."""
    return f'<div style="display:flex; justify-content:center; align-items:center; margin:{margin};">{svg}</div>'


def skull_doodle_svg(width: int = 120) -> str:
    """A little hand-drawn-style skull, because Yorick insisted."""
    svg = f"""
    <svg width="{width}" height="{width}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="50" cy="42" rx="30" ry="28" fill="#e8dcc4" stroke="#241a0e" stroke-width="2.5"/>
        <path d="M22 50 Q20 70 30 78 L70 78 Q80 70 78 50" fill="#e8dcc4" stroke="#241a0e" stroke-width="2.5"/>
        <ellipse cx="37" cy="42" rx="8" ry="10" fill="#241a0e"/>
        <ellipse cx="63" cy="42" rx="8" ry="10" fill="#241a0e"/>
        <path d="M50 48 L46 60 L54 60 Z" fill="#241a0e"/>
        <path d="M32 78 L32 88 M40 78 L40 90 M50 78 L50 90 M60 78 L60 90 M68 78 L68 88"
              stroke="#241a0e" stroke-width="2.5" stroke-linecap="round"/>
        <path d="M28 78 Q50 84 72 78" fill="none" stroke="#241a0e" stroke-width="2"/>
    </svg>
    """
    return _center(svg)


def quill_doodle_svg(width: int = 110) -> str:
    svg = f"""
    <svg width="{width}" height="{width}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M20 85 Q40 60 75 15" stroke="#241a0e" stroke-width="2" fill="none"/>
        <path d="M75 15 C60 20 45 30 35 50 C50 45 65 35 75 15 Z" fill="#d8c48c" stroke="#241a0e" stroke-width="2"/>
        <path d="M75 15 C68 28 55 40 40 48" stroke="#241a0e" stroke-width="1.2" fill="none"/>
        <ellipse cx="18" cy="88" rx="7" ry="3" fill="#241a0e" opacity="0.5"/>
    </svg>
    """
    return _center(svg)


def crown_doodle_svg(width: int = 110) -> str:
    svg = f"""
    <svg width="{width}" height="{width}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M15 70 L20 35 L38 55 L50 25 L62 55 L80 35 L85 70 Z"
              fill="#d8c48c" stroke="#241a0e" stroke-width="2.5" stroke-linejoin="round"/>
        <rect x="15" y="70" width="70" height="10" fill="#d8c48c" stroke="#241a0e" stroke-width="2.5"/>
        <circle cx="20" cy="35" r="4" fill="#7a1f1f" stroke="#241a0e" stroke-width="1.5"/>
        <circle cx="50" cy="25" r="4" fill="#7a1f1f" stroke="#241a0e" stroke-width="1.5"/>
        <circle cx="80" cy="35" r="4" fill="#7a1f1f" stroke="#241a0e" stroke-width="1.5"/>
    </svg>
    """
    return _center(svg)


def castle_doodle_svg(width: int = 140) -> str:
    height = round(width * (110 / 140))
    svg = f"""
    <svg width="{width}" height="{height}" viewBox="0 0 140 110" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="50" width="120" height="55" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <rect x="10" y="40" width="15" height="15" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <rect x="35" y="40" width="15" height="15" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <rect x="60" y="20" width="18" height="35" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <polygon points="60,20 69,5 78,20" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <rect x="90" y="40" width="15" height="15" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <rect x="115" y="40" width="15" height="15" fill="#3a2b1a" stroke="#e8dcc4" stroke-width="1.5"/>
        <rect x="60" y="80" width="20" height="25" fill="#17120f" stroke="#e8dcc4" stroke-width="1.5"/>
        <circle cx="69" cy="8" r="3" fill="#e8c88c"/>
    </svg>
    """
    return _center(svg)


# ══════════════════════════════════════════════════════════════════════════
#  DATA: a handful of Hamlet's most famous lines, deliberately left unfinished
#  so the Oracle can try to finish them. (Trimmed excerpts, not the full text.)
# ══════════════════════════════════════════════════════════════════════════
INCOMPLETE_LINES = [
    "To be or not to be that is the",
    "Something is rotten in the state of",
    "This above all to thine own self be",
    "The lady doth protest too",
    "Though this be madness yet there is",
    "Brevity is the soul of",
    "There is nothing either good or bad but thinking makes it",
    "What a piece of work is",
    "Frailty thy name is",
    "The rest is",
    "Alas poor Yorick I knew him",
    "Neither a borrower nor a lender",
    "Give thy thoughts no tongue nor any unproportioned thought his",
    "Doubt thou the stars are",
    "In my minds",
    "How weary stale flat and unprofitable seem to me all the uses of this",
    "O that this too too solid flesh would",
    "The play's the thing wherein I'll catch the conscience of the",
    "Get thee to a",
    "There are more things in heaven and earth Horatio than are dreamt of in your",
    "When sorrows come they come not single spies but in",
    "Good night sweet prince and flights of angels sing thee to thy",
    "I must be cruel only to be",
    "The time is out of",
    "King. 'Tis deepely",
]


def random_incomplete_line() -> str:
    return random.choice(INCOMPLETE_LINES)


# ══════════════════════════════════════════════════════════════════════════
#  MODEL / TOKENIZER LOADING
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_the_oracles():
    lstm_model = load_model("next_word_lestm.h5")
    gru_model = load_model("next_word_GRU_RNN_1.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return lstm_model, gru_model, tokenizer


# ══════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC (same approach as the notebook, wrapped for reuse)
# ══════════════════════════════════════════════════════════════════════════
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = sequence.pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding="pre"
    )
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


def prophesy(model, tokenizer, seed_text, n_words):
    """Iteratively predicts n_words, one at a time, appending each guess."""
    max_sequence_len = model.input_shape[1] + 1
    text = seed_text
    generated = []
    for _ in range(n_words):
        nxt = predict_next_word(model, tokenizer, text, max_sequence_len)
        if nxt is None:
            break
        generated.append(nxt)
        text = text + " " + nxt
    return generated


# ══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
# NOTE: this key must match the text_input's `key=` below exactly. Streamlit
# ties a widget's displayed value to session_state[key] once that key exists,
# so updating any *other* variable (e.g. a separate "seed_text") is silently
# ignored on rerun. Writing directly to "seed_text_input" is what makes the
# "Summon a line" button actually update the box.
if "seed_text_input" not in st.session_state:
    st.session_state.seed_text_input = "To be or not to be"


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(crown_doodle_svg(90), unsafe_allow_html=True)
    st.markdown("## The Elsinore Oracle")
    st.markdown(
        "Two spirits haunt this castle, both trained upon the text of "
        "*Hamlet, Prince of Denmark*, both sworn to guess the next word "
        "of whatever line you whisper to them."
    )
    st.markdown("---")
    st.markdown(quill_doodle_svg(80), unsafe_allow_html=True)
    st.markdown(
        "**🐍 The Silver Serpent** — an LSTM, long of memory, slow and deliberate, "
        "coiled around every word that came before."
    )
    st.markdown(
        "**🦅 The Swift Raven** — a GRU, lighter and quicker, trusting fewer gates "
        "to carry the past forward."
    )
    st.markdown("---")
    st.markdown(skull_doodle_svg(80), unsafe_allow_html=True)
    st.caption(
        "\"There are more things in heaven and earth... than are dreamt of "
        "in your neural network.\""
    )


# ══════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="oracle-title">THE ELSINORE ORACLE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="oracle-subtitle">— two ghosts, one castle, and the word that comes next —</div>',
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    st.markdown(castle_doodle_svg(220), unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════
try:
    lstm_model, gru_model, tokenizer = load_the_oracles()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(
        "The Oracle could not be summoned. Make sure **next_word_lestm.h5**, "
        "**next_word_GRU_RNN_1.h5**, and **tokenizer.pickle** are sitting next to "
        f"this app.py file.\n\nDetails: {e}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  MAIN INPUT AREA
# ══════════════════════════════════════════════════════════════════════════
st.markdown("### 🕯️ Speak your line into the darkness")

input_col, button_col = st.columns([4, 1])
with input_col:
    seed_text = st.text_input(
        "Your unfinished line",
        key="seed_text_input",
        label_visibility="collapsed",
        placeholder="To be, or not to be...",
    )
with button_col:
    if st.button("🎲 Summon a line", use_container_width=True):
        st.session_state.seed_text_input = random_incomplete_line()
        st.rerun()

n_words = st.slider(
    "How many words shall the Oracle prophesy?", min_value=1, max_value=10, value=1
)

consult = st.button("⚔️ Consult the Oracles", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  RUN PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════
if consult and models_ready:
    if not seed_text.strip():
        st.warning("Even the dead need *something* to work with. Type a line first.")
    else:
        st.markdown("### 📜 The Prophecies")

        col1, col2 = st.columns(2)

        # --- LSTM ---
        with col1:
            st.markdown(
                f"""<div class="scroll-box">
                <div class="scroll-title">🐍 The Silver Serpent (LSTM)</div>
                <div style="font-style:italic; margin-bottom:0.6rem;">"{seed_text}"</div>
                <div id="lstm-words"></div>
                </div>""",
                unsafe_allow_html=True,
            )
            placeholder_lstm = st.empty()

        # --- GRU ---
        with col2:
            st.markdown(
                f"""<div class="scroll-box">
                <div class="scroll-title">🦅 The Swift Raven (GRU)</div>
                <div style="font-style:italic; margin-bottom:0.6rem;">"{seed_text}"</div>
                <div id="gru-words"></div>
                </div>""",
                unsafe_allow_html=True,
            )
            placeholder_gru = st.empty()

        with st.spinner("The spirits are stirring..."):
            lstm_words = prophesy(lstm_model, tokenizer, seed_text, n_words)
            gru_words = prophesy(gru_model, tokenizer, seed_text, n_words)

        # word-by-word reveal, alternating between the two scrolls for drama
        shown_lstm, shown_gru = [], []
        max_len = max(len(lstm_words), len(gru_words))
        for i in range(max_len):
            if i < len(lstm_words):
                shown_lstm.append(lstm_words[i])
            if i < len(gru_words):
                shown_gru.append(gru_words[i])

            lstm_html = "".join(
                f'<span class="predicted-word">{w}</span>' for w in shown_lstm
            )
            gru_html = "".join(
                f'<span class="predicted-word">{w}</span>' for w in shown_gru
            )
            placeholder_lstm.markdown(lstm_html, unsafe_allow_html=True)
            placeholder_gru.markdown(gru_html, unsafe_allow_html=True)
            time.sleep(0.45)

        # --- Verdict ---
        st.markdown("---")
        lstm_full = " ".join(lstm_words) if lstm_words else "..."
        gru_full = " ".join(gru_words) if gru_words else "..."

        if lstm_words and gru_words and lstm_words[0] == gru_words[0]:
            verdict = (
                f"👻 <b>The spirits are in accord.</b> Both ghosts agree the next "
                f"word is <b>\"{lstm_words[0]}\"</b> — a rare moment of harmony "
                f"between Serpent and Raven."
            )
        elif lstm_words and gru_words:
            verdict = (
                f"⚔️ <b>A clash of specters!</b> The Serpent whispers "
                f"<b>\"{lstm_words[0]}\"</b>, while the Raven caws "
                f"<b>\"{gru_words[0]}\"</b>. Two ghosts, two truths — "
                f"choose your haunting."
            )
        else:
            verdict = "🌫️ The mists are too thick tonight. The Oracle saw nothing."

        st.markdown(f'<div class="verdict-box">{verdict}</div>', unsafe_allow_html=True)

        st.markdown("#### Full conjured lines")
        st.markdown(f"**Serpent (LSTM):** *{seed_text} {lstm_full}*")
        st.markdown(f"**Raven (GRU):** *{seed_text} {gru_full}*")

elif consult and not models_ready:
    st.warning("The Oracle is still sleeping — fix the model files above and try again.")


# ══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
foot1, foot2, foot3 = st.columns([1, 3, 1])
with foot1:
    st.markdown(skull_doodle_svg(70), unsafe_allow_html=True)
with foot2:
    st.markdown(
        "<div style='text-align:center; opacity:0.7; font-style:italic;'>"
        "Trained on the words of Shakespeare's Hamlet · Powered by Keras LSTM & GRU · "
        "\"The rest is silence.\"</div>",
        unsafe_allow_html=True,
    )
with foot3:
    st.markdown(quill_doodle_svg(70), unsafe_allow_html=True)