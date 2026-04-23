"""
IPL Match Winner Prediction - Streamlit App
============================================
Loads model.pkl and encoders.pkl, takes match inputs,
computes features, and predicts the winner with probabilities.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Base directory (sports_prediction root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────
# TEAM NAME STANDARDISATION MAP  (must mirror preprocess.py exactly)
# ─────────────────────────────────────────────────────────────────
TEAM_NAME_MAP = {
    "Delhi Daredevils"            : "Delhi Capitals",
    "Kings XI Punjab"             : "Punjab Kings",
    "Deccan Chargers"             : "Sunrisers Hyderabad",
    "Rising Pune Supergiant"      : "Rising Pune Supergiants",
    "Royal Challengers Bangalore" : "Royal Challengers Bengaluru",
}

# Venue cleaning — mirrors preprocess.py exactly (two-step approach):
# Step A is a general comma-strip applied inside load_data().
# Step B handles only the 4 genuine stadium renames.
VENUE_RENAME_MAP = {
    "Feroz Shah Kotla"                   : "Arun Jaitley Stadium",
    "Sardar Patel Stadium"               : "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium" : "Punjab Cricket Association IS Bindra Stadium",
    "M.Chinnaswamy Stadium"              : "M Chinnaswamy Stadium",
}


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL AND ENCODERS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load(BASE_DIR / "model" / "model.pkl")
    encoders = joblib.load(BASE_DIR / "model" / "encoders.pkl")
    return model, encoders

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "ipl.csv")
    # Apply team-name and venue standardisation used during training
    team_cols = ["team1", "team2", "toss_winner", "winner"]
    for col in team_cols:
        df[col] = df[col].replace(TEAM_NAME_MAP)
    df["venue"] = df["venue"].str.split(",").str[0].str.strip()  # Step A: strip city suffix
    df["venue"] = df["venue"].replace(VENUE_RENAME_MAP)           # Step B: genuine renames
    # Keep only valid matches (winner is exactly team1 or team2)
    df = df.dropna(subset=["winner"])
    df = df[(df["winner"] == df["team1"]) | (df["winner"] == df["team2"])].copy()
    return df

model, encoders = load_artifacts()
df_hist         = load_data()

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def compute_win_rate(team: str) -> float:
    """Overall historical win % for a team."""
    games = df_hist[(df_hist["team1"] == team) | (df_hist["team2"] == team)]
    if len(games) == 0:
        return 0.5
    wins = (games["winner"] == team).sum()
    return round(wins / len(games), 4)


def compute_h2h_ratio(team1: str, team2: str) -> float:
    """Head-to-head win % of team1 against team2."""
    h2h = df_hist[
        ((df_hist["team1"] == team1) & (df_hist["team2"] == team2)) |
        ((df_hist["team1"] == team2) & (df_hist["team2"] == team1))
    ]
    if len(h2h) == 0:
        return 0.5
    team1_wins = (h2h["winner"] == team1).sum()
    return round(team1_wins / len(h2h), 4)


def encode_input(team1, team2, toss_winner, toss_decision, venue,
                 team1_win_rate, team2_win_rate):
    """Encode categorical inputs exactly as done during training."""
    row = {
        "team1"          : encoders["team1"].transform([team1])[0],
        "team2"          : encoders["team2"].transform([team2])[0],
        "toss_winner"    : encoders["toss_winner"].transform([toss_winner])[0],
        "toss_decision"  : encoders["toss_decision"].transform([toss_decision])[0],
        "venue"          : encoders["venue"].transform([venue])[0],
        "team1_win_rate" : team1_win_rate,
        "team2_win_rate" : team2_win_rate,
    }
    # Must match training column order exactly
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────
# DERIVE DROPDOWN OPTIONS FROM ENCODERS
# ─────────────────────────────────────────────────────────────────
# LabelEncoder.classes_ is already unique (one entry per class) and
# sorted. We call sorted() explicitly here as a guarantee, using a
# case-insensitive key so the list is human-readable in the UI.
team_list = sorted(encoders["team1"].classes_, key=str.lower)

# Build venue list directly from the encoder — guaranteed unique and consistent
# with training. Apply a comma-strip safeguard in case any suffix survived
# into the encoder (defensive: encoders.pkl should already be clean).
venue_list = sorted(
    {
        v.split(",")[0].strip()              # strip any residual city suffix
        for v in encoders["venue"].classes_  # source of truth = saved encoder
    },
    key=str.lower                            # case-insensitive alphabetical order
)

# ─────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────
st.title("🏏 IPL Match Winner Predictor")
st.markdown("Fill in the match details below and click **Predict Winner**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("🔵 Team 1", team_list, index=0)

with col2:
    # Default team2 to first team that isn't team1
    team2_default = next(
        (i for i, t in enumerate(team_list) if t != team1), 1
    )
    team2 = st.selectbox("🔴 Team 2", team_list, index=team2_default)

toss_winner   = st.selectbox("🪙 Toss Winner", [team1, team2])
toss_decision = st.radio("🏏 Toss Decision", ["bat", "field"], horizontal=True)

# ── Match Venue ──
st.markdown("#### 🏟️ Match Venue")
venue = st.selectbox(
    "Select Venue",
    venue_list,
    label_visibility="collapsed",
    help="Venues are standardised — city suffixes removed, renames applied."
)
st.caption(f"📍 Selected: **{venue}**")

st.divider()

# ─────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────
if st.button("⚡ Predict Winner", use_container_width=True, type="primary"):

    # --- Validate: teams must be different ---
    if team1 == team2:
        st.error("⚠️ Team 1 and Team 2 cannot be the same. Please select different teams.")
        st.stop()

    # --- Compute numerical features ---
    t1_wr  = compute_win_rate(team1)
    t2_wr  = compute_win_rate(team2)
    h2h    = compute_h2h_ratio(team1, team2)

    # --- Encode and predict ---
    try:
        X_input = encode_input(team1, team2, toss_winner, toss_decision,
                               venue, t1_wr, t2_wr)
        proba   = model.predict_proba(X_input)[0]   # [P(team1), P(team2)]
        pred    = model.predict(X_input)[0]          # 0 = team1, 1 = team2
    except ValueError as e:
        st.error(f"Encoding error: {e}\n\nMake sure the selected venue/teams "
                 "were present in the training data.")
        st.stop()

    p_team1 = round(proba[0] * 100, 1)
    p_team2 = round(proba[1] * 100, 1)
    winner  = team1 if pred == 0 else team2
    win_pct = p_team1 if pred == 0 else p_team2

    # ── Result banner ──
    st.success(f"🏆 Predicted Winner: **{winner}**  ({win_pct:.0f}% probability)")

    # ── Probability bars ──
    st.markdown("### Win Probability")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label=f"🔵 {team1}", value=f"{p_team1}%")
        st.progress(int(p_team1))
    with col_b:
        st.metric(label=f"🔴 {team2}", value=f"{p_team2}%")
        st.progress(int(p_team2))

    # ── Additional insights ──
    st.divider()
    st.markdown("### 📊 Match Insights")
    ins1, ins2, ins3 = st.columns(3)
    ins1.metric("🔵 Team 1 Overall Win Rate", f"{t1_wr * 100:.1f}%")
    ins2.metric("🔴 Team 2 Overall Win Rate", f"{t2_wr * 100:.1f}%")
    ins3.metric(f"🔵 {team1} H2H Win Rate", f"{h2h * 100:.1f}%",
                help=f"Historical win % of {team1} when playing against {team2}")
