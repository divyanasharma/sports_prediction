"""
IPL Match Winner Prediction - Preprocessing Script
====================================================
Goal    : Predict which team wins an IPL match.
Target  : winner_binary (0 = team1 wins, 1 = team2 wins)
Features: team1, team2, toss_winner, toss_decision, venue,
          team1_win_rate, team2_win_rate
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Base directory (sports_prediction root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE_DIR / "data" / "ipl.csv")

print("Shape (raw):", df.shape)
print("\nColumn names:\n", df.columns.tolist())

# ─────────────────────────────────────────────────────────────────
# 2. STANDARDISE TEAM NAMES  (fix franchise renames across seasons)
# ─────────────────────────────────────────────────────────────────
# Full audit of ipl.csv revealed 5 inconsistencies:
#   - Delhi Daredevils  renamed to Delhi Capitals  (2019)
#   - Kings XI Punjab   renamed to Punjab Kings     (2021)
#   - Deccan Chargers   folded; replaced by Sunrisers Hyderabad (2013)
#   - Rising Pune Supergiant  (singular typo) vs Rising Pune Supergiants
#   - Royal Challengers Bangalore renamed to Royal Challengers Bengaluru (2024)
TEAM_NAME_MAP = {
    "Delhi Daredevils"         : "Delhi Capitals",
    "Kings XI Punjab"          : "Punjab Kings",
    "Deccan Chargers"          : "Sunrisers Hyderabad",
    "Rising Pune Supergiant"   : "Rising Pune Supergiants",   # singular → plural
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

# Apply mapping to every column that carries a team name
team_cols = ["team1", "team2", "toss_winner", "winner"]
for col in team_cols:
    df[col] = df[col].replace(TEAM_NAME_MAP)

# Verify: print unique team names after standardisation
all_teams = sorted(
    set(df["team1"].dropna()) |
    set(df["team2"].dropna())
)
print(f"\nUnique teams after standardisation ({len(all_teams)} total):")
for t in all_teams:
    print(f"  - {t}")

print("\nFirst few rows (after team standardisation):\n",
      df[["team1", "team2", "toss_winner", "toss_decision", "venue", "winner"]].head())


# ─────────────────────────────────────────────────────────────────
# 3. STANDARDISE VENUE NAMES  (general strip + minimal rename map)
# ─────────────────────────────────────────────────────────────────
# The raw CSV contains 59 venue strings for ~38 actual grounds.
# Two types of duplicates:
#
#   TYPE A – Redundant city/location suffixes after a comma (general rule):
#       "Eden Gardens, Kolkata"             →  "Eden Gardens"
#       "Wankhede Stadium, Mumbai"          →  "Wankhede Stadium"
#       "MA Chidambaram Stadium, Chepauk, Chennai" →  "MA Chidambaram Stadium"
#   Fix: split on the first comma, keep only what's before it.
#
#   TYPE B – Genuine stadium renames (only 4 cases, handled by a small map):
#       "Feroz Shah Kotla"                   →  "Arun Jaitley Stadium"   (2019)
#       "Sardar Patel Stadium"               →  "Narendra Modi Stadium"  (2020)
#       "Punjab Cricket Association Stadium" →  "Punjab Cricket Association IS Bindra Stadium"
#       "M.Chinnaswamy Stadium"              →  "M Chinnaswamy Stadium"  (punctuation)
#
# ORDER: strip city suffix FIRST so the rename map only needs base names.

print(f"\nUnique venues BEFORE standardisation: {df['venue'].nunique()}")

# ── Step A: General comma-strip ──────────────────────────────────
# Split on the first comma only; keep the part before it; trim spaces.
# This single line fixes all "Stadium, City" duplicates automatically.
df["venue"] = df["venue"].str.split(",").str[0].str.strip()

print(f"Unique venues after comma-strip:      {df['venue'].nunique()}")

# ── Step B: Minimal rename map for genuine name changes ──────────
# Only 4 entries needed — comma-strip already resolved all suffix variants.
VENUE_RENAME_MAP = {
    "Feroz Shah Kotla"                   : "Arun Jaitley Stadium",
    "Sardar Patel Stadium"               : "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium" : "Punjab Cricket Association IS Bindra Stadium",
    "M.Chinnaswamy Stadium"              : "M Chinnaswamy Stadium",
}
df["venue"] = df["venue"].replace(VENUE_RENAME_MAP)

print(f"Unique venues AFTER  standardisation: {df['venue'].nunique()}")
print("\nFinal canonical venue list:")
for v in sorted(df["venue"].dropna().unique()):
    print(f"  - {v}")


# ─────────────────────────────────────────────────────────────────
# 4. KEEP ONLY VALID MATCHES  (no null winner, no ties/no-result)
# ─────────────────────────────────────────────────────────────────
# Drop rows where 'winner' is missing (ties, no-result, etc.)
df = df.dropna(subset=["winner"])

# Keep only rows where winner is exactly team1 or team2
mask = (df["winner"] == df["team1"]) | (df["winner"] == df["team2"])
df = df[mask].copy()

print("\nShape after dropping invalid matches:", df.shape)

# ─────────────────────────────────────────────────────────────────
# 5. CLEAN OTHER MISSING VALUES
# ─────────────────────────────────────────────────────────────────
# Fill missing toss_decision with the most frequent value
df["toss_decision"] = df["toss_decision"].fillna(
    df["toss_decision"].mode()[0]
)

# Drop rows with missing team names or venue (very few if any)
df = df.dropna(subset=["team1", "team2", "toss_winner", "venue"])

print("Shape after cleaning NaNs:", df.shape)

# ─────────────────────────────────────────────────────────────────
# 6. COMPUTE HISTORICAL WIN RATES (per-row, using PAST matches only)
# ─────────────────────────────────────────────────────────────────
# Sort by match date so "past" means all rows before the current one
df = df.sort_values("match_date").reset_index(drop=True)

# Running totals: wins and appearances for every team
team_wins  = {}   # team -> cumulative wins seen SO FAR
team_games = {}   # team -> cumulative games seen SO FAR

team1_win_rates = []
team2_win_rates = []

for _, row in df.iterrows():
    t1, t2, w = row["team1"], row["team2"], row["winner"]

    # ── Win rates BEFORE this match is processed ──
    g1 = team_games.get(t1, 0)
    g2 = team_games.get(t2, 0)

    wr1 = team_wins.get(t1, 0) / g1 if g1 > 0 else 0.5
    wr2 = team_wins.get(t2, 0) / g2 if g2 > 0 else 0.5

    team1_win_rates.append(wr1)
    team2_win_rates.append(wr2)

    # ── Update tallies AFTER recording the win rate ──
    team_games[t1] = g1 + 1
    team_games[t2] = g2 + 1

    if w == t1:
        team_wins[t1] = team_wins.get(t1, 0) + 1
    elif w == t2:
        team_wins[t2] = team_wins.get(t2, 0) + 1

df["team1_win_rate"] = team1_win_rates
df["team2_win_rate"] = team2_win_rates

print("\nSample win rates:\n",
      df[["team1", "team2", "team1_win_rate", "team2_win_rate"]].head(10))

# ─────────────────────────────────────────────────────────────────
# 7. CREATE BINARY TARGET VARIABLE
#    0 → team1 wins   |   1 → team2 wins
# ─────────────────────────────────────────────────────────────────
df["winner_binary"] = (df["winner"] == df["team2"]).astype(int)

print("\nTarget distribution:\n", df["winner_binary"].value_counts())

# ─────────────────────────────────────────────────────────────────
# 8. SELECT FEATURE COLUMNS
# ─────────────────────────────────────────────────────────────────
features_cat = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
features_num = ["team1_win_rate", "team2_win_rate"]
target       = "winner_binary"

# Build working dataframe with only needed columns
model_df = df[features_cat + features_num + [target]].copy()

# ─────────────────────────────────────────────────────────────────
# 9. ENCODE CATEGORICAL VARIABLES WITH LabelEncoder
# ─────────────────────────────────────────────────────────────────
label_encoders = {}

for col in features_cat:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    label_encoders[col] = le   # save encoder for later inverse_transform

print("\nEncoded categorical columns:")
for col, le in label_encoders.items():
    print(f"  {col}: {list(le.classes_)[:5]} ...")   # show first 5 classes

# ─────────────────────────────────────────────────────────────────
# 10. FINAL PROCESSED DATAFRAME
# ─────────────────────────────────────────────────────────────────
print("\n-- Final processed dataframe --")
print("Shape :", model_df.shape)
print("Dtypes:\n", model_df.dtypes)
print("\nSample rows:\n", model_df.head())
print("\nNull check:\n", model_df.isnull().sum())

# Separate features (X) and target (y) – ready for model training
X = model_df.drop(columns=[target])
y = model_df[target]

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("\nPreprocessing complete. X and y are ready for model training.")
