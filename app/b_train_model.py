"""
IPL Match Winner Prediction - Model Training Script
=====================================================
Imports preprocessed X, y, and label_encoders from preprocess.py,
trains a RandomForestClassifier, evaluates it, and saves artefacts.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# Base directory (sports_prediction root)
BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────
# 1. IMPORT PREPROCESSED DATA FROM preprocess.py
# ─────────────────────────────────────────────────────────────────
# Running this import executes preprocess.py and gives us
# X (features), y (target), and label_encoders (dict of LabelEncoders)
from a_preprocess import X, y, label_encoders

print("Features shape :", X.shape)
print("Target shape   :", y.shape)

# ─────────────────────────────────────────────────────────────────
# 2. SPLIT INTO TRAIN / TEST (80 / 20)
# ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # reproducible split
    stratify=y         # keep class balance in both splits
)

print(f"\nTrain size : {len(X_train)} samples")
print(f"Test  size : {len(X_test)}  samples")

# ─────────────────────────────────────────────────────────────────
# 3. TRAIN A RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,   # 100 decision trees
    max_depth=6,       # limit depth to avoid overfitting
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1           # use all CPU cores for speed
)

model.fit(X_train, y_train)
print("\nModel training complete.")

# ─────────────────────────────────────────────────────────────────
# 4. EVALUATE THE MODEL
# ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy  = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  Train Accuracy : {train_accuracy * 100:.2f}%")
print(f"  Test  Accuracy : {test_accuracy  * 100:.2f}%")
print(f"{'='*40}")

# ─────────────────────────────────────────────────────────────────
# 5. SAVE MODEL AND ENCODERS WITH JOBLIB
# ─────────────────────────────────────────────────────────────────
joblib.dump(model, BASE_DIR / "model" / "model.pkl")
print(f"\nSaved trained model   -> {BASE_DIR / 'model' / 'model.pkl'}")

joblib.dump(label_encoders, BASE_DIR / "model" / "encoders.pkl")
print(f"Saved label encoders  -> {BASE_DIR / 'model' / 'encoders.pkl'}")

# ─────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (DESCENDING ORDER)
# ─────────────────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nFeature Importances (descending):")
print("-" * 35)
for _, row in importance_df.iterrows():
    bar = "#" * int(row["Importance"] * 50)   # visual bar
    print(f"  {row['Feature']:<20} {row['Importance']:.4f}  {bar}")
