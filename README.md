# 🏏 IPL Match Winner Prediction System

An AI-powered web application that predicts the outcome of IPL matches using historical data and machine learning.

---

## 🚀 Overview

This project uses past IPL match data to predict which team is more likely to win a match based on:

* Teams playing
* Toss result
* Venue
* Historical performance (win rates)

The model is deployed as an interactive web app using **Streamlit**, allowing users to input match conditions and get real-time predictions.

---

## 🎯 Features

* 🔮 Predict match winner
* 📊 Display win probability for both teams
* 📈 Show team win rates
* 🤝 Head-to-head comparison (insight)
* 🧠 Data-driven predictions using Machine Learning
* 🌐 Interactive web interface

---

## 🧠 Machine Learning Approach

* Model: **Random Forest Classifier**

* Features Used:

  * Team 1
  * Team 2
  * Toss Winner
  * Toss Decision
  * Venue
  * Team 1 Win Rate
  * Team 2 Win Rate

* Target:

  * 0 → Team 1 wins
  * 1 → Team 2 wins

---

## 📊 Model Performance

| Metric            | Value |
| ----------------- | ----- |
| Training Accuracy | ~72%  |
| Test Accuracy     | ~53%  |

> ⚠️ Note: Sports outcomes are inherently uncertain, so moderate accuracy is expected.

---

## 🧹 Data Preprocessing

Handled real-world inconsistencies in the dataset:

### ✅ Team Name Standardization

* Delhi Daredevils → Delhi Capitals
* Kings XI Punjab → Punjab Kings
* Deccan Chargers → Sunrisers Hyderabad

### ✅ Venue Cleaning

* Removed location suffixes

  * "Brabourne Stadium, Mumbai" → "Brabourne Stadium"

👉 This improved model performance and ensured consistency.

---

## 🗂️ Project Structure

```bash
sports_prediction/
│
├── app/
│   ├── a_preprocess.py
│   ├── b_train_model.py
│   └── c_frontend.py
│
├── data/
│   └── ipl.csv
│
├── model/
│   ├── model.pkl
│   └── encoders.pkl
│
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## ⚙️ Installation

Using **uv** (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app/c_frontend.py
```

---

## 🌐 Usage

1. Select:

   * Team 1
   * Team 2
   * Toss Winner
   * Toss Decision
   * Venue

2. Click **Predict Winner**

3. View:

   * Predicted winner
   * Winning probability
   * Match insights

---

## 💡 Key Insights

* Historical win rates are the most influential feature
* Venue and toss also impact outcomes
* Model reflects real-world uncertainty rather than overfitting

---

## ⚠️ Limitations

* Does not include player-level data
* Does not consider real-time factors (injuries, pitch, weather)
* Limited feature set

---

## 🚀 Future Improvements

* Add recent form (last 5 matches)
* Include player statistics
* Improve feature engineering
* Try advanced models (XGBoost, LightGBM)

---

## 👨‍💻 Author

Built as a machine learning project to demonstrate:

* Data preprocessing
* Model training
* Real-world deployment

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
