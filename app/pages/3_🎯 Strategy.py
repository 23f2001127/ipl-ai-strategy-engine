import streamlit as st
import pandas as pd
import numpy as np
from app.lstm_predict import load_lstm, predict_lstm

st.title("🎯 Bowling Strategy Advisor")

st.write("""
Choose match situation and available bowlers.
The AI recommends the best bowler based on real IPL data.
""")

@st.cache_resource
def load_dl_model():
    return load_lstm()

@st.cache_resource
def load_impact():
    return pd.read_csv("bowler_impact.csv")

dl_model = load_dl_model()
impact_df = load_impact()

runs = st.number_input("Runs scored", 0, 300, 150)
wickets = st.number_input("Wickets lost", 0, 10, 5)
balls_left = st.number_input("Balls left", 0, 120, 24)
target = st.number_input("Target", 0, 300, 180)

runs_needed = max(0, target - runs)
st.write(f"Need **{runs_needed} runs** in **{balls_left} balls**")

if balls_left > 0:

    required_rr = runs_needed / (balls_left / 6 + 1e-6)
    current_rr = runs / ((120 - balls_left) / 6 + 1e-6)

    seq = np.array([[
        1, 0, runs, wickets, balls_left,
        runs_needed, required_rr, current_rr
    ]] * 12)

    if runs_needed <= 0:
        dl_prob = 1.0
    elif wickets >= 10:
        dl_prob = 0.0
    else:
        dl_prob = predict_lstm(dl_model, seq)

    st.subheader(f"Current Win Probability: {dl_prob:.2f}")

else:
    st.subheader("Match Finished")
    dl_prob = 0

st.markdown("---")
st.subheader("Available Bowlers")

selected = st.multiselect(
    "Choose Bowlers",
    impact_df["bowler"].tolist()
)

if selected:

    SCALE = 2.0
    suggestions = []

    for bowler in selected:

        row = impact_df[impact_df["bowler"] == bowler]
        if len(row) == 0:
            continue

        impact = row["impact"].values[0]

        new_prob = dl_prob - impact * SCALE
        new_prob = max(0, min(1, new_prob))

        change = new_prob - dl_prob
        suggestions.append((bowler, new_prob, change))

    suggestions.sort(key=lambda x: x[1])

    st.write("### Suggested Bowling Order")

    for bowler, prob, change in suggestions:
        sign = "+" if change > 0 else ""
        st.write(f"{bowler} → Win Prob: {prob:.2f} ({sign}{change:.2f})")

    best = suggestions[0][0]
    st.success(f"🏆 Best Bowler to Bowl Next: **{best}**")
