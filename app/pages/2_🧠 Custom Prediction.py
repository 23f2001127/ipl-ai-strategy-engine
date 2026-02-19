import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

st.set_page_config(
    page_title="Custom Prediction",
    page_icon="🧠"
)

CURRENT_DIR = os.path.dirname(__file__)          
APP_DIR = os.path.dirname(CURRENT_DIR)           
PROJECT_ROOT = os.path.dirname(APP_DIR)          

sys.path.append(APP_DIR)

from lstm_predict import load_lstm, predict_lstm


st.title("🧠 Custom Match Prediction")

@st.cache_resource
def load_dl_model():
    return load_lstm()

dl_model = load_dl_model()

runs = st.number_input("Runs scored", 0, 300, 150)
wickets = st.number_input("Wickets lost", 0, 10, 5)
balls_left = st.number_input("Balls left", 0, 120, 24)
target = st.number_input("Target", 0, 300, 180)

last_over = st.text_input(
    "Last 6 balls (example: 4 1 W 2 0 6)",
    "1 1 0 2 0 1"
)

runs_needed = max(0, target - runs)
st.write(f"Need **{runs_needed} runs** in **{balls_left} balls**")

def parse_last_over(text):
    balls = text.split()
    runs = []
    wickets = []

    for b in balls[:6]:
        if b.upper() == "W":
            runs.append(0)
            wickets.append(1)
        else:
            try:
                runs.append(int(b))
                wickets.append(0)
            except:
                runs.append(0)
                wickets.append(0)

    while len(runs) < 6:
        runs.append(0)
        wickets.append(0)

    return runs, wickets

last_runs, last_wkts = parse_last_over(last_over)

if balls_left > 0:

    required_rr = runs_needed / (balls_left / 6 + 1e-6)
    current_rr = runs / ((120 - balls_left) / 6 + 1e-6)

    seq = []
    for r, w in zip(last_runs, last_wkts):
        seq.append([
            r,
            w,
            runs,
            wickets,
            balls_left,
            runs_needed,
            required_rr,
            current_rr
        ])

    seq = np.array(seq * 2)

    if runs_needed <= 0:
        dl_prob = 1.0
    elif wickets >= 10:
        dl_prob = 0.0
    else:
        dl_prob = predict_lstm(dl_model, seq)

    st.subheader(f"Chasing Team's Win Probability: {dl_prob:.2f}")

else:
    st.subheader("Match Finished")
    dl_prob = 0

if balls_left > 0:

    st.markdown("---")
    st.subheader("📈 Win Probability Projection of Chasing Team")

    sims = 120
    curve = np.zeros(balls_left + 1)

    for _ in range(sims):

        temp_runs = runs
        temp_balls = balls_left

        for i in range(temp_balls, -1, -1):

            temp_needed = max(0, target - temp_runs)

            if i == 0:
                prob = 0 if temp_needed > 0 else 1
                curve[i] += prob
                break

            required_rr_proj = temp_needed / (i / 6 + 1e-6)
            current_rr_proj = temp_runs / ((120 - i) / 6 + 1e-6)

            seq_proj = []
            for r, w in zip(last_runs, last_wkts):
                seq_proj.append([
                    r,
                    w,
                    temp_runs,
                    wickets,
                    i,
                    temp_needed,
                    required_rr_proj,
                    current_rr_proj
                ])

            seq_proj = np.array(seq_proj * 2)

            prob = predict_lstm(dl_model, seq_proj)
            curve[i] += prob

            run = np.random.choice(
                [0, 1, 2, 4, 6],
                p=[0.30, 0.40, 0.10, 0.15, 0.05]
            )
            temp_runs += run

    curve = curve / sims
    st.line_chart(curve)
