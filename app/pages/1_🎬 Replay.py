import streamlit as st
import pandas as pd
import numpy as np
from app.lstm_predict import load_lstm, predict_lstm

st.title("🎬 Replay Real IPL Match")

@st.cache_resource
def load_dl_model():
    return load_lstm()

@st.cache_resource
def load_dataset():
    return pd.read_csv("ipl_features_advanced.csv")

@st.cache_resource
def load_match_info():
    return pd.read_csv("match_info.csv")

dl_model = load_dl_model()
df_all = load_dataset()
info_df = load_match_info()

match_ids = df_all["match_id"].unique()
selected_match = st.selectbox("Choose Match ID", match_ids)

if selected_match:

    info = info_df[info_df["match_id"] == selected_match]
    if len(info):
        row = info.iloc[0]
        st.write(f"### {row.team1} vs {row.team2}")
        st.write(f"Season: {row.season}")
        st.write(f"Venue: {row.venue}")

    match_df = df_all[df_all["match_id"] == selected_match].reset_index(drop=True)

    probs = []

    for _, r in match_df.iterrows():

        seq = np.array([[ 
            r["runs"],
            r["wicket"],
            r["runs_so_far"],
            r["wickets_so_far"],
            r["balls_left"],
            r["runs_remaining"],
            r["required_rr"],
            r["current_rr"]
        ]] * 12)

        prob = predict_lstm(dl_model, seq)
        probs.append(prob)

    st.line_chart(probs)
    st.caption("Win probability throughout the match.")

    st.subheader("🔥 Key Turning Points")

    changes = []
    for i in range(1, len(probs)):
        diff = abs(probs[i] - probs[i-1])
        changes.append((i, diff))

    changes.sort(key=lambda x: x[1], reverse=True)
    top = changes[:3]

    for idx, diff in top:
        st.write(f"Ball {idx}: Win probability changed by {diff:.2f}")
