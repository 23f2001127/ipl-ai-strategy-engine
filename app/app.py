import streamlit as st

st.set_page_config(page_title="AI Cricket Strategy Engine", layout="centered")

st.title("🏏 AI Cricket Strategy Engine")

st.write("""
Welcome to the AI Cricket Strategy Engine.

This app uses **Deep Learning on IPL ball-by-ball data**
to predict match outcomes and recommend strategies.

### Features
• Replay real IPL matches  
• Predict custom match situations  
• Detect turning points  
• Suggest best bowler  
• Deep Learning win probability  

Use the sidebar to explore.
""")

st.success("Built using IPL CricSheet data + PyTorch LSTM + Streamlit.")
