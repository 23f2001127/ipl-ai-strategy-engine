import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="📊"
)

st.title("📊 About This Project")

st.write("""
### 🏏 AI Cricket Strategy Engine

This project predicts win probability in IPL matches using 
**Machine Learning and Deep Learning** on real ball-by-ball data.
""")

st.markdown("---")

st.header("📂 Dataset")

st.write("""
• Source: CricSheet IPL ball-by-ball dataset  
• Matches: 1100+ IPL games  
• Deliveries: 270,000+ balls  

Each delivery includes:
- Runs scored  
- Wicket info  
- Match situation  
- Teams, venue, season  
""")

st.markdown("---")

st.header("🤖 Models Used")

st.write("""
### 1️⃣ Baseline Model
Logistic Regression predicting win probability.

### 2️⃣ Deep Learning Model
PyTorch LSTM trained on last-12-ball sequences to capture momentum.

ROC-AUC improved from **0.79 → 0.83**.
""")

st.markdown("---")

st.header("🎯 Strategy Engine")

st.write("""
Bowler recommendations use **real IPL bowler impact stats**.

Impact = League average runs per ball − Bowler runs per ball.

This shows how much each bowler improves or worsens win chances.
""")

st.markdown("---")

st.header("📈 Features")

st.write("""
✔ Replay real IPL matches  
✔ Detect key turning points  
✔ Predict custom match situations  
✔ Suggest best bowler  
✔ Win probability projection graph  
✔ Deep Learning momentum model  
""")

st.markdown("---")

st.header("🛠️ Tech Stack")

st.write("""
• Python  
• Pandas + NumPy  
• Scikit-Learn  
• PyTorch  
• Streamlit  
""")

st.markdown("---")

st.success("Built by Antareep Ghosh")
