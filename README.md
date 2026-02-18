# 🏏 AI Cricket Strategy Engine

An end-to-end Machine Learning & Deep Learning system that predicts IPL match win probability in real time and recommends bowling strategies.

Built using ball-by-ball IPL data from CricSheet.

---

## 🚀 Features

- Replay real IPL matches ball-by-ball  
- Deep Learning win probability prediction  
- Detect key turning points in matches  
- Bowler strategy recommendation using player impact metrics  
- Monte-Carlo win probability projection  
- Interactive multi-page Streamlit dashboard  

---

## 📊 Dataset

**Source:** CricSheet IPL ball-by-ball dataset  

- 1100+ IPL matches  
- 270,000+ deliveries  

Data pipeline includes:

- JSON → CSV parsing  
- Feature engineering  
- Sequence generation for LSTM  

---

## 🤖 Models

### Baseline Model
Logistic Regression predicting win probability.  
ROC-AUC ≈ **0.79**

### Deep Learning Model
PyTorch LSTM trained on last-12-ball sequences to capture match momentum.  
ROC-AUC improved to ≈ **0.83**

---

## 🎯 Strategy Engine

Bowler recommendations use a data-driven impact metric:

**Impact = League average runs per ball − Bowler runs per ball**

This estimates how each bowler changes win probability in similar match situations.

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- PyTorch  
- Streamlit  

---

## 📂 Project Structure

```
ipl-ai-strategy-engine/
│
├── app/
│     ├── app.py
│     └── pages/
│
├── src/
│     └── data processing & training scripts
│
├── data/
│     └── processed IPL datasets
│
├── models/
│     └── trained ML & DL models
│
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```
pip install -r requirements.txt
streamlit run app/app.py
```

Then open the browser link shown in terminal.

---

## 🌐 Future Improvements

- Include batter & venue features  
- Add first-innings prediction  
- Live match integration  
- Transformer-based sequence model  
- Improved UI/UX  

---

## 👤 Author

*Antareep Ghosh**  
 
---

## ⭐ Acknowledgements

- CricSheet for IPL dataset  
- Streamlit for dashboard framework  
- PyTorch & Scikit-Learn communities  
