import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("ipl_features.csv")

features = ["runs_so_far", "wickets_so_far", "balls_left"]
X = df[features]
y = df["won"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)


match_id = df["match_id"].iloc[0]
match_df = df[df["match_id"] == match_id]

teams = match_df["batting_team"].unique()
winner = match_df["winner"].iloc[0]

print("Teams:", teams)
print("Winner:", winner)

probs = model.predict_proba(match_df[features])[:, 1]

plt.plot(probs)
innings_break = match_df["ball_id"].iloc[120]
plt.axvline(x=innings_break, linestyle="--")
plt.title(f"{teams[0]} vs {teams[1]} | Winner: {winner}")
plt.xlabel("Ball Number")
plt.ylabel("Win Probability")
plt.show()
