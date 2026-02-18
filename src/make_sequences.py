import pandas as pd
import numpy as np

df = pd.read_csv("ipl_features_advanced.csv")

features = [
    "runs",
    "wicket",
    "runs_so_far",
    "wickets_so_far",
    "balls_left",
    "runs_remaining",
    "required_rr",
    "current_rr"
]

SEQ_LEN = 12

X = []
y = []

for match in df["match_id"].unique():

    match_df = df[df["match_id"] == match].reset_index(drop=True)

    for i in range(SEQ_LEN, len(match_df)):
        seq = match_df.iloc[i-SEQ_LEN:i][features].values
        label = match_df.iloc[i]["won"]

        X.append(seq)
        y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X_seq.npy", X)
np.save("y_seq.npy", y)

print("Done!")
print("Shape X:", X.shape)
print("Shape y:", y.shape)
