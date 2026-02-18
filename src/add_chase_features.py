import pandas as pd

df = pd.read_csv("ipl_features.csv")

df["innings_number"] = df.groupby("match_id")["batting_team"].transform(
    lambda x: (x != x.shift()).cumsum()
)

first_innings_totals = (
    df[df["innings_number"] == 1]
    .groupby("match_id")["runs"]
    .sum()
    .reset_index()
    .rename(columns={"runs": "first_innings_runs"})
)

df = df.merge(first_innings_totals, on="match_id", how="left")

df["target"] = df["first_innings_runs"] + 1

df["runs_remaining"] = df["target"] - df["runs_so_far"]

df["required_rr"] = df["runs_remaining"] / (df["balls_left"] / 6 + 1e-6)

df["current_rr"] = df["runs_so_far"] / ((120 - df["balls_left"]) / 6 + 1e-6)

df_2nd = df[df["innings_number"] == 2]

df_2nd.to_csv("ipl_features_advanced.csv", index=False)

print("Done! Advanced features created.")