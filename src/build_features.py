import pandas as pd

df = pd.read_csv("ipl_ball_by_ball.csv")

df = df.sort_values(["match_id", "batting_team", "over"]).reset_index(drop=True)

df["ball_id"] = df.groupby(["match_id", "batting_team"]).cumcount() + 1

df["runs_so_far"] = df.groupby(["match_id", "batting_team"])["runs"].cumsum()

df["wickets_so_far"] = df.groupby(["match_id", "batting_team"])["wicket"].cumsum()

df["balls_left"] = 120 - df["ball_id"]

df["won"] = (df["batting_team"] == df["winner"]).astype(int)

df.to_csv("ipl_features.csv", index=False)

print("Done! Features created")
