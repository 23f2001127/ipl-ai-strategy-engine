import pandas as pd

print("Loading data...")
df = pd.read_csv("ipl_ball_by_ball.csv")

print("Total deliveries:", len(df))

bowler_stats = (
    df.groupby("bowler")
    .agg(
        balls=("runs", "count"),
        runs_conceded=("runs", "sum"),
        wickets=("wicket", "sum")
    )
)

bowler_stats["runs_per_ball"] = bowler_stats["runs_conceded"] / bowler_stats["balls"]

bowler_stats["wicket_prob"] = bowler_stats["wickets"] / bowler_stats["balls"]

bowler_stats["economy"] = bowler_stats["runs_per_ball"] * 6

MIN_BALLS = 300  
bowler_stats = bowler_stats[bowler_stats["balls"] >= MIN_BALLS]

bowler_stats = bowler_stats.sort_values("runs_per_ball")

bowler_stats.to_csv("bowler_stats.csv")

print("\nTop 10 Best Economy Bowlers:\n")
print(bowler_stats.head(10))

print("\nSaved to bowler_stats.csv")
