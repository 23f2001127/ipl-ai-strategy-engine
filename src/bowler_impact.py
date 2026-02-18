import pandas as pd

print("Loading data...")

df = pd.read_csv("ipl_features_advanced.csv")


df = df[df["runs"].notna()]

print("Total balls:", len(df))


league_rpb = df["runs"].mean()
print("League runs per ball:", round(league_rpb, 3))


bowler_stats = (
    df.groupby("bowler")
    .agg(
        balls=("runs", "count"),
        runs_conceded=("runs", "sum"),
    )
)

bowler_stats["runs_per_ball"] = (
    bowler_stats["runs_conceded"] / bowler_stats["balls"]
)

bowler_stats["impact"] = league_rpb - bowler_stats["runs_per_ball"]


bowler_stats = bowler_stats[bowler_stats["balls"] > 200]

bowler_stats = bowler_stats.sort_values("impact", ascending=False)

print("\nTop bowlers by impact:")
print(bowler_stats.head(10))

bowler_stats.to_csv("bowler_impact.csv")
print("\nSaved to bowler_impact.csv")
