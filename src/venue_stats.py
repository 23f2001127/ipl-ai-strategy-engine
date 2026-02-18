import pandas as pd

print("Loading data...")

df = pd.read_csv("ipl_features_advanced.csv")

print("Total balls:", len(df))

venue_stats = (
    df.groupby("venue")
    .agg(
        balls=("runs", "count"),
        total_runs=("runs", "sum")
    )
)

venue_stats["runs_per_ball"] = venue_stats["total_runs"] / venue_stats["balls"]

venue_stats = venue_stats.sort_values("runs_per_ball", ascending=False)

print("\nTop high scoring venues:")
print(venue_stats.head())

venue_stats.to_csv("venue_stats.csv")
print("\nSaved venue_stats.csv")
