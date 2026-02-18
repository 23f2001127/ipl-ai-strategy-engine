import pandas as pd
import json
import glob

files = glob.glob("ipl_male_json/*.json")

rows = []

for f in files:
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)

    info = data["info"]

    match_id = f.split("\\")[-1].replace(".json","")

    team1, team2 = info["teams"]

    venue = info.get("venue", "Unknown")

    season = info.get("season", "Unknown")

    rows.append({
        "match_id": match_id,
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "season": season
    })

df = pd.DataFrame(rows)
df.to_csv("match_info.csv", index=False)

print("Saved match_info.csv")
