import json
import pandas as pd
import glob

files = glob.glob("ipl_male_json/*.json")

rows = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    match_id = file.split("\\")[-1].replace(".json", "")

    winner = data["info"].get("outcome", {}).get("winner", None)

    for innings in data["innings"]:
        team = innings["team"]

        for over in innings["overs"]:
            over_num = over["over"]

            for ball in over["deliveries"]:
                runs = ball["runs"]["total"]
                batter = ball.get("batter", None)
                bowler = ball.get("bowler", None)
                wicket = 1 if "wickets" in ball else 0

                rows.append({
                    "match_id": match_id,
                    "batting_team": team,
                    "over": over_num,
                    "runs": runs,
                    "batter": batter,
                    "bowler": bowler,
                    "wicket": wicket,
                    "winner": winner
                })

df = pd.DataFrame(rows)
df.to_csv("ipl_ball_by_ball.csv", index=False)

print("DONE. Rows:", len(df))
