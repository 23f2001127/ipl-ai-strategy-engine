import pandas as pd
import numpy as np

print("Loading bowler stats...")
stats = pd.read_csv("bowler_stats.csv")


runs_so_far = 140
wickets = 5
balls_left = 24
target = 170

print("\nMatch Situation:")
print(f"Need {target - runs_so_far} runs in {balls_left} balls")



def sample_runs(mean_runs):

    
    probs = [0.30, 0.40, 0.10, 0.15, 0.05]

    
    if mean_runs < 1.0:
        probs = [0.40, 0.35, 0.10, 0.10, 0.05]


    elif mean_runs > 1.3:
        probs = [0.20, 0.40, 0.15, 0.15, 0.10]

    return np.random.choice([0, 1, 2, 4, 6], p=probs)



def simulate_bowler(bowler_name, sims=4000):

    row = stats[stats["bowler"] == bowler_name]
    if len(row) == 0:
        return None

    mean_runs = row["runs_per_ball"].values[0]
    wicket_prob = row["wicket_prob"].values[0]

    wins = 0

    for _ in range(sims):

        runs = runs_so_far
        wkts = wickets
        balls = balls_left

    
        next_over = min(6, balls)

        for _ in range(next_over):

            run = sample_runs(mean_runs)
            runs += run

            if np.random.rand() < wicket_prob:
                wkts += 1

            balls -= 1

    
        avg_runs = 1.2
        avg_wicket_prob = 0.03

        while balls > 0 and wkts < 10 and runs < target:

            run = sample_runs(avg_runs)
            runs += run

            if np.random.rand() < avg_wicket_prob:
                wkts += 1

            balls -= 1

        if runs >= target:
            wins += 1

    return wins / sims


print("\nBowler Suggestions:\n")

for bowler in stats.head(10)["bowler"]:
    prob = simulate_bowler(bowler)
    print(f"{bowler:20s} -> Win Prob: {prob:.3f}")
