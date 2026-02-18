import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("ipl_features_advanced.csv")

features = [
    "runs_so_far",
    "wickets_so_far",
    "balls_left",
    "runs_remaining",
    "required_rr",
    "current_rr"
]

df = df.dropna()

X = df[features]
y = df["won"]

model = LogisticRegression(max_iter=2000)
model.fit(X, y)


runs_so_far = 140
wickets = 5
balls_left = 24
target = 170  


def sample_run(strategy):

    if strategy == "defensive":
        probs = [0.45, 0.35, 0.1, 0.07, 0.03]

    elif strategy == "normal":
        probs = [0.35, 0.4, 0.1, 0.1, 0.05]

    else:  
        probs = [0.25, 0.35, 0.15, 0.15, 0.1]

    return np.random.choice([0, 1, 2, 4, 6], p=probs)


def simulate(strategy, simulations=3000):

    wins = 0

    for _ in range(simulations):

        runs = runs_so_far
        wkts = wickets
        balls = balls_left

        while balls > 0 and wkts < 10 and runs < target:

            run = sample_run(strategy)
            runs += run

            # wicket chance
            if np.random.rand() < 0.03:
                wkts += 1

            balls -= 1

        if runs >= target:
            wins += 1

    return wins / simulations


print("Defensive bowling win prob:", simulate("defensive"))
print("Normal bowling win prob:", simulate("normal"))
print("Aggressive bowling win prob:", simulate("aggressive"))
