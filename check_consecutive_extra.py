"""Check if any two or more consecutive draws share the same extra number."""

import pandas as pd

df = pd.read_csv("data/raw_draws.csv")

runs = []
start = 0
for i in range(1, len(df)):
    if df.loc[i, "extra"] != df.loc[i - 1, "extra"]:
        if i - start >= 2:
            runs.append(df.iloc[start:i])
        start = i
# Check final run
if len(df) - start >= 2:
    runs.append(df.iloc[start:])

if not runs:
    print("No consecutive draws with the same extra number found.")
else:
    for run in runs:
        print(f"--- {len(run)} consecutive draws with extra = {run.iloc[0]['extra']} ---")
        for _, row in run.iterrows():
            print(f"  {row['date']}  draw {row['draw']}  extra={row['extra']}")
        print()
