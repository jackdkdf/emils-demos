import pandas as pd

# Load the csv file for a single team
df = pd.read_csv('3dmax.csv')

# Ensure column names are clean if there is any whitespace
df.columns = [col.strip() for col in df.columns]

# Normalize the 'result' column to identify wins
df['is_win'] = df['result'].str.upper().str.startswith('W')

# 1. Win percentage against each opponent
win_vs_opponent = (
    df.groupby('opponentname')['is_win']
    .agg(['count', 'sum'])
    .reset_index()
    .rename(columns={'count': 'num_matches', 'sum': 'num_wins'})
)
win_vs_opponent['win_percentage_vs_opponent'] = win_vs_opponent['num_wins'] / win_vs_opponent['num_matches'] * 100

# 2. Win percentage on each map
win_on_map = (
    df.groupby('mapname')['is_win']
    .agg(['count', 'sum'])
    .reset_index()
    .rename(columns={'count': 'num_matches', 'sum': 'num_wins'})
)
win_on_map['win_percentage_on_map'] = win_on_map['num_wins'] / win_on_map['num_matches'] * 100

# Show results
print("Win % vs Opponents:")
print(win_vs_opponent[['opponentname', 'num_matches', 'num_wins', 'win_percentage_vs_opponent']])
print("\nWin % on Maps:")
print(win_on_map[['mapname', 'num_matches', 'num_wins', 'win_percentage_on_map']])