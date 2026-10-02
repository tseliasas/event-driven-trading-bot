import pandas as pd
from train_model import load_data, add_episode_id, add_barrier_labels, FEE_PCT

CPI_EVENTS = ['CPI (MoM)', 'Core CPI (MoM)', 'CPI (YoY)']
SL_MULT = 5
HOLDOUT = 20

df = add_episode_id(load_data())
cpi = df[df['event'].isin(CPI_EVENTS)].copy()

long_df = add_barrier_labels(cpi, SL_MULT, 'long')
short_df = add_barrier_labels(cpi, SL_MULT, 'short')

long_1 = long_df[long_df['minutes_from_release'] == 1]
short_1 = short_df[short_df['minutes_from_release'] == 1]

moments = long_1.groupby('release_time').agg(
    score=('surprise_score', 'mean'),
    long_result=('trade_outcome_pct', 'first')
)
moments['short_result'] = short_1.groupby('release_time')['trade_outcome_pct'].first()
moments = moments.dropna(subset=['score']).sort_index()

history = moments.iloc[:-HOLDOUT].copy()
holdout = moments.iloc[-HOLDOUT:].copy()

bins = [float('-inf'), -1, -0.3, 0.3, 1, float('inf')]
names = ['very cold', 'cold', 'in line', 'hot', 'very hot']
history['bucket'] = pd.cut(history['score'], bins=bins, labels=names)

summary = history.groupby('bucket', observed=True).agg(
    releases=('score', 'size'),
    long_avg=('long_result', 'mean'),
    short_avg=('short_result', 'mean')
)
summary['long_avg'] = summary['long_avg'] - FEE_PCT
summary['short_avg'] = summary['short_avg'] - FEE_PCT

print(f"{len(history)} CPI releases studied, {len(holdout)} locked away\n")
print(summary.round(3))
