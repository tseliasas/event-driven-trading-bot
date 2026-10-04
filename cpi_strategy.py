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

SCORE_MIN = 1
EMERGENCY_STOP_PCT = 2

candles = cpi.drop_duplicates(['release_time', 'minutes_from_release'])
close = candles.pivot(index='release_time', columns='minutes_from_release', values='close')
high = candles.pivot(index='release_time', columns='minutes_from_release', values='high')


def short_result(release_time):
    entry = close.loc[release_time, 1]
    stop_price = entry * (1 + EMERGENCY_STOP_PCT / 100)
    later_highs = high.loc[release_time, 2:60]

    if (later_highs >= stop_price).any():
        return -EMERGENCY_STOP_PCT
    return (entry - close.loc[release_time, 60]) / entry * 100


def run_rule(part, label):
    trades = part[part['score'] > SCORE_MIN]
    results = pd.Series([short_result(t) for t in trades.index], index=trades.index, dtype=float) - FEE_PCT

    print(f"\n===== {label}: short if score > {SCORE_MIN}, enter minute 1, exit T+60, emergency stop {EMERGENCY_STOP_PCT}% =====")
    print(f"{len(part)} CPI releases | {len(results)} trades | win rate {(results > 0).mean():.0%} | avg {results.mean():.3f}% | total {results.sum():.2f}%")
    for t, r in results.items():
        print(f"  {t:%Y-%m-%d} | score {trades.loc[t, 'score']:+.2f} | {r:+.3f}%")

    return results


run_rule(history, "HISTORY")
run_rule(holdout, "HOLDOUT (first and only look)")
