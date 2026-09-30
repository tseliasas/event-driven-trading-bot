import pandas as pd

# Question: does the news move happen so fast that it's already over by the time we can enter (close of minute 1)?

CHECKPOINTS = [0, 1, 5, 15, 30, 60]

df = pd.read_csv("XGBoost_Episodic_Data.csv", usecols=['timestamp', 'open', 'close', 'event', 'importance', 'surprise_score', 'minutes_from_release'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['release_time'] = df['timestamp'] - pd.to_timedelta(df['minutes_from_release'], unit='m')

# 1. One row per news EVENT with the price at each checkpoint
# pivot = turn the minutes into columns, so each event becomes one row: close_0, close_1, close_5...
prices = df[df['minutes_from_release'].isin(CHECKPOINTS)].pivot_table(
    index=['release_time', 'event'], columns='minutes_from_release', values='close'
)
prices.columns = [f"close_{int(c)}" for c in prices.columns]

# The price the exact second the news drops = the OPEN of the T=0 candle
open_0 = df[df['minutes_from_release'] == 0].set_index(['release_time', 'event'])['open']
info = df[df['minutes_from_release'] == 0].set_index(['release_time', 'event'])[['importance', 'surprise_score']]
events = prices.join(open_0.rename('open_0')).join(info).dropna(subset=[f"close_{c}" for c in CHECKPOINTS] + ['open_0'])

# 2. How far has the price moved since the news dropped, at each checkpoint (in %)
for c in CHECKPOINTS:
    events[f"move_{c}"] = (events[f"close_{c}"] - events['open_0']) / events['open_0'] * 100

# What's left for US: from our earliest entry (close of minute 1) to the end (T+60)
events['after_entry'] = (events['close_60'] - events['close_1']) / events['close_1'] * 100

# Price stuff only needs one row per news MOMENT (reports at the same minute share candles)
moments = events.reset_index().drop_duplicates('release_time')

print(f"{len(moments)} news moments, {len(events)} events\n")

print("A) How big is the move so far? (median size, in %)")
for c in CHECKPOINTS:
    print(f"   by minute {c:>2}: {moments[f'move_{c}'].abs().median():.3f}%")
print(f"   AFTER our entry (minute 1 -> 60): {moments['after_entry'].abs().median():.3f}%")

print("\nB) Is the final direction already decided early? (same up/down at minute X as at T+60)")
for c in [0, 1, 5, 15, 30]:
    same = ((moments[f'move_{c}'] > 0) == (moments['move_60'] > 0)).mean()
    print(f"   minute {c:>2}: {same:.1%}  (50% = coin flip)")

print("\nC) Does a BIGGER surprise mean a BIGGER move? (rank correlation: 0 = no link, 1 = perfect)")
print("   Only events that have a surprise score")
scored = events.reset_index().dropna(subset=['surprise_score']).copy() # reset_index turns 'event' back into a normal column
scored['abs_score'] = scored['surprise_score'].abs()
for imp in ['high', 'medium', 'low']:
    s = scored[scored['importance'] == imp]
    first_min = s['abs_score'].corr(s['move_1'].abs(), method='spearman')
    after = s['abs_score'].corr(s['after_entry'].abs(), method='spearman')
    print(f"   {imp:<6} ({len(s):>4} events): surprise vs first-minute move {first_min:+.2f} | surprise vs move AFTER our entry {after:+.2f}")

print("\nD) Direction for the biggest reports: when the number beats the forecast, which way does BTC go?")
print("   (+ = higher number -> BTC up, - = higher number -> BTC down)")
top = scored[scored['importance'] == 'high']['event'].value_counts().head(8).index
for ev in top:
    s = scored[scored['event'] == ev]
    first_min = s['surprise_score'].corr(s['move_1'], method='spearman')
    after = s['surprise_score'].corr(s['after_entry'], method='spearman')
    print(f"   {ev[:40]:<40} ({len(s):>3}): first minute {first_min:+.2f} | after our entry {after:+.2f}")
