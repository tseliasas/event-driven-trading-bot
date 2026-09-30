import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

categorical = ['event', 'importance']
num = ['surprise', 'minutes_from_release', 'pct_change_since_release','pre_news_return_pct','pre_news_volatility','return_last_5m_pct','high_since_release_pct','low_since_release_pct','volume_ratio_since_release']

FEE_PCT = 0.1
RR = 2
SL_MULTS = [3, 5, 8, 12]
MIN_SL_PCT = 0.3


def load_data(csv_path="XGBoost_Episodic_Data.csv"):
    df = pd.read_csv(csv_path)
    df = df[df['minutes_from_release'] >= 1].copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

def add_episode_id(df):
    minutes_since_news = pd.to_timedelta(df['minutes_from_release'], unit='m')
    df['release_time'] = df['timestamp'] - minutes_since_news

    return df

def add_barrier_labels(df, sl_mult):
    df = df.copy() # We run this once per stop size, so don't mess up the original

    # How far away the stop is (in %), wider when the market was jumpy before the news
    df['sl_pct'] = df['pre_news_volatility'] * sl_mult
    df['sl_pct'] = df['sl_pct'].clip(lower=MIN_SL_PCT) # clip(lower=x) = anything smaller than x becomes x
    df['tp_pct'] = df['sl_pct'] * RR

    candles = df.drop_duplicates(subset=['release_time', 'minutes_from_release'])

    results = []

    for release_time, ep in candles.groupby('release_time'):
        high = ep['high'].to_numpy() # to_numpy() means plain array...faster to loop through
        low = ep['low'].to_numpy()
        close = ep['close'].to_numpy()
        minutes = ep['minutes_from_release'].to_numpy()
        sl = ep['sl_pct'].iloc[0] # Same for the whole episode
        tp = ep['tp_pct'].iloc[0]

        for i in range(len(ep) - 1): # imagine we buy at every candle except for the last one
            entry = close[i]
            stop_price = entry * (1 - sl / 100)
            target_price = entry * (1 + tp / 100)

            # Only candles AFTER the entry can hit anything
            future_low = low[i + 1:]
            future_high = high[i + 1:]

            # np.flatnonzero = the positions where it's True, [0] = the first one
            sl_hits = np.flatnonzero(future_low <= stop_price)
            tp_hits = np.flatnonzero(future_high >= target_price)
            first_sl = sl_hits[0] if len(sl_hits) > 0 else 999 # 999 = never hit
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 999

            if first_sl == 999 and first_tp == 999:
                outcome = (close[-1] - entry) / entry * 100 # Neither hit, sell at T+60
                hit_tp = 0
            elif first_sl <= first_tp: # Stop first, OR both in the same candle (assume the worst)
                outcome = -sl
                hit_tp = 0
            else:
                outcome = tp
                hit_tp = 1

            results.append((release_time, minutes[i], outcome, hit_tp))

    # 6. Glue the answers back onto every row (minute 60 gets NaN because it's never an entry)
    results_df = pd.DataFrame(results, columns=['release_time', 'minutes_from_release', 'trade_outcome_pct', 'target_hit_tp'])
    df = df.merge(results_df, on=['release_time', 'minutes_from_release'], how='left')

    return df

def split_by_time(df, val_frac=0.15, test_frac=0.15):
    unique_times = df['release_time'].drop_duplicates().sort_values().reset_index(drop=True)

    val_start = unique_times.iloc[int(len(unique_times) * (1 - val_frac - test_frac))]
    test_start = unique_times.iloc[int(len(unique_times) * (1 - test_frac))]

    train_df = df[df['release_time'] < val_start].copy()
    val_df = df[(df['release_time'] >= val_start) & (df['release_time'] < test_start)].copy()
    test_df = df[df['release_time'] >= test_start].copy()

    return train_df, val_df, test_df

def get_X_y(df):
    df = df.dropna(subset=['target_hit_tp'])

    X = df[categorical + num]
    y = df['target_hit_tp'].astype(int)

    return X, y

def build_model(pos_weight):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=pos_weight))
    ])

    return model

def replay(model, df, threshold):
    df = df.dropna(subset=['target_hit_tp']).copy()

    X, y = get_X_y(df)
    df['proba'] = model.predict_proba(X)[:,1]

    minutes = df.groupby(['release_time', 'minutes_from_release']).agg(
        proba=('proba', 'mean'),
        trade_outcome=('trade_outcome_pct', 'first')
    ).reset_index()

    signals = minutes[minutes['proba'] > threshold]
    trades = signals.groupby('release_time').head(1)

    return trades['trade_outcome'] - FEE_PCT

def pick_threshold(model, val_df):
    print("\nThreshold search (validation):")
    best_threshold = None
    best_total = float('-inf') # -infinity so the first real result always beats it

    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        profits = replay(model, val_df, threshold)

        if len(profits) < 30:
            continue

        total = profits.sum()
        print(f"  {threshold}: {len(profits)} trades | win rate {(profits > 0).mean():.1%} | avg {profits.mean():.3f}% | total {total:.1f}%")

        if total > best_total:
            best_threshold = threshold
            best_total = total

    return best_threshold

df = load_data()
df = add_episode_id(df)

results = []

for sl_mult in SL_MULTS:
    print(f"\n===== Stop = {sl_mult} x volatility | TP = {sl_mult * RR} x volatility =====")

    labeled = add_barrier_labels(df, sl_mult)
    train_df, val_df, test_df = split_by_time(labeled)

    X_train, y_train = get_X_y(train_df)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_model(pos_weight)
    model.fit(X_train, y_train)

    threshold = pick_threshold(model, val_df)
    if threshold is None:
        continue

    val_total = replay(model, val_df, threshold).sum()
    results.append({'sl_mult': sl_mult, 'threshold': threshold, 'val_total': val_total, 'model': model, 'test_df': test_df})

best = max(results, key=lambda r: r['val_total'])
print(f"\nBest on validation: stop {best['sl_mult']}x, threshold {best['threshold']}, total {best['val_total']:.1f}%")

if best['val_total'] <= 0:
    print("Nothing made money on validation -> not opening the test set yet")
else:
    profits = replay(best['model'], best['test_df'], best['threshold'])
    print(f"TEST: {len(profits)} trades | win rate {(profits > 0).mean():.1%} | avg {profits.mean():.3f}% | total {profits.sum():.1f}%")
