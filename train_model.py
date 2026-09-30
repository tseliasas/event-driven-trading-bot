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
SL_MULTS = [3, 5, 8, 12]   # stop sizes to try, in multiples of pre_news_volatility
MIN_SL_PCT = 0.3   # min sl


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

df = load_data()
print(len(df))
print(df['minutes_from_release'].min(), df['minutes_from_release'].max())

df = add_episode_id(df)

labeled = add_barrier_labels(df, 5)
print(labeled['target_hit_tp'].isna().sum()) # should be 10193 (one minute-60 row per episode)
print(labeled['target_hit_tp'].mean()) # share of minutes where TP came first
print(labeled['trade_outcome_pct'].describe())