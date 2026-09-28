import pandas as pd
import numpy as np

def clean_economic_data(csv_path="Economic_calendar_US.csv"):
    print("Loading and cleaning Economic Calendar...")
    df = pd.read_csv(csv_path)
    
    # 1. Filter for US Dollar events and drop "All Day" times
    df = df[df['currency'] == 'USD'].copy()
    df = df[df['time'] != 'All Day'].copy()
    
    # 2. Fix the Timezone (EST to UTC)
    df['datetime_str'] = df['date'] + ' ' + df['time']
    
    # Use errors='coerce' to turn "Tentative" strings into NaT (Not a Time)
    df['datetime'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    
    # Drop the rows that couldn't be parsed
    df = df.dropna(subset=['datetime']).copy()
    
    # Localize to Eastern Time, then convert to UTC to match Binance
    df['datetime'] = df['datetime'].dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward')
    df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    
    # Remove the timezone "awareness" so it matches the naive Binance timestamps perfectly
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    # 3. Build a parser for strings like "3.5%", "250K", "-1.2M"
    def parse_value(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().upper()
        multiplier = 1
        
        if '%' in val:
            val = val.replace('%', '')
        if 'K' in val:
            val = val.replace('K', '')
            multiplier = 1_000
        if 'M' in val:
            val = val.replace('M', '')
            multiplier = 1_000_000
        if 'B' in val:
            val = val.replace('B', '')
            multiplier = 1_000_000_000
            
        try:
            return float(val) * multiplier
        except ValueError:
            return np.nan

    # Apply the parser to the Actual and Forecast columns
    df['actual_num'] = df['actual'].apply(parse_value)
    df['forecast_num'] = df['forecast'].apply(parse_value)
    
    # 4. ENGINEER THE CORE FEATURE: Surprise Factor
    df['surprise'] = df['actual_num'] - df['forecast_num']
    
    # Drop rows where we couldn't calculate a surprise (e.g., missing forecast)
    df = df.dropna(subset=['surprise']).copy()
    
    # Sort chronologically
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Keep only what we need for the ML model
    df = df[['datetime', 'event', 'importance', 'actual', 'forecast', 'surprise']]
    
    return df

clean_news_df = clean_economic_data()
print(f"\nSuccessfully cleaned {len(clean_news_df)} actionable US events.")
print(clean_news_df.head(10))

def build_episodic_dataset():
    # 1. Get the clean news
    news_df = clean_economic_data()
    
    # 2. Load the continuous Bitcoin data
    print("Loading Binance BTC data...")
    btc_df = pd.read_csv("BTCUSDT_1m_2Y.csv")
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    
    # We must set the timestamp as the index to quickly slice the 120-minute windows
    btc_df.set_index('timestamp', inplace=True)
    
    all_episodes = []
    
    print(f"Extracting 120-minute episodes for {len(news_df)} events...")
    
    # 3. Iterate through every single news event
    for _, news_row in news_df.iterrows():
        news_time = news_row['datetime']
        
        # Calculate the start and end of our 120-minute window
        window_start = news_time - pd.Timedelta(minutes=60)
        window_end = news_time + pd.Timedelta(minutes=60)
        
        # Slice the continuous Bitcoin data to get just this 120-minute episode
        # Use .copy() to avoid SettingWithCopy warnings later
        episode_df = btc_df.loc[window_start:window_end].copy()
        
        # If the exchange was down or we have missing data, skip this event
        if len(episode_df) < 120:
            continue
            
        # 4. Inject the News Data into every row of the episode
        # The model needs to know what the news was, even at minute +45
        episode_df['event'] = news_row['event']
        episode_df['importance'] = news_row['importance']
        episode_df['surprise'] = news_row['surprise']
        
        # 5. Create the "Clock" feature
        # Calculate how many minutes each row is from the actual release time (T=0)
        episode_df['minutes_from_release'] = (episode_df.index - news_time).total_seconds() / 60.0
        
        # 6. Engineer Dynamic Features
        # How much has the price moved since the news dropped?
        # Get the closing price exactly at T=0
        try:
            price_at_release = episode_df.loc[news_time, 'close']
            episode_df['pct_change_since_release'] = (episode_df['close'] - price_at_release) / price_at_release * 100
        except KeyError:
            # If the exact T=0 candle is missing, skip the episode
            continue
            
        # 7. Engineer the Dynamic Target (For rows after T=0)
        # If I enter a trade right now, what is the return at the end of the episode (T+60)?
        price_at_end = episode_df.iloc[-1]['close']
        episode_df['target_return_to_end_pct'] = (price_at_end - episode_df['close']) / episode_df['close'] * 100
        
        # We only want the bot to look for entries AFTER the news drops.
        # We will set the target to NaN for the pre-news rows so we don't train on them.
        episode_df.loc[episode_df['minutes_from_release'] <= 0, 'target_return_to_end_pct'] = np.nan
        episode_df['target_direction'] = (episode_df['target_return_to_end_pct'] > 0).astype(int)
        
        # Reset index to turn timestamp back into a normal column
        episode_df.reset_index(inplace=True)
        
        all_episodes.append(episode_df)
        
    # Combine all individual episodes into one massive training dataset
    final_df = pd.concat(all_episodes, ignore_index=True)
    
    return final_df

final_ml_data = build_episodic_dataset()
final_ml_data.to_csv("XGBoost_Episodic_Data.csv", index=False)
print(f"\nSaved {len(final_ml_data)} rows to XGBoost_Episodic_Data.csv")