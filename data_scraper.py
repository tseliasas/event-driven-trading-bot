import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def get_2_years_binance_data(symbol="BTCUSDT", interval="1m", years=2):
    print(f"Fetching {years} years of {interval} data for {symbol}...")
    # Just a notice to know its working properly
    url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_date = datetime.now() - timedelta(days=365 * years)
    start_time = int(start_date.timestamp() * 1000)
    # Basically when to start and end
    all_candles = [] 
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }) # Because binance kept on getting suspicious
    
    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000, # 1000 candles at a time
            "startTime": start_time,
            "endTime": end_time
        }
        
        try:
            response = session.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data:
                break
                
            all_candles.extend(data)
            start_time = data[-1][0] + 1
            
            # Print progress so you know it hasn't frozen
            if len(all_candles) % 50000 == 0:
                print(f"Downloaded {len(all_candles)} candles so far...")
                
            time.sleep(0.1)
            
        except requests.exceptions.ConnectionError:
            print("\nConnection dropped by Binance! Sleeping for 5 seconds and retrying...")
            time.sleep(5)
            continue # This jumps back to the top of the loop and tries the exact same request again
            
    # Structure the final dataset
    columns = [
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    
    df = pd.DataFrame(all_candles, columns=columns)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    return df

btc_df = get_2_years_binance_data()
print(f"\nSuccessfully downloaded {len(btc_df)} rows of data.")

csv_name = "BTCUSDT_1m_2Y.csv"
btc_df.to_csv(csv_name)
print(f"Data saved locally to {csv_name}")