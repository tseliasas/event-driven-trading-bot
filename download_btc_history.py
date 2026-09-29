import io
import time
import zipfile
import requests
import pandas as pd

# Binance posts every month of 1m candles as one zip file here, way faster than hitting the API 1000 candles at a time
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"

def download_month(session, symbol, month):
    url = BASE_URL.format(symbol=symbol, month=month)

    for attempt in range(3): # Try 3 times before giving up on a month
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status() # Turns a 404/500 into an actual error instead of silently saving garbage
            break
        except requests.exceptions.RequestException:
            print(f"  {month} failed, retrying in 5 seconds...")
            time.sleep(5)
    else:
        raise RuntimeError(f"Could not download {month} after 3 tries")

    # The zip lives in memory, we just open it and read the one CSV inside
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open(z.namelist()[0]) as f:
            # These files have NO header row so we name the columns ourselves (same order as data_scraper.py)
            columns = [
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            df = pd.read_csv(f, header=None, names=columns)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]

def get_btc_history(symbol="BTCUSDT", start="2017-08", end="2024-12"):
    # 1. Build the list of months we need, e.g. ["2017-08", "2017-09", ... "2024-12"]
    months = pd.period_range(start=start, end=end, freq="M").strftime("%Y-%m")
    print(f"Downloading {len(months)} months of {symbol} 1m data from Binance...")

    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'}) # Same trick as data_scraper.py so Binance doesn't get suspicious

    # 2. Download every month one by one
    all_months = []
    for month in months:
        all_months.append(download_month(session, symbol, month))
        print(f"  {month} done") # Just so we know it hasn't frozen

    df = pd.concat(all_months, ignore_index=True)

    # 3. Timestamps come in as milliseconds since 1970 (UTC), turn them into normal dates
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # During some old outages Binance saved fake zero-volume candles at weird times like 06:00:20.799
    # A real 1m candle always starts exactly on the minute, so we throw out anything that doesn't
    off_minute = df["timestamp"] != df["timestamp"].dt.floor("min")
    print(f"Dropping {off_minute.sum()} fake off-minute candles")
    df = df[~off_minute]

    return df

# 4. Grab the old history and glue it onto the 2Y file we already have
history_df = get_btc_history()
recent_df = pd.read_csv("BTCUSDT_1m_2Y.csv", parse_dates=["timestamp"])

full_df = pd.concat([history_df, recent_df], ignore_index=True)

# The two files overlap from 2024-09-28 to 2024-12-31, so kick out the repeated minutes
full_df = full_df.drop_duplicates(subset="timestamp", keep="first")
full_df = full_df.sort_values("timestamp").reset_index(drop=True)

# 5. Sanity check: how many minutes are missing (Binance had a few outages over the years)
expected_minutes = int((full_df["timestamp"].iloc[-1] - full_df["timestamp"].iloc[0]).total_seconds() / 60) + 1
print(f"\nRange: {full_df['timestamp'].iloc[0]} -> {full_df['timestamp'].iloc[-1]}")
print(f"Rows: {len(full_df)} | Missing minutes: {expected_minutes - len(full_df)}")

csv_name = "BTCUSDT_1m_full.csv"
full_df.to_csv(csv_name, index=False)
print(f"Data saved locally to {csv_name}")
