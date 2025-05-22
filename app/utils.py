import httpx
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_latest_candles(polygon_api_key, symbol="SOLUSD", multiplier=15, timespan="minute", limit=100):
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}/range/{multiplier}/{timespan}/now"
    params = {"apiKey": polygon_api_key, "limit": limit}
    resp = httpx.get(url, params=params)
    data = resp.json()

    bars = []
    for bar in data.get("results", []):
        bars.append({
            "timestamp": datetime.utcfromtimestamp(bar["t"] / 1000),
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar["v"]
        })

    return pd.DataFrame(bars)

def compute_features(df):
    df = df.copy()
    df["rsi"] = df["close"].rolling(14).apply(lambda x: ta_rsi(x), raw=False)
    df["ad"] = (2 * df["close"] - df["low"] - df["high"]) / (df["high"] - df["low"]).replace(0, 1) * df["volume"]
    df["ad"] = df["ad"].cumsum()
    df["ad_ema"] = df["ad"].ewm(span=12).mean()
    df["price_ema"] = df["close"].ewm(span=12).mean()
    df["trend_ema"] = df["close"].ewm(span=25).mean()
    df["avg_volume"] = df["volume"].rolling(20).mean()

    df["rsi_roc"] = df["rsi"].diff()
    df["ad_roc"] = df["ad"].diff()
    df["ad_ema_roc"] = df["ad_ema"].diff()
    df["price_ema_roc"] = df["price_ema"].diff()
    df["trend_ema_roc"] = df["trend_ema"].diff()
    df["rel_volume"] = df["volume"] / df["avg_volume"]

    df["support"] = df["low"].rolling(10).min()
    df["resistance"] = df["high"].rolling(10).max()
    df["pos_above_support"] = df["close"] > df["support"]
    df["pos_below_resistance"] = df["close"] < df["resistance"]

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    return df

def ta_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
