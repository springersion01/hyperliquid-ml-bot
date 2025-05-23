import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from hyperliquid.wallet import Wallet
from hyperliquid.exchange import Exchange
import pickle
import requests
import logging

logging.basicConfig(level=logging.INFO)
logging.info("🔁 Starting model setup...")

load_dotenv()

# Load environment variables
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# === Model download helper ===
def download_model(url, local_path):
    try:
        if not os.path.exists(local_path):
            logging.info(f"⬇️ Downloading model from {url} ...")
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
        else:
            logging.info(f"✅ Model already exists at {local_path}")
    except Exception as e:
        logging.error(f"❌ Failed to download {url}: {e}")
        raise

# === Model URLs ===
long_url = "https://hyperliquid-models.s3.ap-southeast-1.amazonaws.com/long_model_xgb.pkl"
short_url = "https://hyperliquid-models.s3.ap-southeast-1.amazonaws.com/short_model_xgb.pkl"
long_model_path = "models/long_model_xgb.pkl"
short_model_path = "models/short_model_xgb.pkl"

# === Load models ===
try:
    download_model(long_url, long_model_path)
    with open(long_model_path, "rb") as f:
        long_model = pickle.load(f)
    logging.info("✅ Long model loaded")
except Exception as e:
    logging.error("❌ Failed to load long model: %s", e)
    long_model = None

try:
    download_model(short_url, short_model_path)
    with open(short_model_path, "rb") as f:
        short_model = pickle.load(f)
    logging.info("✅ Short model loaded")
except Exception as e:
    logging.error("❌ Failed to load short model: %s", e)
    short_model = None

# === Initialize wallet & exchange ===
try:
    wallet = Wallet.from_private_key(PRIVATE_KEY)
    exchange = Exchange(wallet)
    logging.info("✅ Hyperliquid wallet initialized")
except Exception as e:
    logging.error("❌ Failed to initialize Hyperliquid wallet: %s", e)
    exchange = None

# === Fetch candles from Polygon ===
def fetch_latest_candles(polygon_api_key, symbol="SOLUSD", multiplier=15, timespan="minute", limit=100):
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}/range/{multiplier}/{timespan}/now"
    params = {"apiKey": polygon_api_key, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=10)
        logging.info(f"📦 Polygon raw response (truncated): {response.text[:200]}")
        if 'application/json' not in response.headers.get('Content-Type', ''):
            logging.error("❌ Unexpected content type. Possibly HTML or error page.")
            return pd.DataFrame()
        data = response.json()
        if not data.get("results"):
            logging.error("❌ Polygon API returned no results")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"❌ Failed to fetch candles: {e}")
        return pd.DataFrame()

    bars = []
    for bar in data.get("results", []):
        bars.append({
            "timestamp": pd.to_datetime(bar["t"], unit="ms"),
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar["v"]
        })
    return pd.DataFrame(bars)

# === Technical indicators ===
def compute_features(df):
    df = df.copy()
    if "close" not in df.columns:
        raise ValueError("Missing 'close' column in DataFrame")

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

# === Trading Logic ===
def run_trading_logic():
    if long_model is None or short_model is None or exchange is None:
        return {"status": "error", "message": "Model or wallet initialization failed."}

    df = fetch_latest_candles(POLYGON_API_KEY)

    if df.empty:
        return {"status": "error", "message": "Empty DataFrame from Polygon"}

    if "close" not in df.columns:
        logging.error(f"❌ Missing 'close' column. Columns found: {df.columns.tolist()}")
        return {"status": "error", "message": "Missing 'close' column"}

    try:
        df = compute_features(df)
    except Exception as e:
        logging.error(f"❌ Feature computation error: {e}")
        return {"status": "error", "message": f"Feature error: {e}"}

    if len(df) < 20:
        return {"status": "error", "message": "Not enough data"}

    features = [
        "close", "volume", "rsi", "rsi_roc", "ad", "ad_roc",
        "ad_ema", "ad_ema_roc", "price_ema", "price_ema_roc",
        "trend_ema", "trend_ema_roc", "rel_volume",
        "pos_above_support", "pos_below_resistance", "hour", "dayofweek"
    ]

    seq = df.iloc[-20:][features].values
    if np.isnan(seq).any():
        return {"status": "error", "message": "NaNs in input sequence"}

    X = seq.reshape(1, -1)
    long_pred = long_model.predict(X)[0]
    long_conf = long_model.predict_proba(X)[0][1]
    short_pred = short_model.predict(X)[0]
    short_conf = short_model.predict_proba(X)[0][1]

    coin = "SOL"
    side = None

    if long_pred == 1 and short_pred == 0:
        side = "buy"
    elif short_pred == 1 and long_pred == 0:
        side = "sell"
    else:
        return {"status": "ok", "message": "No trade signal"}

    size = 0.5
    try:
        order = exchange.order(coin=coin, is_buy=(side == "buy"), sz=size, limit_px=None, reduce_only=False)
    except Exception as e:
        return {"status": "error", "message": f"Order failed: {e}"}

    return {
        "status": "success",
        "coin": coin,
        "side": side,
        "confidence": long_conf if side == "buy" else short_conf,
        "order_result": order
    }

