import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from hyperliquid import Wallet, Exchange
import pickle
import requests
from utils import fetch_latest_candles, compute_features

load_dotenv()

# Load environment variables
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Model download helper
def download_model(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path} from {url}")
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)

# Download models if not present
long_url = "https://hyperliquid-models.s3.ap-southeast-1.amazonaws.com/long_model_xgb.pkl"
short_url = "https://hyperliquid-models.s3.ap-southeast-1.amazonaws.com/short_model_xgb.pkl"
long_model_path = "app/models/long_model_xgb.pkl"
short_model_path = "app/models/short_model_xgb.pkl"

download_model(long_url, long_model_path)
download_model(short_url, short_model_path)

# Load models
with open(long_model_path, "rb") as f:
    long_model = pickle.load(f)
with open(short_model_path, "rb") as f:
    short_model = pickle.load(f)

# Hyperliquid setup
wallet = Wallet.from_private_key(PRIVATE_KEY)
exchange = Exchange(wallet)

def run_trading_logic():
    df = fetch_latest_candles(POLYGON_API_KEY)
    df = compute_features(df)

    if len(df) < 20:
        return "Not enough data"

    features = [
        "close", "volume", "rsi", "rsi_roc", "ad", "ad_roc",
        "ad_ema", "ad_ema_roc", "price_ema", "price_ema_roc",
        "trend_ema", "trend_ema_roc", "rel_volume",
        "pos_above_support", "pos_below_resistance", "hour", "dayofweek"
    ]

    seq = df.iloc[-20:][features].values
    if np.isnan(seq).any():
        return "NaNs in input sequence"

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
        return "No trade signal"

    # Place order with fixed amount for now
    size = 0.5  # replace with dynamic sizing if needed
    order = exchange.order(coin=coin, is_buy=(side=="buy"), sz=size, limit_px=None, reduce_only=False)

    return {
        "coin": coin,
        "side": side,
        "confidence": long_conf if side == "buy" else short_conf,
        "order_result": order
    }
