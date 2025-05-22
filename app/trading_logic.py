import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from hyperliquid import Wallet, Exchange
import pickle
from utils import fetch_latest_candles, compute_features

load_dotenv()

# Load environment variables
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Load models
with open("app/models/long_model_xgb_v2.pkl", "rb") as f:
    long_model = pickle.load(f)
with open("app/models/short_model_xgb.pkl", "rb") as f:
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
