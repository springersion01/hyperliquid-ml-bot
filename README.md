# hyperliquid-ml-bot

A 24/7 trading bot for Hyperliquid using machine learning models.

## What It Does

- Pulls 15-minute candle data from Polygon.io
- Computes indicators like RSI, AD, EMA
- Uses trained XGBoost models to decide long/short/no trade
- Places market orders on Hyperliquid via wallet
- Runs as a FastAPI server (`/run` endpoint)

## Setup

1. Add your environment variables:
   - `PRIVATE_KEY`: your wallet private key (used for signing)
   - `POLYGON_API_KEY`: your Polygon.io API key

2. Install dependencies:
