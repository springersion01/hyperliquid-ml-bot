import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from trading_logic import run_trading_logic

app = FastAPI()

@app.get("/")
def hello():
    return {"status": "alive"}

@app.get("/run")
def run():
    try:
        result = run_trading_logic()
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
