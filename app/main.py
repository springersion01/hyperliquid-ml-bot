import os
import uvicorn
from fastapi import FastAPI
from trading_logic import run_trading_logic

app = FastAPI()

@app.get("/run")
def run():
    result = run_trading_logic()
    return {"status": "success", "result": result}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
