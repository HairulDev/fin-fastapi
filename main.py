from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from services.fmp_service import get_prediction, get_company_comparison

import os, torch, joblib, numpy as np
from models.model_lstm import StockLSTM
from utils.utils import fetch_historical_prices, load_json_prices

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI"}

@app.get("/predict/{symbol}")
async def predict_price(symbol: str):
    return await get_prediction(symbol)

@app.get("/compare")
async def compare_companies(symbols: str = Query(...)):
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    return await get_company_comparison(symbol_list)

# ----------------------------------------------------------------------
# Helper prediksi LSTM untuk satu symbol
def predict_single_symbol(symbol: str) -> dict:
    symbol = symbol.upper()
    model_path = f"stock_lstm_{symbol}.pt"
    scaler_path = f"price_scaler_{symbol}.gz"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"symbol": symbol, "error": "Model or scaler not found."}

    model = StockLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    scaler = joblib.load(scaler_path)

    # gabungkan data API + JSON lokal (opsional)
    df_api = fetch_historical_prices(symbol)
    df_json = load_json_prices(symbol)
    df = (df_api if df_json.empty else
          df_api.append(df_json, ignore_index=True).drop_duplicates("date").sort_values("date"))

    prices = df["close"].values[-30:].reshape(-1, 1)
    seq_scaled = scaler.transform(prices)
    x = torch.tensor(seq_scaled).float().unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x).item()
    pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
    return {"symbol": symbol, "prediksi_besok_lstm": round(float(pred_real), 2)}

# ----------------------------------------------------------------------
# GET /predict-lstm/AAPL
@app.get("/predict-lstm/{symbol}")
def predict_lstm(symbol: str):
    """
    Prediksi 1 symbol menggunakan model & scaler masing-masing.
    """
    return predict_single_symbol(symbol)

# GET /predict-lstm-multi?symbols=AAPL,MSFT,TSLA
@app.get("/predict-lstm-multi")
def predict_lstm_multi(symbols: str = Query(...)):
    """
    Prediksi banyak symbol sekaligus.
    Contoh: /predict-lstm-multi?symbols=AAPL,MSFT,TSLA
    """
    out = []
    for sym in [s.strip().upper() for s in symbols.split(",")]:
        out.append(predict_single_symbol(sym))
    return {"results": out}
