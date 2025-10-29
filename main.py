from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import asyncio
import os, torch, joblib, numpy as np
import shutil
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from models.model_lstm import StockLSTM
from utils.utils import fetch_historical_prices, load_json_prices, fetch_price
from services.fmp_service import get_prediction, get_company_comparison
from services.wage_prediction_service import train_and_predict_minimum_wage
from services.wage_prediction_service import get_paginated_predictions


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
    df = pd.concat([df_api, df_json], ignore_index=True).drop_duplicates("date").sort_values("date") \
        if not df_json.empty else df_api

    prices = df["close"].values[-30:].reshape(-1, 1)
    seq_scaled = scaler.transform(prices)
    x = torch.tensor(seq_scaled).float().unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(x).item()
    pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
    return {"symbol": symbol, "tommorow_prediction": round(float(pred_real), 2)}

async def train_api():
    """
    Generator async yang men-stream log training LSTM untuk beberapa symbol.
    """
    symbols = ["AAPL", "MSFT", "GOOGL"]
    seq_len = 30

    for symbol in symbols:
        yield f"\n=== Training {symbol} ===\n"
        await asyncio.sleep(0)  # beri kesempatan event loop

        # ambil data gabungan API + JSON lokal
        df = fetch_price(symbol)
        prices = df["close"].values[::-1].reshape(-1, 1)

        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices)

        X, y = [], []
        for i in range(len(prices_scaled) - seq_len):
            X.append(prices_scaled[i:i + seq_len])
            y.append(prices_scaled[i + seq_len])

        X = torch.tensor(np.array(X)).float()
        y = torch.tensor(np.array(y)).float()
        loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

        model = StockLSTM()
        loss_fn = torch.nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(20):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            msg = f"{symbol} epoch {epoch + 1} loss {loss.item():.4f}\n"
            yield msg
            await asyncio.sleep(0)

        torch.save(model.state_dict(), f"stock_lstm_{symbol}.pt")
        joblib.dump(scaler, f"price_scaler_{symbol}.gz")
        yield f"Model & scaler for {symbol} saved.\n"
        await asyncio.sleep(0)
# ----------------------------------------------------------------------

@app.get("/predict-wage")
def predict_minimum_wage(
    province: str | None = Query(None, description="Province name (optional)"),
    page: int = Query(1, ge=1, description="Page number for pagination"),
    limit: int = Query(10, ge=1, le=999, description="Number of items per page")
):
    return get_paginated_predictions(province, page, limit)

# GET /predict-lstm/AAPL
@app.get("/predict-lstm/{symbol}")
def predict_lstm(symbol: str):
    """
    Prediksi 1 symbol menggunakan model & scaler masing-masing.
    """
    return predict_single_symbol(symbol)

# GET /predict-lstm-multi?symbols=AAPL,MSFT,GOOGL
@app.get("/predict-lstm-multi")
def predict_lstm_multi(symbols: str = Query(...)):
    """
    Prediksi banyak symbol sekaligus.
    Contoh: /predict-lstm-multi?symbols=AAPL,MSFT,GOOGL
    """
    out = []
    for sym in [s.strip().upper() for s in symbols.split(",")]:
        out.append(predict_single_symbol(sym))
    return {"results": out}

@app.post("/train-lstm")
async def train_lstm():
    """
    Trigger training semua symbol dan kirim log secara streaming (SSE).
    Test dengan: curl -N -X POST http://localhost:8000/train-lstm
    atau Postman (tab 'Raw', keep-alive).
    """
    return StreamingResponse(train_api(), media_type="text/plain")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)

    file_path = os.path.join("data", file.filename)
    is_overwrite = os.path.exists(file_path)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "message": "File overwritten successfully" if is_overwrite else "File uploaded successfully"
    }