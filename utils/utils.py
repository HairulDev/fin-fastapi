import os
import json
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from dotenv import load_dotenv

# Load environment variables dari .env
load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")
BASE_URL = os.getenv("API_FMP", "https://financialmodelingprep.com/api/v3")


def fetch_historical_prices(symbol: str) -> pd.DataFrame:
    """Ambil data harga dari FMP API"""
    url = f"{BASE_URL}/historical-price-full/{symbol}?apikey={FMP_API_KEY}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json().get("historical", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df[['date', 'close']].sort_values("date")
    df['close'] = df['close'].astype(float)
    return df


def load_json_prices(symbol: str, folder: str = "data") -> pd.DataFrame:
    """Baca file JSON lokal & ambil harga 'companyProfile.price' jika ada."""
    path = os.path.join(folder, f"{symbol}.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        j = json.load(f)
    price = j.get("companyProfile", {}).get("price")
    if price is None:
        return pd.DataFrame()
    # Buat 1 baris pseudo-date agar bisa digabung
    return pd.DataFrame([{"date": "extra", "close": float(price)}])


def fetch_price(symbol: str) -> pd.DataFrame:
    """Gabungkan data FMP API dan JSON lokal menjadi satu DataFrame."""
    df_api = fetch_historical_prices(symbol)
    df_json = load_json_prices(symbol)
    if df_json.empty:
        df = df_api
    else:
        df = pd.concat([df_api, df_json], ignore_index=True)
        df = df.drop_duplicates("date").sort_values("date")
    return df


def build_dataset(df: pd.DataFrame, look_back=60):
    """Buat dataset X,y untuk training LSTM"""
    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i - look_back:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y, scaler


def train_and_predict(symbol: str) -> float:
    """Latih model LSTM dan prediksi harga terakhir"""
    df_api = fetch_historical_prices(symbol)
    df_local = load_json_prices(symbol)
    df = pd.concat([df_api, df_local], ignore_index=True).drop_duplicates('date').sort_values("date")

    if len(df) < 61:
        raise ValueError(f"Data {symbol} kurang untuk training")

    X, y, scaler = build_dataset(df)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_seq = X[-1].reshape(1, X.shape[1], 1)
    pred_scaled = model.predict(last_seq, verbose=0)
    return float(scaler.inverse_transform(pred_scaled)[0][0])
