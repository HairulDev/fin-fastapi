import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch, numpy as np, joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from models.model_lstm import StockLSTM
from utils.utils import fetch_price  # <-- sudah ada di utils

symbols = ["AAPL", "MSFT", "GOOGL"]

seq_len = 30

for symbol in symbols:
    print(f"\n=== Training {symbol} ===")
    
    # ambil data gabungan API + JSON lokal
    df = fetch_price(symbol)
    prices = df['close'].values[::-1].reshape(-1,1)

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(len(prices_scaled) - seq_len):
        X.append(prices_scaled[i:i+seq_len])
        y.append(prices_scaled[i+seq_len])
    X, y = torch.tensor(np.array(X)).float(), torch.tensor(np.array(y)).float()
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
        print(f"{symbol} epoch {epoch+1} loss {loss.item():.4f}")

    torch.save(model.state_dict(), f"stock_lstm_{symbol}.pt")
    joblib.dump(scaler, f"price_scaler_{symbol}.gz")
    print(f"Model & scaler for {symbol} saved.")
