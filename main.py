from fastapi import FastAPI, Query
from services.fmp_service import get_prediction, get_company_comparison
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS if used from frontend
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
    # Split string seperti "AAPL,MSFT" menjadi list ['AAPL', 'MSFT']
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    return await get_company_comparison(symbol_list)
