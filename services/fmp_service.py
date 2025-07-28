import httpx
import os
import time
from dotenv import load_dotenv
from models.ml_model import simple_moving_average

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = os.getenv("API_FMP")

# cache dictionary
_cache = {
    "profile": {},
    "metrics": {}
}
TTL = 60 * 60  # 1 jam

def is_cache_valid(cache_entry):
    return time.time() - cache_entry["timestamp"] < TTL

async def get_profile(symbol: str):
    if symbol in _cache["profile"] and is_cache_valid(_cache["profile"][symbol]):
        return _cache["profile"][symbol]["data"]

    url = f"{BASE_URL}/profile/{symbol}?apikey={FMP_API_KEY}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    _cache["profile"][symbol] = {"data": data, "timestamp": time.time()}
    return data


async def get_key_metrics(symbol: str):
    if symbol in _cache["metrics"] and is_cache_valid(_cache["metrics"][symbol]):
        return _cache["metrics"][symbol]["data"]

    url = f"{BASE_URL}/key-metrics-ttm/{symbol}?limit=1&apikey={FMP_API_KEY}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    _cache["metrics"][symbol] = {"data": data, "timestamp": time.time()}
    return data


async def get_company_comparison(symbols: list[str]):
    result = []

    for symbol in symbols:
        try:
            profile = await get_profile(symbol)
            metrics = await get_key_metrics(symbol)

            profile_data = profile[0] if profile else {}
            metrics_data = metrics[0] if metrics else {}

            # Hitung profit margin manual jika tersedia
            revenue = metrics_data.get("revenuePerShareTTM")
            income = metrics_data.get("netIncomePerShareTTM")
            profit_margin = income / revenue if income and revenue else None

            result.append({
                "symbol": symbol,
                "companyName": profile_data.get("companyName"),
                "peRatio": metrics_data.get("peRatioTTM"),
                "roe": metrics_data.get("roeTTM"),
                "debtEquity": metrics_data.get("debtToEquityTTM"),
                "profitMargin": profit_margin
            })

        except Exception as e:
            result.append({"symbol": symbol, "error": str(e)})

    return result


async def get_prediction(symbol: str):
    url = f"{BASE_URL}/historical-price-full/{symbol}?apikey={FMP_API_KEY}&serietype=line"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

    data = response.json()
    historical = data.get("historical", [])

    if not historical or len(historical) < 5:
        return {"error": "Data historis tidak cukup"}

    closing_prices = [item["close"] for item in historical[:5]]
    prediction = simple_moving_average(closing_prices)

    return {
        "symbol": symbol,
        "prediction": prediction,
        "used_data": closing_prices
    }
