import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression


def train_and_predict_minimum_wage():
    base_path = Path(__file__).resolve().parent.parent / "data"
    gold_price_path = base_path / "gold" / "gold_price.json"
    wage_path = base_path / "umk" / "minimum_wage_idr.json"

    with open(gold_price_path, "r") as f:
        gold_data = json.load(f)
    with open(wage_path, "r") as f:
        wage_data = json.load(f)

    gold_dict = {entry["year"]: entry["average_price_idr"] for entry in gold_data}
    years_common = sorted(list(set(gold_dict.keys()) & set(map(int, wage_data.keys()))))
    output = {}

    # === Rata-rata upah per tahun untuk inflasi riil ===
    avg_wage_per_year = {}
    for year in years_common:
        wages = [p["minimum_wage"] for p in wage_data[str(year)]]
        avg_wage_per_year[year] = np.mean(wages)

    # === Hitung inflasi riil tenaga kerja ===
    real_inflation_by_year = {}
    prev_year = None
    for year in years_common:
        if prev_year:
            gold_growth = (gold_dict[year] - gold_dict[prev_year]) / gold_dict[prev_year]
            wage_growth = (avg_wage_per_year[year] - avg_wage_per_year[prev_year]) / avg_wage_per_year[prev_year]
            real_inflation = ((1 + gold_growth) / (1 + wage_growth)) - 1
            real_inflation_by_year[year] = round(real_inflation * 100, 2)
        else:
            real_inflation_by_year[year] = None
        prev_year = year

    # === Prediksi tiap provinsi ===
    for year in years_common:
        year_str = str(year)
        output[year_str] = []
        gold_price_current = gold_dict[year]

        for province_entry in wage_data[year_str]:
            province = province_entry["province"]
            minimum_wage = province_entry["minimum_wage"]

            X, y = [], []
            for y_train in years_common:
                if y_train <= year:
                    g_price = gold_dict[y_train]
                    wages_for_year = wage_data.get(str(y_train), [])
                    for w in wages_for_year:
                        if w["province"] == province:
                            X.append([g_price])
                            y.append(w["minimum_wage"])
                            break

            if len(X) >= 3:
                model = LinearRegression()
                model.fit(np.array(X), np.array(y))
                gold_price_target = gold_dict.get(2025, gold_price_current)
                predicted_wage = model.predict(np.array([[gold_price_target]]))[0]
            else:
                predicted_wage = minimum_wage

            output[year_str].append({
                "province": province,
                "provincial_minimum_wage": minimum_wage,
                "minimum_wage_prediction": round(float(predicted_wage)),
                "average_price_gold": gold_price_current,
                "real_inflation_rate": real_inflation_by_year[year]
            })

    return output


def get_paginated_predictions(province: str = None, page: int = 1, limit: int = 10):
    data = train_and_predict_minimum_wage()
    results = []

    # Gabungkan semua tahun jadi 1 list
    for year, entries in data.items():
        for item in entries:
            item_with_year = {"year": year, **item}
            results.append(item_with_year)

    # Filter berdasarkan provinsi jika ada
    if province:
        results = [r for r in results if r["province"].lower() == province.lower()]

    total_items = len(results)
    total_pages = (total_items + limit - 1) // limit
    start = (page - 1) * limit
    end = start + limit

    paginated = results[start:end]

    return {
        "page": page,
        "limit": limit,
        "total_items": total_items,
        "total_pages": total_pages,
        "data": paginated
    }
