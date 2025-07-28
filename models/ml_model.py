def simple_moving_average(prices: list[float]) -> float:
    return round(sum(prices) / len(prices), 2)
