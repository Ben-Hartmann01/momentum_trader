from src.data_loader import load_data, get_monthly_prices, get_monthly_returns
from src.signals import momentum_signal
from src.portfolio import compute_weights
from src.backtest import compute_returns, apply_transaction_costs
from src.metrics import performance_metrics, drawdown

import matplotlib.pyplot as plt

# Stocks
tickers = [
    "AAPL", "MSFT", "AMZN", "META", "NVDA",
    "JPM", "GS", "XOM", "KO", "WMT"
]

start_date = "2018-01-01"
end_date = "2025-01-01"

# Pipeline
data = load_data(tickers, start_date, end_date)
prices = get_monthly_prices(data)
returns = get_monthly_returns(prices)

train_prices = prices.loc[: "2021-12-31"]
test_prices = prices.loc["2022-01-31":]

signal = momentum_signal(prices)
weights = compute_weights(signal)

strategy_returns = compute_returns(weights, returns)
strategy_returns_net, turnover = apply_transaction_costs(weights, strategy_returns)

cumulative = (1 + strategy_returns).cumprod()
cumulative_net = (1 + strategy_returns_net).cumprod()

# Metrics
ann_ret, ann_vol, sharpe = performance_metrics(strategy_returns_net)
dd, max_dd = drawdown(cumulative_net)

print("Annual return:", round(ann_ret, 4))
print("Annual vol:", round(ann_vol, 4))
print("Sharpe:", round(sharpe, 4))
print("Max drawdown:", round(max_dd, 4))

# Plot
plt.figure(figsize=(10,6))
plt.plot(cumulative, label="Gross")
plt.plot(cumulative_net, label="Net")
plt.legend()
plt.title("Momentum Strategy")
plt.grid()
plt.show()