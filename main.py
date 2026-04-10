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

# Data
data = load_data(tickers, start_date, end_date)
prices = get_monthly_prices(data)
returns = get_monthly_returns(prices)

returns = returns.dropna()
prices = prices.loc[returns.index]

# 70 / 30 Split
split_idx = int(len(prices) * 0.8)

train_prices = prices.iloc[:split_idx].copy() # dont mess up the real prices & returns
test_prices = prices.iloc[split_idx:].copy()

train_returns = returns.iloc[:split_idx].copy()
test_returns = returns.iloc[split_idx:].copy()

lookbacks = [6, 9, 12]
quantiles = [0.2, 0.3, 0.4]

best_result = None
best_params = None

for lb in lookbacks:
    for q in quantiles:
        signals = momentum_signal(prices, lookback=lb)   # would need to adapt your function
        train_signals = signals.iloc[:split_idx].copy()

        train_weights = compute_weights(
            train_signals,
            long_quantile=q,
            short_quantile=q
        )

        train_rets = compute_returns(train_weights, train_returns)
        train_rets_net, _ = apply_transaction_costs(train_weights, train_rets)

        ann_ret, ann_vol, sharpe = performance_metrics(train_rets_net.dropna())

        if best_result is None or sharpe > best_result: # Measure best result with sharpe and take the corr. params
            best_result = sharpe
            best_params = {"lookback": lb, "quantile": q}

best_lb = best_params["lookback"]
best_q = best_params["quantile"]

# Test
signals = momentum_signal(prices, best_lb)
test_signals = signals.iloc[split_idx:].copy()
test_weights = compute_weights(test_signals, best_q, best_q)

test_strategy_returns = compute_returns(test_weights, test_returns)
test_strategy_returns_net, test_turnover = apply_transaction_costs(
    test_weights, test_strategy_returns
)

# Metrics on Test
cumulative_test = (1 + test_strategy_returns.fillna(0)).cumprod()
cumulative_test_net = (1 + test_strategy_returns_net.fillna(0)).cumprod()

ann_ret, ann_vol, sharpe = performance_metrics(test_strategy_returns_net.dropna())
dd, max_dd = drawdown(cumulative_test_net)


print("Best Lookback =", best_lb)
print("Best Quantile", best_q)
print("TEST RESULTS")
print("Annual return:", round(ann_ret, 4))
print("Annual vol:", round(ann_vol, 4))
print("Sharpe:", round(sharpe, 4))
print("Max drawdown:", round(max_dd, 4))

# Plot TEST only
plt.figure(figsize=(10, 6))
plt.plot(cumulative_test_net, label="Test Net")
plt.legend()
plt.title("Momentum Strategy - Test Set")
plt.grid()
plt.show()