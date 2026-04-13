from src.data_loader import load_data, get_monthly_prices, get_monthly_returns
from src.signals import momentum_signal
from src.portfolio import compute_weights, compute_weights_signal_weighted
from src.backtest import compute_returns, apply_transaction_costs
from src.metrics import performance_metrics, drawdown
from src.random_portfolio import compute_random_weights, compute_random_weights_signal_based
from src.benchmark import compute_equal_weight_long_only_weights

import matplotlib.pyplot as plt
import pandas as pd

# Data
tickers = [
    # Tech
    "AAPL", "MSFT", "AMZN", "META", "NVDA", "GOOGL", "GOOG", "TSLA",
    "ADBE", "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM", "TXN", "AVGO",

    # Finance
    "JPM", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW",

    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY", "GILD",

    # Consumer
    "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",

    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX",

    # Industrials
    "CAT", "BA", "GE", "HON", "UPS", "FDX", "DE", "LMT", "RTX", "MMM",

    # Utilities / Others
    "NEE", "DUK", "SO", "AEP", "EXC",

    # Communication / Media
    "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",

    # Extra diversification
    "SPGI", "ICE", "ADP", "INTU", "ISRG", "MU", "PYPL", "AMAT", "KLAC", "LRCX",
]

start_date = "2014-01-01"
end_date = "2026-04-10"

lookbacks = [3, 6, 9, 12]
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]
net_exposures = [0.0, 0.5, 1.0]
target_gross_exposure = 2

train_window = 36
# 36 months training

test_window = 6
# 6 months testing


# Variable functions depending on method and strat/bm
def get_weight_function(method: str, benchmark: bool = False):
    if benchmark:
        if method == "equal":
            return compute_random_weights
        elif method == "signal":
            return compute_random_weights_signal_based
    else:
        if method == "equal":
            return compute_weights
        elif method == "signal":
            return compute_weights_signal_weighted

    raise ValueError(f"Unknown method: {method}")

# Grid search
def select_best_params(train_prices, train_returns, lookbacks, quantiles, method="equal", target_net_exposure=0.0, target_gross_exposure=2.0):
    best_sharpe = None
    best_params = None

    weight_func = get_weight_function(method=method, benchmark=False)

    for lb in lookbacks:
        for q in quantiles:
            signals = momentum_signal(train_prices, lookback=lb)

            weights = weight_func(
                signals,
                long_quantile=q,
                short_quantile=q,
                target_net_exposure=target_net_exposure,
                target_gross_exposure=target_gross_exposure,
            )

            rets = compute_returns(weights, train_returns)
            rets_net, _ = apply_transaction_costs(weights, rets)

            _, _, sharpe = performance_metrics(rets_net.dropna())

            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {"lookback": lb, "quantile": q}

    return best_params, best_sharpe

# Runners
def run_test_block(prices, returns, test_start, test_end, lookback, quantile, method="equal", benchmark=False, target_net_exposure=0.0, target_gross_exposure=2.0):
    block_prices = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals = momentum_signal(block_prices, lookback=lookback)
    weight_func = get_weight_function(method=method, benchmark=benchmark)

    weights = weight_func(
        signals,
        long_quantile=quantile,
        short_quantile=quantile,
        target_net_exposure=target_net_exposure,
        target_gross_exposure=target_gross_exposure,
    )

    test_weights = weights.iloc[test_start:test_end].copy()
    test_returns = block_returns.iloc[test_start:test_end].copy()

    strategy_returns = compute_returns(test_weights, test_returns)
    strategy_returns_net, turnover = apply_transaction_costs(test_weights, strategy_returns)

    return test_weights, strategy_returns_net, turnover

# New BM
def run_equal_weight_benchmark_block(prices, returns, test_start, test_end):
    block_prices = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    benchmark_weights = compute_equal_weight_long_only_weights(block_prices) # signals dont matter due to benchmark strat

    test_weights = benchmark_weights.iloc[test_start:test_end].copy()
    test_returns = block_returns.iloc[test_start:test_end].copy()

    benchmark_returns = compute_returns(test_weights, test_returns)
    benchmark_returns_net, turnover = apply_transaction_costs(test_weights, benchmark_returns)

    return test_weights, benchmark_returns_net, turnover

# Results
def summarize_results(returns_series: pd.Series):
    cumulative = (1 + returns_series.fillna(0)).cumprod()
    ann_ret, ann_vol, sharpe = performance_metrics(returns_series.dropna())
    dd, max_dd = drawdown(cumulative)

    return {
        "returns": returns_series,
        "cumulative": cumulative,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "drawdown": dd,
        "max_dd": max_dd,
    }

# Training/testing 36 + 6
def run_walk_forward(prices, returns, lookbacks, quantiles, train_window, test_window, method="equal", target_net_exposure=0.0, target_gross_exposure=2.0):
    all_strategy_returns = []
    all_benchmark_returns = []
    selected_params_history = []

    n_periods = len(prices)

    for test_start in range(train_window, n_periods - test_window + 1, test_window):
        train_start = test_start - train_window
        test_end = test_start + test_window

        train_prices = prices.iloc[train_start:test_start].copy()
        train_returns = returns.iloc[train_start:test_start].copy()

        best_params, best_sharpe = select_best_params(
            train_prices=train_prices,
            train_returns=train_returns,
            lookbacks=lookbacks,
            quantiles=quantiles,
            method=method,
            target_net_exposure=target_net_exposure,
            target_gross_exposure=target_gross_exposure,
        )

        best_lb = best_params["lookback"]
        best_q = best_params["quantile"]

        _, strategy_returns_net, _ = run_test_block(
            prices=prices,
            returns=returns,
            test_start=test_start,
            test_end=test_end,
            lookback=best_lb,
            quantile=best_q,
            method=method,
            benchmark=False,
            target_net_exposure=target_net_exposure,
            target_gross_exposure=target_gross_exposure,
        )

        _, benchmark_returns_net, _ = run_equal_weight_benchmark_block(
            prices=prices,
            returns=returns,
            test_start=test_start,
            test_end=test_end,
        )

        all_strategy_returns.append(strategy_returns_net)
        all_benchmark_returns.append(benchmark_returns_net)

        selected_params_history.append(
            {
                "test_start": prices.index[test_start],
                "test_end": prices.index[test_end - 1],
                "lookback": best_lb,
                "quantile": best_q,
                "train_sharpe": best_sharpe,
            }
        )

    strategy_oos_returns = pd.concat(all_strategy_returns).sort_index()
    benchmark_oos_returns = pd.concat(all_benchmark_returns).sort_index()

    strategy_summary = summarize_results(strategy_oos_returns)
    benchmark_summary = summarize_results(benchmark_oos_returns)
    params_df = pd.DataFrame(selected_params_history)

    return {
        "strategy": strategy_summary,
        "benchmark": benchmark_summary,
        "params": params_df,
    }

# Key numbers printer for all Portfolios
def print_summary(title, summary):
    print(f"\n{title}")
    print("Annual return:", round(summary["ann_ret"], 4))
    print("Annual vol:", round(summary["ann_vol"], 4))
    print("Sharpe:", round(summary["sharpe"], 4))
    print("Max drawdown:", round(summary["max_dd"], 4))


# Run everything
def main():
    # Load data
    data = load_data(tickers, start_date, end_date)
    prices = get_monthly_prices(data)
    returns = get_monthly_returns(prices)

    returns = returns.dropna()
    prices = prices.loc[returns.index]

    # Run both strategies
    equal_results_by_net = {}
    signal_results_by_net = {}

    for net in net_exposures:
        equal_results_by_net[net] = run_walk_forward(
            prices=prices,
            returns=returns,
            lookbacks=lookbacks,
            quantiles=quantiles,
            train_window=train_window,
            test_window=test_window,
            method="equal",
            target_net_exposure=net,
            target_gross_exposure=target_gross_exposure,
        )

        signal_results_by_net[net] = run_walk_forward(
            prices=prices,
            returns=returns,
            lookbacks=lookbacks,
            quantiles=quantiles,
            train_window=train_window,
            test_window=test_window,
            method="signal",
            target_net_exposure=net,
            target_gross_exposure=target_gross_exposure,
        )

    benchmark_results = equal_results_by_net[1.0]["benchmark"]

    # Print strategy summaries
    for net, results in equal_results_by_net.items():
        print_summary(f"WALK-FORWARD RESULTS EQUALLY WEIGHTED (NET={net})", results["strategy"])

    for net, results in signal_results_by_net.items():
        print_summary(f"WALK-FORWARD RESULTS SIGNAL-WEIGHTED (NET={net})", results["strategy"])

    # Print benchmark summaries
    print_summary("BENCHMARK RESULTS (EQUAL-WEIGHT LONG-ONLY)", benchmark_results)

    # Print selected parameters
    for net, results in equal_results_by_net.items():
        print(f"\nSELECTED PARAMETERS BY WINDOW (Equal-weighted, NET={net})")
        print(results["params"])

    for net, results in signal_results_by_net.items():
        print(f"\nSELECTED PARAMETERS BY WINDOW (Signal-weighted, NET={net})")
        print(results["params"])

    # Plot cum_returns
    plt.figure(figsize=(10, 6))

    for net, results in equal_results_by_net.items():
        plt.plot(results["strategy"]["cumulative"], label=f"Equal-weighted (Net={net})")

    for net, results in signal_results_by_net.items():
        plt.plot(results["strategy"]["cumulative"], label=f"Signal-weighted (Net={net})")

    plt.plot(benchmark_results["cumulative"], label="Benchmark (Equal-weight Long-only)")

    plt.legend()
    plt.title("Walk-Forward Strategies by Net Exposure")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()


