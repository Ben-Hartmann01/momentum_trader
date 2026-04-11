from src.data_loader import load_data, get_monthly_prices, get_monthly_returns
from src.signals import momentum_signal
from src.portfolio import compute_weights, compute_weights_signal_weighted, check_portfolio
from src.backtest import compute_returns, apply_transaction_costs
from src.metrics import performance_metrics, drawdown
from src.random_portfolio import compute_random_weights, compute_random_weights_signal_based
import matplotlib.pyplot as plt
import pandas as pd

# Stocks
tickers = [
    # Tech
    "AAPL","MSFT","AMZN","META","NVDA","GOOGL","GOOG","TSLA",
    "ADBE","CRM","ORCL","CSCO","INTC","AMD","QCOM","TXN","AVGO",

    # Finance
    "JPM","GS","MS","BAC","WFC","C","BLK","SCHW",

    # Healthcare
    "JNJ","PFE","MRK","ABBV","LLY","TMO","ABT","DHR","BMY","GILD",

    # Consumer
    "KO","PEP","WMT","COST","HD","MCD","NKE","SBUX","TGT","LOW",

    # Energy
    "XOM","CVX","COP","SLB","EOG","PSX",

    # Industrials
    "CAT","BA","GE","HON","UPS","FDX","DE","LMT","RTX","MMM",

    # Utilities / Others
    "NEE","DUK","SO","AEP","EXC",

    # Communication / Media
    "NFLX","DIS","CMCSA","VZ","T","TMUS",

    # Extra diversification
    "SPGI","ICE","ADP","INTU","ISRG","MU","PYPL","AMAT","KLAC","LRCX"
]


start_date = "2014-01-01"
end_date = "2026-04-10"

# Data
data = load_data(tickers, start_date, end_date)
prices = get_monthly_prices(data)
returns = get_monthly_returns(prices)

returns = returns.dropna()
prices = prices.loc[returns.index]

# Optimization
def select_best_params(train_prices, train_returns, lookbacks, quantiles):
    best_result = None
    best_params = None

    for lb in lookbacks:
        for q in quantiles:
            signals = momentum_signal(train_prices, lookback=lb)
            weights = compute_weights(signals, long_quantile=q, short_quantile=q)

            rets = compute_returns(weights, train_returns)
            rets_net, _ = apply_transaction_costs(weights, rets)

            ann_ret, ann_vol, sharpe = performance_metrics(rets_net.dropna())

            if best_result is None or sharpe > best_result:
                best_result = sharpe
                best_params = {"lookback": lb, "quantile": q}

    return best_params, best_result

def select_best_params_signal_weighted(train_prices, train_returns, lookbacks, quantiles):
    best_result_signal_weighted = None
    best_params_signal_weighted = None

    for lb in lookbacks:
        for q in quantiles:
            signals = momentum_signal(train_prices, lookback=lb)
            weights_signal_weighted = compute_weights_signal_weighted(signals, long_quantile=q, short_quantile=q)

            rets_signal_weighted = compute_returns(weights_signal_weighted, train_returns)
            rets_net_signal_weighted, _ = apply_transaction_costs(weights_signal_weighted, rets_signal_weighted)

            ann_ret, ann_vol, sharpe = performance_metrics(rets_net_signal_weighted.dropna())

            if best_result_signal_weighted is None or sharpe > best_result_signal_weighted:
                best_result_signal_weighted = sharpe
                best_params_signal_weighted = {"lookback": lb, "quantile": q}

    return best_params_signal_weighted, best_result_signal_weighted

# Test
def run_test_block(prices, returns, test_start, test_end, lookback, quantile):
    block_prices = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals = momentum_signal(block_prices, lookback)
    weights = compute_weights(signals, quantile, quantile)

    test_weights = weights.iloc[test_start:test_end].copy()

    #check_portfolio(test_weights, "Equal-weight strategy")

    test_returns = block_returns.iloc[test_start:test_end].copy()

    strategy_returns = compute_returns(test_weights, test_returns)
    strategy_returns_net, turnover = apply_transaction_costs(test_weights, strategy_returns)

    return test_weights, strategy_returns_net, turnover

def run_test_block_signal_weighted(prices, returns, test_start, test_end, lookback, quantile):
    block_prices = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals = momentum_signal(block_prices, lookback)
    weights_signal_weighted = compute_weights_signal_weighted(signals, quantile, quantile)

    test_weights_signal_weighted = weights_signal_weighted.iloc[test_start:test_end].copy()

    #check_portfolio(test_weights_signal_weighted, "Signal-weight strategy")

    test_returns = block_returns.iloc[test_start:test_end].copy()

    strategy_returns_signal_weighted = compute_returns(test_weights_signal_weighted, test_returns)
    strategy_returns_net_signal_weighted, turnover_signal_weighted = apply_transaction_costs(test_weights_signal_weighted, strategy_returns_signal_weighted)

    return test_weights_signal_weighted, strategy_returns_net_signal_weighted, turnover_signal_weighted

# Benchmark
def run_random_benchmark(prices, returns, test_start, test_end, lookback, quantile):
    block_prices = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals = momentum_signal(block_prices, lookback)
    weights = compute_random_weights(signals, quantile, quantile)

    test_weights = weights.iloc[test_start:test_end].copy()

    #check_portfolio(test_weights, "Random BM")

    test_returns = block_returns.iloc[test_start:test_end].copy()

    strategy_returns = compute_returns(test_weights, test_returns)
    strategy_returns_net, turnover = apply_transaction_costs(test_weights, strategy_returns)

    return test_weights, strategy_returns_net, turnover

def run_random_benchmark_signal_based(prices, returns, test_start, test_end, lookback, quantile):
    block_prices = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals = momentum_signal(block_prices, lookback)
    weights_signal_based = compute_random_weights_signal_based(signals, quantile, quantile)

    test_weights_signal_based = weights_signal_based.iloc[test_start:test_end].copy()

    #check_portfolio(test_weights_signal_based, "Signal-based BM")

    test_returns = block_returns.iloc[test_start:test_end].copy()

    strategy_returns_signal_based = compute_returns(test_weights_signal_based, test_returns)
    strategy_returns_net_signal_based, turnover_signal_based = apply_transaction_costs(test_weights_signal_based, strategy_returns_signal_based)

    return test_weights_signal_based, strategy_returns_net_signal_based, turnover_signal_based

lookbacks = [3, 6, 9, 12]
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5]

train_window = 36   # 36 months of training
test_window = 6    # 6 months of testing

all_test_returns = []
all_bm_returns = []
selected_params_history = []

n_periods = len(prices)

for test_start in range(train_window, n_periods - test_window + 1, test_window): # every possible 6 month window in whole data set
    train_start = test_start - train_window # use 36 prior months for training
    test_end = test_start + test_window

    train_prices = prices.iloc[train_start:test_start].copy()
    train_returns = returns.iloc[train_start:test_start].copy()

    best_params, best_result = select_best_params(
        train_prices, train_returns, lookbacks, quantiles
    )

    best_lb = best_params["lookback"]
    best_q = best_params["quantile"]

    # Filter returns
    test_weights, test_strategy_returns_net, _ = run_test_block(
        prices=prices,
        returns=returns,
        test_start=test_start,
        test_end=test_end,
        lookback=best_lb,
        quantile=best_q
    )

    bm_weights, bm_strategy_returns_net, _ = run_random_benchmark(
        prices=prices,
        returns=returns,
        test_start=test_start,
        test_end=test_end,
        lookback=best_lb,
        quantile=best_q
    )


    all_test_returns.append(test_strategy_returns_net)
    all_bm_returns.append(bm_strategy_returns_net)


    # add params to dates
    selected_params_history.append({
        "test_start": prices.index[test_start],
        "test_end": prices.index[test_end - 1],
        "lookback": best_lb,
        "quantile": best_q,
        "train_sharpe": best_result
    })

# Out of sample returns for strat and BM
oos_returns = pd.concat(all_test_returns).sort_index()
bm_oos_returns = pd.concat(all_bm_returns).sort_index()

oos_cumulative = (1 + oos_returns.fillna(0)).cumprod()
bm_cumulative = (1 + bm_oos_returns.fillna(0)).cumprod()

ann_ret, ann_vol, sharpe = performance_metrics(oos_returns.dropna())
dd, max_dd = drawdown(oos_cumulative)

bm_ann_ret, bm_ann_vol, bm_sharpe = performance_metrics(bm_oos_returns.dropna())
bm_dd, bm_max_dd = drawdown(bm_cumulative)


# test signal-weighted
all_test_returns_signal_weighted = []
all_bm_returns_signal_based = []
selected_params_history_signal_weighted = []

for test_start in range(train_window, n_periods - test_window + 1,
                        test_window):  # every possible 6 month window in whole data set
    train_start = test_start - train_window  # use 36 prior months for training
    test_end = test_start + test_window

    train_prices = prices.iloc[train_start:test_start].copy()
    train_returns = returns.iloc[train_start:test_start].copy()

    best_params_signal_weighted, best_result_signal_weighted = select_best_params_signal_weighted(
        train_prices, train_returns, lookbacks, quantiles
    )

    best_lb_signal_weighted = best_params_signal_weighted["lookback"]
    best_q_signal_weighted = best_params_signal_weighted["quantile"]

    # Filter returns
    test_weights_signal_based, test_strategy_returns_net_signal_weighted, _ = run_test_block_signal_weighted(
        prices=prices,
        returns=returns,
        test_start=test_start,
        test_end=test_end,
        lookback=best_lb_signal_weighted,
        quantile=best_q_signal_weighted
    )

    bm_weights_signal_based, bm_strategy_returns_net_signal_based, _ = run_random_benchmark_signal_based(
        prices=prices,
        returns=returns,
        test_start=test_start,
        test_end=test_end,
        lookback=best_lb_signal_weighted,
        quantile=best_q_signal_weighted
    )

    #print(bm_weights, bm_weights_signal_based)

    all_test_returns_signal_weighted.append(test_strategy_returns_net_signal_weighted)
    all_bm_returns_signal_based.append(bm_strategy_returns_net_signal_based)

    # add params to dates
    selected_params_history_signal_weighted.append({
        "test_start": prices.index[test_start],
        "test_end": prices.index[test_end - 1],
        "lookback": best_lb_signal_weighted,
        "quantile": best_q_signal_weighted,
        "train_sharpe": best_result_signal_weighted
    })

    # Out of sample returns for strat and BM
    oos_returns_signal_weighted = pd.concat(all_test_returns_signal_weighted).sort_index()
    bm_oos_returns_signal_based = pd.concat(all_bm_returns_signal_based).sort_index()

    oos_cumulative_signal_weighted = (1 + oos_returns_signal_weighted.fillna(0)).cumprod()
    bm_cumulative_signal_based = (1 + bm_oos_returns_signal_based.fillna(0)).cumprod()

    ann_ret_signal_weighted, ann_vol_signal_weighted, sharpe_signal_weighted = performance_metrics(oos_returns_signal_weighted.dropna())
    dd_signal_weighted, max_dd_signal_weighted = drawdown(oos_cumulative_signal_weighted)

    bm_ann_ret_signal_based, bm_ann_vol_signal_based, bm_sharpe_signal_based = performance_metrics(bm_oos_returns_signal_based.dropna())
    bm_dd_signal_based, bm_max_dd_signal_based = drawdown(bm_cumulative_signal_based)

print("WALK-FORWARD RESULTS EQUALLY WEIGHTED")
print("Strategy Annual return:", round(ann_ret, 4))
print("Strategy Annual vol:", round(ann_vol, 4))
print("Strategy Sharpe:", round(sharpe, 4))
print("Strategy Max drawdown:", round(max_dd, 4))

print("\nWALK-FORWARD RESULTS WEIGHTED SIGNAL-BASED")
print("Strategy Annual return:", round(ann_ret_signal_weighted, 4))
print("Strategy Annual vol:", round(ann_vol_signal_weighted, 4))
print("Strategy Sharpe:", round(sharpe_signal_weighted, 4))
print("Strategy Max drawdown:", round(max_dd_signal_weighted, 4))

print("\nBENCHMARK RESULTS")
print("Benchmark Annual return:", round(bm_ann_ret, 4))
print("Benchmark Annual vol:", round(bm_ann_vol, 4))
print("Benchmark Sharpe:", round(bm_sharpe, 4))
print("Benchmark Max drawdown:", round(bm_max_dd, 4))

print("\nBENCHMARK RESULTS SIGNAL-BASED")
print("Benchmark Annual return:", round(bm_ann_ret_signal_based, 4))
print("Benchmark Annual vol:", round(bm_ann_vol_signal_based, 4))
print("Benchmark Sharpe:", round(bm_sharpe_signal_based, 4))
print("Benchmark Max drawdown:", round(bm_max_dd_signal_based, 4))

params_df = pd.DataFrame(selected_params_history)
print("\nSELECTED PARAMETERS BY WINDOW (Equal-weighted)")
print(params_df)

params_df_signal_weighted = pd.DataFrame(selected_params_history_signal_weighted)
print("\nSELECTED PARAMETERS BY WINDOW (Signal-weighted)")
print(params_df_signal_weighted)

plt.figure(figsize=(10, 6))
plt.plot(oos_cumulative, label="Strategy", c="red")
plt.plot(oos_cumulative_signal_weighted, label="Signal-weighed Strat", c="black")
plt.plot(bm_cumulative, label="Benchmark", c="blue")
plt.plot(bm_cumulative_signal_based, label="Benchmark signal-based", c="purple")
plt.legend()
plt.title("Walk-Forward Strategy equal-weighted vs Walk-Forward Strategy signal-weighted vs. Benchmark")
plt.grid()
plt.show()

