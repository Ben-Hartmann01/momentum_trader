# Quantitative Momentum Strategy Backtesting (In Progress)

## Status

This project is currently under active development. Core functionality is implemented, but the framework is still being improved and extended. Results should not be considered final or production-ready.

---

## Project Overview

This project implements a systematic long/short equity strategy based on cross-sectional momentum, together with a modular backtesting framework.

The main objectives are:

* to build a clean and extensible research pipeline
* to evaluate strategy robustness using walk-forward testing
* to compare performance against a benchmark 
* to assess whether the signal and portfolio construction provide meaningful value
---

## Strategies Description

The current 2 compared strategies are cross-sectional momentum/mean_reversion strategies:
1. Equal-weight Strategy
   1. A signal is computed based on past price performance over a given lookback window 

   2. Assets are ranked according to the signal
      * we use a momentum signal and a momentum x mean reversion signal and compare both

   3. Portfolio construction:

      * top quantile is taken long
      * bottom quantile is taken short
      * everything in between stays neutral
      * positions are equal-weighted within each side
      * Net exposures 0, 0.5, 1 get tested (gross expsoure = 2) -> weights within each direction are equal, but cross-directional only in the case of net exposure being 0

   4. The portfolio is rebalanced on a monthly basis (ME)

2. Signal-based-weight Strategy
   1. A signal is computed based on past price performance over a given lookback window 

   2. Assets are ranked according to the signal
      * we use a momentum signal and a momentum x mean reversion signal and compare both

   3. Portfolio construction:

      * top quantile assets are taken as long candidates
      * bottom quantile assets are taken as short candidates
      * everything in between stays neutral
      * positions are weighted based on their normalized signal (signal / sum(signals) in this quantile)
      * Net exposures 0, 0.5, 1 get tested (gross expsoure = 2) 

   4. The portfolio is rebalanced on a monthly basis (ME)

---

## Walk-Forward Backtesting

Instead of a single train/test split, the project uses a rolling (walk-forward) validation approach:

* train on a fixed window (right now 36 months)
* select the best parameters (lookback and quantile) on the training window (Grid is adjustable)
* test on the subsequent period (right now 6 months)
* repeat this process through time (Training on months 0-35, Test on months 36-41; Training on months 6-41; Test on months 42-47)

This ensures:

* no look-ahead bias
* no reuse of test data
* a more realistic out-of-sample evaluation

---

## Performance Evaluation

The following metrics are computed:

* annualized return (mean monthly return multiplied by 12)
* annualized volatility
* Sharpe ratio
* maximum drawdown
* turnover (based on changes in portfolio weights)

Returns are calculated using lagged weights to avoid look-ahead bias. Transaction costs are incorporated based on portfolio turnover.

---

## Benchmarks

Buy and hold strategy as Benchmark

* every asset gets weight 1 / n
* hold each position until the end of the maturity
* we therefore have no transaction costs


---

## Project Structure

```text
src/
├── data_loader.py
├── signals.py
├── portfolio.py
├── backtest.py
├── benchmark.py
├── metrics.py
├── random_portfolio.py

main.py
```

---

## Current Limitations

* simplified transaction cost model 
* weak benchmark (Its tough to not outperform the first BM)
* no statistical significance testing
* there is an obvious improvement of the strategies due to more net exposure

---

## Planned Improvements

* clean plot
* improved transaction cost modeling (check realistic values)
* different benchmarks 
* additional signals
* improved reporting and visualization
* get rid of the market exposure bias

---

## How to Run

1. Install dependencies:

   ```
   pip install pandas numpy yfinance matplotlib
   ```

2. Run the main script:

   ```
   python main.py
   ```
