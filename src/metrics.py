import numpy as np

def performance_metrics(returns):
    mean_monthly = returns.mean() # returns are given on a monthly basis already
    vol_monthly = returns.std()

    annual_return = mean_monthly * 12
    annual_vol = vol_monthly * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    return annual_return, annual_vol, sharpe

def drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax() # stores the highest value reached so far
    dd = cumulative_returns / running_max - 1 # current value / max value - 1; dd_min is max drawdown - lowest negative number
    return dd, dd.min()


