# create trading signals

def momentum_signal(prices, lookback):
    return prices.shift(1) / prices.shift(lookback) - 1 # P_i(t-1) / P_i(t-12) - 1; change of price from 12 months ago to 1 month ago in %

