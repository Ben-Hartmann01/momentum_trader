# create trading signals

def momentum_signal(prices, lookback):
    return prices.shift(1) / prices.shift(lookback) - 1 # P_i(t-1) / P_i(t-12) - 1; change of price from 12 months ago to 1 month ago in %

def mean_reversion_signal(prices, lookback=1):
    return -(prices.shift(1) / prices.shift(1 + lookback) - 1) # -(P_i(t-1) / P_i(t-2) - 1); we assume some mean reversion after short-term price increasements

def composite_signal(prices, momentum_lookback=12, reversion_lookback=1, # combine signals
                     momentum_weight=0.7, reversion_weight=0.3):
    mom = momentum_signal(prices, momentum_lookback)
    rev = mean_reversion_signal(prices, reversion_lookback)

    # Z-scores: (Stock - mu(all stocks)) / sigma
    mom_z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)
    rev_z = rev.sub(rev.mean(axis=1), axis=0).div(rev.std(axis=1), axis=0) # for each date, take mean over all assets, subtract it from all assets; take sigma of all assets, divide

    return momentum_weight * mom_z + reversion_weight * rev_z