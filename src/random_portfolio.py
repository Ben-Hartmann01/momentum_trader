import pandas as pd
import numpy as np

def get_random_weights(signal_row, long_quantile, short_quantile, rng):
    signal_row = signal_row.dropna()

    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype = float)

    n_long = max(1, int(n * long_quantile))
    n_short = max(1, int(n * short_quantile)) # we go long / short on 0.3 * n assets and at least 1 each

    names = signal_row.index.values

    if len(names) < (n_short + n_long):
        return pd.Series(0.0, index = signal_row.index) # make sure there are enough assets to apply the strat

    choice = rng.choice(names, size = n_long + n_short, replace = False) # generates random sample of assets

    long_stocks = choice[:n_long]
    short_stocks = choice[n_long:] # half of them short, half long

    weights = pd.Series(0.0, index = signal_row.index) # initializes a pd.Series with 0.0s anywhere; with indices equal to signal:row
    weights.loc[long_stocks] = 1.0 / n_long
    weights.loc[short_stocks] =  -1.0 / n_short # every taken asset gets weight equal to 1 / n_s/l; rest stays 0 - keep it in the pf

    return weights

def compute_random_weights(signal_df, long_quantile, short_quantile):
    rng = np.random.default_rng() # generate random number generator for random choice

    weights_list = []
    for date in signal_df.index: # creates a matrix with the calculated weights dates x stocks
        w = get_random_weights(signal_df.loc[date], long_quantile, short_quantile, rng) # sample weights for all stocks at this point in time
        w.name = date
        weights_list.append(w)

    return pd.DataFrame(weights_list).fillna(0.0)