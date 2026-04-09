import pandas as pd

def get_weights(signal_row, long_quantile = 0.3, short_quantile = 0.3):
    signal_row = signal_row.dropna()

    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype = float) # no signals, return no data

    n_long = max(1, int(n * long_quantile))
    n_short = max(1, int(n * short_quantile)) # we go long / short on 0.3 * n assets and at least 1 each

    ranked = signal_row.sort_values()

    short_stocks = ranked.index[:n_short]
    long_stocks = ranked.index[-n_long:] # take worst n_short assets and best n_long assets; remember their position via .index

    weights = pd.Series(0.0, index = signal_row.index) # initializes a pd.Series with 0.0s anywhere; with indices equal to signal:row
    weights.loc[long_stocks] = 1.0 / n_long
    weights.loc[short_stocks] =  -1.0 / n_short # every taken asset gets weight equal to 1 / n_s/l; rest stays 0 - keep it in the pf

    return weights

def compute_weights(signal_df):
    weights_list = []
    for date in signal_df.index: # creates a matrix with the calculated weights dates x stocks
        w = get_weights(signal_df.loc[date]) # sample weights for all stocks at this point in time
        w.name = date
        weights_list.append(w)

    return pd.DataFrame(weights_list).fillna(0.0)




