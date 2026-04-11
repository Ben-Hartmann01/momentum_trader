import pandas as pd

# basic fixed number, same weight approach
def get_weights(signal_row, long_quantile, short_quantile):
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

def compute_weights(signal_df, long_quantile, short_quantile):
    weights_list = []
    for date in signal_df.index: # creates a matrix with the calculated weights dates x stocks
        w = get_weights(signal_df.loc[date], long_quantile, short_quantile) # sample weights for all stocks at this point in time
        w.name = date
        weights_list.append(w)

    return pd.DataFrame(weights_list).fillna(0.0)

# signal-weighted approach
def get_weights_signal_weighted(signal_row, long_quantile, short_quantile):
    signal_row = signal_row.dropna()

    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long = max(1, int(n * long_quantile))
    n_short = max(1, int(n * short_quantile)) # we still define candidates

    ranked = signal_row.sort_values()

    short_stocks = ranked.index[:n_short]
    long_stocks = ranked.index[-n_long:]

    weights = pd.Series(0.0, index=signal_row.index)

    # make sure signs are positive / negative --> no "wrong" moves
    long_signals = signal_row.loc[long_stocks].clip(lower=0)
    short_signals = (-signal_row.loc[short_stocks]).clip(lower=0)

    # fallback in case sums are zero or negative for some reason
    if long_signals.sum() <= 0:
        weights.loc[long_stocks] = 1.0 / n_long
    else:
        weights.loc[long_stocks] = long_signals / long_signals.sum()

    if short_signals.sum() <= 0:
        weights.loc[short_stocks] = -1.0 / n_short
    else:
        weights.loc[short_stocks] = -short_signals / short_signals.sum()

    return weights

def compute_weights_signal_weighted(signal_df, long_quantile, short_quantile):
    weights_list = []
    for date in signal_df.index: # creates a matrix with the calculated weights dates x stocks
        w = get_weights_signal_weighted(signal_df.loc[date], long_quantile, short_quantile) # sample weights for all stocks at this point in time
        w.name = date
        weights_list.append(w)

    return pd.DataFrame(weights_list).fillna(0.0)

def check_portfolio(weights, name):
    long_sum = weights.clip(lower=0).sum(axis=1)
    short_sum = -weights.clip(upper=0).sum(axis=1)
    net_sum = weights.sum(axis=1)

    print(name)
    print("avg long exposure :", long_sum.mean())
    print("avg short exposure:", short_sum.mean())
    print("avg net exposure  :", net_sum.mean())