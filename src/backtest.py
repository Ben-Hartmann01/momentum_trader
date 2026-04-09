def compute_returns(weights, returns):
    strategy_returns = (weights.shift(1) * returns).sum(axis = 1) # weights from t-1 for the returns afterwards
    return strategy_returns

def apply_transaction_costs(weights, strategy_returns, cost_rate = 0.001):
    turnover = weights.diff().abs().sum(axis = 1) # abs sum of all weight changes, meaning sells and buys
    costs = turnover * cost_rate

    net_returns = strategy_returns - costs
    return net_returns, turnover