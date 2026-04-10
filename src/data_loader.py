import yfinance as yf

# load the data from yf
def load_data(tickers, start_date, end_date):
    data = yf.download(tickers, start = start_date, end = end_date, auto_adjust = True)[("Close")] # adjust prices by dividends
    return data

# filter monthly prices out of data
def get_monthly_prices(data):
    return data.resample("ME").last() # group into monthly buckets and get last available price at this month

# filter returns out of the prices
def get_monthly_returns(prices):
    return prices.pct_change() # vector of changes in %
