

###########################################################
# Libraries
###########################################################
import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

###########################################################
# Functions
###########################################################


# Take Stock Data From Yahoo
def get_stock(stock_code, start_date=[2019,1,1], end_date=[2019,12,31]):
    """This function gets the stock data within the desired date range from Yahoo Finance. The data includes
    High, Low, Open, Close, Volume, Adj Close  prices. It returns a DataFrame

    :param stock_code: It is a string. The code of the stock on Yahoo Fınace, For example for  Amazon; "AMZN"
    :param start_date: It is a List. The starting date of data.  For example ; [2019,1,1]. [2019,1,1] by default
    :param end_date: It is a List. The  end date of data.  For example ; [2019,12,31]. [2019,12,31] by default
    :return: returns the all data (opened, closed, volume, adj close etc) of stock
    """
    stock=pdr.get_data_yahoo(symbols=stock_code,
                            start=datetime(start_date[0],start_date[1],start_date[2],),
                            end=datetime(end_date[0],end_date[1],end_date[2],))
    return stock


# Creat a portfolio with wanted stocks
def creat_portfolio(stock_list,start_date,end_date,price_type="Adj Close"):
    """ Creats porfolio within the desired stocks and  date range from Yahoo Finance, Gets the Stocks' price and puts on DataFrame,
    Price type can be adjusted. For examaple it can be open, closed, adjusted close, high ect.

    :param stock_list: It is a list or array. Is has toks code which will be included in portfolio. For example ["AAPL","AMZN","T"]
    :param start_date: It is a List. The starting date of data.  For example ; [2019,1,1].
    :param end_date: It is a List. The  end date of data.  For example ; [2019,12,31].
    :param price_type: which price type is the target price ( "Open", "Close", "Adj Close").  "Adj Close" by default
    :return: A DataFrame. Retuns the portfolio dataframe
    """
    portfolio = pd.DataFrame()
    for stock in stock_list:
        portfolio[stock] = pdr.get_data_yahoo(symbols=stock,
                                              start=datetime(start_date[0],start_date[1],start_date[2]),
                                              end=datetime(end_date[0],end_date[1],end_date[2],))[price_type]
    return portfolio



# Visualize the Portfolio or Stock Data
def plot_pf_price(portfolio):
    """It is plot the price of all stocks in portfolio

    :param portfolio: Portfolio DataFrame that include stock prices
    :return: Plot the daily price of all stocks
    """
    fig, ax=plt.subplots(figsize=(15,6))
    for stock in portfolio.columns:
        ax.plot(portfolio.loc[:,stock],label=stock);
        ax.legend(fontsize="x-large")
    plt.show()


# Calculate the portfolio return with wanted wegihts
def w_portfolio(portfolio, weight):
    """It calculates the daily return of portfolio with desired weights.

    :param portfolio: Portfolio DataFrame that include stock prices
    :param weight: array or list, desired weights
    :return: A pandas series. Daily returns of portfolio
    """
    return_daily=portfolio.pct_change()
    portfolio_returns = return_daily.mul(weight, axis=1).sum(axis=1)
    return portfolio_returns


# Calculate portfolio volatility
def portf_vol(portfolio, weight,st_exch_day=260):
    """ It calucltes portfolio volatility by covariance matrix and weights

    :param portfolio:It is a DataFrame, Portfolio DataFrame that include stock prices
    :param weight: A numpy array or list. Portfolio weight set
    :param st_exch_day: An integer or float. Is is stock excahnge days in a year.
    :return: An integer. Portfolio volatility
    """
    cov_matrix_annual=(portfolio.pct_change()).cov()*st_exch_day
    portfolio_volatility=np.sqrt(np.dot(weight.T,
                                        np.dot(cov_matrix_annual,
                                               weight)))
    return portfolio_volatility



# Creat random portrfolio weight sets with wanted range
def creat_random_portf(portfolio,size=5000):
    """It creats up to size diffirent weight sets for each stocks in a dataframe.

    :param portfolio:It is a DataFrame, Portfolio DataFrame that include stock prices
    :param size: An integer. Number of Diffirent Potfolio sets
    :return: A DataFrame. Is include size diffirent portfolio weight sets
    """
    RandomPortfolio = pd.DataFrame()
    for i in range(size):
        weight = np.random.random(portfolio.shape[1])
        weight /= np.sum(weight)
        RandomPortfolio.loc[i,portfolio.columns] = weight
    return RandomPortfolio


# Create Random Portfolio, Calculate  the portfolio metrics and sharp Ratio and plot efficent frontier
def markowitz_portfolio(portfolio, risk_free=0, size=5000, plot=False):
    """ Creats a random portfolio weight sets and calculates Markowitz porfolio metrics and save on DataFrame.

    :param portfolio:It is a DataFrame, Portfolio DataFrame that include stock prices
    :param risk_free: An integer or float. Risk-free rate on stock market
    :param plot: A boolen. İf True, then plots some portfolio cum price and Markowitz Random portfolio
    return-volatility scatter plot
    :return: Return two data frame, first Random Portfolio Dataframe, second is weight-returns sets Dataframe
    """

    portfolio_return = portfolio.pct_change()
    col = portfolio.columns
    RandomPortfolio = pd.DataFrame()
    for i in range(size):
        weight = np.random.random(portfolio.shape[1])
        weight /= np.sum(weight)
        RandomPortfolio.loc[i, col] = weight

    RandomPortfolio["Return"] = np.dot(RandomPortfolio.loc[:, col], portfolio_return.sum())
    RandomPortfolio["Volatility"] = RandomPortfolio.apply(lambda x: portf_vol(portfolio, x[0:portfolio.shape[1]]),
                                                          axis=1)
    RandomPortfolio['Sharpe'] = (RandomPortfolio["Return"] - risk_free) / RandomPortfolio["Volatility"]

    numstock = portfolio.shape[1]
    e_w = np.repeat(1 / numstock, numstock)
    msr_w = np.array(RandomPortfolio.sort_values(by="Sharpe", ascending=False).iloc[0:1, 0:numstock])
    gmv_w = np.array(RandomPortfolio.sort_values(by="Volatility").iloc[0:1, 0:numstock])

    df = portfolio.pct_change()
    port_returns = portfolio.pct_change()
    port_returns["Return_EW"] = port_returns.loc[:, col].mul(e_w, axis=1).sum(axis=1)
    port_returns["Return_MSR"] = port_returns.loc[:, col].mul(msr_w, axis=1).sum(axis=1)
    port_returns["Return_GMV"] = port_returns.loc[:, col].mul(gmv_w, axis=1).sum(axis=1)
    weight_set = pd.DataFrame()
    weight_set.loc["e_w", col] = e_w
    weight_set.loc["msr_w", col] = msr_w
    weight_set.loc["gmv_w", col] = gmv_w
    weight_set.loc["e_w", "Return"] = port_returns["Return_EW"].sum()
    weight_set.loc["msr_w", "Return"] = port_returns["Return_MSR"].sum()
    weight_set.loc["gmv_w", "Return"] = port_returns["Return_GMV"].sum()

    if plot:
        ((1 + port_returns[["Return_EW", "Return_MSR", "Return_GMV"]]).cumprod() - 1).plot()
        plt.show()
        sns.scatterplot(x="Volatility", y="Return", data=RandomPortfolio)
        sns.scatterplot(x="Volatility", y="Return",
                        data=RandomPortfolio.sort_values(by="Sharpe", ascending=False).iloc[0:1], color="red",
                        marker="*", s=500)
        sns.scatterplot(x="Volatility", y="Return", data=RandomPortfolio.sort_values(by="Volatility").iloc[0:1],
                        color="orange", marker="*", s=500)
        plt.show()

    return RandomPortfolio, weight_set



