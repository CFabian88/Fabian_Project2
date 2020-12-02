import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from datetime import date
import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np 
import time
import math

# Ignore PyPlot Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#### FUNCTIONS ####
# Price to get data from Yahoo Finance
def line_graph(y_col):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df.index, df['Close'], label=f'Close')
    plt.title(f'Closing Daily Prices of {stock}')
    plt.xlabel('Date')
    plt.xticks(rotation = 45)
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    st.pyplot()

# Creates histogram of data 
def hist_norm_curve(y_col):
    # Plot data as histogram
    plt.hist(y_col, bins = 25, density = True, color = 'g')
    # Create normal standardized values from data
    mu, std = norm.fit(y_col)
    # Find upper and lower bounds of x-axis
    xmin, xmax = plt.xlim()
    # Returns 100 values evenly spaced between the upper and 
    # lower bounds of the x-axis
    x = np.linspace(xmin, xmax, 100)
    # Returns pdf values based on mean and std of data
    p = norm.pdf(x, mu, std)
    # plot normal distribution curve
    plt.plot(x, p, 'k', linewidth=2)
    title = f'Histogram of Daily Closing Prices for {stock}: mu = %.2f,  std = %.2f' % (mu, std)
    plt.title(title)
    st.pyplot()

def make_lognorm_dist(col):
    st_dev = col.std()
    mean = col.mean()
    return lognorm([st_dev],loc = mean)

# Title of app
st.title('Stock Neural Networks')

# First Header
st.header('Choose stock to analyze.')

# Choose stock
try:
    stock = st.text_input('What stock would you like to create a neural network for? ', 'aapl')
    stock = stock.upper()
except:
    st.write('Please choose a valid stock.')


# Used as a command to improve processing speed
# Must be followed by a function
@st.cache(allow_output_mutation = True)
def get_data(stock):
    stock = str(stock)
    ticker = stock.upper()
    current_time = time.strftime('%Y-%m-%d', time.localtime())
    df = web.DataReader(ticker, data_source = 'yahoo', start = '2000-01-01', end = current_time)
    return pd.DataFrame(df)

# Load data
df = get_data(stock)

# Delete unwanted columns
#df.drop(['Adj_close'], axis = 1)

# Create Daily Returns Column
df['Daily_return'] = df['Close'] - df['Open']
st.write(df)

cols = list(df.columns.values)
line_graph(df['Close'])

hist_norm_curve(df['Daily_return'])