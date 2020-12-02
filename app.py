from scipy.stats import lognorm, norm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import date
import streamlit as st
from PIL import Image
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
    title = f'Histogram of Daily Returns for {stock}: mu = %.2f,  std = %.2f' % (mu, std)
    plt.title(title)
    st.pyplot()

def make_lognorm_dist(col):
    st_dev = col.std()
    mean = col.mean()
    return lognorm([st_dev],loc = mean)

def post_image(image, caption = '', width = None):
    image = Image.open(image)
    st.image(image, caption = caption, width = width)

# Title of app
st.title('Stock Return Analysis')

# First Header
st.header('Choose stock to analyze.')

# Choose stock
try:
    stock = st.text_input(
        'What stock would you like to analyze? Enter ticker symbol.', 
        'aapl')
    stock = stock.upper()
except:
    st.write('Please choose a valid stock.')

# DESCRIP: data
st.write('''
    As we can below, our data consists of daily prices including the daily: 
    open, high, low, close and return. For our return statistic. We are using the 
    formula below, where P(i) = the current period\'s closing price and P(i-1) = 
    the previous period\'s closing price. The formula yields the change in closing 
    price as a percentage of the previous period\'s price.
    ''')

# IMAGE: daily_returns
post_image('daily_return.jpg', 'Daily return formula')

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
df = df.drop(['Adj Close', 'Volume'], axis = 1)

# Create Daily Returns Column
df['Returns'] = df['Close'].pct_change()
df = df.dropna(axis = 0)
st.write(df)

# Create line graph of daily closing prices
line_graph(df['Close'])

# Histogram of daily returns + fitted normal dist curve
hist_norm_curve(df['Returns'])

