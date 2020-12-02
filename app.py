import pandas_datareader.data as web
import matplotlib.pyplot as plt
from scipy.stats import lognorm
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
    '''x = range(len(y_col) + 1)
    y = range(int(round(y_col.max(), 0)))'''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df.index, df['Close'], label=f'Close')
    plt.title(f'Closing Daily Prices of {stock}')
    plt.xlabel('Date')
    plt.xticks(rotation = 45)
    plt.ylabel('Price')
    plt.legend(loc='upper left')
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
df.drop(['Adj Close'], axis = 1)

# Create Daily Returns Column
df['Daily_return'] = df['Close'] - df['Open']
st.write(df)

cols = list(df.columns.values)
line_graph(df['Close'])

