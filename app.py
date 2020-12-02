import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import datetime as dt
import pandas as pd
import numpy as np 
import math

# Ignore PyPlot Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#### FUNCTIONS ####
# Price to get data from Yahoo Finance
def get_data(stock, start, end):
    stock = str(stock)
    ticker = stock.upper()
    start = str(start)
    end = str(end)
    df = web.DataReader(ticker, data_source = 'yahoo', start = start, end = end)
    return df

def create_scatter(df):
    x = df.index.tolist()
    y = df['Close'].tolist()
    y2 = df['Open'].tolist()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x, y, s = 5, c = 'b', label = 'Close')
    ax1.scatter(x, y2, s = 5, c = 'r', label = 'Open')
    plt.legend(loc = 'upper right')
    plt.show()

# Title of app
st.title('Stock Neural Networks')

# First Header
st.header('Choose stock and dates to analyze.')

# Choose stock
stock = st.text_input('What stock would you like to create a neural network for? ', 'aapl')
stock = stock.upper()
start_date = st.date_input('Start Date: ')
end_date = st.date_input('End Date: ')
df = get_data(stock, start_date, end_date)


# Used as a command to improve processing speed
# Must be followed by a function
@st.cache(allow_output_mutation = True)
def load_data(df):
    return pd.DataFrame(df)

df = load_data(df)
st.write(df)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(df.index, df['Close'], s=10, c='b', marker="s", label='Close')
plt.title(f'Closing Prices of {stock}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left');
plt.show()