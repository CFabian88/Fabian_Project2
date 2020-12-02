import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import datetime as dt
from datetime import date
import pandas as pd
import numpy as np 
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

# Title of app
st.title('Stock Neural Networks')

# First Header
st.header('Choose stock and dates to analyze.')

# Choose stock
stock = st.text_input('What stock would you like to create a neural network for? ', 'aapl')
stock = stock.upper()


# Used as a command to improve processing speed
# Must be followed by a function
@st.cache(allow_output_mutation = True)
def get_data(stock):
    stock = str(stock)
    ticker = stock.upper()
    today = date.today
    today = today.strftime('%Y-%m-%d')
    df = web.DataReader(ticker, data_source = 'yahoo', start = '2000-01-01', end = today)
    return pd.DataFrame(df)

df = get_data(stock)
st.write(df)

choice = st.text_input('Choose Type of Data to Show', df.columns)
line_graph(df[choice])