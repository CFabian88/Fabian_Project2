from scipy.stats import lognorm, norm, skew, kurtosis, skewtest, kurtosistest
import pandas_datareader.data as web
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import date
import streamlit as st
from PIL import Image
import datetime as dt
import pandas as pd
import numpy as np 
import time
import math
import os

# Ignore PyPlot Warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

#### FUNCTIONS ####
# Price to get data from Yahoo Finance
def line_graph(y_col):
    graph = px.line(
        x = df.index, 
        y = y_col,
        title = f'Closing Prices for {stock}',
        labels = {
            'y' : 'Closing Price ($)', 
            'x' : 'Date'
        }
        )
    st.plotly_chart(graph)

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
    title = f'Histogram of Daily Returns for {stock}'
    plt.title(title)
    st.pyplot()

def boxplot(y_col, label = '', title = ''):
    plt.boxplot(y_col,labels = [label])
    plt.title(title)
    st.pyplot()

def make_lognorm_dist(col):
    st_dev = col.std()
    mean = col.mean()
    return lognorm([st_dev],loc = mean)

def post_image(image, caption = '', width = None):
    image = Image.open(image)
    st.image(image, caption = caption, width = width)

def make_summary_table(y_col):
    stat_dict = {
        'Mean' : np.mean(y_col),
        'Median' : np.quantile(y_col, 0.5),
        'Std' : np.std(y_col),
        'Skew' : skew(y_col),
        'Kurtosis' : kurtosis(y_col)
    }
    dat = pd.DataFrame(stat_dict, index = [f'{stock}'])
    st.table(dat)

def skew_test(y_col):
    stat, p_val = skewtest(y_col)
    stat_dict = {
        'Test Statistic' : stat,
        'P-Value' : p_val
    }
    dat = pd.DataFrame(stat_dict, index = [f'{stock}'])
    st.table(dat)

# Title of app
st.title('Stock Return Analysis')

# First Header
st.header('Choose stock to analyze.')

# Choose stock
try:
    stock = st.text_input(
        'What stock would you like to analyze? Enter ticker symbol.', 
        'aapl'
        )
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

# Box plot of returns
boxplot(df['Returns'], 'Returns', 'Boxplot of Daily Returns')

st.header('Data Observables')
st.write('''
Lets calculate some main statistics for our data.
''')
make_summary_table(df['Returns'])
st.write('''
The first thing to do is compare our mean and median. If our median is greater than 
our mean, then we know that our data is left(negative)-skewed which means that there 
are more data points above the mean than below. But, this also means that there are extreme
values on the lower end of the data that are pulling the mean value down. On the other
hand, if our median is less than our mean, then we know that our data is right(positive)-skewed
which means that there are more data points below the mean than above. But, this also
means that there are extreme values on the upper end of the data that are pulling the
mean higher. The greater the difference between the mean and median, the greater the
skewness. Please refer to the \"Examples of skewness\" image below for examples.
''')
post_image('skew.jpg', caption = 'Examples of skewness')
st.write('''
Another statistic to evaluate is Skew, which is a measurement of the symmetry of our 
data. Skew has a range of all real postive and negative numbers. The smaller the statistic,
the more left-skewed the data is and the greater the statistic the more right-skewed
the data is. A perfectly symmetrical distribution will have a skew value of 0. Please refer 
to the \"Examples of skewness\" image above for examples.
''')
st.write('''
Lastly, we will look at the statistic Kurtosis. Kurtosis is a measure of how much the
tails of the data differ from that of a normal distribution. Kurtosis values can be any
real positive or negative number. The smaller a kurtosis value is, the more Platykurtic
the distribution is. A Platykurtic Distribution is one that looks shorter and wider than
a normal distribution. The larger a kurtosis value is, the more Leptokurtic the distribution 
is. A Leptokurtic Distribution is one that looks taller and skinnier than a normal
distribution. Please refer to the \"Examples of kurtosis\" image below for examples.
''')
post_image('kurtosis.jpg', 'Examples of kurtosis')

st.write('''
Next, we will run a skew test.
''')
skew_test(df['Returns'])