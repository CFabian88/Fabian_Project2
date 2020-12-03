from scipy.stats import lognorm, norm, skew, kurtosis, skewtest, kurtosistest
from statsmodels.graphics.tsaplots import plot_pacf
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
def line_graph(y_col, y_label = '', x_label = '', title = ''):
    graph = px.line(
        x = df.index, 
        y = y_col,
        title = title,
        labels = {
            'y' : y_label, 
            'x' : x_label
        }
        )
    st.plotly_chart(graph)

def rolling_line_graph(y_col, roll_per = 20):
    graph = px.line(
        x = df.index, 
        y = y_col.rolling(roll_per).std(),
        title = f'{roll_per}-Days Rolling Standard Deviation of Returns',
        labels = {
                'y' : f'Standard Deviation', 
                'x' : 'Time'
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

def make_pacf_plot(y_col, title = '', y_label = '', x_label = '', lags = 20):
    plot_pacf(y_col, lags = lags)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    st.pyplot()

def qq_plot(y_col, title = ''):
    # Create set of 1000 numbers equally spaced betweed 0.01 and 0.99
    x = np.linspace(0.01,0.99,1000)
    # Create Sample Quantiles
    q1 = np.quantile(y_col, x)
    # Create Theoretical Quantiles of normally distributed data
    q2 = norm.ppf(x, loc=np.mean(y_col), scale=np.std(y_col))
    # Plot Sample quantiles as x-variable and Theoretical quantiles as y-variable
    plt.plot(q1,q2)
    # Plot y = x
    plt.plot([min(q1),max(q1)],[min(q2),max(q2)])
    # Set limits on x and y axes
    plt.xlim((min(q1),max(q1)))
    plt.ylim((min(q2),max(q2)))
    # Title + axis labels
    plt.title(title)
    plt.xlabel('Sample Quantiles')
    plt.ylabel('Theoretical Quantiles')
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

def kurtosis_test(y_col):
    stat, p_val = kurtosistest(y_col)
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
post_image('pics/daily_return.jpg', 'Daily return formula')

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
line_graph(
    df['Close'], 
    y_label = 'Closing Price ($)', 
    x_label = 'Time',
    title = f'Closing Prices for {stock}'
    )

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
post_image('pics/skew.jpg', caption = 'Examples of skewness')
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
post_image('pics/kurtosis.jpg', 'Examples of kurtosis')

st.header('Hypothesis Tests')
st.write('''
Next, we will run a skew test with the following hypotheses:
''')
post_image('pics/skew_test_hypoth.jpg')
skew_test(df['Returns'])
st.write('''
For all hypothesis tests we will be using 0.05 for our alpha threshold. This means that
if the p-value is less than 0.05, we will reject the null hypothesis and accept the
alternative one as true. If the p-value is greater than or equal to 0.05, we will NOT
reject the null hypothesis and will continue to accept it as true. In this case, if the
p-value is less than 0.05, then we will reject the null hypothesis that says the data
is symmetrical, and accept the alternative which states that the data is NOT
symmetrical. And if the p-value is greater than or equal to 0.05, then we will
accept the conclusion that the data is symmetrical.
''')

st.write('''
Now that we know if our data is skewed or not, we can now test if our data is normally
distributed or not. For this we will run a kurtosis test with the following hypotheses:
''')
post_image('pics/kurtosis_test_hypoth.jpg')
kurtosis_test(df['Returns'])
st.write('''
If the p-value is less than 0.05, then we can reject the null hypothesis that states the
data is normally distributed and accept the alternative that says the data is NOT normally
distributed. If the p-value is greater than or equal to 0.05, then we can continue to
accept the null hypothesis that our data is indeed normally distributed.
''')

st.header('QQ-Plot')
st.write('''
Although we can tell a lot about a dataset's distribution based on our skew and kurtosis
tests. QQ (Quantile-Quantile) Plots give us a visual confirmation for these tests. They 
are plots that show a datasets quantile values versus the theoretical quantile values 
of the same data if it were normally distributed. Lets take a look. 
''')
qq_plot(df['Returns'], title = f'QQ-Plot of {stock}')
st.write('''
If our data set is normally distributed, then the line for our sample data will be very
similar to the line y=x. This is because our y-variable is the theoretical quantiles of
our data set if it was truly normally distributed and our x-variable is what our dataset's
quantiles actually are. Therefore, if we want our data to be normally distributed, we would
want our 'x' and 'y' values to be the same for each coordinate. Hence why we want the line
to match that of y=x.
''')
st.write('''
Generally, it is very rare to find a stock whose returns are normally distributed. If 
you find that your data is not normally distributed, then we can only use stochastic
models as an approximation for future returns because they take the assumption that data
is distributed normally.
''')
st.write('''
But what do we mean when we say a stochastic model? In short, a stochastic model is any model
that describes the evolution in time of a random phenomenon. Even though the stock market
is not random and is affected by real world events, these factors are so numerous and 
complex that when we attempt to model them, we are better off treating them like 
random phenomena.
''')
st.write('''
One example of a common stochastic process used in finance is the Geometric Brownian
Motion model. The Brownian Motion, often called the Wiener Process, is simply just the
calculated random movement of a thing given the mean and standard deviation of that 
thing. In this case, we are refering to stock prices. Due to it's ability to account 
for randomness, it is easy to see why it is such a common tool in the finance industry.
Below is an example of various Brownian Motion models, all with varying variances 
and a mean value of 1. The graph shows how the model can change as the variance of the 
data does.
''')
post_image('pics/brownian_motion.jpg', caption = 'Brownian Motion Example')

st.header('Volatility')
st.write('''
As we saw above, the Brownian Motion can change quite drastically with changes in
volatility. So, if we want to try and predict future stock prices using this model, 
it would be useful to know if our data's volitility has varied throughout time. Lets 
look at the raw time series data of our returns.
''')
line_graph(
    df['Returns'], 
    y_label = 'Returns (%)',
    x_label = 'Time',
    title = f'Time Series Data of Returns for {stock}'
)
st.write('''
As we can see, the data is nonsequential the entire time and our data's volatility is 
clearly varying throughout time. This phenomenon is known as volatility clustering, 
and it is very common throughout the stock market.
''')
st.write('''
Because the volatility of a stock price can change so rapidly, it is hard to gain an
understanding of any patterns that may be present. In order to smooth this data and 
get a better understanding of the behavior of the price's volatility, we can use the 
30-day rolling standard deviation. Instead of calculating volatility using all previous
data points and constantly changing our time interval, we calculate volatility using only
the previous 30 data points. This way, the time variable in our calculation will remain
constant. This will drastically smooth our plot from above.
''')
rolling_line_graph(df['Returns'], roll_per = 30)
st.write('''
As we can see, our suspicions about the volatility varying throughout time are true.
It is certainly not a constant value by any means. In fact, we see quite a bit of spikeage
and oscillation. The most likely cause for these dramatic changes is most likely due to
the extreme outliers in the data set. Because these points are so influential to the model,
we cannot neglect these outliers when producing a model. 
''')

st.header('Partial Autocorrelation Function (PACF)')
st.write('''
Now, we can use our partial autocorrelation function to see if there is any correlation
between observations at two time spots given that we consider both observation are
correlated to observations at other time spots. For example, the stock price today
can be correlated to the day before yesterday, and yesterday can also be correlated to
the day before yesterday. If this is the case, then the partial autocorrelation function
of yesterday is the \"real\" correlation between today and yesterday after taking out
the influence of the day before.
''')
st.write('''
You may be asking yourself what that even means and why anyone would care about this.
Well, the reason why PACF is so widely used in predicting stock prices, is because it
allows us to get a measurement of correlation between today's and yesterday's stock
price without the influence of the day before yesterday's correlation. In this way,
calculating the PACF is the only way to understand the \"real\" correlation between
today's and yesterday's prices because we are removing the influence from previous days. 
By doing so, we can determine which data points we should use in our ARIMA model.
''')
st.write('''
In order to visualize this, we will plot our PACF function below. On the graph, each
vertical line represents the partial autocorrelation for that point. You will also notice
a blue area around the x-axis. Only the PACF points that extend past the shaded blue area
are considered significant. The line at lag 1 is a sanity checker for our plot because it
represents the correlation between today's stock price and itself, so it should always be
equal to 1. Every lag after that represent each day backwards. So, lag 20 represents the
correlation between the stock price 20 days ago and now. In order to choose which points
we want for our AMIRA model, we must go lag to lag in order. When we eventually reach one
that does not extend beyond the blue area (meaning it is insignificant), we stop and take
all data points from the previous lags. For example, if the first and second lags extend
beyond the blue area but the third lag does not, then we use the data points
from lag 1 and lag 2 in our AMIRA model. However, if the second lag is insignificant, then
it is ill-advised to use an AMIRA model.
''')

make_pacf_plot(
    df['Returns'], 
    title = f'Partial Autocorrelation Function for {stock}',
    y_label= f'Correlation',
    x_label= f'Lag #'
)

st.write('''
Thank you for reading my AMIRA model. At the end of this I hope that you are able to 
enter any stock and quickly analyze if an AMIRA model is appropriate for predicting
future stock price behavior. I also hope that you were able to learn something new
along the way.
''')