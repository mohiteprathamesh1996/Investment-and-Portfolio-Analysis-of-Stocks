import pandas as pd
import numpy as np
import pandas_datareader as pdr
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
from PIL import Image
# To fetch URLs i.e. uniform resource locators
from urllib.request import urlopen, Request

# To fetch data from HTML and/or XML files
from bs4 import BeautifulSoup

# For sentiment analysis and download vader_lexicon to measure sentiment intensity
# import nltk; nltk.download()
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import requests
from itertools import groupby
from operator import itemgetter
from tqdm import tqdm
warnings.filterwarnings("ignore")


## Adding title bar
st.title("Analyze your stocks portfolio!")
st.markdown("Get accurate predictions within minutes...")

## Adding image
st.image(Image.open("C:\\Users\\Prathamesh\\Desktop\\Side projects\\Predicting-Closing_Price-of-Stocks-using-LSTMs\stockstock1.jpg"),
use_column_width=True)
st.markdown("*[Image Courtesy : {}]*".format("Yahoo Finance"))



## Create a sidebar for user input
st.sidebar.header("User input")

## Get user input
tickers = st.sidebar.text_input("Ticker","AAPL")
start_date = st.sidebar.text_input("From","2015-01-01")
end_date = st.sidebar.text_input("To",str(datetime.date.today()))

## Fetch live data
df = pdr.get_data_yahoo(tickers, 
                        start=pd.to_datetime(start_date),
                        end=pd.to_datetime(end_date))

st.header("Showing results for {}".format(tickers))

st.markdown("Closing Price")
st.line_chart(df["Close"])




## Sentiment analysis
def find_missing(lst): 
    return [i for x, y in zip(lst, lst[1:])  
        for i in range(x + 1, y) if y - x > 1] 



finviz_url = 'https://finviz.com/quote.ashx?t='

def news_table(finviz_url, tickers):
    df = []

    news_tables = {}

    for ticker in tqdm(tickers):
        news_tables[ticker] = BeautifulSoup(urlopen(Request(url=finviz_url + ticker,
                                                            headers={"user-agent":"my-app"})),
                                            "html").find(id="news-table")



        df.append(pd.DataFrame([(row.a.get_text(), 
                                 row.td.text,
                                 ticker) for index, row in enumerate(news_tables[ticker].findAll("tr"))], 
                               columns = ["Title","Timestamp","Ticker"]))

    df = pd.concat(df).reset_index(drop=True)

    for k, g in groupby(enumerate([i for i in range(len(df["Timestamp"])) if len(df["Timestamp"][i])!=19]), 
                        lambda i_x: i_x[0] - i_x[1]):
        l = list(map(itemgetter(1), g))
        v = df.iloc[l[0]-1,:]["Timestamp"].split(" ")[0]+" "+df.iloc[l,:]["Timestamp"]
        df.iloc[l,1] = v

    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.date
    
    vader = SentimentIntensityAnalyzer()
    
    df["compound"] = [vader.polarity_scores(titles)["compound"] for titles in df["Title"]]
    
    return df.groupby(["Ticker","Timestamp"]).mean().unstack().xs("compound", axis="columns").transpose()

st.markdown("Live Sentiment tracker *[source: finviz.com]*")

sent_df = news_table(finviz_url, tickers=["{}".format(tickers)])

st.bar_chart(sent_df["{}".format(tickers)])



## LSTMs to forecast future value of closing price

# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df.reset_index()["Close"]).reshape(-1,1))

# Split train test
percentage_training_data = 0.70
train_data = df1[:int(percentage_training_data*df.shape[0])]
test_data = df1[int(percentage_training_data*df.shape[0]):]

# Features and targets
def create_features_targets(data, timesteps):
    features = []
    target = []
    for i in range(len(data)-timesteps-1):
        features.append(list(data[i:i+timesteps]))
        target.append(data[i+timesteps])
    return np.array(features), np.array(target)

timesteps = 30

X_train, y_train = create_features_targets(train_data, timesteps=timesteps)
X_test, y_test = create_features_targets(test_data, timesteps=timesteps)

# Reshaping
## Reshaping input for LSTM as : [sample_size, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

## Creating the stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(timesteps,
               return_sequences=True,
               input_shape=(timesteps, 1)))

model.add(LSTM(timesteps))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam")

## Fit the model
model = model.fit(X_train, y_train, 
          validation_data=(X_test, y_test), 
          epochs=100, 
          batch_size=50, 
          verbose=0)

## Model predictions on test data
train_predictions = model.model.predict(X_train)
test_predictions = model.model.predict(X_test)

## Tranforming predictions back to original values
train_predictions = scaler.inverse_transform(train_predictions) 
test_predictions = scaler.inverse_transform(test_predictions)



x_input = test_data[len(test_data)-timesteps:].reshape(1,-1)

temp_input = list(x_input)[0].tolist()

lst_output=[]
n_steps=timesteps
i=0

days_ahead = int(st.sidebar.text_input("Prediction for days","11"))

while(i<days_ahead):
    
    if(len(temp_input)>timesteps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
#         print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.model.predict(x_input, verbose=0)
#         print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.model.predict(x_input, verbose=0)
#         print(yhat[0])
        temp_input.extend(yhat[0].tolist())
#         print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

future = scaler.inverse_transform(lst_output)

df_prediction = pd.concat([pd.DataFrame([df.index.max()+datetime.timedelta(i+1) for i in range(days_ahead)],
                                        columns=["Date"]), 
                           pd.DataFrame(future,
                                        columns=["Prediction"])], axis=1).set_index("Date")

final_df_with_prediction = pd.concat([df, df_prediction])


st.markdown("Forecast for next {} days".format(days_ahead))
st.line_chart(final_df_with_prediction["Prediction"])


