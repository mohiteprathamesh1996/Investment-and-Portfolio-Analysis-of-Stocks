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
st.markdown("[Image Courtesy : {}]".format("Yahoo Finance"))



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

st.header("*Showing results for {}*".format(tickers))



st.markdown("Closing Price")
st.line_chart(df["Close"])


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

st.markdown("Live Sentiment tracker")

sent_df = news_table(finviz_url, tickers=["{}".format(tickers)])

st.bar_chart(sent_df["{}".format(tickers)])












