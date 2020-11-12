import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from textblob import TextBlob
import requests 
from bs4 import BeautifulSoup
# from sys import 



term = 'Arnab goswami'
page=1

newss = []

while (page<20):

    url ='https://news.search.yahoo.com/search?q={}&pz=10&b={}'.format(term,page)
#     print(url)
    page = page + 10
    response = requests.get(url,verify=False)
    if response.status_code !=200:
        break
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all("div")

    d = []
    for h in tqdm(headlines):
        d.append(TextBlob(h.get_text()))
    newss.append(d)

    print(newss[0][0].split("...")[0])