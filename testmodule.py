import tweepy 
from tweepy import OAuthHandler 
from tweepy import Cursor
from textblob import TextBlob 
import re
import csv
import datetime
import os 
import sys
import json
import time
import math
import pymongo
from pymongo import MongoClient
import numpy as np
import scipy
import sklearn
from sklearn import svm
from sklearn import tree
from sklearn import dummy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from iexfinance import Stock
from iexfinance.stocks import Stock


from datetime import datetime
from iexfinance.stocks import get_historical_data

histoDay = 7
histoMonth = 1
histoYear = 2019

start = datetime(histoYear, histoMonth, histoDay)
end = datetime(histoYear, histoMonth, histoDay)

df = get_historical_data("TSLA", start, end)
ef = df["2019-01-07"]["close"] - df["2019-01-07"]["open"]
ff = df["2019-01-07"]["open"]
gf = float(ef)/float(ff)
print(gf)