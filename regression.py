import datetime
import os 
import sys
import json
import time
import math
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
from iexfinance.stocks import get_historical_data
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import selenium
from datetime import datetime
import smtplib
from selenium import webdriver
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

def getStocks():


    stock_list = []
    stock_list.append('AAPL')
    stock_list.append('ADBE')
    stock_list.append('AMZN')
    
    #Using the stock list to predict the future price of the stock a specificed amount of days
    number = 0
    for i in stock_list:
        print("Number: " + str(number))
        predictData(i,1)
        number += 1

def predictData(stock, days):
    print(stock)

    start = datetime(2017, 1, 1)
    end = datetime.now()

    #Outputting the Historical data into a .csv for later use
    df = get_historical_data(stock, start=start, end=end, output_format='pandas')
    
    if os.path.exists('./Exports'):
        csv_name = ('Exports/' + stock + '_Export.csv')
    else:
        os.mkdir("Exports")
        csv_name = ('Exports/' + stock + '_Export.csv')
    df.to_csv(csv_name)

    Xp = np.array(df)[-1]
    print(Xp)

    df['prediction'] = df['close'].shift(-1)
    df.dropna(inplace=True)
  
    
    forecast_time = int(days)

    #Predicting the Stock price in the future
    X = np.array(df.drop(['prediction'], 1))
    print(X[-1])
    print(X.shape)

    X = np.vstack([X, Xp])
    print(X[-1])
    print(X.shape)

    Y = np.array(df['prediction'])
    Xs = preprocessing.scale(X)

    Xp = Xs[-1]
    Xs = Xs[:-1, :]
    print(Xs.shape)
    print(Xs[-1])
    print(Xp)
    
    X_prediction = np.reshape(Xp,(1, 5))
    X_train, X_test, Y_train, Y_test = train_test_split(Xs, Y, test_size=0.2)

    #Performing the Regression on the training data
    clf = LinearRegression()
    clfcv = LinearRegression()

    clf.fit(Xs, Y)
    prediction = (clf.predict(X_prediction))

    clfcv.fit(X_train,Y_train)
    print(clfcv.score(X_test,Y_test))

    #last_row = df.tail(1)
    
    #print(last_row['close'])
    print(prediction)
    change = float(prediction)-float(Y[-1])
    print(change)

if __name__ == '__main__':
    getStocks()