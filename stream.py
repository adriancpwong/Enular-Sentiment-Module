import tweepy 
from tweepy import OAuthHandler 
from tweepy import Cursor
from textblob import TextBlob 
import re
import csv
import datetime
from datetime import datetime as dt
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
from iexfinance.stocks import get_historical_data

#todo
#impliment stocker api to get stock prices
#create key words dictionary
#figure out a more reliable filtering method
#add stock code to post
#add stock statistics to post

#HOW STOCKS NIGHT BEFORE AFFECTS PRICE IN THE MORNING


class MyStreamListener(tweepy.StreamListener):
    
    def on_data(self, data):
        
        global feed
        global companyName
        global searchQuery
        global dateTodayStr
        global dateTodayStrRev
        global histoDay
        global histoMonth
        global histoYear
        global stockCode
        
        #loads json data in a library format
        pydata = json.loads(data)

        #loads userdata from pydata in a library format
        try:
            userdata = pydata['user']
        except:
            pass
        #    print("userdata failed")
        
        #currently not used
        #try:
        #    rtdata = pydata['retweeted_status']
        #except:
        #    pass
        #    print("rtdata failed")

        #currently not used
        try:
            tDate = pydata['created_at']
        except:
            pass
        #    print("tDAte failed")

        #tweet content
        try:
            tText = pydata['text']
        except:
            pass
        #    print("tText failed failed")

        #confirms company name in stock text
        companyName = ''
        stockCode = ''
        try:
            for term in searchQuery:
                stockSearch = stockCodeDict[term]
                stockRelated = stockRelatedDict[term]
                if term in tText or stockSearch in tText or stockRelated in tText:
                    companyName = term
                    stockCode = stockCodeDict[companyName]

                    print(feed)
                    print(companyName)
                    print(stockCode)

                else:
                    continue
        except:
            pass
        #    print("string search failed")

        
        if companyName != '':
            print("Company name OK")

            try:
                tUserFollowers = userdata['followers_count']
            except:
                pass
                tUserFollowers = 0
                print("tUserFollowers failed")
            try:
                tUserFriends = userdata['friends_count']
            except:
                pass
                tUserFriends = 0
                print("tUserFriends failed")
            try:
                tUsersLists = userdata['listed_count']
            except:
                pass
                tUsersLists = 0
                print("tUsersLists failed")
            try:
                tUserFavourites = userdata['favourites_count']
            except:
                pass
                tUserFavourites = 0
                print("tUserFavourites failed")
            try:
                tUserStatuses = userdata['statuses_count']
            except:
                pass
                tUserStatuses = 0
                print("tUserStatuses failed")

            try:
                tQuotes = pydata['quote_count']
            except:
                pass
                tQuotes = 0
                print("tQuotes failed")
            try:
                tReplies = pydata['reply_count']
            except:
                pass
                tReplies = 0
                print("tReplies failed")
            try:
                tRetweets = pydata['retweet_count']
            except:
                pass
                tRetweets = 0
                print("tRetweets failed")
            try:
                tFavourites = pydata['favorite_count']
            except:
                pass
                tFavourites = 0
                print("tFavourites failed")

            try:
                tSentiment = "{:.2f}".format(get_tweet_sentiment(tText))
                tLength = get_tweet_length(tText)
                tMentionsOfSearch = tText.count(companyName)
                tNumberOfHash = tText.count("#")
                tNumberOfAt = tText.count("@")
                tNumberOfSpace = tText.count(" ")
            except:
                pass
                print("tText segment failed somehow")
            
            #stockChange = get_stock_history(stockCode)
            stockChange = get_stock_change(stockCode)
            print(stockChange)

            try:
                


                predict = 0

                post = {
                    
                    'Predict':predict,

                    'Change':stockChange,

                    'Company':companyName,
                    'Code':stockCode,
                    #'Date':tDate,
                    'DateStr':dateTodayStr,
                    'Text':tText,

                    'Followers':tUserFollowers,
                    'Friends':tUserFriends,
                    'Lists':tUsersLists,
                    'Userfavourites':tUserFavourites,
                    'Statuses':tUserStatuses,

                    'Quotes':tQuotes,
                    'Replies':tReplies,
                    'Retweets':tRetweets,
                    'Favourites':tFavourites,

                    'Sentiment':tSentiment,
                    'Length':tLength,
                    'Mentions':tMentionsOfSearch,
                    'Hashes':tNumberOfHash,
                    'Ats':tNumberOfAt,
                    'Words':tNumberOfSpace
                
                }

                post_id = posts.insert_one(post).inserted_id

                print(tText)
                print("=====")
                print("SUCCESS")
                print("=====")

            except:
                pass
                print("post failed")


        feed += 1


feed = 1
companyName = ''
stockCode = ''

companyNameOne = 'apple'
companyNameTwo = 'intel'
companyNameThree = 'amazon'
companyNameFour = 'microsoft'
companyNameFive = 'alibaba'
companyNameSix = 'facebook'
companyNameSeven = 'alphabet'
companyNameEight = 'disney'
companyNameNine = 'micron'
companyNameTen = 'netflix'

stockCodeDict = {
    'apple':'AAPL',
    'intel':'INTC',
    'amazon':'AMZN',
    'microsoft':'MSFT',
    'alibaba':'BABA',
    'facebook':'FB',
    'alphabet':'GOOGL',
    'disney':'DIS',
    'micron':'MU',
    'netflix':'NFLX'
}

stockRelatedDict = {
    'apple':'iphone',
    'intel':'i7',
    'amazon':'alexa',
    'microsoft':'windows',
    'alibaba':'alipay',
    'facebook':'zuckerberg',
    'alphabet':'google',
    'disney':'marvel',
    'micron':'cpu',
    'netflix':'birdbox'
}

searchQuery = [companyNameOne,companyNameTwo,companyNameThree,companyNameFour,companyNameFive,companyNameSix,companyNameSeven,companyNameEight,companyNameNine,companyNameTen]

dateToday = datetime.date.today()
#dateTodayStr = dateToday.strftime('%d-%m-%Y')
#dateTodayStrRev = dateToday.strftime('%Y-%m-%d')
dateTodayStr = "06-02-2019"
#for history stock price
dateTodayStrRev = "2019-02-06"
histoDay = 6
histoMonth = 2
histoYear = 2019

consumer_key = "6nzlaKRpWYKpCvVegHbgfJFsb"
consumer_secret = "PCui0esGRhV9CDZ7HW3ldlKdvQ1qa9uJHF3PKqpHzwjT8RPabZ"
access_token = "1049145042810667008-sFVbw3FfQ12PTam6F6D2pz64l5SnCd"
access_token_secret = "LopeA1dv0NKYtLI336clfoYJ7DWBeBc28mEdtUhCeYKAl"

try:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
except: 
    print("Error: Authentication Failed")

client = MongoClient()
db = client.database_one
collection = db.collection_one
posts = db.posts

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])(\w+:\/\/\S+)", " ", tweet).split()) 

def get_tweet_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def get_tweet_length(tweet):
    return len(tweet)

def get_stock_change(stockCode):
    stock = Stock(stockCode)
    stockStats = stock.get_quote()
    stockChange = stockStats['changePercent']
    return stockChange

def get_stock_history(stockCode):

    global dateTodayStrRev
    global histoDay
    global histoMonth
    global histoYear

    start = dt(histoYear, histoMonth, histoDay)
    end = dt(histoYear, histoMonth, histoDay)

    df = get_historical_data(stockCode, start, end)
    print(df)
    ef = df[dateTodayStrRev]["close"] - df[dateTodayStrRev]["open"]
    ff = df[dateTodayStrRev]["open"]
    change = float(ef)/float(ff)
    return change


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
streamSearch = myStream.filter(track=searchQuery, async=False)

