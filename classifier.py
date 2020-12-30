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
from iexfinance.stocks import get_historical_data
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

client = MongoClient()
db = client.database_one
collection = db.collection_one
posts = db.posts

#for date in dates
#   for company in companies
#       deconstruct features to temp 2d array
#       aggregate to 1d array
#       add to xtrain 2d array
#       add stock price to ytrain 1d array

predict = 'adobe'

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

companies = [companyNameOne,companyNameTwo,companyNameThree,companyNameFour,companyNameFive,companyNameSix,companyNameSeven,companyNameEight,companyNameNine,companyNameTen]
dates = ["07-01-2019","08-01-2019","09-01-2019","10-01-2019","11-01-2019",
"14-01-2019","15-01-2019","16-01-2019","17-01-2019","18-01-2019",
"22-01-2019","23-01-2019","24-01-2019","25-01-2019",
"28-01-2019","29-01-2019","30-01-2019","31-01-2019","01-02-2019",
"01-02-2019","04-02-2019","05-02-2019","06-02-2019"]

#"28-01-2019"
#"29-01-2019"
#"25-01-2019"
#"05-02-2019"
#"06-02-2019"

dateToday = datetime.date.today()
dateTodayStr = dateToday.strftime('%d-%m-%Y')

x_train_temp = []
y_train_temp = []
test_load = []

feed = 0
for date in dates:
    for company in companies:

        aggregation = [float(0.00)]

        for post in posts.find({"Company":company,"DateStr":date,"Predict":0}):
            change = post['Change']
            aggregation.append(float(change))
        
        npagg = np.asarray(aggregation)
        y_train_temp.append(math.ceil(np.mean(npagg)))
        feed += 1

print(feed)
print(y_train_temp)

for date in dates:
    for company in companies:

        followsArray = [float(0.00)]
        friendsArray = [float(0.00)]
        listsArray = [float(0.00)]
        #userfavouritesArray = [float(0.00)]
        statusesArray = [float(0.00)]
        quotesArray = [float(0.00)]
        repliesArray = [float(0.00)]
        retweetsArray = [float(0.00)]
        favouritesArray = [float(0.00)]
        sentimentArray = [float(0.00)]
        lengthArray = [float(0.00)]
        mentionsArray = [float(0.00)]
        hashesArray = [float(0.00)]
        atsArray = [float(0.00)]
        wordArray = [float(0.00)]

        tempArray = []

        for post in posts.find({"Company":company,"DateStr":date,"Predict":0}):

            followsArray.append(float(post['Followers']))
            friendsArray.append(float(post['Friends']))
            listsArray.append(float(post['Lists']))
            #userfavouritesArray.append(float('0'))
            statusesArray.append(float(post['Statuses']))
            quotesArray.append(float(post['Quotes']))
            repliesArray.append(float(post['Replies']))
            retweetsArray.append(float(post['Retweets']))
            favouritesArray.append(float(post['Favourites']))
            sentimentArray.append(float(post['Sentiment']))
            lengthArray.append(float(post['Length']))
            mentionsArray.append(float(post['Mentions']))
            hashesArray.append(float(post['Hashes']))
            atsArray.append(float(post['Ats']))
            wordArray.append(float(post['Words']))

        tempArray.append(np.mean(np.asarray(followsArray)))
        tempArray.append(np.mean(np.asarray(friendsArray)))
        tempArray.append(np.mean(np.asarray(listsArray)))
        #tempArray.append(np.mean(np.asarray(userfavouritesArray)))
        tempArray.append(np.mean(np.asarray(statusesArray)))
        tempArray.append(np.mean(np.asarray(quotesArray)))
        tempArray.append(np.mean(np.asarray(repliesArray)))
        tempArray.append(np.mean(np.asarray(retweetsArray)))
        tempArray.append(np.mean(np.asarray(favouritesArray)))
        tempArray.append(np.mean(np.asarray(sentimentArray)))
        tempArray.append(np.mean(np.asarray(lengthArray)))
        tempArray.append(np.mean(np.asarray(mentionsArray)))
        tempArray.append(np.mean(np.asarray(hashesArray)))
        tempArray.append(np.mean(np.asarray(atsArray)))
        tempArray.append(np.mean(np.asarray(wordArray)))

        x_train_temp.append(tempArray)

print(x_train_temp)

if predict != '':

    feeds = 0

    followsArray = [float(0.00)]
    friendsArray = [float(0.00)]
    listsArray = [float(0.00)]
    #userfavouritesArray = [float(0.00)]
    statusesArray = [float(0.00)]
    quotesArray = [float(0.00)]
    repliesArray = [float(0.00)]
    retweetsArray = [float(0.00)]
    favouritesArray = [float(0.00)]
    sentimentArray = [float(0.00)]
    lengthArray = [float(0.00)]
    mentionsArray = [float(0.00)]
    hashesArray = [float(0.00)]
    atsArray = [float(0.00)]
    wordArray = [float(0.00)]

    tempArray = []

    for post in posts.find({"Company":predict,"DateStr":dateTodayStr,"Predict":1}):

        followsArray.append(float(post['Followers']))
        friendsArray.append(float(post['Friends']))
        listsArray.append(float(post['Lists']))
        #userfavouritesArray.append(float(0))
        statusesArray.append(float(post['Statuses']))
        quotesArray.append(float(post['Quotes']))
        repliesArray.append(float(post['Replies']))
        retweetsArray.append(float(post['Retweets']))
        favouritesArray.append(float(post['Favourites']))
        sentimentArray.append(float(post['Sentiment']))
        lengthArray.append(float(post['Length']))
        mentionsArray.append(float(post['Mentions']))
        hashesArray.append(float(post['Hashes']))
        atsArray.append(float(post['Ats']))
        wordArray.append(float(post['Words']))

    tempArray.append(np.mean(np.asarray(followsArray)))
    tempArray.append(np.mean(np.asarray(friendsArray)))
    tempArray.append(np.mean(np.asarray(listsArray)))
    #tempArray.append(np.mean(np.asarray(userfavouritesArray)))
    tempArray.append(np.mean(np.asarray(statusesArray)))
    tempArray.append(np.mean(np.asarray(quotesArray)))
    tempArray.append(np.mean(np.asarray(repliesArray)))
    tempArray.append(np.mean(np.asarray(retweetsArray)))
    tempArray.append(np.mean(np.asarray(favouritesArray)))
    tempArray.append(np.mean(np.asarray(sentimentArray)))
    tempArray.append(np.mean(np.asarray(lengthArray)))
    tempArray.append(np.mean(np.asarray(mentionsArray)))
    tempArray.append(np.mean(np.asarray(hashesArray)))
    tempArray.append(np.mean(np.asarray(atsArray)))
    tempArray.append(np.mean(np.asarray(wordArray)))

    test_load.append(tempArray)
else:
    test_load = [[50,50]]

print(feeds)
print(test_load)
#loads files

x_train_load_pre = np.asarray(x_train_temp)
y_train_load_pre = np.asarray(y_train_temp)

#shifts data one day so it is predicts tomorrow's stock price
x_train_load = x_train_load_pre[:-10, :]
y_train_load = np.delete(y_train_load_pre,[0,1,2,3,4,5,6,7,8,9])

print(x_train_load.shape)
print(y_train_load.shape)

#scalling datasets
scaler = StandardScaler()


x_train_scaled = scaler.fit_transform(x_train_load)

'''

REMEMBER TO SCALE THE X DATA AND TEST DATA TOGETHER
PCA TOGETHER OR SEPERATELY?
ASK IADH

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

REMEMBER TO SHIFT THE Y DATA BACKWARDS ONE
SO THAT X DATA IS PREDICTING TOMORROWS PRICE
INSTEAD OF YESTERDAY'S
REMEMBER
'''

test_scaled = scaler.fit_transform(test_load)


y_train_scaled = y_train_load.ravel()
#splitting datasets for cross validation
x_train_pre, x_test_pre, y_train, y_test = train_test_split(x_train_scaled, y_train_scaled, test_size=0.5, random_state=0)

#reducing features
pca = PCA(n_components=2)
pca.fit(x_train_scaled)
x_train = pca.transform(x_train_pre)
x_test = pca.transform(x_test_pre)
test = pca.transform(test_scaled)

print("Metrics: F1, Accuracy, Precision, Recall")
metrics = tree.DecisionTreeClassifier(max_depth=10)
metrics.fit(x_train,y_train)
y_true = y_test
y_pred = metrics.predict(x_test)
f1metric = f1_score(y_true, y_pred, average='weighted')
accuracymetric = accuracy_score(y_true, y_pred)
precisionmetric = precision_score(y_true, y_pred, average='weighted')
recallmetric = recall_score(y_true, y_pred, average='weighted') 
print(f1metric)
print(accuracymetric)
print(precisionmetric)
print(recallmetric)

dtc=tree.DecisionTreeClassifier(max_depth=10)
dtc.fit(x_train,y_train)
print ("DTC")
print(dtc.score(x_test,y_test))
	
smc=svm.SVC(gamma='scale',C=2.0)
smc.fit(x_train,y_train)
print ("SVM")
print(smc.score(x_test,y_test))

#knn=KNeighborsClassifier(n_neighbors=5,algorithm='auto',leaf_size=30,p=2)
#knn.fit(x_train,y_train)
#print ("KNN")
#print(knn.score(x_test,y_test))

gnb = GaussianNB()
gnb.fit(x_train,y_train)
print ("GNB")
print(gnb.score(x_test,y_test))

rfc = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
rfc.fit(x_train,y_train)
print ("RFC")
print(rfc.score(x_test,y_test))

etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(x_train,y_train)
print("ETC")
print(etc.score(x_test,y_test))

mlp = MLPClassifier(max_iter=1000,hidden_layer_sizes=(1000,))
mlp.fit(x_train,y_train)
print("MLP")
print(mlp.score(x_test,y_test))

dummy = DummyClassifier(strategy='uniform')
dummy.fit(x_train,y_train)
print("Dummy")
print(dummy.score(x_test,y_test))

results = dtc.predict(test)

print(results)
#np.savetxt('results.csv', results, fmt='%d', delimiter=",", comments="")

#Compare metrics between classifiers
#Group features into categories
#Perform ablation study: remove groups of features then run classifier with groups missing
#Then gather matrics for each study without groups
#Find out which features are useless and decrease the accuracy
#Find out which features are heavy and usefull

#Once all this is done, have a fixed test set
#Train with everytihng else, minus the useless features we found out from ablation study
#Evaluate classifier on last several days
#Give it gradually less days of training
#Make sure that training with less days gives a worse performance
#make sure you're using the same test set for each training set

#Part 4 is up to you
#filtering noise? eg ignore all tweets at contain less than 3 words? ignore all spam? more than 10 hashtags?
#rank tweets by their relevance to the company
#heuristics and rules
#THEN FILTER THEM OUT FROM THE TEST SET
#SEE IF IT GOES UP OR DOWN
#IF UP, THEN NOISE FILTERING WAS SUCESSFUL

#IR/TERRIER
#Rank tweets by relevance, only the top 80%
#Remove last 20%
#what if i remove the last 30%? or 10%? How it affects the f1 metrics?

#Maybe its good for apple, bad for alibaba? Due to no data? Better at some data than others? Investigate.

#Finally, combine

#continue collection data? YES
#Other predictors
#Web app/possibility
#Disseration
#Reference

#For dissertation, describe failures