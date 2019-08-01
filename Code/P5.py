"""Problem 5"""

import numpy as np
import json
from datetime import datetime
import time
import statsmodels.api as sm

timeStart = int(time.mktime(datetime(2015, 2, 1, 8, 0, 0).timetuple()))
timeEnd = int(time.mktime(datetime(2015, 2, 1, 20, 0, 0).timetuple()))
hashtagLinReg = {'#gohawks': {},
                 '#gopatriots': {},
                 '#nfl': {},
                 '#patriots': {},
                 '#sb49': {},
                 '#superbowl': {}}


numOfTweets = {}
numOfRetweets = {}
numOfFollowers = {}
maxNumOfFollowers = {}
numOfFriends = {}
numOfStatuses = {}
numOfRankScore = {}
featureVecPeriod1 = []
labelVecPeriod1 = []
featureVecPeriod2 = []
labelVecPeriod2 = []
featureVecPeriod3 = []
labelVecPeriod3 = []

for hashtag in hashtagLinReg:
    with open('tweet_data/tweets_%s.txt' % hashtag, 'r') as tweetFile:
        print('Hashtag: %s File Loaded' % hashtag)
        for tweetLine in tweetFile:
            tweet = json.loads(tweetLine)
            t = tweet['firstpost_date']
            hour_ = datetime.time(datetime.fromtimestamp(t)).hour
            minute_ = datetime.time(datetime.fromtimestamp(t)).minute
            second_ = datetime.time(datetime.fromtimestamp(t)).second
            tWind = t - minute_ * 60 - second_

            if (tWind in numOfTweets):
                numOfTweets[tWind] += 1
                numOfRetweets[tWind] += tweet['metrics']['citations']['total']
                numOfFollowers[tWind] += tweet['author']['followers']
                if (tweet['author']['followers'] > maxNumOfFollowers[tWind]):
                    maxNumOfFollowers[tWind] = tweet['author']['followers']
                numOfFriends[tWind] += tweet['tweet']['user']['friends_count']
                numOfStatuses[tWind] += tweet['tweet']['user']['statuses_count']
                numOfRankScore[tWind] += tweet['metrics']['ranking_score']
            else:
                numOfTweets[tWind] = 1
                numOfRetweets[tWind] = tweet['metrics']['citations']['total']
                numOfFollowers[tWind] = tweet['author']['followers']
                maxNumOfFollowers[tWind] = tweet['author']['followers']
                numOfFriends[tWind] = tweet['tweet']['user']['friends_count']
                numOfStatuses[tWind] = tweet['tweet']['user']['statuses_count']
                numOfRankScore[tWind] = tweet['metrics']['ranking_score']

# Get feature and label vectors
for tWind in numOfTweets.keys():
    if (tWind < timeStart):
        if ((tWind + 3600) in numOfTweets):
            labelVecPeriod1.append(numOfTweets[tWind + 3600])
        else:
            labelVecPeriod1.append(0)

        tmpNumOfTweets = 0
        tmpNumOfRetweets = 0
        tmpNumOfFollowers = 0
        tmpMaxNumOfFollowers = 0
        tmpNumOfFriends = 0
        tmpNumOfStatuses = 0
        tmpNumOfRankScore = 0

        for k in range(6):
            if ((tWind - k * 3600) in numOfTweets):
                tmpNumOfTweets += numOfTweets[tWind - k * 3600]
                tmpNumOfRetweets += numOfRetweets[tWind - k * 3600]
                tmpNumOfFollowers += numOfFollowers[tWind - k * 3600]
                tmpMaxNumOfFollowers += maxNumOfFollowers[tWind - k * 3600]
                tmpNumOfFriends += numOfFriends[tWind - k * 3600]
                tmpNumOfStatuses += numOfStatuses[tWind - k * 3600]
                tmpNumOfRankScore += numOfRankScore[tWind - k * 3600]

        featureVecPeriod1.append([1,
                                  tmpNumOfTweets,
                                  tmpNumOfRetweets,
                                  tmpNumOfFollowers,
                                  tmpMaxNumOfFollowers,
                                  tmpNumOfFriends,
                                  tmpNumOfStatuses,
                                  tmpNumOfRankScore])
    elif (tWind < timeEnd):
        if ((tWind + 3600) in numOfTweets):
            labelVecPeriod2.append(numOfTweets[tWind + 3600])
        else:
            labelVecPeriod2.append(0)

        tmpNumOfTweets = 0
        tmpNumOfRetweets = 0
        tmpNumOfFollowers = 0
        tmpMaxNumOfFollowers = 0
        tmpNumOfFriends = 0
        tmpNumOfStatuses = 0
        tmpNumOfRankScore = 0

        for k in range(6):
            if ((tWind - k * 3600) in numOfTweets):
                tmpNumOfTweets += numOfTweets[tWind - k * 3600]
                tmpNumOfRetweets += numOfRetweets[tWind - k * 3600]
                tmpNumOfFollowers += numOfFollowers[tWind - k * 3600]
                tmpMaxNumOfFollowers += maxNumOfFollowers[tWind - k * 3600]
                tmpNumOfFriends += numOfFriends[tWind - k * 3600]
                tmpNumOfStatuses += numOfStatuses[tWind - k * 3600]
                tmpNumOfRankScore += numOfRankScore[tWind - k * 3600]

        featureVecPeriod2.append([1,
                                  tmpNumOfTweets,
                                  tmpNumOfRetweets,
                                  tmpNumOfFollowers,
                                  tmpMaxNumOfFollowers,
                                  tmpNumOfFriends,
                                  tmpNumOfStatuses,
                                  tmpNumOfRankScore])
    else:
        if ((tWind + 3600) in numOfTweets):
            labelVecPeriod3.append(numOfTweets[tWind + 3600])
        else:
            labelVecPeriod3.append(0)

        tmpNumOfTweets = 0
        tmpNumOfRetweets = 0
        tmpNumOfFollowers = 0
        tmpMaxNumOfFollowers = 0
        tmpNumOfFriends = 0
        tmpNumOfStatuses = 0
        tmpNumOfRankScore = 0

        for k in range(1):
            if ((tWind - k * 3600) in numOfTweets):
                tmpNumOfTweets += numOfTweets[tWind - k * 3600]
                tmpNumOfRetweets += numOfRetweets[tWind - k * 3600]
                tmpNumOfFollowers += numOfFollowers[tWind - k * 3600]
                tmpMaxNumOfFollowers += maxNumOfFollowers[tWind - k * 3600]
                tmpNumOfFriends += numOfFriends[tWind - k * 3600]
                tmpNumOfStatuses += numOfStatuses[tWind - k * 3600]
                tmpNumOfRankScore += numOfRankScore[tWind - k * 3600]

        featureVecPeriod3.append([1,
                                  tmpNumOfTweets,
                                  tmpNumOfRetweets,
                                  tmpNumOfFollowers,
                                  tmpMaxNumOfFollowers,
                                  tmpNumOfFriends,
                                  tmpNumOfStatuses,
                                  tmpNumOfRankScore])

# Linear Regression for each model
featureVecPeriod1 = np.asarray(featureVecPeriod1)
labelVecPeriod1 = np.asarray(labelVecPeriod1)
labelVecPeriod1 = np.transpose(labelVecPeriod1)

linRegPeriod1 = sm.OLS(labelVecPeriod1, featureVecPeriod1)
linRegResultsPeriod1 = linRegPeriod1.fit()
print(linRegResultsPeriod1.summary())

featureVecPeriod2 = np.asarray(featureVecPeriod2)
labelVecPeriod2 = np.asarray(labelVecPeriod2)
labelVecPeriod2 = np.transpose(labelVecPeriod2)

linRegPeriod2 = sm.OLS(labelVecPeriod2, featureVecPeriod2)
linRegResultsPeriod2 = linRegPeriod2.fit()
print(linRegResultsPeriod2.summary())

featureVecPeriod3 = np.asarray(featureVecPeriod3)
labelVecPeriod3 = np.asarray(labelVecPeriod3)
labelVecPeriod3 = np.transpose(labelVecPeriod3)

linRegPeriod3 = sm.OLS(labelVecPeriod3, featureVecPeriod3)
linRegResultsPeriod3 = linRegPeriod3.fit()
print(linRegResultsPeriod3.summary())

# Prediction
testFileSet = ['sample1_period1.txt',
               'sample4_period1.txt',
               'sample5_period1.txt',
               'sample8_period1.txt']
numOfTweets = 0
numOfRetweets = 0
numOfFollowers = 0
maxNumOfFollowers = 0
numOfFriends = 0
numOfStatuses = 0
numOfRankScore = 0

for testFile in testFileSet:
    with open('test_data/%s' % testFile, 'r') as tweetFile:
        print('Testfile: %s Loaded' % testFile)
        for tweetLine in tweetFile:
            tweet = json.loads(tweetLine)

            numOfTweets += 1
            numOfRetweets += tweet['metrics']['citations']['total']
            numOfFollowers += tweet['author']['followers']
            if (tweet['author']['followers'] > maxNumOfFollowers):
                maxNumOfFollowers = tweet['author']['followers']
            numOfFriends += tweet['tweet']['user']['friends_count']
            numOfStatuses += tweet['tweet']['user']['statuses_count']
            numOfRankScore += tweet['metrics']['ranking_score']

        featureVecTest = [1,
                          numOfTweets,
                          numOfRetweets,
                          numOfFollowers,
                          maxNumOfFollowers,
                          numOfFriends,
                          numOfStatuses,
                          numOfRankScore]

        featureVecTest = np.asarray(featureVecTest)

        labelVecTest = np.transpose(linRegResultsPeriod1.predict(featureVecTest))
        print('The predicted number of tweets for the next hour of test file: %s is %f' % (testFile, labelVecTest))

testFileSet = ['sample2_period2.txt']
"""
               'sample6_period2.txt',
               'sample9_period2.txt']
"""
numOfTweets = 0
numOfRetweets = 0
numOfFollowers = 0
maxNumOfFollowers = 0
numOfFriends = 0
numOfStatuses = 0
numOfRankScore = 0

for testFile in testFileSet:
    with open('test_data/%s' % testFile, 'r') as tweetFile:
        print('Testfile: %s Loaded' % testFile)
        for tweetLine in tweetFile:
            tweet = json.loads(tweetLine)

            numOfTweets += 1
            numOfRetweets += tweet['metrics']['citations']['total']
            numOfFollowers += tweet['author']['followers']
            if (tweet['author']['followers'] > maxNumOfFollowers):
                maxNumOfFollowers = tweet['author']['followers']
            numOfFriends += tweet['tweet']['user']['friends_count']
            numOfStatuses += tweet['tweet']['user']['statuses_count']
            numOfRankScore += tweet['metrics']['ranking_score']

        featureVecTest = [1,
                          numOfTweets,
                          numOfRetweets,
                          numOfFollowers,
                          maxNumOfFollowers,
                          numOfFriends,
                          numOfStatuses,
                          numOfRankScore]

        featureVecTest = np.asarray(featureVecTest)

        labelVecTest = np.transpose(linRegResultsPeriod2.predict(featureVecTest))
        print('The predicted number of tweets for the next hour of test file: %s is %f' % (testFile, labelVecTest))

testFileSet = ['sample3_period3.txt',
               'sample7_period3.txt',
               'sample10_period3.txt']
numOfTweets = 0
numOfRetweets = 0
numOfFollowers = 0
maxNumOfFollowers = 0
numOfFriends = 0
numOfStatuses = 0
numOfRankScore = 0

for testFile in testFileSet:
    with open('test_data/%s' % testFile, 'r') as tweetFile:
        print('Testfile: %s Loaded' % testFile)
        for tweetLine in tweetFile:
            tweet = json.loads(tweetLine)

            numOfTweets += 1
            numOfRetweets += tweet['metrics']['citations']['total']
            numOfFollowers += tweet['author']['followers']
            if (tweet['author']['followers'] > maxNumOfFollowers):
                maxNumOfFollowers = tweet['author']['followers']
            numOfFriends += tweet['tweet']['user']['friends_count']
            numOfStatuses += tweet['tweet']['user']['statuses_count']
            numOfRankScore += tweet['metrics']['ranking_score']

        featureVecTest = [1,
                          numOfTweets,
                          numOfRetweets,
                          numOfFollowers,
                          maxNumOfFollowers,
                          numOfFriends,
                          numOfStatuses,
                          numOfRankScore]

        featureVecTest = np.asarray(featureVecTest)

        labelVecTest = np.transpose(linRegResultsPeriod3.predict(featureVecTest))
        print('The predicted number of tweets for the next hour of test file: %s is %f' % (testFile, labelVecTest))