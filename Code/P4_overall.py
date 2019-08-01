"""Problem 4"""

import numpy as np
import json
from datetime import datetime
import time
import statsmodels.api as sm

kFold = 10
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
featureVec = []
labelVec = []
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

for tWind in numOfTweets.keys():
    if (tWind < timeStart):
        if ((tWind + 3600) in numOfTweets):
            labelVecPeriod1.append(numOfTweets[tWind + 3600])
        else:
            labelVecPeriod1.append(0)

        featureVecPeriod1.append([1,
                                  numOfTweets[tWind],
                                  numOfRetweets[tWind],
                                  numOfFollowers[tWind],
                                  maxNumOfFollowers[tWind],
                                  numOfFriends[tWind],
                                  numOfStatuses[tWind],
                                  numOfRankScore[tWind]])
    elif (tWind < timeEnd):
        if ((tWind + 3600) in numOfTweets):
            labelVecPeriod2.append(numOfTweets[tWind + 3600])
        else:
            labelVecPeriod2.append(0)

        featureVecPeriod2.append([1,
                                  numOfTweets[tWind],
                                  numOfRetweets[tWind],
                                  numOfFollowers[tWind],
                                  maxNumOfFollowers[tWind],
                                  numOfFriends[tWind],
                                  numOfStatuses[tWind],
                                  numOfRankScore[tWind]])
    else:
        if ((tWind + 3600) in numOfTweets):
            labelVecPeriod3.append(numOfTweets[tWind + 3600])
        else:
            labelVecPeriod3.append(0)

        featureVecPeriod3.append([1,
                                  numOfTweets[tWind],
                                  numOfRetweets[tWind],
                                  numOfFollowers[tWind],
                                  maxNumOfFollowers[tWind],
                                  numOfFriends[tWind],
                                  numOfStatuses[tWind],
                                  numOfRankScore[tWind]])

    if ((tWind + 3600) in numOfTweets):
        labelVec.append(numOfTweets[tWind + 3600])
    else:
        labelVec.append(0)

    featureVec.append([1,
                       numOfTweets[tWind],
                       numOfRetweets[tWind],
                       numOfFollowers[tWind],
                       maxNumOfFollowers[tWind],
                       numOfFriends[tWind],
                       numOfStatuses[tWind],
                       numOfRankScore[tWind]])
# The whole period
lenOfFeatureVec = len(featureVec)
lenFold = lenOfFeatureVec / kFold
indices = np.arange(lenOfFeatureVec)
np.random.shuffle(indices)
errorPrediction = 0

for k in range(kFold):
    indicesTest = indices[range(k * lenFold, (k + 1) * lenFold)]
    featureVecTest = [featureVec[i] for i in indicesTest]
    labelVecTest = [labelVec[i] for i in indicesTest]
    indicesTrain = np.delete(indices, range(k * lenFold, (k + 1) * lenFold))
    featureVecTrain = [featureVec[i] for i in indicesTrain]
    labelVecTrain = [labelVec[i] for i in indicesTrain]

    featureVecTrain = np.asarray(featureVecTrain)
    labelVecTrain = np.asarray(labelVecTrain)
    labelVecTrain = np.transpose(labelVecTrain)

    linReg = sm.OLS(labelVecTrain, featureVecTrain)
    linRegResults = linReg.fit()

    labelVecPredict = np.transpose(linRegResults.predict(featureVecTest))

    errorPrediction += sum(abs(labelVecPredict - labelVecTest)) / float(lenFold)

print('The average prediction error overall is %f' % (errorPrediction/kFold))

# 3 time periods
# Period 1
lenOfFeatureVec = len(featureVecPeriod1)
lenFold = lenOfFeatureVec / kFold
indices = np.arange(lenOfFeatureVec)
np.random.shuffle(indices)
errorPrediction = 0

for k in range(kFold):
    indicesTest = indices[range(k * lenFold, (k + 1) * lenFold)]
    featureVecTest = [featureVecPeriod1[i] for i in indicesTest]
    labelVecTest = [labelVecPeriod1[i] for i in indicesTest]
    indicesTrain = np.delete(indices, range(k * lenFold, (k + 1) * lenFold))
    featureVecTrain = [featureVecPeriod1[i] for i in indicesTrain]
    labelVecTrain = [labelVecPeriod1[i] for i in indicesTrain]

    featureVecTrain = np.asarray(featureVecTrain)
    labelVecTrain = np.asarray(labelVecTrain)
    labelVecTrain = np.transpose(labelVecTrain)
    linReg = sm.OLS(labelVecTrain, featureVecTrain)
    linRegResults = linReg.fit()

    labelVecPredict = np.transpose(linRegResults.predict(featureVecTest))

    errorPrediction += sum(abs(labelVecPredict - labelVecTest)) / float(lenFold)

print('The average prediction error overall in period 1 is %f' % (errorPrediction/kFold))

# Period 2
lenOfFeatureVec = len(featureVecPeriod2)
lenFold = lenOfFeatureVec / kFold
indices = np.arange(lenOfFeatureVec)
np.random.shuffle(indices)
errorPrediction = 0

for k in range(kFold):
    indicesTest = indices[range(k * lenFold, (k + 1) * lenFold)]
    featureVecTest = [featureVecPeriod2[i] for i in indicesTest]
    labelVecTest = [labelVecPeriod2[i] for i in indicesTest]
    indicesTrain = np.delete(indices, range(k * lenFold, (k + 1) * lenFold))
    featureVecTrain = [featureVecPeriod2[i] for i in indicesTrain]
    labelVecTrain = [labelVecPeriod2[i] for i in indicesTrain]

    featureVecTrain = np.asarray(featureVecTrain)
    labelVecTrain = np.asarray(labelVecTrain)
    labelVecTrain = np.transpose(labelVecTrain)
    linReg = sm.OLS(labelVecTrain, featureVecTrain)
    linRegResults = linReg.fit()

    labelVecPredict = np.transpose(linRegResults.predict(featureVecTest))

    errorPrediction += sum(abs(labelVecPredict - labelVecTest)) / float(lenFold)

print('The average prediction error in period 2 is %f' % (errorPrediction/kFold))

# Period 3
lenOfFeatureVec = len(featureVecPeriod3)
lenFold = lenOfFeatureVec / kFold
indices = np.arange(lenOfFeatureVec)
np.random.shuffle(indices)
errorPrediction = 0

for k in range(kFold):
    indicesTest = indices[range(k * lenFold, (k + 1) * lenFold)]
    featureVecTest = [featureVecPeriod3[i] for i in indicesTest]
    labelVecTest = [labelVecPeriod3[i] for i in indicesTest]
    indicesTrain = np.delete(indices, range(k * lenFold, (k + 1) * lenFold))
    featureVecTrain = [featureVecPeriod3[i] for i in indicesTrain]
    labelVecTrain = [labelVecPeriod3[i] for i in indicesTrain]

    featureVecTrain = np.asarray(featureVecTrain)
    labelVecTrain = np.asarray(labelVecTrain)
    labelVecTrain = np.transpose(labelVecTrain)
    linReg = sm.OLS(labelVecTrain, featureVecTrain)
    linRegResults = linReg.fit()

    labelVecPredict = np.transpose(linRegResults.predict(featureVecTest))

    errorPrediction += sum(abs(labelVecPredict - labelVecTest)) / float(lenFold)

print('The average prediction error overall in period 3 is %f' % (errorPrediction/kFold))
