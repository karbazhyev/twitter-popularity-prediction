"""Problem 2"""

import numpy as np
import json
from datetime import datetime
import time
import statsmodels.api as sm

hashtagLinReg = {'#gohawks': {},
                 '#gopatriots': {},
                 '#nfl': {},
                 '#patriots': {},
                 '#sb49': {},
                 '#superbowl': {}}

for hashtag in hashtagLinReg:
    numOfTweets = {}
    numOfRetweets = {}
    numOfFollowers = {}
    maxNumOfFollowers = {}
    timeOfDay = {}
    featureVec = []
    labelVec = []
    with open('tweet_data/tweets_%s.txt' % hashtag, 'r') as tweetFile:
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
            else:
                numOfTweets[tWind] = 1
                numOfRetweets[tWind] = tweet['metrics']['citations']['total']
                numOfFollowers[tWind] = tweet['author']['followers']
                maxNumOfFollowers[tWind] = tweet['author']['followers']
                timeOfDay[tWind] = datetime.time(datetime.fromtimestamp(t)).hour

            print('Hashtag: %s, Time of the Day: %d, Number of Tweets: %d' % (hashtag, timeOfDay[tWind], numOfTweets[tWind]))

    # Linear Regression

    for tWind in numOfTweets.keys():
        if ((tWind + 3600) in numOfTweets):
            labelVec.append(numOfTweets[tWind + 3600])
        else:
            labelVec.append(0)

        featureVec.append([numOfTweets[tWind],
                           numOfRetweets[tWind],
                           numOfFollowers[tWind],
                           maxNumOfFollowers[tWind],
                           timeOfDay[tWind]])

    featureVec = sm.add_constant(featureVec)
    featureVec = np.asarray(featureVec)
    labelVec = np.asarray(labelVec)
    labelVec = np.transpose(labelVec)
    linReg = sm.OLS(labelVec, featureVec)
    linRegResults = linReg.fit()

    hashtagLinReg[hashtag]['numOfTweets'] = numOfTweets
    hashtagLinReg[hashtag]['numOfRetweets'] = numOfRetweets
    hashtagLinReg[hashtag]['numOfFollowers'] = numOfFollowers
    hashtagLinReg[hashtag]['maxNumOfFollowers'] = maxNumOfFollowers
    hashtagLinReg[hashtag]['timeOfDay'] = timeOfDay
    hashtagLinReg[hashtag]['featureVec'] = featureVec
    hashtagLinReg[hashtag]['labelVec'] = labelVec
    hashtagLinReg[hashtag]['linRegResults'] = linRegResults
    #print linRegResults.summary()

"""
            timeOfDayVec = np.zeros(24)
            timeOfDayVec[timeOfDay[tWind]] = 1
            tmpFeatureVec = np.array(
                [numOfTweets[tWind], numOfRetweets[tWind], numOfFollowers[tWind], maxNumOfFollowers[tWind]])
            tmpFeatureVec = np.hstack((tmpFeatureVec, timeOfDayVec))
            featureVec.append(tmpFeatureVec)
"""