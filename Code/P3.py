"""Problem 3"""

import numpy as np
import json
from datetime import datetime
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
    numOfFriends = {}
    numOfStatuses = {}
    numOfRankScore = {}
    featureVec = []
    labelVec = []
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
        if ((tWind + 3600) in numOfTweets):
            labelVec.append(numOfTweets[tWind + 3600])
        else:
            labelVec.append(0)

        featureVec.append([numOfTweets[tWind],
                           numOfRetweets[tWind],
                           numOfFollowers[tWind],
                           maxNumOfFollowers[tWind],
                           numOfFriends[tWind],
                           numOfStatuses[tWind],
                           numOfRankScore[tWind]])

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
    hashtagLinReg[hashtag]['numOfFriends'] = numOfFriends
    hashtagLinReg[hashtag]['numOfStatuses'] = numOfStatuses
    hashtagLinReg[hashtag]['numOfRankScore'] = numOfRankScore
    hashtagLinReg[hashtag]['featureVec'] = featureVec
    hashtagLinReg[hashtag]['labelVec'] = labelVec
    hashtagLinReg[hashtag]['linRegResults'] = linRegResults

# Plotting
#"""
hashtag = '#superbowl'
featureIndex = 6
featureVec = hashtagLinReg[hashtag]['featureVec']
labelVec = hashtagLinReg[hashtag]['labelVec']

plt.scatter(np.log10(featureVec[:, featureIndex]), np.log10(labelVec))
plt.ylabel('$log_{10}$(Number of tweets for next hour)')
plt.xlabel('$log_{10}$(Number of statuses)')
plt.title(hashtag)
plt.grid(True)
plt.show()

#"""