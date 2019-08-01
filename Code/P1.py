"""Problem 1"""

import numpy as np
import json
import matplotlib.pyplot as plt


hashtagStat = {'#gohawks': {},
              '#gopatriots': {},
              '#nfl': {},
              '#patriots': {},
              '#sb49': {},
              '#superbowl': {}}

for hashtag in hashtagStat:
    numOfTweets = 0
    numOfFollowers = 0
    maxNumOfFollowers = 0
    numOfRetweets = 0
    tStart = 0
    tIndex = []
    tEnd = 0
    with open('tweet_data/tweets_%s.txt' % hashtag, 'r') as tweetFile:
        for tweetLine in tweetFile:
            tweet = json.loads(tweetLine)
            numOfTweets += 1
            print('hashtag: %s, Number of Tweets: %d' % (hashtag, numOfTweets))
            numOfFollowers += tweet['author']['followers']
            if (tweet['author']['followers'] > maxNumOfFollowers):
                maxNumOfFollowers = tweet['author']['followers']
            numOfRetweets += tweet['metrics']['citations']['total']

            if (hashtag == '#superbowl') or (hashtag == '#nfl'):
                tIndex.append(tweet['firstpost_date'])

            if (numOfTweets == 1):
                tStart = tweet['firstpost_date']

        tEnd = tweet['firstpost_date']

    if (hashtag == '#superbowl') or (hashtag == '#nfl'):
        hashtagStat[hashtag]['tIndex'] = tIndex

    hashtagStat[hashtag]['numOfTweets'] = numOfTweets
    hashtagStat[hashtag]['numOfFollowers'] = numOfFollowers
    hashtagStat[hashtag]['maxNumOfFollowers'] = maxNumOfFollowers
    hashtagStat[hashtag]['numOfRetweets'] = numOfRetweets
    hashtagStat[hashtag]['tStart'] = tStart
    hashtagStat[hashtag]['tEnd'] = tEnd
    hashtagStat[hashtag]['avgNumOfTweetsPerHour'] = float(numOfTweets)/(tEnd - tStart)*3600
    hashtagStat[hashtag]['avgNumOfFollowersPerTweet'] = float(numOfFollowers)/numOfTweets
    hashtagStat[hashtag]['avgNumOfRetweetsPerTweet'] = float(numOfRetweets)/numOfTweets

# Plotting
tArray = (np.array(hashtagStat['#superbowl']['tIndex']) - hashtagStat['#superbowl']['tStart'])/3600.0
tBins = np.arange(min(tArray), max(tArray) + 1, 1)
plt.hist(tArray, tBins)
plt.grid(True)
plt.xlabel('Time (hour)')
plt.ylabel('Number of tweets')
plt.title('#SuperBowl')
plt.show()

tArray = (np.array(hashtagStat['#nfl']['tIndex']) - hashtagStat['#nfl']['tStart'])/3600.0
tBins = np.arange(min(tArray), max(tArray) + 1, 1)
plt.hist(tArray, tBins)
plt.grid(True)
plt.xlabel('Time (hour)')
plt.ylabel('Number of tweets')
plt.title('#NFL')
plt.show()
