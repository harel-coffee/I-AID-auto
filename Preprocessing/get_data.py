import os
import json
import pprint

import numpy as np
import pandas as pd
pd.options.display.float_format = '{:20,.2f}'.format
from pandas.io.json import json_normalize

events = ['albertaFloods2013', 'australiaBushfire2013', 'bostonBombings2013', 'chileEarthquake2014',
           'flSchoolShooting2018','guatemalaEarthquake2012', 'italyEarthquakes2012', 'joplinTornado2011',
           'manilaFloods2013','nepalEarthquake2015','parisAttacks2015','philipinnesFloods2012',
           'queenslandFloods2013','typhoonHagupit2014','typhoonYolanda2013']

class Dataset_Verification:

    def validate_json_files(self):
        for event in events:
            with open('Data/TREC_recheck/trecis2018-test.' +event+'.json', 'r') as file:
                data = [json.loads(line) for line in file]
                json.dump(data, open('Data/TREC_recheck/'+event+'.json', 'w'))
        print('Done.')

    # 1) correct json encoding by first loading into list, then saving it as another json file
    # 2) filter properties from allProperties, retain 'topic' in the final dataframe
    # 3) save tweet_id as key and rest of the properties as values
    # metadata = ['user.verified', 'entities.urls', 'entities.hashtags', 'retweet_count', 'entities.symbols', 'id',
    # 'text', 'user.location', 'user.followers_count', 'created_at', 'retweeted_status.entities.hashtags' ]

    def fetch_trec_tweets_event_wise(self):
        for event in events:
            tweet_dict = {} # id -> tweet data + metadata

            with open('Data/TREC_recheck/'+ event +'.json', 'r') as f:
                data = json.load(f)

                for tweet in data: # for every tweet, fetch id and metadata
                    tdict = {}
                    k = tweet['allProperties']
                    #tdict['id'] = str(k['id'])
                    tdict['user.verified'] = k['user.verified']
                    tdict['entities.urls'] = k['entities.urls']
                    tdict['entities.hashtags'] = k['entities.hashtags']
                    tdict['retweet_count'] = k['retweet_count']
                    tdict['text'] = k['text']
                    tdict['user.location'] = k['user.location']
                    tdict['user.followers_count'] = k['user.followers_count']
                    tdict['created_at'] = k['created_at']
                    tdict['topic'] = tweet['topic']
                    tdict['truncated'] = k['truncated']
                    tdict['user.url'] = k['user.url']
                    tdict['user.description'] = k['user.description']
                    if 'retweeted_status.entities.hashtags' in k:
                        tdict['retweeted_status.entities.hashtags'] = k['retweeted_status.entities.hashtags']
                        tdict['retweeted_status.text'] = k['retweeted_status.text']
                    else:
                        tdict['retweeted_status.entities.hashtags'] = None
                        tdict['retweeted_status.text'] = None
                    tweet_dict[str(k['id'])] = tdict

                tweet_df = pd.DataFrame.from_dict(tweet_dict, orient='index')
                tweet_df.index.map(str)
                print(tweet_df.dtypes)
                tweet_df.to_csv('Data/TREC_recheck/tweets/'+ event +'.csv',index=True)

    def fetch_trec_labels(self, path = 'Data/TRECIS-2018-TestEvents-Labels'):
        seen_files = set()
        for f in os.listdir(path):
            with open(os.path.join(path,f), 'r') as file:
                print(f)
                data = json.load(file)
                for event in data['events']:
                    event_name = event['eventid']
                    tweet_labels = {}
                    for tweet in event['tweets']:
                        tdict ={}
                        #tdict['postID'] = str(tweet['postID'])
                        tdict['categories'] = tweet['categories']
                        tdict['priority'] = tweet['priority']
                        tweet_labels[tweet['postID']] = tdict
                    tweet_df = pd.DataFrame.from_dict(tweet_labels, orient='index')
                    tweet_df.index = [str(x) for x in tweet_df.index]
                    for ind, row in tweet_df.iterrows():
                        print(ind, type(ind))

                    if event_name not in seen_files:
                        seen_files.add(event_name)
                        tweet_df.to_csv('Data/TREC_recheck/labels/' + event_name + '.csv', index=True)
                    else:
                        with open('Data/TREC_recheck/labels/' + event_name + '.csv', 'a') as fil:
                            tweet_df.to_csv(fil, index=True, header='false')
                    print(f'length of {event_name} df : {len(tweet_df)}')

    def merge_all_labels_or_tweets(self, path ='Data/TREC_recheck/labels', fname='all_labels.csv'):
        all_paths = [os.path.join(path, file) for file in os.listdir(path) ]

        frames = []
        for fpath in all_paths:
            try:
                df = pd.read_csv(fpath, header=0, dtype=object, engine='python', index_col=0)
                df.index = [str(x) for x in df.index]
                for ind, row in df.iterrows():
                    print(ind, type(ind))
                    if '.' in ind:
                        print(fpath)
                        break
                frames.append(df)
            except:
                continue

        # frames = [pd.read_csv(
        #     fpath, header=0,  engine='python') for fpath in all_paths]

        combined_df = pd.concat(frames, axis=0, sort=False, ignore_index=False)
        combined_df.to_csv(path + '/' + fname, index=True)

        print(combined_df.isna().sum())
        print('combined length', len(combined_df))

    def combine_tweets_with_labels(self):
        tweet_df = pd.read_csv('Data/TREC_recheck/tweets/all_tweets.csv', header=0,  engine='python', index_col=0)
        tweet_df.index = [str(x) for x in tweet_df.index]
        label_df = pd.read_csv('Data/TREC_recheck/labels/all_labels.csv', header=0,  engine='python', index_col=0)
        label_df.index = [str(x) for x in label_df.index]
        print(tweet_df.dtypes)
        print(label_df.dtypes)
        combined_df = pd.merge(label_df, tweet_df, left_index=True, right_index=True, how='inner')
        combined_df.to_csv('Data/TREC_recheck/trec_data.csv', index=True)
        print(combined_df.isna().sum())
        print('combined length', len(combined_df))


if __name__ == '__main__':
    dv = Dataset_Verification()
    dv.fetch_trec_labels()
    dv.fetch_trec_tweets_event_wise()
    dv.merge_all_labels_or_tweets()
    dv.merge_all_labels_or_tweets(path='Data/TREC_recheck/tweets', fname='all_tweets.csv')
    dv.combine_tweets_with_labels()

