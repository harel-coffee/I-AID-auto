from os import listdir
import json

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

from twarc import Twarc


class Dataset_Reader:

    def load_TREC_Data(self, fileName):
        '''
        This function loads TREC dataset (Event ID, Tweet ID, Indicator Terms, Priority, Cateorgries)
        '''
        with open(fileName, 'r') as file:
            json_data = json.load(file)

        json_df = pd.DataFrame.from_dict(
            json_normalize(json_data), orient='columns')
        json_df.set_index('eventid', inplace=True)
        return json_df

    def load_tweets(self, fileName, consumer_key, consumer_secret, access_token, access_token_secret):
        # Tweets_df that contain tweets retrieved from Twarc

        tweets_df = pd.read_csv(fileName, delimiter='\t', index_col='tweet_id')

        tweets_df['full_text'] = None
        tweets_df['date'] = None

        # Read tweets IDs from tsv file into pandas dataframe
        tweets_Ids = tweets_df.index.values.tolist()

        twarc_loader = Twarc(consumer_key, consumer_secret,
                             access_token, access_token_secret)

        tweets = twarc_loader.hydrate(iter(tweets_Ids))

        # Iterate tweets from Twarc
        for tweet in tweets:
            tweets_df.full_text.loc[tweet['id']] = tweet['full_text']
            tweets_df.date.loc[tweet['id']] = tweet['created_at']

        # save into file
        fileName = fileName.split('/')
        tweets_df.to_csv('Data/eventsTweets_df/'+fileName[-1])
        

    def load_tweets_events(self):
        consumer_key = 'Q7j4eN16sx7NWXfIysgjz4bJv'
        consumer_secret = 'pNgJvYXIEunIPnQPHiYR3HXmCcLOgpffwYKAvHCWjeKpGHGLkI'
        access_token = '53767406-fgupotwM59YIC5UrxAP5yWpE4fDwqhm987T8fI2XP'
        access_token_secret = '5ijj5OrVDvuIpUIpSIuU9fzSgsjJX6DVwJouY9OTSkKzY'

        # Loading tweets per event and save into separate dataframe
        print('--Loading tweets per event and save into separate dataframe--')

        for file in listdir('Data/event_tweetsIDs'):
            print('Processing tweets for event:'+file[:-4])
            self.load_tweets('Data/event_tweetsIDs/'+file, consumer_key,
                             consumer_secret, access_token, access_token_secret)
        print('Done...')

    def save_TRECData_to_csv(self):
        '''
        # save events TREC data as dataframes in csv files.
        path = 'Data/TRECIS-2018-TestEvents-Labels/'
        for file in listdir(path):

            trec_data = self.load_TREC_Data(path+file)

            for index, row in trec_data.iterrows():

                json_df = pd.DataFrame(row)
                json_df = json_df[index]

        
            #Saving TREC data into json file per event.
        
            fname = 'Data/events_trec/'+index+'.json'
            with open(fname, 'w') as f:
                f.write(json_df.to_json(orient='records'))
        '''
        
        #Load events json file as dataframe
        
        path1 = 'Data/events_trec_json/'
        path2 = 'Data/events_trec_df/'

        for file in listdir(path1):
            json_df = pd.read_json(path1+file, orient='records')
            json_df.to_csv(path2+file[:-4]+'csv', header=True, index=False)

    def merge_trecData_tweets(self, path, event_name):
        #tweets_path = path+event_name+'/'+event_name+'_tweets.csv'
        print(event_name)
        tweets_path = path+event_name+'/trecis2018-test.'+event_name+'.csv' # to load tweets retrieved by TREC-Downloader.jar
        trec_path = path+event_name+'/'+event_name+'.csv'

        tweets_df = pd.read_csv(tweets_path, header=0, index_col='tweet_id',engine='python')
        trec_df = pd.read_csv(trec_path, header=0, index_col='postID',engine='python')

        
        trec_df['full_text'] = None

        # convert index type from int64 to str
        if type(trec_df.index) != type(tweets_df.index):
            trec_df.index = trec_df.index.map(str)

        print(len(tweets_df), len(trec_df))
        count = 0


        for index, row in tweets_df.iterrows():
            trec_df.loc[index, 'full_text'] = row['Text']
            #print(row['full_text'])

        trec_df.to_csv(path+event_name+'/'+event_name+'_all.csv')

    def combine_all_events(self, path, event_list):
        '''
        :param path: to save the common dataframe
        :param event_list: all the events in train/test dataset
        :return: save the combined df in the given path
        '''
        # event_df_path = path + event_list[0] + '/' + event_list[0] + '_all.csv'
        # combined_df = pd.read_csv(event_df_path, header=0, index_col='tweet_id', engine='python')

        file_paths = []
        for event in event_list:
            file_paths.append(path + event +'/'+ event +'_all.csv')

        frames = [ pd.read_csv(fpath, header=0, engine='python') for fpath in file_paths ]
        combined_df = pd.concat(frames)

        combined_df.to_csv(path+'/'+'all_events.csv')
        print(len(combined_df))


if __name__ == "__main__":
    TREC_Data = Dataset_Reader()
    events_names = ['albertaFloods2013', 'australiaBushfire2013', 'bostonBombings2013', 'chileEarthquake2014',
                    'flSchoolShooting2018',
                    'guatemalaEarthquake2012', 'italyEarthquakes2012', 'joplinTornado2011', 'manilaFloods2013',
                    'nepalEarthquake2015', 'parisAttacks2015', 'philipinnesFloods2012', 'queenslandFloods2013',
                    'typhoonHagupit2014', 'typhoonYolanda2013']

    TREC_Data.combine_all_events('Data/TREC_Data/', events_names)

    #TREC_Data.load_tweets_events()
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/', 'bostonBombings2013')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','australiaBushfire2013')
    #
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','albertaFloods2013')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','bostonBombings2013')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','chileEarthquake2014')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','guatemalaEarthquake2012')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','italyEarthquakes2012')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/', 'flSchoolShooting2018')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/', 'queenslandFloods2013')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/', 'nepalEarthquake2015')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','joplinTornado2011')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','manilaFloods2013')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','parisAttacks2015')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','philipinnesFloods2012')
    #
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','typhoonHagupit2014')
    # TREC_Data.merge_trecData_tweets('Data/TREC_Data/','typhoonYolanda2013')



