from os import listdir
import json

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


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

    def merge_trecData_tweets(self, path, event_name):
        tweets_path = path+event_name+'/trecis2018-test.'+event_name + '.csv'
        trec_path = path+event_name+'/'+event_name+'.csv'

        tweets_df = pd.read_csv(tweets_path, header=0,
                                index_col='tweet_id')
        trec_df = pd.read_csv(trec_path, header=0,
                              index_col='postID')

        trec_df['full_text'] = None

        for index, row in trec_df.iterrows():
            if index in tweets_df.index.values:
                row['full_text'] = tweets_df.loc[index, 'Text']

        trec_df.to_csv(path+event_name+'/'+event_name+'_all.csv')

        print(event_name, ' length ', len(trec_df))

    def combine_all_events(self, path, event_list):
        '''
        :param path: to save the common dataframe
        :param event_list: all the events in train/test dataset
        :return: save the combined df in the given path
        '''

        file_paths = []
        for event in event_list:
            file_paths.append(path + event + '/' + event + '_all.csv')

        frames = [pd.read_csv(
            fpath, header=0, index_col='postID') for fpath in file_paths]

        combined_df = pd.concat(frames)
        combined_df.to_csv(path+'/'+'all_events.csv')

        print(combined_df.isna().sum())
        print('length of combined events', len(combined_df))


if __name__ == "__main__":

    TREC_Data = Dataset_Reader()
    events_list = ['albertaFloods2013', 'australiaBushfire2013', 'bostonBombings2013', 'chileEarthquake2014',
                   'flSchoolShooting2018','guatemalaEarthquake2012', 'italyEarthquakes2012', 'joplinTornado2011', 
                   'manilaFloods2013','nepalEarthquake2015','parisAttacks2015','philipinnesFloods2012', 
                   'queenslandFloods2013','typhoonHagupit2014','typhoonYolanda2013']

    # merge each event with its tweets

    for event in events_list:
        TREC_Data.merge_trecData_tweets('Data/TREC_Data/', event)

    TREC_Data.combine_all_events('Data/TREC_Data/', events_list)

    file_paths = []
    for event in events_list:
        file_paths.append('Data/TREC_Data/' + event + '/' + event + '.csv')

        frames = [pd.read_csv(
            fpath, header=0, index_col='postID', engine='python') for fpath in file_paths]

    combined_df = pd.concat(frames)
    

    '''
    nepal_list=['nepalEarthquake2015', 'nepalEarthquake2015S1', 'nepalEarthquake2015S2','nepalEarthquake2015S3','nepalEarthquake2015S4']
    hagupit_list=['typhoonHagupit2014', 'typhoonHagupit2014S1', 'typhoonHagupit2014S2']

    file_paths = []

    for event in nepal_list:
        file_paths.append('Data/TREC_Data/nepalEarthquake2015/' + event + '.csv')
        frames = [pd.read_csv(fpath, header=0, index_col='postID', engine='python') for fpath in file_paths]

        combined_df = pd.concat(frames)
        combined_df.to_csv('Data/TREC_Data/nepalEarthquake2015/combined_nepalEarthquake2015.csv')

        print (len(combined_df))


    file_paths = []
    for event in hagupit_list:
        file_paths.append('Data/TREC_Data/typhoonHagupit2014/' + event + '.csv')
        frames = [pd.read_csv(fpath, header=0, index_col='postID', engine='python') for fpath in file_paths]

        combined_df = pd.concat(frames)
        combined_df.to_csv('Data/TREC_Data/typhoonHagupit2014/combined_hagupit.csv')        

        print (len(combined_df))

    '''









