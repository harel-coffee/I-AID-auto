from twarc import Twarc
import pandas as pd
import numpy as np
from os import listdir


class Dataset_Reader:
    def __init__(self, consumer_key=None, consumer_secret=None,access_token=None, access_token_secret=None, trec_path=None, tweets_dir=None):

        self.consumer_key=consumer_key
        self.consumer_secret=consumer_secret
        self.access_token=access_token
        self.access_token_secret=access_token_secret
        self.trec_path=trec_path
        self.tweets_dir=tweets_dir
        return None

    def load_TREC_Data(self):
        '''
        This function loads TREC dataset (Event ID, Tweet ID, Indicator Terms, Priority, Cateorgries)
        '''
        return None

    def load_tweets(self,fileName=None):
        #Tweets_df that contain tweets retrieved from Twarc        
        column_names=['event_id', 'event_name', 'tweet_id']
        tweets_df=pd.read_csv(fileName,delimiter='\t',index_col='tweet_id')
        

        tweets_df['full_text']=None
        tweets_df['date']=None
    

        #Read tweets IDs from tsv file into pandas dataframe
        tweets_Ids=tweets_df.index.values.tolist()

        twarc_loader=Twarc(self.consumer_key, self.consumer_secret, self.access_token, self.access_token_secret)

        tweets=twarc_loader.hydrate(iter(tweets_Ids))#

        #Iterate tweets from Twarc

        for tweet in tweets:
            tweets_df.full_text.loc[tweet['id']]=tweet['full_text']
            tweets_df.date.loc[tweet['id']]=tweet['created_at']

        #save into file
        fileName=fileName.split('/')
        tweets_df.to_csv('Data/events_df/'+fileName[-1])
        return tweets_df
    
if __name__ == "__main__":
    #from secrets import consumer_key, consumer_secret, access_token, access_token_secret

    consumer_key='Q7j4eN16sx7NWXfIysgjz4bJv'
    consumer_secret='pNgJvYXIEunIPnQPHiYR3HXmCcLOgpffwYKAvHCWjeKpGHGLkI'
    access_token='53767406-fgupotwM59YIC5UrxAP5yWpE4fDwqhm987T8fI2XP'
    access_token_secret='5ijj5OrVDvuIpUIpSIuU9fzSgsjJX6DVwJouY9OTSkKzY'

    TREC_data=Dataset_Reader(consumer_key,consumer_secret, access_token, access_token_secret)

    #Loading tweets per event and save into separate dataframe
    print ('--Loading tweets per event and save into separate dataframe--')
    for file in listdir('Data/event_tweetsIDs'):      
        print ('Processing tweets for event:'+file[:-4])
        TREC_data.load_tweets('Data/event_tweetsIDs/'+file)
    print('Done...')

  
    for index, row in trec_data.iterrows():
        #row
        print (row)
        json_df=pd.DataFrame(row)
        json_df=json_df[index]

        
    