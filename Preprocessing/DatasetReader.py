from twarc import Twarc
import pandas as pd
import numpy as np

class DataSet_Reader:
    def __init__(self, consumer_key=None, consumer_secret=None,access_token=None, access_token_secret=None, trec_path=None, tweets_dir=None):

        self.consumer_key=consumer_key
        self.consumer_secret=consumer_secret
        self.access_token=access_token
        self.access_token_secret=access_token_secret
        self.trec_path=trec_path
        self.tweets_dir=tweets_dir
        return None

    def load_TREC_Data():
        '''
        This function loads TREC dataset (Event ID, Tweet ID, Indicator Terms, Priority, Cateorgries)
        '''
        return None

    def get_Tweets(self,fileName=None):

        #Tweets_df that contain tweets retrieved from Twarc
        Tweets_df=pd.DataFrame()

        #Read tweets IDs from tsv file
        tweets_file=open(fileName,'r')

        #save TWeets_IDs into list
        tweets_IDs=[]


        twarc_loader=Twarc(self.consumer_key, self.consumer_secret, self.access_token, self.access_token_secret)

        tweets=twarc_loader.hydrate(iter(tweets_IDs))#

        #Iterate tweets from Twarc

        for tweet in tweets:
            print (tweet)
            Tweets_df['full_text']=tweet['full_text']
            Tweets_df['date']=tweet['created_at']

        #save into file
        
        Tweets_df.to_csv('./')
        return Tweets_df

