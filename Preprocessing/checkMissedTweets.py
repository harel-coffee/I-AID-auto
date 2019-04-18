import pandas as pd

def check_missed_Tweets(path, event_name):

    tweets_path = path+event_name+'/trecis2018-test.'+event_name+'.csv' # to load tweets retrieved by TREC-Downloader.jar
    tweets_path2=path+event_name+'/'+event_name+'_tweets.csv'
    trec_path = path+event_name+'/'+event_name+'.csv'

    tweets_df = pd.read_csv(tweets_path, header=0, index_col='tweet_id',engine='python')
    tweets_df2=pd.read_csv(tweets_path2, header=0, index_col='tweet_id',engine='python') # load tweets downloaded by TREC tool

    trec_df = pd.read_csv(trec_path, header=0, index_col='postID',engine='python')

     # convert index type from int64 to str
    if type(trec_df.index) != type(tweets_df.index):
        trec_df.index = trec_df.index.map(str)

    missed_tweets=pd.DataFrame(columns=['event_name','tweet_id'])
    
    i=0 #count
    for index, row in trec_df.iterrows():

        if index not in tweets_df.index and index not in tweets_df2.index:
            missed_tweets.loc[i]=[event_name,index]
            i+=1
    
    missed_tweets.to_csv(path+event_name+'/miss_tweets.csv',index=None)


if __name__ == "__main__":
     #TREC_Data.load_tweets_events()
    check_missed_Tweets('Data/TREC_Data/', 'bostonBombings2013')
    check_missed_Tweets('Data/TREC_Data/','australiaBushfire2013')
    check_missed_Tweets('Data/TREC_Data/','albertaFloods2013')
    check_missed_Tweets('Data/TREC_Data/','bostonBombings2013')
    check_missed_Tweets('Data/TREC_Data/','chileEarthquake2014')
    check_missed_Tweets('Data/TREC_Data/','guatemalaEarthquake2012')
    check_missed_Tweets('Data/TREC_Data/','italyEarthquakes2012')
    check_missed_Tweets('Data/TREC_Data/','joplinTornado2011')
    check_missed_Tweets('Data/TREC_Data/','manilaFloods2013')
    check_missed_Tweets('Data/TREC_Data/','parisAttacks2015')
    check_missed_Tweets('Data/TREC_Data/','philipinnesFloods2012')
    check_missed_Tweets('Data/TREC_Data/','typhoonHagupit2014')
    check_missed_Tweets('Data/TREC_Data/','typhoonYolanda2013')




