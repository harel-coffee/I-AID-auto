import string
import nltk

import pandas as pd

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class Linguistic_Features:

    def __init__(self):

        self.transtable = {
            ord(c): None for c in string.punctuation + string.digits}
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        # if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
        tokens = [word for word in nltk.word_tokenize(
            text.translate(self.transtable)) if len(word) > 1]
        stems = [self.stemmer.stem(item) for item in tokens]
        return stems

    def tfIdf_feature(self, event_df, features_cnt):
        '''
        -   event_df : the event dataframe which contains trec data combined with tweets
        -   max_features: max of feaures to generate

        return a dictionaly of (word, tfIdf)
        '''

        tfidf = TfidfVectorizer(tokenizer=self.tokenize,  ngram_range=(
            1, 2), stop_words='english', max_features=features_cnt, sublinear_tf=True)

        features = pd.DataFrame(tfidf.fit_transform(event_df['full_text']).toarray(
        ), columns=['tfidf_'+name for name in tfidf.get_feature_names()])

        drop_columns = ['event_name', 'event_id', 'full_text', 'date']
        event_df = event_df.drop(drop_columns, axis=1)

        features = event_df.join(features)

        return features


if __name__ == "__main__":

    events_path = 'Data/TREC_Data/'
    events_names = ['albertaFloods2013', 'australiaBushfire2013', 'bostonBombings2013', 'chileEarthquake2014', 'flSchoolShooting2018',
                    'guatemalaEarthquake2012', 'italyEarthquakes2012', 'joplinTornado2011', 'manilaFloods2013', 'nepalEarthquake2015', 'parisAttacks2015', 'philipinnesFloods2012', 'queenslandFloods2013', 'typhoonHagupit2014', 'typhoonYolanda2013']



    #compute TF-IDF features:
    ling_feat = Linguistic_Features()
    
    for event in events_names:
        event_tweets_path=events_path+event+'/'+event+'_tweets.csv'

        try:
            event_tweets_df=pd.read_csv(event_tweets_path)
        except:
            event_tweets_df=pd.read_csv(open(event_tweets_path,'rU'), encoding='utf-8', engine='c')


        # drop rows with null values if any
        event_tweets_df.dropna(inplace=True)

        event_TFIDF_feat=ling_feat.tfIdf_feature(event_tweets_df,20) # save to 20 TF-IDF features

        event_TFIDF_feat.to_csv(events_path+event+'/features/'+event+'_tfidf.csv',index=False)
