import pandas as pd
import itertools
import numpy as np
import ast
from matplotlib import pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from sklearn.preprocessing import MultiLabelBinarizer

class DataAnalysis:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    #TODO: find the most used words in each category for every event separately
    def perform_data_analysis(self, event_tweets_df, event, visualize=True):
        data = event_tweets_df
        analysis = dict()
        analysis['null_values'] = data.isnull().sum()
        analysis['before dropping values'] = len(data)

        all_categories = ['Advice', 'CleanUp', 'ContinuingNews', 'Discussion', 'Donations', 'EmergingThreats',
                              'Factoid', 'FirstPartyObservation', 'Hashtags', 'InformationWanted', 'Irrelevant',
                              'KnownAlready', 'MovePeople', 'MultimediaShare', 'Official', 'PastNews', 'SearchAndRescue',
                              'Sentiment', 'ServiceAvailable', 'SignificantEventChange', 'ThirdPartyObservation',
                              'Unknown', 'Volunteer', 'Weather', 'GoodsServices']

        data.dropna(inplace=True, subset=['categories', 'text'])  # drop missing values
        analysis['after dropping values'] = len(data)

        if visualize:
            self.viz_TweetsPerClass(all_categories, event, data)

        data['categories'] = self.mlb.fit_transform(data['categories'].str.strip('[]').str.split(', '))
        analysis['num_subclasses'] = len(list(self.mlb.classes_))
        analysis['subclasses'] = list(self.mlb.classes_)

        print(analysis)

    def visualize_word_cloud(self, tweets_in_category):
        plt.figure(figsize=(40, 25))
        # clean
        for cat in tweets_in_category:
            cloud_toxic = WordCloud(
                stopwords=STOPWORDS,
                background_color='black',
                collocations=False,
                max_words=200,
                width=2500,
                height=1800
            ).generate(" ".join(tweets_in_category[cat]))
            plt.axis('off')
            plt.title("Clean", fontsize=40)
            plt.show(cloud_toxic)


    def viz_TweetsPerClass(self, all_categories, event, data):
        tweets_in_category = dict()

        for ind, row in data.iterrows():
            row['categories'] = ast.literal_eval(row['categories'])
            for category in row['categories']:
                if category in tweets_in_category:
                    tweets_in_category[category].append(row['text'])
                else:
                    tweets_in_category[category] = [row['text']]

        self.visualize_word_cloud(tweets_in_category)

        for cat in all_categories:
            if cat not in tweets_in_category:
                tweets_in_category[cat] = []

        indexes = np.arange(len(all_categories))
        plt.figure(figsize=(15, 5))
        plt.bar(indexes, len(tweets_in_category.values()), alpha=0.5, align='center')
        plt.xticks(indexes, all_categories, rotation='vertical')
        plt.tight_layout()
        plt.savefig('Data/TREC_Data/' + event + '/' + event + '_analysis.pdf')

if __name__ == '__main__':
    events_path = 'Data/TREC_Data/'
    events_names = ['albertaFloods2013', 'australiaBushfire2013', 'bostonBombings2013', 'chileEarthquake2014',
                    'flSchoolShooting2018',
                    'guatemalaEarthquake2012', 'italyEarthquakes2012', 'joplinTornado2011', 'manilaFloods2013',
                    'nepalEarthquake2015', 'parisAttacks2015', 'philipinnesFloods2012', 'queenslandFloods2013',
                    'typhoonHagupit2014', 'typhoonYolanda2013']

    da = DataAnalysis()
    path = 'Data/trec_data.csv'
    all_tweets = pd.read_csv(path, header=0,  engine='python')
    da.perform_data_analysis(all_tweets, event='all_events', visualize=False)

    # extract Linguistic features

    # for event in events_names:
    #     print(event)
    #     event_tweets_path = events_path + event + '/' + event + '_all.csv'
    #
    #     try:
    #         event_tweets_df = pd.read_csv(event_tweets_path)
    #     except:
    #         event_tweets_df = pd.read_csv(
    #             open(event_tweets_path, 'rU'), encoding='utf-8', engine='c')
    #     da = DataAnalysis()
    #     da.perform_data_analysis(event_tweets_df, event)