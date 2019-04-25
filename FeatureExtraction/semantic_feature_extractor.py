import pandas as pd
import numpy as np
import spacy

from sklearn.preprocessing import MultiLabelBinarizer
from FeatureExtraction.helper_featureExtractor import Helper_FeatureExtraction


class FeatureExtraction:
    def __init__(self, df=None, event=None):
        if df is None:
            self.df = pd.read_csv('Data/TREC_Data/all_events.csv',header=0, index_col='postID')
        else:
            self.df = df
        self.event_name = event
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        self.hfe = Helper_FeatureExtraction()
        self.nlp = spacy.load('en_vectors_web_lg', disable=['tagger', 'parser', 'ner', 'textcat'])
        # n_vectors = 105000  # number of vectors to keep
        # self.nlp.vocab.prune_vectors(n_vectors)
        self.norm_df = self.create_dataframe_for_normalized_tweets()

    def create_dataframe_for_normalized_tweets(self):
        data = self.df
        print(f'before dropping values {len(data)}')
        data.dropna(subset=['categories', 'full_text'], inplace=True)  # drop missing values
        print(f'after dropping values {len(data)}')
        data['categories'] = data['categories'].str.strip('[]').str.split(', ')
        for ind, row in data.iterrows():
            row['categories'] = set([label.replace("'", '') for label in row['categories']])

        data['categories'] = self.mlb.fit_transform(data['categories'])

        return data

    def concept2vec(self):
        concept_df = pd.DataFrame(index=self.norm_df.index)
        concept_df = concept_df[:50]
        print(len(concept_df))
        concept_df['categories'] = self.norm_df[:50]['categories']
        concept_matrix = []

        i = 0
        for ind, row in self.norm_df[:50].iterrows():
            concepts = self.hfe.get_microsoft_concept(row['full_text'])
            mag_concept = np.asarray([self.nlp(word).vector for word in concepts], dtype=object)
            avg_mag_vector = np.mean(mag_concept, axis=0)
            concept_matrix.append(avg_mag_vector)
            concept_df.at[ind, 'tweet_id'] = ind
            # concept_df.at[ind, 'categories'] = row['categories']
            # print(row['categories'])
            i += 1
            print(i)

        print(len(concept_matrix))
        print(len(concept_df))
        concept_df['concept_vec'] = concept_matrix
        concept_df.to_csv('Data/TREC_Data/all_events_microsoft_concepts.csv', index=0)


if __name__ == '__main__':
    fe = FeatureExtraction()
    fe.concept2vec()
