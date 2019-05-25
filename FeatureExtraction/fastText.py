import fasttext
import ast
import pandas as pd
import numpy as np
from random import shuffle

from sklearn.model_selection import train_test_split
from FeatureExtraction.semantic_feature_extractor import FeatureExtraction
from Preprocessing.preprocess_tweets import TweetPrepocesser
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

class FastText(object):
    def __init__(self):
        self.tp = TweetPrepocesser()

    def prepare_dataset(self):
        data = pd.read_csv('Data/trec_data_first_labels.csv',header=0, index_col='id')
        print(data.columns.values)
        # data.dropna(subset=['text'], inplace=True)
        print(len(data))
        data.dropna(subset=['categories', 'text'], inplace=True)  # drop missing values
        labels = data['categories'].tolist()
        labels = np.asarray([[x.strip() for x in item[1:-1].split(',')] for item in labels])
        print(labels[0])

        removed_cat_tweets = set()
        with open('Data/crisis_tweets.txt', 'w+') as file:
            for i, (ind, row) in enumerate(data.iterrows()):
                lab, tags = [], []
                for label in labels[i]:
                    if 'Unknown' in label and len(labels) == 1: # remove tweets which have been categorized as labels only
                        removed_cat_tweets.add(ind)
                        print(row['text'])
                        break
                    elif 'Unknown' in label: # remove if present as one of the labels
                        print(f'labels: {labels[i]}')
                        continue
                    lab.append(' __label__' + label.replace("'", ""))

                if ind in removed_cat_tweets:
                    continue

                # for tag in ast.literal_eval(row['entities.hashtags']):
                #     print(tag)
                #     tags.append(tag['text'])

                sent = ' '.join(lab) + ' ' + self.tp.preprocess_tweet(row['text']) #+ ' '.join(tags)
                print(sent)
                #sent = self.tp.preprocess_tweet(row['Text'], '')
                file.write(sent.strip()+'\n')

    def prepare_train_test_val(self):
        with open('Data/crisis_tweets.txt', 'r') as file:
            data = file.readlines()
            split_ind = int(len(data)*0.8)
            shuffle(data)
            train = data[:split_ind]
            test = data[split_ind:]

            with open('Data/crisis.train.txt', 'w+') as f:
                f.write(' '.join(train))
            with open('Data/crisis.test.txt', 'w+') as f:
                f.write(' '.join(test))

            return train, test

    def train_classifier(self):
        pass


if __name__ == '__main__':
    ft = FastText()
    ft.prepare_dataset()
    ft.prepare_train_test_val()
