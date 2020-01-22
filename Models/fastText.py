import fasttext
import ast
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import pprint

from sklearn.model_selection import train_test_split
from FeatureExtraction.semantic_feature_extractor import FeatureExtraction
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Preprocessing.preprocess_tweets import TweetPrepocesser
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

class FastText(object):
    def __init__(self):
        self.tp = TweetPrepocesser()

    def prepare_dataset(self):
        # -- for CrisisLex dataset --
        # crisis_train = pd.read_csv('Data/crisisLex_train.csv', index_col='Tweet ID')
        # ## dropout Not Related Tweets.
        # crisis_train = crisis_train[crisis_train[
        #                                 ' Informativeness'] != 'Not related']  # drop Not related tweets 11% of total size (2284 out 22354)
        # crisis_train = crisis_train.sample(frac=1)  # to randomly shuffle the dataset
        #
        # crisis_train = crisis_train[[' Tweet Text', ' Information Type']]
        # crisis_train.dropna(inplace=True)
        #
        # train_data = crisis_train[' Tweet Text']
        #
        # # preprocess data labels:
        # train_labels = crisis_train[' Information Type'].tolist()  # +extra_tweets_labels
        # self.prepare_input_data(crisis_train, train_labels, 'Data/fast/paperCrisis_tweets_train.txt')
        #
        # crisis_test = pd.read_csv('Data/crisisLex_test.csv', index_col='Tweet ID')
        # ## dropout Not Related Tweets.
        # crisis_test = crisis_test[crisis_test[
        #                                 ' Informativeness'] != 'Not related']  # drop Not related tweets 11% of total size (2284 out 22354)
        # crisis_test = crisis_test.sample(frac=1)  # to randomly shuffle the dataset
        #
        # crisis_test = crisis_test[[' Tweet Text', ' Information Type']]
        # crisis_test.dropna(inplace=True)
        #
        # test_data = crisis_test[' Tweet Text']
        # test_labels = crisis_train[' Information Type'].tolist()
        # self.prepare_input_data(crisis_test, test_labels, 'Data/fast/paperCrisis_tweets_test.txt')

        # --- for TREC dataset ----
        train_data = pd.read_csv('Data/trec_train.csv',header=0, index_col='id', engine='python')
        train_data = train_data[train_data['categories'] != '[\'Unknown\']']  # delete Unknown label
        test_data = pd.read_csv('Data/trec_test.csv',header=0, index_col='postID', engine='python')
        # data.dropna(subset=['text'], inplace=True)
        print(len(train_data),len(test_data))
        train_data.dropna(subset=['categories', 'text'], inplace=True)  # drop missing values
        test_data.dropna(subset=['categories', 'text'], inplace=True)
        train_labels = train_data['categories'].tolist()
        train_labels = np.asarray([[x.strip() for x in item[1:-1].split(',')] for item in train_labels])
        test_labels = test_data['categories'].tolist()
        test_labels = np.asarray([[x.strip() for x in item[1:-1].split(',')] for item in test_labels])

        # remove_labels = ['CleanUp', 'InformationWanted', 'GoodsServices', 'KnownAlready', 'ServiceAvailable',
        #                 'MovePeople', 'Official', 'SearchAndRescue', 'Unknown', 'Volunteer', 'SignificantEventChange',
        #                  'EmergingThreats', 'Donations']

        #removed_cat_tweets = set()
        self.prepare_input_data(train_data, train_labels, 'Data/fast/papertrec_tweets_train.txt')
        self.prepare_input_data(test_data, test_labels, 'Data/fast/papertrec_tweets_test.txt')

    def prepare_testData(self):
        import os
        path = 'Data/TREC-is run test B/CSV Format'  # directory of prediction files
        all_paths = [os.path.join(path, file) for file in os.listdir(path)]
        for p in all_paths:
            event_name = p[p.find('.') + 1:p.find('_')]
            df = pd.read_csv(p,header=0, index_col='tweet_id')
            with open('Data/fast/paper/'+event_name+'.txt', 'w+') as file:
                for i, (ind, row) in enumerate(df.iterrows()):
                    sent = self.tp.preprocess_tweet(row['Text'])
                    file.write(sent.strip()+'\n')

    def prepare_input_data(self, data, labels, path):
        with open(path, 'w+') as file:
            for i, (ind, row) in enumerate(data.iterrows()):
                lab, tags = [], []
                for label in labels[i]:
                    lab.append(' __label__' + label.replace("'", ""))

                # for tag in ast.literal_eval(row['entities.hashtags']):
                #     print(tag)
                #     tags.append(tag['text'])

                sent = ' '.join(lab) + ' ' + self.tp.preprocess_tweet(row['text'])  # + ' '.join(tags) # Replace ' Tweet Text' with 'text' for TREC data
                print(sent)
                # sent = self.tp.preprocess_tweet(row['Text'], '')
                file.write(sent.strip() + '\n')

    def prepare_prediction_file(self):
        import os
        path = 'Data/fast/paper/prediction_results/'  # directory of prediction files
        all_paths = [os.path.join(path, file) for file in os.listdir(path)]
        # test_dfs = ['trecis2019-A-test.earthquakeBohol2013.csv', 'trecis2019-A-test.earthquakeCalifornia2014.json.csv',
        #     'trecis2019-A-test.fireAndover2019.json.csv', 'trecis2019-A-test.fireYMM2016.json.csv',
        #     'trecis2019-A-test.floodChoco2019.json.csv', 'trecis2019-A-test.hurricaneFlorence2018.csv',
        #     'trecis2019-A-test.shootingDallas2017.json.csv']
        for p in all_paths:
            event_name = p[p.rfind('/') + 1:p.find('.')]
            with open(p, 'r') as file:
                #test_event = ''
                # for test_file in test_dfs:
                #     if event_name.lower() in test_file.lower():
                #         print(test_file, event_name)
                #         test_event = test_file
                #         break

                df = pd.read_csv('Data/TREC-is run test B/CSV Format/trecis2019-B-test.' + event_name + '_remoDupli.csv', header=0, index_col='tweet_id', engine='python')
                data = file.readlines()
                with open('Evaluation/predictions/fasttext/paper_'+event_name+'.txt', 'w+') as wf:
                    self.prepare_dataframeFormat(data, df, wf)

    #def predictClasses(self):
    #  ./fastText-0.9.1/fasttext predict TRECmodel_autotune.ftz Data/fast/paper/southAfricaFloods2019.txt -1 0.09 > Data/fast/paper/prediction_results/southAfricaFloods2019.txt

    def prepare_dataframeFormat(self, fastText_outfile, df_with_tweet_ids, writeFile):
        for (ind, row), line in zip(df_with_tweet_ids.iterrows(), fastText_outfile):
            sent = str(ind)
            terms = line.split()
            labels = []
            #prob = []
            for i, term in enumerate(terms):
                print(term)
                term = term.replace('__label__', '')
                labels.append(term)

            sent += '\t' + str(labels)
            writeFile.write(sent + '\n')

    def prepare_train_test_val(self):
        with open('Data/fast/papertrec_tweets_train.txt', 'r') as file:
            data = file.readlines()
            split_ind = int(len(data)*0.8)
            shuffle(data)
            train = data[:split_ind]
            test = data[split_ind:]

            with open('Data/pTrec.train.txt', 'w+') as f:
                f.write(' '.join(train))
            with open('Data/pTrec.val.txt', 'w+') as f:
                f.write(' '.join(test))

            return train, test

    def generate_classification_report(self):
        col_names = ['id', 'labels', 'scores']
        test_df = pd.read_csv('Data/Trec-test_data.csv', header=0, index_col='id')
        df = pd.read_csv('Data/fast/preds/test_dataformat.txt', delimiter='\t', names=col_names)
        labels = df['labels'].tolist()
        mlb = MultiLabelBinarizer()
        labels = np.asarray([ [ x.strip().replace("'",'') for x in item[1:-1].split(',') if len(x)!=0] for item in labels])
        print(labels[:5])
        Y = mlb.fit_transform(labels)
        test_labels = test_df['categories'].tolist()
        test_labels = np.asarray([[x.strip().replace("'",'') for x in item[1:-1].split(',')] for item in test_labels])
        print(test_labels[:5])
        test_Y = mlb.transform(test_labels)
        print(test_Y)
        print(Y)
        print(len(mlb.classes_))
        report = classification_report(test_Y, Y, target_names=mlb.classes_, output_dict=True)
        print(type(report))
        report_df = pd.DataFrame.from_dict(report, orient='index')
        pprint.pprint(classification_report(test_Y, Y, target_names=mlb.classes_))
        report_df.to_csv('Data/fast/preds/classification_report.csv', index=True)

        pass




if __name__ == '__main__':
    ft = FastText()
    ft.prepare_dataset()
    ft.prepare_train_test_val()
    #ft.prepare_testData()
    #ft.prepare_prediction_file()
    #ft.generate_classification_report()
    model = fasttext.train_supervised(input='Data/pTrec.train.txt', autotuneValidationFile='Data/pTrec.val.txt', autotunePredictions=-1, autotuneDuration=600)
    model.save_model('TRECmodel_autotune.ftz')
    model.test('Data/fast/papertrec_tweets_test.txt')

