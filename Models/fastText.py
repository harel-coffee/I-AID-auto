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
        trec_train = pd.read_csv('Data/trec_data.csv', index_col='Tweet_ID')

        trec_train = trec_train[['Text', 'Categories']]
        trec_train=trec_train[trec_train['Categories']!='[\'Unknown\']'] # delete Unknown label
        trec_train.dropna(inplace=True)
        trec_train2 = trec_train.groupby(trec_train.index).first()

        trec_copy = trec_train2.copy()
        trec_train = trec_copy.sample(frac=0.80, random_state=42)
        trec_test = trec_copy.drop(trec_train.index)
        # combine both files - ucd and trec_train
        # mdf = pd.read_csv('Data/generated_samples/ucd_combined.csv', sep='\t', header=0, index_col=0, engine='python')
        # mdf.index.name = 'id'
        # balanced_train = pd.concat([trec_train, mdf])  # merged data frame
        # #print(result)
        # balanced_train.to_csv('Data/ucd_balanced_train.csv')
        trec_test.to_csv('Data/trec_test.csv')
        trec_train.to_csv('Data/trec_train.csv')


        print(len(trec_train),len(trec_test)) # len(balanced_train))

        train_labels = trec_train['Categories'].tolist()
        train_labels = np.asarray([[x.strip() for x in item[1:-1].split(',')] for item in train_labels])
        test_labels = trec_test['Categories'].tolist()
        test_labels = np.asarray([[x.strip() for x in item[1:-1].split(',')] for item in test_labels])

        self.prepare_input_data(trec_train, train_labels, 'Data/fast/papertrec_tweets_train.txt')
        self.prepare_input_data(trec_test, test_labels, 'Data/fast/papertrec_tweets_test.txt') # generate classification report for this dataset

    def prepare_testData(self):
        '''
        generates prediction file for evaluating against TREC-IS notebook
        :return:
        '''
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

                sent = ' '.join(lab) + ' ' + self.tp.preprocess_tweet(row['Text'])  # + ' '.join(tags) # Replace ' Tweet Text' with 'text' for TREC data
                print(sent)
                file.write(sent.strip() + '\n')

    def prepare_prediction_file(self):
        import os
        path = 'Data/fast/paper/prediction_results/on_trec_test.txt'  # directory of prediction files
        with open(path, 'r') as file:
            df = pd.read_csv('Data/trec_test.csv', header=0, index_col='Tweet_ID', engine='python')
            data = file.readlines()
            with open('Evaluation/predictions/fasttext/paper_trec_test.txt', 'w+') as wf:
                self.prepare_dataframeFormat(data, df, wf)


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
        '''using scikit documentation'''
        col_names = ['Tweet_ID', 'labels', 'scores']
        test_df = pd.read_csv('Data/trec_test.csv', header=0, index_col='Tweet_ID')
        pred_df = pd.read_csv('Evaluation/predictions/fasttext/paper_trec_test.txt', delimiter='\t', names=col_names) # Data/fast/preds/test_dataformat.txt
        pred_labels = pred_df['labels'].tolist()
        mlb = MultiLabelBinarizer()
        pred_labels = np.asarray([ [ x.strip().replace("'",'') for x in item[1:-1].split(',') if len(x)!=0] for item in pred_labels])
        test_labels = test_df['Categories'].tolist()
        test_labels = np.asarray([[x.strip().replace("'",'') for x in item[1:-1].split(',')] for item in test_labels])
        test_Y = mlb.fit_transform(test_labels)
        pred_Y = mlb.transform(pred_labels)
        print(pred_Y.shape, test_Y.shape)
        print(len(mlb.classes_))
        report = classification_report(test_Y, pred_Y, target_names=mlb.classes_, output_dict=True)
        #print(type(report))
        report_df = pd.DataFrame.from_dict(report, orient='index')
        pprint.pprint(classification_report(test_Y, pred_Y, target_names=mlb.classes_))
        print("accuracy score: "  ,str(accuracy_score(test_Y, pred_Y)))
        #report_df.to_csv('Data/fast/preds/trec_train12_classification_report.csv', index=True) # uncomment to generate the report
        from sklearn.metrics import jaccard_similarity_score
        from sklearn.metrics import hamming_loss
        jac_score = jaccard_similarity_score(test_Y, pred_Y)
        loss = hamming_loss(test_Y, pred_Y)
        print(jac_score, loss)



if __name__ == '__main__':
    ft = FastText()
    ft.prepare_dataset()
    ft.prepare_train_test_val()
    ft.prepare_testData()
    model = fasttext.train_supervised(input='Data/pTrec.train.txt', autotuneValidationFile='Data/pTrec.val.txt', autotunePredictions=-1, autotuneDuration=1200)
    model.save_model('TRECmodel_autotune.ftz')
    ## -- optional ---
    #model = fasttext.load_model('TRECmodel_autotune.ftz')
    #print(model.test('Data/fast/papertrec_tweets_test.txt', k= -1))
    ## ---------------
    ## run the command from the terminal to generate prediction files using the generated model - change the file names according to the dataset
    # ./fastText-0.9.1/fasttext predict TRECmodel_autotune.ftz Data/fast/papertrec_tweets_test.txt -1 0.2 > Data/fast/paper/prediction_results/on_trec_test.txt
    ## -- after running the command on the terminal, run the following two lines of code
    #ft.prepare_prediction_file()
    #ft.generate_classification_report()

