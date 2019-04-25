from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
import pandas as pd

data = pd.read_csv('Data/TREC_Data/all_events_microsoft_concepts.csv',header=0)

train, test = train_test_split(data, random_state=42, test_size=0.15, shuffle=True)
print(len(train), len(test))

x_train = train.drop(labels=['categories', 'tweet_id'], axis=1)
print(x_train[:1])
y_train = train.drop(labels = ['tweet_id','concept_vec'], axis=1)
print(y_train[:1])
x_test = test.drop(labels=['categories', 'tweet_id'], axis=1)
y_test = test.drop(labels = ['tweet_id','concept_vec'], axis=1)

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

# train
classifier.fit(x_train, y_train)
# predict
predictions = classifier.predict(x_test)
# accuracy
print("F1-score = ",f1_score(y_test,predictions, average='macro'))