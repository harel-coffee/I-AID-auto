import re
import operator
import requests
from Preprocessing.preprocess_tweets import TweetPrepocesser

from Preprocessing.Data_Preprocesser import Data_Preprocessor
from Preprocessing.secrets import babelnet_key


class Helper_FeatureExtraction:
    def __init__(self):
        self.pt = TweetPrepocesser()

    def get_microsoft_concept(self, text):
        concepts = []
        text = self.pt.preprocess_tweet(text)
        text = re.sub('RT', '', text)

        for token in text.strip().split():
            payload = {'instance': token, 'topK': 10}
            r = requests.get('https://concept.research.microsoft.com/api/Concept/ScoreByProb', params=payload)
            res = r.json()
            if len(res) != 0:
                concepts.append(max(res.items(), key = operator.itemgetter(1))[0])
        return(concepts)


if __name__ == '__main__':
    hfe = Helper_FeatureExtraction()
    hfe.get_microsoft_concept('RT @7NewsBrisbane: PHOTO: Dozens of cars are on fire at Sydney Olympic Park Aquatic Centre http://t.co/Lx2ZkZgzO1')