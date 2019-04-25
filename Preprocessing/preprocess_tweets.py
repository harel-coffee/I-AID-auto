import string, nltk, re
from spacy.lang.en import English

class TweetPrepocesser(object):
    def __init__(self):
        self.nlp = English()

    def remove_stopwords_and_punctuations(self, text):
        '''
        text = "It's going be a rainy week for Davao. #PabloPH http://t.co/XnObb62J"
        output = "It going rainy week Davao PabloPH http://t.co/XnObb62J"
        :param text:
        :return:
        '''

        customize_spacy_stop_words = ["'ve", "n't", "\n", "'s"] #removed "rt" from the list

        for w in customize_spacy_stop_words:
            self.nlp.vocab[w].is_stop = True
        parsed_text = self.nlp(text)
        tokens = [(token.text) for token in parsed_text if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def lemmatize_text(self, text):
        '''
        text = "It's going be a rainy week for Davao. #PabloPH http://t.co/XnObb62J"
        lem_text = "It be go be a rainy week for davao . # pabloph http://t.co/xnobb62j"
        :return:
        '''
        text = self.nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    def preprocess_tweet(self, text):
        text = text.strip()  # remove whitespaces
        re.sub('RT', '', text)
        text = re.sub(' +', ' ', text)  # remove extra whitespace
        text = re.sub('@[^\s]+', '.', text) #remove username
        text = re.sub(r"http\S+", "", text) # remove url
        text = self.lemmatize_text(text)
        text = self.remove_stopwords_and_punctuations(text)
        return text
