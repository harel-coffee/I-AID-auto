import emoji
import string, nltk, re
from spacy.lang.en import English

class TweetPrepocesser(object):
    def __init__(self):
        self.nlp = English()

    def remove_special_symbols(self, text):
        '''
        removes arabic, tamil, latin symbols and dingbats
        :param text:
        :return:
        '''
        special_symbols = re.compile(r"[\u0600-\u06FF\u0B80-\u0BFF\u25A0-\u25FF\u2700-\u27BF]+", re.UNICODE)
        text = special_symbols.sub('', text)
        return text

    def extract_emojis_from_text(self, text):
        emoji_list = ''.join(c for c in text if c in emoji.UNICODE_EMOJI)
        return list(set(emoji_list))

    def emoji_to_text(self, text):
        text = emoji.demojize(text)
        text = text.replace("::", " ") #for emojis that don't have space between them
        return text

    def remove_numbers(self, text):
        text = re.sub(r'\d+', '', text)
        return text

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
        '''
        1) find named-entities/phrases. consider them as single entities
        2) demojise tweets
        3) process links
        4) remove multiple lines to make it one line tweet
        5) if tweet truncated, use retweet!!
        :param text:
        :param retweeted_text:
        :return:
        '''
        text = text.strip()  # remove whitespaces
        print(f'before process: {text}')
        text = " ".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
        print(f'lines reduced: {text}')
        text = text.lower()
        #re.sub('rt', '', text)
        text = re.sub(' +', ' ', text)  # remove extra whitespace
        text = re.sub('@[^\s]+', '.', text) #remove username
        text = re.sub(r"http\S+", "", text) # remove url
        text = self.lemmatize_text(text)
        text = self.emoji_to_text(text)
        text = self.remove_special_symbols(text)
        text = self.remove_stopwords_and_punctuations(text)
        text = re.sub(' +', ' ', text)  # remove extra whitespace
        print(f'processed: {text}')
        return text

if __name__ == '__main__':
    tp = TweetPrepocesser()