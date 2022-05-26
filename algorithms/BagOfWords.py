from nltk import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from DataUtils import get_stopwords
from algorithms.BaseAlgorithm import BaseVectorizer


class BagOfWords(BaseVectorizer):
    def __init__(self, corpus):
        BaseVectorizer.__init__(self, corpus)

    def _get_vectorizer(self):
        return CountVectorizer(stop_words=get_stopwords(), tokenizer=casual_tokenize)
