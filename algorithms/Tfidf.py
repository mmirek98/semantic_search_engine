from nltk import casual_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from DataUtils import get_stopwords
from algorithms.BaseAlgorithm import BaseVectorizer


class TfidfAlgorithm(BaseVectorizer):
    def __init__(self, corpus):
        BaseVectorizer.__init__(self, corpus)

    def _get_vectorizer(self):
        return TfidfVectorizer(stop_words=get_stopwords(), tokenizer=casual_tokenize)
