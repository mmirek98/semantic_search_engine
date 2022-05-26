from nltk import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from DataUtils import get_stopwords
from algorithms.BaseAlgorithm import BaseVectorizer


class NGram(BaseVectorizer):
    def __init__(self, corpus):
        BaseVectorizer.__init__(self, corpus)
        self.ngram_range = (1, 3)

    def _get_vectorizer(self):
        return CountVectorizer(stop_words=get_stopwords(), tokenizer=casual_tokenize, ngram_range=self.ngram_range)
