from SemanticSearchEngine import SemanticSearchEngine
from algorithms.BagOfWords import BagOfWords
from algorithms.NGram import NGram
from algorithms.Tfidf import TfidfAlgorithm


def create_tfidf_search(corpus, titles):
    return SemanticSearchEngine(
        corpus,
        TfidfAlgorithm(corpus),
        titles
    )
def create_bag_of_words_search(corpus, titles):
    return SemanticSearchEngine(
        corpus,
        BagOfWords(corpus),
        titles
    )

def create_ngrams_search(corpus, titles):
    return SemanticSearchEngine(
        corpus,
        NGram(corpus),
        titles
    )
