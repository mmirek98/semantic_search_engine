from nltk import casual_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from DataUtils import prepare_stop_words
from SemanticSearchEngine import SemanticSearchEngine


def create_tfidf_search(thread_name, corpus, titles, n_components):
    stop_words = list(prepare_stop_words(thread_name))
    return SemanticSearchEngine(
        corpus,
        TfidfVectorizer(min_df=1, stop_words=stop_words, tokenizer=casual_tokenize),
        n_components,
        # 100,
        titles
    )

def create_bag_of_words_search(thread_name, corpus, titles, n_components):
    stop_words = list(prepare_stop_words(thread_name))
    return SemanticSearchEngine(
        corpus,
        CountVectorizer(min_df=1, stop_words=stop_words, tokenizer=casual_tokenize),
        n_components,
        titles
    )

def create_ngrams_search(thread_name, corpus, titles, n_components, ngram_range):
    stop_words = list(prepare_stop_words(thread_name))
    return SemanticSearchEngine(
        corpus,
        CountVectorizer(min_df=1, stop_words=stop_words, tokenizer=casual_tokenize, ngram_range=ngram_range),
        n_components,
        titles
    )
