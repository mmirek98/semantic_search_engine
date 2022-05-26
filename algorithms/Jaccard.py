import nltk
from algorithms.BaseAlgorithm import BaseAlgorithm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")


class Jaccard(BaseAlgorithm):
    def __init__(self, corpus):
        BaseAlgorithm.__init__(self, corpus)

    def perform_search(self, querry):
        self._corpus = self._pipe(self._corpus)
        querry = self._pipe([querry])
        self._results = self._find_best_result(querry)

    def _pipe(self, corpus):
        _corpus = self._lower_case_corpus(corpus)
        _corpus = self._remove_punctuation_in_corpus(_corpus)
        # _corpus = self._tokenize_corpus(_corpus)
        _corpus = self._remove_stop_words_in_corpus(_corpus)
        _corpus = self._lemmatize_corpus(_corpus)
        return _corpus

    def _find_best_result(self, querry):
        scores = []
        for doc in self._corpus:
            scores.append(self._calculate_jaccard([doc], querry))
        # max_value = max(scores)
        # max_idx = scores.index(max_value)
        return scores

    def _lower_case_corpus(self, corpus):
        lower_cased_corpus = []
        for doc in corpus:
            lower_cased_corpus.append(doc.lower())
        return lower_cased_corpus

    def _tokenize_corpus(self, corpus):
        tokenized_corpus = []
        for doc in corpus:
            tokenized_corpus.append(nltk.word_tokenize(doc))
        return tokenized_corpus

    def _remove_stop_words_in_corpus(self, tokenized_corpus):
        stop_words = set(stopwords.words('english'))
        corpus_without_stop_words = [[]] * len(tokenized_corpus)
        print(tokenized_corpus)

        for tokens, i in zip(tokenized_corpus, range(len(tokenized_corpus))):
            for token in tokens:
                corpus_without_stop_words[i].append([w for w in token if not w.lower() in stop_words])
        return corpus_without_stop_words

    def _remove_punctuation_in_corpus(self, corpus):
        tokenizer = RegexpTokenizer(r'\w+')
        corpus_without_punctuation = [[]] * len(corpus)
        for doc, i in zip(corpus, range(len(corpus))):
            corpus_without_punctuation[i].append(tokenizer.tokenize(doc))
        return corpus_without_punctuation

    def _lemmatize_corpus(self, corpus):
        lemmatizer = WordNetLemmatizer()
        lemmatized_corpus = [[]] * len(corpus)
        for doc, i in zip(corpus, range(len(corpus))):
            for token in doc:
                lemmatized_corpus[i].append(lemmatizer.lemmatize(token))
        return lemmatized_corpus

    def _perform_intersection(self, corpus, querry):
        intersection = set()
        for w in corpus:
            if w in querry:
                intersection.add(w)
        return intersection

    def _calculate_jaccard(self, document, querry):
        merged = document[0] + querry[0]
        union = set(merged)
        intersection = self._perform_intersection(document, querry)
        if any(intersection):
            print('Hooray!')
        jaccard_score = len(intersection) / len(union)
        return jaccard_score
