from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BaseAlgorithm(ABC):
    def __init__(self, corpus):
        self._results = None
        self._corpus = corpus

    def get_best_results(self, n_results=1):
        results = sorted(self._results, reverse=True)[:n_results]
        keys = np.argsort(-self._results)[:n_results]

        summary = {}
        for r, k in zip(results, keys):
            summary[k] = r

        return summary

    @abstractmethod
    def perform_search(self, querry):
        pass


class BaseVectorizer(BaseAlgorithm):
    def __init__(self, corpus):
        BaseAlgorithm.__init__(self, corpus)

    def perform_search(self, querry):
        vectorizer = self._get_vectorizer()
        fitted = self._fit_vectorizer(vectorizer, querry)
        self._results = self._calculate_cosine_similarities(fitted[0:1], fitted[1:])

    @abstractmethod
    def _get_vectorizer(self):
        pass

    def _fit_vectorizer(self, vectorizer, querry):
        return vectorizer.fit_transform(np.concatenate(([querry], self._corpus)))

    def _calculate_cosine_similarities(self, querry, corpus):
        return cosine_similarity(querry, corpus).flatten()

    def _calculate_highest_score(self, cos_sim):
        highest_score = 0
        highest_score_index = 0
        for i, score in enumerate(cos_sim):
            if highest_score < score:
                highest_score = score
                highest_score_index = i
        return highest_score, highest_score_index
