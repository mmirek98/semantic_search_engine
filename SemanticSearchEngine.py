from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np


class SemanticSearchEngine:
    def __init__(self, corpus, algorithm, articles_label):
        self._corpus = corpus
        self._algorithm = algorithm
        self._articles_label = articles_label

    def query(self, expression, n_results=10):
        self._algorithm.perform_search(expression)
        results = self._algorithm.get_best_results(n_results)
        print("\n\n", self._algorithm.__class__.__name__)
        print("==== ", expression, " ====")
        for i in results.keys():
            print("{}. ".format(i+1), self._articles_label[i])

    def _print_results(self, results, query):
        print("Results for: ", query)
        for res in results:
            print(res)
