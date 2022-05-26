import webbrowser

from wikipedia import wikipedia


class SemanticSearchEngine:
    def __init__(self, corpus, algorithm, articles_label):
        self._corpus = corpus
        self._algorithm = algorithm
        self._articles_label = articles_label

    def query(self, expression, n_results=10, open_browser=False):
        self._algorithm.perform_search(expression)
        results = self._algorithm.get_best_results(n_results)
        print("\n\n", self._algorithm.__class__.__name__)
        print("==== ", expression, " ====")
        for idx, no in zip(results.keys(), range(1, len(results.keys()) + 1)):
            print("{}. ".format(no), self._articles_label[idx])
        if open_browser:
            best_article_idx = list(results.keys())[0]
            self._open_result_for_best_score(self._articles_label[best_article_idx])

    def _print_results(self, results, query):
        print("Results for: ", query)
        for res in results:
            print(res)

    def _open_result_for_best_score(self, title):
        page = wikipedia.page(title)
        webbrowser.open(page.url, new=0, autoraise=True)