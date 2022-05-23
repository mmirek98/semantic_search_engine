from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


class SemanticSearchEngine:
    def __init__(self, corpus, vectorizer, n_components, articles_label):
        self._corpus = corpus
        self._vectorizer = vectorizer
        self._lsa = TruncatedSVD(n_components=n_components)
        self._articles_label = articles_label

    def learn(self):
        vec = self._prepare_words_vector()
        self._fit_lsa_model(vec)
        self._topic_vectors = self._calculate_topic_vectors(vec)

    def query(self, expression, n_results=10):
        question = [expression]
        question_df = pd.DataFrame(self._vectorizer.transform(raw_documents=question).toarray())
        lsa_q = self._lsa.transform(question_df)


        if lsa_q[0].all() == 0:
            print("No results for ", expression, "! Please enter another query...")
            return
        results = self._calculate_best_results(lsa_q, n_results)
        self._print_results(results, expression)

    def _prepare_words_vector(self):
        vec = pd.DataFrame(self._vectorizer.fit_transform(raw_documents=self._corpus).toarray())
        id_words = [(i, w) for (w, i) in self._vectorizer.vocabulary_.items()]
        vec.columns = list(zip(*sorted(id_words)))[1]
        return vec

    def _fit_lsa_model(self, vec):
        self._lsa.fit_transform(vec.values)

    def _calculate_topic_vectors(self, vec):
        topic_vectors = self._transform_to_topics_vector(vec)
        topic_labels = self._prepare_topic_columns()
        return pd.DataFrame(topic_vectors, columns=topic_labels, index=self._articles_label)

    def _transform_to_topics_vector(self, vec):
        return self._lsa.transform(vec)

    def _prepare_topic_columns(self):
        return ["topic{}".format(i) for i in range(self._lsa.n_components)]

    def _find_best_topic_idx(self, results):
        c = 0
        minimal = 1000000
        for i, co in zip(results[0], range(len(results[0]))):
            x = abs(i)
            if minimal > x:
                minimal = x
            c = co
        return c

    def _calculate_best_results(self, lsa_q, n_results):
        final_vector = list(map(lambda x: abs(x[1]-lsa_q[0]), self._topic_vectors.items()))
        res = {}
        i = 0
        for f in final_vector:
            val = np.linalg.norm(f)
            res[i] = val
            i += 1

        s = sorted(res, key=res.get)[:n_results]
        best_results = final_vector[s[0]].to_dict()
        return sorted(best_results, key=best_results.get)[:n_results]

    def _print_results(self, results, query):
        print("Results for: ", query)
        for res in results:
            print(res)
