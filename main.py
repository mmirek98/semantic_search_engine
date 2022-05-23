import wikipedia
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




def get_articles(thread, results=100):
    return wikipedia.search(thread, results=results)


def fetch_wikipedia_articles(thread, n_articles, summary_only=False, debug_logs=False):
    pages_content = {}
    body = []
    titles = []
    for title in get_articles(thread, n_articles):
        if debug_logs:
            print("[Debug]: article title: ", title)
        try:
            page = wikipedia.page(title)
            if len(page.summary) < 250:
                continue

            if summary_only:
                pages_content[title] = page.summary
                body.append(page.summary)
            else:
                pages_content[title] = page.content
                body.append(page.content)
            titles.append(title)

        except Exception:
            if debug_logs:
                print("[Error]: cannot find page with title: ", title)
    if debug_logs:
        print("[DEBUG]: fetched ", len(titles), " articles")
    return body, pages_content


def save_to_file(corpus, titles, filename):
    f = open("{}.txt".format(filename), "w")
    for article, title in zip(corpus, titles):
        print(article)
        body = str.join(" ", article.splitlines())
        f.write("{}\n".format(title.encode()))
        f.write("{}\n".format(body.encode()))
    f.close()


def read_from_file(filename):
    titles = []
    body = []
    pages_content = {}
    f = open("{}.txt".format(filename), "r")
    i = 0
    # with open("{}.txt".format(filename), "r") as f:
    #     for line in f:
    for line in f.readlines():
        # line = str(line, 'utf-8')
        if i % 2 == 0:
            titles.append(line.rstrip()[2:-1])
        else:
            body.append(line.rstrip()[2:-1])
        i += 1

    for title, content in zip(titles, body):
        pages_content[title] = content

    return body, pages_content


class SemanticSearchEngine:
    def __init__(self, corpus, vectorizer, n_components, articles_label):
        self._corpus = corpus
        self._vectorizer = vectorizer
        self._lsa = TruncatedSVD(n_components=n_components)
        self._articles_label = articles_label

    def learn(self):
        vec = self.prepare_words_vector()
        self.fit_lsa_model(vec)
        self._topic_vectors = self.calculate_topic_vectors(vec)

    def query(self, expression, n_results=10):
        question = [expression]
        question_df = pd.DataFrame(self._vectorizer.transform(raw_documents=question).toarray())
        lsa_q = self._lsa.transform(question_df)
        lsa_q *= 10
        print("LSA_Q: ", lsa_q)


        if lsa_q[0].all() == 0:
            print(lsa_q)
            print("No results for ", expression, "! Please enter another query...")
            return
        results = self._calculate_best_results(lsa_q, n_results)
        self._print_results(results, expression)

    def prepare_words_vector(self):
        vec = pd.DataFrame(self._vectorizer.fit_transform(raw_documents=self._corpus).toarray())
        id_words = [(i, w) for (w, i) in self._vectorizer.vocabulary_.items()]
        vec.columns = list(zip(*sorted(id_words)))[1]
        return vec

    def fit_lsa_model(self, vec):
        self._lsa.fit_transform(vec)

    def calculate_topic_vectors(self, vec):
        topic_vectors = self.transform_to_topics_vector(vec)
        topic_labels = self.prepare_topic_columns()
        return pd.DataFrame(topic_vectors, columns=topic_labels, index=self._articles_label)

    def transform_to_topics_vector(self, vec):
        return self._lsa.transform(vec)

    def prepare_topic_columns(self):
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


def prepare_stop_words(thread_name):
    sw = stopwords.words('english')
    # TODO: change to thread name
    sw.append(thread_name)
    return sw

def create_tfidf_search(corpus, titles, n_components):
    stop_words = list(prepare_stop_words(thread_name))
    return SemanticSearchEngine(
        corpus,
        TfidfVectorizer(min_df=1, stop_words=stop_words, tokenizer=casual_tokenize),
        n_components,
        # 100,
        titles
    )

def create_bag_of_words_search(corpus, titles, n_components):
    stop_words = list(prepare_stop_words(thread_name))
    return SemanticSearchEngine(
        corpus,
        CountVectorizer(min_df=1, stop_words=stop_words, tokenizer=casual_tokenize),
        n_components,
        titles
    )

def create_ngrams_search(corpus, titles, n_components, ngram_range):
    stop_words = list(prepare_stop_words(thread_name))
    return SemanticSearchEngine(
        corpus,
        CountVectorizer(min_df=1, stop_words=stop_words, tokenizer=casual_tokenize, ngram_range=ngram_range),
        n_components,
        titles
    )


nltk.download('stopwords')

thread_name = "Internet"
# body, pages_content = fetch_wikipedia_articles(thread_name, 300, summary_only=True, debug_logs=True)
# titles = list(pages_content.keys())

# save_to_file(body, titles, thread_name)

body, pages_content = read_from_file(thread_name)
titles = list(pages_content.keys())


tfidf_search = create_tfidf_search(body, titles, len(titles))
ngram_search = create_ngrams_search(body, titles, len(titles), (2, 3))
bag_of_words_search = create_bag_of_words_search(body, titles, len(titles))

search_engines = {
    "BAG_OF_WORDS": bag_of_words_search,
    "NGRAM": ngram_search,
    "TFIDF": tfidf_search,
}

queries = [
    # "american microblogging and online social media and social networking service",
    "best communicators",
    "toxic person",
    "internet protocols",
    "music in india",
    "video streaming",
    "social media",
    "how to download movies",
    "best horror movies",
    "free software",
    "elon musk",
]

for type, se in search_engines.items():
    print("Using search engine with words vector type: ", type)
    se.learn()
    for q in queries:
        print("===== SEARCHING FOR ", q)
        se.query(q)
#
# tfidf_search.learn()
# tfidf_search.query("counter strike")
# tfidf_search.query("ipv4")
