import nltk

import DataUtils as utils
import SemanticSearchEngineFactory as factory

nltk.download("punkt")
nltk.download('stopwords')

# ==== Config ====
thread_name = "Internet"
queries = [
    "Animals",
    "troll",
    "Telecommunications in African countries",
    "Telecommunications Monaco",
    "History of internet",
    # "Displaying HTML element in website"
    "internet protocols",
    "music in india",
    "video streaming",
    "social media",
    "how to download movies",
    "best horror movies",
    "free software",
    "elon musk",
]

# ==== Fetch from WIKIPEDIA_API ====
# body, pages_content = utils.fetch_wikipedia_articles(thread_name, 100000, summary_only=True, debug_logs=True)
# titles = list(pages_content.keys())

# ==== Save to file fetched data ====
# utils.save_to_file(body, titles, thread_name)

# ==== Read from file if exists ====
body, pages_content = utils.read_from_file(thread_name)
titles = list(pages_content.keys())


# ==== Create search engines ====
tfidf_search = factory.create_tfidf_search(body, titles)
ngram_search = factory.create_ngrams_search(body, titles)
bag_of_words_search = factory.create_bag_of_words_search(body, titles)
jaccard_search = factory.create_jaccard_search(body, titles)

search_engines = {
    # "BAG_OF_WORDS": bag_of_words_search,
    # "NGRAM": ngram_search,
    # "TFIDF": tfidf_search,
    "JACCARD": jaccard_search
}

# print("type anython")
# var = input()
# print("you typed: ", var)
#
# print("next ...")



# ==== Main loop ====
for type, se in search_engines.items():
    for q in queries:
        se.query(q, 3)
