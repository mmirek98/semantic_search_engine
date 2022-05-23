import DataUtils as utils
import SemanticSearchEngineFactory as factory

# ==== Config ====
thread_name = "Internet"
queries = [
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

# ==== Fetch from WIKIPEDIA_API ====
# body, pages_content = fetch_wikipedia_articles(thread_name, 300, summary_only=True, debug_logs=True)
# titles = list(pages_content.keys())

# ==== Save to file fetched data ====
# save_to_file(body, titles, thread_name)

# ==== Read from file if exists ====
body, pages_content = utils.read_from_file(thread_name)
titles = list(pages_content.keys())


# ==== Create search engines ====
tfidf_search = factory.create_tfidf_search(thread_name, body, titles, len(titles))
ngram_search = factory.create_ngrams_search(thread_name, body, titles, len(titles), (2, 3))
bag_of_words_search = factory.create_bag_of_words_search(thread_name, body, titles, len(titles))
search_engines = {
    "BAG_OF_WORDS": bag_of_words_search,
    "NGRAM": ngram_search,
    "TFIDF": tfidf_search,
}

# ==== Main loop ====
for type, se in search_engines.items():
    print("Using search engine with words vector type: ", type)
    se.learn()
    for q in queries:
        print("===== SEARCHING FOR =====", q)
        se.query(q)

