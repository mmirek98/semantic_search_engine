import nltk
import wikipedia
from nltk.corpus import stopwords

nltk.download('stopwords')


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
    for line in f.readlines():
        if i % 2 == 0:
            titles.append(line.rstrip()[2:-1])
        else:
            body.append(line.rstrip()[2:-1])
        i += 1

    for title, content in zip(titles, body):
        pages_content[title] = content

    return body, pages_content

def prepare_stop_words(thread_name):
    sw = stopwords.words('english')
    # TODO: change to thread name
    sw.append(thread_name)
    return sw
