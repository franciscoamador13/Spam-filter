from sklearn.base import BaseEstimator, TransformerMixin
import re
from collections import Counter
from scipy.sparse import csr_matrix
import nltk
import urlextract
from bs4 import BeautifulSoup
import re
import numpy as np

def html_to_plain_text(msg):
    def extract_email_body(msg):
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type in ["text/html", "text/plain"]:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset()
                    if not charset or charset.lower() == "default":
                        charset = "utf-8"
                    try:
                        return payload.decode(charset, errors='ignore')
                    except LookupError:
                        return payload.decode("utf-8", errors='ignore')
        else:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset()
            if not charset or charset.lower() == "default":
                charset = "utf-8"
            try:
                return payload.decode(charset, errors='ignore')
            except LookupError:
                return payload.decode("utf-8", errors='ignore')
        return ""

    raw_body = extract_email_body(msg)
    try:
        soup = BeautifulSoup(raw_body, "html.parser")
        text = soup.get_text(separator=' ', strip=True)
    except (ValueError, Exception):
        # If BeautifulSoup fails to parse (e.g., malformed HTML), strip tags with regex
        text = re.sub(r'<[^>]+>', ' ', raw_body)
        text = re.sub(r'\s+', ' ', text).strip()
    return text

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_case=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = html_to_plain_text(email) or ""
            if self.lower_case:
                text = text.lower()
            url_extractor = urlextract.URLExtract()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            stemmer = nltk.PorterStemmer()
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1
                            for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(X), self.vocabulary_size + 1))