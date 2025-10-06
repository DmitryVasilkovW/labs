import contextlib
import io
import logging
from collections import Counter

import nltk
import pandas as pd
import stanza
import yake
from matplotlib import pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

from lab1.src.constant.fetching import KEYWORDS, TITLE, SUMMARY


class PagesProcessor:
    _nlp = None
    _stop_words = None

    def __init__(self, pages):
        self.pages = pages
        self.df = pd.DataFrame(self.pages)
        self.yake_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.85, top=20)

        logging.getLogger("stanza").setLevel(logging.ERROR)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if PagesProcessor._nlp is None:
                stanza.download('en', verbose=False)
                PagesProcessor._nlp = stanza.Pipeline(
                    'en',
                    processors='tokenize,pos,lemma',
                    use_gpu=False,
                    verbose=False,
                    download_method=None
                )
            if PagesProcessor._stop_words is None:
                nltk.download('stopwords', quiet=True)
                PagesProcessor._stop_words = set(stopwords.words('english'))

        self.nlp = PagesProcessor._nlp
        self.stop_words = PagesProcessor._stop_words

    def show_pages_with_keywords_count(self):
        count = 0
        for page in self.pages:
            if page.get(KEYWORDS):
                count += 1

        print(f"Найдено {count} статьи/ей с ключевыми словами")

    def get_samples(self, count: int = 10):
        return self.df.sample(count)

    def update_keywords(self):
        self.df[KEYWORDS] =\
            self.df.apply(self.__get_updated_keywords, axis=1)

    def show_most_common_keywords(self):
        keyword_freq = Counter(self.get_keywords_as_list())
        most_common_keywords = keyword_freq.most_common(30)
        df_plot = pd.DataFrame(most_common_keywords, columns=["keyword", "count"])

        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=df_plot,
            x="count",
            y="keyword",
            color="skyblue"
        )
        plt.xlabel('Частота')
        plt.ylabel('Ключевые слова')
        plt.title('Топ-30 самых частых ключевых слов')
        plt.show()

    def get_title(self):
        return self.df[TITLE]

    def get_title_keyword_df(self):
        return self.df[[TITLE, KEYWORDS]]

    def get_keywords(self):
        return self.df[KEYWORDS]

    def get_keywords_as_list(self):
        all_keywords = []
        for keyword_list in self.get_keywords():
            for keyword in keyword_list:
                all_keywords.append(keyword)

        return all_keywords

    def get_unique_keywords(self):
        return set(self.get_keywords_as_list())

    def lemmatize_keywords(self):
        self.df[KEYWORDS] = self.df[KEYWORDS].apply(self.__lemmatize_keywords_stanza)

    def __lemmatize_keywords_stanza(self, keywords):
        if not isinstance(keywords, list):
            return []

        lemmatized_phrases = []
        for phrase in keywords:
            doc = self.nlp(phrase)
            lemmatized_words = [word.lemma for sent in doc.sentences for word in sent.words]
            lemmatized_phrases.append(" ".join(lemmatized_words))
        return lemmatized_phrases

    def clean_keywords_from_stopwords(self):
        self.df[KEYWORDS] = self.df[KEYWORDS].apply(self.__clean_keywords)

    def __clean_keywords(self, keywords):
        if not isinstance(keywords, list):
            return []

        cleaned_keywords = []
        for phrase in keywords:
            words = phrase.split()
            filtered = [word for word in words if word.lower() not in self.stop_words]
            if filtered:
                cleaned_keywords.append(" ".join(filtered))
        return cleaned_keywords

    def iterate(self):
        return self.df.iterrows()

    def __get_updated_keywords(self, row) -> list:
        existing_keywords = set(row.get(KEYWORDS) or [])
        new_keywords = set(self.__extract_keywords(row[TITLE] + " " + row[SUMMARY]))

        updated_keywords = list(existing_keywords | new_keywords)
        return updated_keywords

    def __extract_keywords(self, text) -> list:
        text = text.lower()
        keywords = self.yake_extractor.extract_keywords(text)

        keywords_without_weights = []
        for keyword in keywords:
            keywords_without_weights.append(keyword[0])

        return keywords_without_weights

