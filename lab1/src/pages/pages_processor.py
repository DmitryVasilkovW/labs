from collections import Counter

import pandas as pd
import yake
from matplotlib import pyplot as plt
import seaborn as sns


class PagesProcessor:
    def __init__(self, pages):
        self.pages = pages
        self.df = pd.DataFrame(self.pages)
        self.yake_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.85, top=20)

    def show_pages_with_keywords_count(self):
        count = 0
        for page in self.pages:
            if page.get('keywords'):
                count += 1

        print(f"Найдено {count} статьи/ей с ключевыми словами")

    def get_samples(self, count: int = 10):
        return self.df.sample(count)

    def update_keywords(self):
        self.df["keywords"] =\
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
        return self.df["title"]

    def get_title_keyword_df(self):
        return self.df[["title", "keywords"]]

    def get_keywords(self):
        return self.df["keywords"]

    def get_keywords_as_list(self):
        all_keywords = []
        for keyword_list in self.get_keywords():
            for keyword in keyword_list:
                all_keywords.append(keyword)

        return all_keywords

    def get_unique_keywords(self):
        return set(self.get_keywords_as_list())

    def iterate(self):
        return self.df.iterrows()

    def __get_updated_keywords(self, row) -> list:
        existing_keywords = set(row.get("keywords") or [])
        new_keywords = set(self.__extract_keywords(row["title"] + " " + row["summary"]))

        updated_keywords = list(existing_keywords | new_keywords)
        return updated_keywords

    def __extract_keywords(self, text) -> list:
        text = text.lower()
        keywords = self.yake_extractor.extract_keywords(text)

        keywords_without_weights = []
        for keyword in keywords:
            keywords_without_weights.append(keyword[0])

        return keywords_without_weights

