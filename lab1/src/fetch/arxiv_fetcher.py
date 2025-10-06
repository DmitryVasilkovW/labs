import urllib.request as url_request
import requests
import feedparser
import re
import time
import io
import pdfplumber

from lab1.src.constant.arxiv import BASE_URL, CATEGORY
from lab1.src.constant.fetching import BATCH_SIZE, RESULTS, AUTHORS, SUMMARY, TITLE, KEYWORDS


class ArxivFetcher:
    __keyword_regex: str = r"(?i)(?:keywords?|KEYWORDS?):\s*([^\n]+)"
    __research_flag: re.RegexFlag = re.IGNORECASE | re.DOTALL
    __pages = None

    @classmethod
    def fetch_arxiv_pages_or_get_cashed(
            cls,
            category: str = CATEGORY,
            max_results: int = RESULTS,
            batch_size: int = BATCH_SIZE,
    ):
        if cls.__pages is None:
            return cls.fetch_arxiv_pages(
                category=category,
                max_results=max_results,
                batch_size=batch_size,
            )
        return cls.__pages

    @classmethod
    def fetch_arxiv_pages(
            cls,
            category: str = CATEGORY,
            max_results: int = RESULTS,
            batch_size: int = BATCH_SIZE,
    ):
        pages: list = []

        for start in range(0, max_results, batch_size):
            print(f"качаю {start + batch_size} статей")
            query = f"search_query=cat:{category}&start={start}&max_results={batch_size}"
            url = BASE_URL + query

            response = requests.get(url)
            data = response.text
            parsed_data = feedparser.parse(data)

            for entry in parsed_data.entries:
                page_data = cls.__build_page_data(entry)
                pages.append(page_data)

            time.sleep(1)
            print(f"скачал: {start + batch_size}")

        cls.__pages = pages
        return pages

    @classmethod
    def __build_page_data(cls, entry) -> dict[str, list | None | list[str]]:
        full_url = entry.id
        pdf_url = full_url.replace("abs", "pdf")
        keywords = cls.__fetch_keywords_from_pdf(pdf_url)
        authors = []
        for author in entry.authors:
            authors.append(author.name)

        return {
            TITLE: entry.title,
            AUTHORS: authors,
            SUMMARY: entry.summary,
            KEYWORDS: keywords,
        }

    @classmethod
    def __fetch_keywords_from_pdf(cls, pdf_url) -> list[str] | None:
        try:
            response = url_request.urlopen(pdf_url)
            pdf_data = response.read()
            pdf_file = pdfplumber.open(io.BytesIO(pdf_data))

            page_text = []
            for page in pdf_file.pages:
                extracted_text: str = page.extract_text()
                if extracted_text:
                    page_text.append(extracted_text)

            full_text = "\n".join(page_text)
            return cls.__get_keywords(full_text)

        except Exception as e:
            print("pdf parsing error" + str(e))
            return None

    @classmethod
    def __get_keywords(cls, text: str) -> list[str] | None:
        matched: re.Match[str] | None = re.search(
            pattern=cls.__keyword_regex,
            string=text,
            flags=cls.__research_flag,
        )

        return cls.__build_keywords_array(matched)

    @classmethod
    def __build_keywords_array(cls, keywords: re.Match[str] | None) -> list[str] | None:
        if keywords is None:
            return keywords

        grouped_keywords: str = keywords.group(1)
        raw_list: list[str] = grouped_keywords.split(",")

        result: list[str] = []
        for keyword in raw_list:
            result.append(keyword.strip())

        return result
