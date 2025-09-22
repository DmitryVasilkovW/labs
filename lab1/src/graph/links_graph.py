import itertools
import pandas as pd
import networkx as nx
from cosmograph import cosmo
from lab1.src.pages.pages_processor import PagesProcessor


class LinksGraph:
    def __init__(self, page_processor: PagesProcessor):
        self.page_processor = page_processor
        self.vertices = self.__init_vertices()
        self.edges = self.__init_edges()
        self.graph = nx.DiGraph()

    def __init_vertices(self):
        df = self.page_processor.get_title_keyword_df()

        keyword_to_title = {}
        all_keywords = []

        for _, row in df.iterrows():
            title = row["title"]
            keywords = row["keywords"]
            if not keywords:
                continue
            for kw in keywords:
                kw_clean = kw.strip().lower()
                all_keywords.append(kw_clean)
                if kw_clean not in keyword_to_title:
                    keyword_to_title[kw_clean] = title

        unique_keywords = list(set(all_keywords))

        ids = list(range(len(unique_keywords)))
        counts = [all_keywords.count(k) for k in unique_keywords]
        titles = [keyword_to_title[k] for k in unique_keywords]

        vertices = pd.DataFrame({
            'id': ids,
            'keywords': unique_keywords,
            'count': counts,
            'title': titles,
        })

        return vertices

    def __init_edges(self):
        keyword_to_id = dict(zip(self.vertices['keywords'], self.vertices['id']))

        edges = []
        for keywords in self.page_processor.get_keywords():
            if not keywords:
                continue
            keywords_clean = [kw.strip().lower() for kw in keywords if kw]
            for pair in itertools.combinations(keywords_clean, 2):
                keyword_from, keyword_to = pair
                id_from = keyword_to_id.get(keyword_from)
                id_to = keyword_to_id.get(keyword_to)
                if id_from is not None and id_to is not None:
                    edges.append({
                        'source': id_from,
                        'target': id_to,
                        'value': 1.0,
                    })

        return pd.DataFrame(edges)

    def show(self):
        widget = cosmo(
            points=self.vertices,
            links=self.edges,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            point_color_by='title',
            point_include_columns=['count'],
            point_label_by='keywords',
            link_include_columns=['value'],
        )
        return widget
