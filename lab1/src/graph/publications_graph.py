import networkx as nx
from matplotlib import pyplot as plt

from lab1.src.pages.pages_processor import PagesProcessor


class PublicationsGraph:
    def __init__(self, pages_processor: PagesProcessor):
        self.graph = nx.Graph()
        self.pages_processor = pages_processor
        self.__init_vertex()
        self.__init_edges()

    def __init_vertex(self):
        for i, row in self.pages_processor.iterate():
            self.graph.add_node(
                i,
                title=row['title'],
                authors=row['authors'],
                summary=row['summary'],
            )

    def __init_edges(self):
        for i, row in self.pages_processor.iterate():
            keywords = set(row['keywords'])
            for j, other_row in self.pages_processor.iterate():
                self.__add_edge_if_not_common(i, j, keywords, other_row)

    def __add_edge_if_not_common(self, first_index, second_index, keywords, other_row):
        if first_index != second_index:
            other_keywords = set(other_row['keywords'])
            common_keywords = keywords.intersection(other_keywords)
            if common_keywords:
                weight = len(common_keywords)
                self.graph.add_edge(first_index, second_index, weight=weight)

    def find_neighbors(self, publication: int):
        neighbors = list(self.graph.neighbors(publication))
        publication_title = self.graph.nodes[publication]['title']
        titles = []
        for n in neighbors:
            title = self.graph.nodes[n]['title']
            titles.append((n, title))
        return {
            "publication": publication,
            "publication_title": publication_title,
            "titles": titles,
        }

    def show(self):
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_color='black')
        plt.title("Граф публикаций", fontsize=18)
        plt.show()
