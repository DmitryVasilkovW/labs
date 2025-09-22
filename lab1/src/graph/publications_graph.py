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
        for idx, row in self.pages_processor.iterate():
            keywords = set(row['keywords'])
            for other_idx, other_row in self.pages_processor.iterate():
                if idx != other_idx:
                    other_keywords = set(other_row['keywords'])
                    common_keywords = keywords.intersection(other_keywords)
                    if common_keywords:
                        weight = len(common_keywords)
                        self.graph.add_edge(idx, other_idx, weight=weight)

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
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_nodes(self.graph, pos, node_size=50, node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, width=0.5, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_color='black')
        plt.title("Граф публикаций")
        plt.show()
