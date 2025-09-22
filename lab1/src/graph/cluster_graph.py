import networkx as nx
import community as community_louvain
from matplotlib import pyplot as plt


class ClusterGraph:
    def __init__(self, edges, vertices):
        self.graph = nx.Graph()
        self.__setup_edges(edges)
        self.__setup_vertices(vertices)
        self.community_louvain = community_louvain.best_partition(self.graph)
        self.partition = community_louvain.best_partition(self.graph)
        self.modularity = community_louvain.modularity(self.partition, self.graph)

    def __setup_edges(self, edges):
        for _, row in edges.iterrows():
            self.graph.add_edge(row['source'], row['target'], weight=row['value'])

    def __setup_vertices(self, vertices):
        for _, row in vertices.iterrows():
            self.graph.add_node(row['id'], title=row['title'], count=row['count'])

    def show_info(self):
        plt.figure(figsize=(6, 1.5))
        plt.barh([0], [self.modularity], color='skyblue', height=0.5)
        plt.xlim(-1, 1)
        plt.xlabel("Модульность")
        plt.yticks([])
        plt.title(f"Модульность графа: {self.modularity:.4f}")
        plt.show()

        print("Информация о кластерах")

        info = {}
        for node, cluster in self.partition.items():
            title = self.graph.nodes[node]['title']
            info.setdefault(cluster, []).append(title)
        for cluster, keywords in info.items():
            print(f"Кластер: {cluster}\nНекоторые ключевые слова: {', '.join(keywords[:1])}...")

    def show(self):
        colors = []
        for node in self.graph.nodes():
            colors.append(self.partition[node])

        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, cmap=plt.cm.rainbow, node_size=50, alpha=0.7)
        nx.draw_networkx_labels(self.graph, pos, font_size=7, font_color='black')
        nx.draw_networkx_edges(self.graph, pos, width=0.5, alpha=0.5, edge_color='gray')
        plt.title(f"Граф кластеризации ключевых слов")
        plt.show()
