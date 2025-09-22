import networkx as nx


class KeywordScoringProcessor:
    def __init__(self, cluster_graph):
        self.cluster_graph = cluster_graph.graph

    def show_scoring_results(self):
        degree_centrality = nx.degree_centrality(self.cluster_graph)
        closeness_centrality = nx.closeness_centrality(self.cluster_graph)
        eigenvector_centrality = nx.eigenvector_centrality(self.cluster_graph, weight='weight')

        self.__show_sorted_scoring_result(
            result=degree_centrality,
            scoring_method="degree centrality",
        )
        self.__show_sorted_scoring_result(
            result=closeness_centrality,
            scoring_method="closeness centrality",
        )
        self.__show_sorted_scoring_result(
            result=eigenvector_centrality,
            scoring_method="eigenvector centrality",
        )

    def __show_sorted_scoring_result(self, result, scoring_method: str, count=5):
        print(f"Топ {count} наборов ключевых слов по {scoring_method}:")
        it = count
        for node, centrality in self.__sort_scoring_result(result)[:count]:
            print(f"{self.cluster_graph.nodes[node]['title']}: {centrality:.4f}")
            it -= 1
            if it == 0:
                break
        print('\n\n\n')

    @staticmethod
    def __sort_scoring_result(result):
        return sorted(result.items(), key=lambda x: x[1], reverse=True)
