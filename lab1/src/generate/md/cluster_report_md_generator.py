from collections import Counter
from datetime import datetime

from lab1.src.constant.fetching import KEYWORDS
from lab1.src.graph.cluster_graph import ClusterGraph


class ClusterReportGenerator:
    def __init__(self, cluster_graph: ClusterGraph):
        self.cluster_graph = cluster_graph
        self.cluster_interpretations = {}

    def generate_md_report(self, filename="cluster_analysis_report.md"):
        if not self.cluster_interpretations:
            self.__analyze_clusters()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Анализ кластеров научных публикаций\n\n")
            f.write(f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Общая информация\n\n")
            f.write(f"- **Модульность графа:** {self.cluster_graph.modularity:.4f}\n")
            f.write(f"- **Количество кластеров:** {len(self.cluster_interpretations)}\n")
            f.write(f"- **Общее количество узлов:** {len(self.cluster_graph.graph.nodes())}\n")
            f.write(f"- **Общее количество связей:** {len(self.cluster_graph.graph.edges())}\n\n")

            f.write("## Интерпретация кластеров\n\n")

            for cluster_id, interpretation in sorted(self.cluster_interpretations.items()):
                f.write(f"### Кластер {cluster_id}\n\n")
                f.write(f"**Размер:** {interpretation['size']} узлов\n\n")

                f.write("**Топ-10 ключевых слов:**\n")
                for i, (keyword, _) in enumerate(interpretation['top_keywords'], 1):
                    f.write(f"{i}. {keyword}\n")
                f.write("\n")

                f.write("**Примеры узлов:**\n")
                for node in interpretation['sample_nodes']:
                    node_keywords = self.cluster_graph.graph.nodes[node].get(KEYWORDS, [])
                    f.write(f"- {node}: {node_keywords}\n")
                f.write("\n")

            f.write("## Статистика по кластерам\n\n")
            cluster_sizes = [interpretation['size'] for interpretation in
                             self.cluster_interpretations.values()]

            f.write(f"- **Самый большой кластер:** {max(cluster_sizes)} узлов\n")
            f.write(f"- **Самый маленький кластер:** {min(cluster_sizes)} узлов\n")
            f.write(f"- **Средний размер кластера:** {sum(cluster_sizes) / len(cluster_sizes):.1f} узлов\n\n")

        print(f"Отчет сохранен в файл: {filename}")
        return filename

    def __analyze_clusters(self):
        cluster_data = {}

        for node, cluster_id in self.cluster_graph.partition.items():
            keywords = self.cluster_graph.graph.nodes[node].get(KEYWORDS, [])
            if isinstance(keywords, str):
                if ',' in keywords:
                    keywords = [kw.strip() for kw in keywords.split(',')]
                else:
                    keywords = [keywords]

            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = {
                    'nodes': [],
                    'all_keywords': [],
                    'node_count': 0,
                }
            cluster_data[cluster_id]['nodes'].append(node)
            cluster_data[cluster_id]['all_keywords'].extend(keywords)
            cluster_data[cluster_id]['node_count'] += 1

        for cluster_id, data in cluster_data.items():
            keyword_counter = Counter(data['all_keywords'])
            top_keywords = keyword_counter.most_common(10)

            self.cluster_interpretations[cluster_id] = {
                'size': data['node_count'],
                'top_keywords': top_keywords,
                'sample_nodes': data['nodes'][:5]
            }

        return self.cluster_interpretations
