import numpy as np


class Metrics:
    @staticmethod
    def precision_at_k(recommended, actual, k=1000):
        """Precision@K - точность среди топ-K рекомендаций"""
        if len(recommended) == 0:
            return 0.0
        recommended_k = recommended[:k]
        relevant = set(actual) & set(recommended_k)
        return len(relevant) / k

    @staticmethod
    def recall_at_k(recommended, actual, k=1000):
        """Recall@K - полнота среди топ-K рекомендаций"""
        if len(actual) == 0:
            return 0.0
        recommended_k = recommended[:k]
        relevant = set(actual) & set(recommended_k)
        return len(relevant) / len(actual)

    @staticmethod
    def ndcg_at_k(recommended, actual, k=1000):
        """NDCG@K - учитывает порядок рекомендаций"""
        if len(actual) == 0:
            return 0.0

        recommended_k = recommended[:k]
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in actual:
                dcg += 1 / np.log2(i + 2)

        idcg = 0.0
        for i in range(min(len(actual), k)):
            idcg += 1 / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def coverage(recommended_lists, catalog_size):
        """Coverage - покрытие каталога рекомендациями"""
        all_recommended = set()
        for recs in recommended_lists:
            all_recommended.update(recs)
        return len(all_recommended) / catalog_size

    @staticmethod
    def novelty(recommended_lists, item_popularity):
        """Novelty - новизна рекомендаций (рекомендации менее популярных товаров)"""
        novelty_scores = []
        for recs in recommended_lists:
            if len(recs) == 0:
                continue
            avg_popularity = np.mean([item_popularity.get(item, 0) for item in recs])
            novelty_scores.append(1 / (1 + avg_popularity))

        return np.mean(novelty_scores) if novelty_scores else 0.0
