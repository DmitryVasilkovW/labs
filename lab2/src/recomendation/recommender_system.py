import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from collections import defaultdict


class RecommenderSystem:
    def __init__(self):
        self.models = {}

    def most_popular(self, user_item_matrix, n_recommendations=1500):
        if user_item_matrix.shape[1] == 0:
            return lambda user_id: []

        item_popularity = user_item_matrix.sum(axis=0)

        valid_items = [item for item in item_popularity.index if item in user_item_matrix.columns]
        item_popularity = item_popularity[valid_items]

        popular_items = item_popularity.sort_values(ascending=False).head(n_recommendations).index.tolist()

        print(f"Most Popular: {len(popular_items)} рекомендаций")
        print(f"Примеры: {popular_items[:3]}")

        return lambda user_id: popular_items

    def item_knn(self, user_item_matrix, n_neighbors=1500, n_recommendations=1500):
        if user_item_matrix.shape[1] == 0:
            return lambda user_id: []

        print(f"Вычисление схожести для {user_item_matrix.shape[1]} товаров...")

        try:
            item_item_similarity = cosine_similarity(user_item_matrix.T)
            item_sim_df = pd.DataFrame(
                item_item_similarity,
                index=user_item_matrix.columns,
                columns=user_item_matrix.columns
            )

            def get_recommendations(user_id):
                if user_id not in user_item_matrix.index:
                    return []

                user_vector = user_item_matrix.loc[user_id]
                user_items = user_vector[user_vector > 0].index

                if len(user_items) == 0:
                    item_popularity = user_item_matrix.sum(axis=0)
                    return item_popularity.sort_values(ascending=False).head(n_recommendations).index.tolist()

                scores = defaultdict(float)
                for item in user_items:
                    if item in item_sim_df.columns:
                        similar_items = item_sim_df[item].sort_values(ascending=False).iloc[1:n_neighbors+1]
                        for similar_item, similarity in similar_items.items():
                            if similar_item not in user_items:
                                scores[similar_item] += similarity * user_vector[item]

                recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
                return [item for item, score in recommended]

            return get_recommendations

        except Exception as e:
            print(f"Ошибка в ItemKNN: {e}")
            return lambda user_id: []

    def als_recommendations(self, user_item_matrix, n_factors=1500, n_recommendations=1500):
        if user_item_matrix.shape[1] == 0 or user_item_matrix.shape[0] == 0:
            return lambda user_id: []

        sparse_matrix = sp.csr_matrix(user_item_matrix.values.astype('double'))

        model = AlternatingLeastSquares(
            factors=min(n_factors, user_item_matrix.shape[1], user_item_matrix.shape[0]),
            regularization=0.01,
            iterations=15,
            random_state=42
        )

        try:
            model.fit(sparse_matrix)
            print("ALS модель успешно обучена")
        except Exception as e:
            print(f"Ошибка при обучении ALS: {e}")
            return lambda user_id: []

        def get_recommendations(user_id):
            if user_id not in user_item_matrix.index:
                return []

            user_idx = user_item_matrix.index.get_loc(user_id)

            try:
                result = model.recommend(
                    user_idx,
                    sparse_matrix[user_idx],
                    N=min(n_recommendations, user_item_matrix.shape[1]),
                    filter_already_liked_items=True
                )

                if len(result) == 2:
                    item_indices, scores = result
                else:
                    print(f"Неожиданный формат результата ALS: {len(result)} элементов")
                    item_indices = result[0] if len(result) > 0 else []

                return [user_item_matrix.columns[item_idx] for item_idx in item_indices]

            except Exception as e:
                print(f"Ошибка ALS для пользователя {user_id}: {e}")
                return []

        return get_recommendations
