import numpy as np

from lab2.src.recomendation.scoring.metrics import Metrics


def evaluation_with_fixes(recommender, test_matrix, train_matrix, k=1000):
    results = {}
    metrics = Metrics()

    catalog_size = train_matrix.shape[1]
    item_popularity = train_matrix.sum(axis=0).to_dict()

    active_test_users = test_matrix[test_matrix.sum(axis=1) > 0].index
    test_users_sample = active_test_users[:min(50, len(active_test_users))]

    print(f"оценка на {len(test_users_sample)} пользователях")

    for model_name in ['most_popular', 'item_knn', 'als']:
        print(f"Оценка {model_name}...")

        recommendations = {}
        model_func = recommender.models[model_name]

        successful_users = 0
        for user in test_users_sample:
            try:
                recs = model_func(user)
                recommendations[user] = recs
                successful_users += 1
            except Exception as e:
                print(f"Ошибка {model_name} для {user}: {e}")
                recommendations[user] = []

        print(f"  Успешных пользователей: {successful_users}/{len(test_users_sample)}")

        precisions = []
        recalls = []
        ndcgs = []
        all_recommendations = []

        for user in test_users_sample:
            actual = test_matrix.loc[user][test_matrix.loc[user] > 0].index.tolist()
            recommended = recommendations.get(user, [])

            precisions.append(metrics.precision_at_k(recommended, actual, k))
            recalls.append(metrics.recall_at_k(recommended, actual, k))
            ndcgs.append(metrics.ndcg_at_k(recommended, actual, k))
            all_recommendations.append(recommended)

        coverage_score = metrics.coverage(all_recommendations, catalog_size)
        novelty_score = metrics.novelty(all_recommendations, item_popularity)

        results[model_name] = {
            'precision@k': np.mean(precisions),
            'recall@k': np.mean(recalls),
            'ndcg@k': np.mean(ndcgs),
            'coverage': coverage_score,
            'novelty': novelty_score,
            'successful_users': successful_users
        }

    return results


def show_scoring_results(recommender, test_matrix, train_matrix):
    evaluation_results = evaluation_with_fixes(recommender, test_matrix, train_matrix)

    for model_name, metrics in evaluation_results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            if metric == 'successful_users':
                print(f"  {metric}: {value}")
            else:
                print(f"  {metric}: {value:.4f}")
