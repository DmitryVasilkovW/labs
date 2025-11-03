import pandas as pd


def generate_final_recommendations(recommender, test_users, train_matrix, method='als'):
    final_recommendations = {}

    print(f"Генерация рекомендаций методом: {method}")
    print(f"Количество тестовых пользователей: {len(test_users)}")

    if method == 'most_popular':
        popular_items = recommender.models['most_popular']
        for user in test_users:
            if callable(popular_items):
                final_recommendations[user] = popular_items(user)
            else:
                final_recommendations[user] = popular_items
    else:
        model_func = recommender.models[method]
        successful = 0
        for i, user in enumerate(test_users):
            try:
                if method == 'als':
                    recommendations = model_func(user)
                else:
                    if user in train_matrix.index:
                        user_vector = train_matrix.loc[user]
                        recommendations = model_func(user_vector)
                    else:
                        recommendations = recommender.models['most_popular']
                        if callable(recommendations):
                            recommendations = recommendations(user)

                final_recommendations[user] = recommendations
                successful += 1
            except Exception as e:
                fallback_recs = recommender.models['most_popular']
                if callable(fallback_recs):
                    fallback_recs = fallback_recs(user)
                final_recommendations[user] = fallback_recs
                if i < 5:
                    print(f"Ошибка для пользователя {user}: {e}")

        print(f"Успешно сгенерировано рекомендаций: {successful}/{len(test_users)}")

    return final_recommendations


def print_recommendations_summary(recommendations_dict, model_name, max_users=5):
    print(f"\n{'='*60}")
    print(f"МОДЕЛЬ: {model_name.upper()}")
    print(f"{'='*60}")

    items_list = list(recommendations_dict.items())

    for i, (user, items) in enumerate(items_list[:max_users]):
        print(f"Пользователь {user}:")
        if callable(items):
            items = items()
        for j, item in enumerate(items[:10]):
            print(f"   {j+1:2d}. Товар: {item}")

    total_users = len(recommendations_dict)
    all_items = []
    for user, items in recommendations_dict.items():
        if callable(items):
            items = items()
        all_items.extend(items)

    total_recommendations = len(all_items)
    unique_items = len(set(all_items))

    print(f"\n Статистика для {model_name}:")
    print(f"   • Всего пользователей: {total_users}")
    print(f"   • Всего рекомендаций: {total_recommendations}")
    print(f"   • Уникальных товаров: {unique_items}")
    if total_users > 0:
        print(f"   • Рекомендаций на пользователя: {total_recommendations/total_users:.1f}")


def show_results(test_users, train_matrix, evaluation_results, recommender):
    if hasattr(test_users, 'columns'):
        print(f"Колонки test_users: {test_users.columns.tolist()}")
        if 'cookie' in test_users.columns:
            test_users_list = test_users['cookie'].values
        else:
            print("Предупреждение: Столбец 'cookie' не найден в test_users")
            test_users_list = test_users.iloc[:, 0].values if len(test_users.columns) > 0 else test_users.index.values
    else:
        test_users_list = test_users

    print(f"Список тестовых пользователей: {test_users_list[:10]}")

    if 'evaluation_results' in locals() and evaluation_results:
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['ndcg@k'])[0]
        print(f"\n Лучшая модель для финальных рекомендаций: {best_model}")
    else:
        print("\nОценка моделей не проведена, используем most_popular")
        best_model = 'most_popular'

    models = ["item_knn", "most_popular", "als"]

    for model in models:
        print(f"\n{'#' * 80}")
        print(f"ФОРМИРОВАНИЕ РЕКОМЕНДАЦИЙ ДЛЯ МОДЕЛИ: {model.upper()}")
        print(f"{'#' * 80}")

        test_recommendations = generate_final_recommendations(
            recommender,
            test_users_list,
            train_matrix,
            method=model
        )

        if model == best_model:
            print("\n ЛУЧШАЯ МОДЕЛЬ")

        print_recommendations_summary(test_recommendations, model, max_users=5)

        print(f"\n Первые 100 элементов словаря рекомендаций для {model}:")
        items_list = list(test_recommendations.items())[:100]
        for i, (user, items) in enumerate(items_list):
            if callable(items):
                items = items()
            print(f"{i + 1:3d}. User {user}: {items[:5]}{'...' if len(items) > 5 else ''}")

    print(f" ЛУЧШАЯ МОДЕЛЬ - {best_model.upper()}")

    final_recommendations = generate_final_recommendations(
        recommender,
        test_users_list,
        train_matrix,
        method=best_model
    )

    print(f"\nФинальные рекомендации сгенерированы с использованием модели: {best_model}")

    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    comparison_data = []
    for model in models:
        temp_recs = generate_final_recommendations(recommender, test_users_list, train_matrix, method=model)

        total_recs = 0
        all_items = []
        for user, items in temp_recs.items():
            if callable(items):
                items = items()
            total_recs += len(items)
            all_items.extend(items)

        unique_items = len(set(all_items))

        comparison_data.append({
            'Модель': model,
            'Пользователи': len(temp_recs),
            'Всего рекомендаций': total_recs,
            'Уникальные товары': unique_items,
            'Рек. на пользователя': f"{total_recs / len(temp_recs):.1f}" if len(temp_recs) > 0 else "0",
            'Лучшая': '' if model == best_model else ''
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
