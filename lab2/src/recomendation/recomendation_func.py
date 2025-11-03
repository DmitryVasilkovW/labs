from lab2.src.recomendation.recommender_system import RecommenderSystem


def fit(train_matrix):
    recommender = RecommenderSystem()

    print("Обучение Most Popular...")
    recommender.models['most_popular'] = recommender.most_popular(train_matrix)

    print("Обучение ItemKNN...")
    recommender.models['item_knn'] = recommender.item_knn(train_matrix)

    print("Обучение ALS...")
    recommender.models['als'] = recommender.als_recommendations(train_matrix)

    return recommender