import pandas as pd


class Eda:
    def __init__(
            self,
            text_features,
            cat_features,
            clickstream,
            events,
            test_users,
            cache,
    ):
        self.text_features = text_features
        self.cat_features = cat_features
        self.clickstream = clickstream
        self.events = events
        self.test_users = test_users
        self.cache = cache

    def show(self, use_cache=True):
        print("EDA")

        print("\n1. АНАЛИЗ ТЕКСТОВЫХ ФИЧ:")
        print(f"Уникальные объявления: {self.text_features['item'].nunique()}")
        print(f"Размерность векторов: {len(self.text_features['title_projection'].iloc[0])}")
        print(f"Пропуски: {self.text_features.isnull().sum().sum()}")

        print("\n2. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ФИЧ:")
        print(f"Уникальные локации: {self.cat_features['location'].nunique()}")
        print(f"Уникальные категории: {self.cat_features['category'].nunique()}")
        print(f"Уникальные ноды: {self.cat_features['node'].nunique()}")
        print(f"Пропуски: {self.cat_features.isnull().sum()}")

        print("\n3. АНАЛИЗ CLEAN_PARAMS:")
        if use_cache:
            self.cat_features = self.cache.get_dataframe('cat_features')
        else:
            self.cat_features['params_count'] = self.cat_features['clean_params'].apply(
                lambda x: len(eval(x)) if pd.notna(x) else 0
            )
            self.cache.set_dataframe('cat_features', self.cat_features)
            self.cache.save()

        print(f"Среднее количество параметров: {self.cat_features['params_count'].mean():.2f}")
        print(f"Максимальное количество параметров: {self.cat_features['params_count'].max()}")

        print("\n4. АНАЛИЗ СОБЫТИЙ:")
        print("Распределение типов событий:")
        print(self.events['event'].value_counts())
        print("\nКонтактные события:")
        print(self.events['is_contact'].value_counts())

        print("\n5. АНАЛИЗ КЛИКСТРИМА:")
        print(f"Период данных: {self.clickstream['event_date'].min()} - {self.clickstream['event_date'].max()}")
        print(f"Уникальные пользователи: {self.clickstream['cookie'].nunique()}")
        print(f"Уникальные объявления: {self.clickstream['item'].nunique()}")
        print(f"Уникальные события: {self.clickstream['event'].nunique()}")

        print("\nРаспределение платформ:")
        print(self.clickstream['platform'].value_counts())
        print("\nРаспределение поверхностей:")
        print(self.clickstream['surface'].value_counts())
