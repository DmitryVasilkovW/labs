from datetime import timedelta


class DataMatrix:

    @staticmethod
    def temporal_split_data(clickstream_df, test_days=7):
        max_date = clickstream_df['event_date'].max()
        cutoff_date = max_date - timedelta(days=test_days)

        train_data = clickstream_df[clickstream_df['event_date'] <= cutoff_date]
        test_data = clickstream_df[clickstream_df['event_date'] > cutoff_date]

        print(f"Исходное разделение:")
        print(f"  Train: {len(train_data)} взаимодействий, {train_data['cookie'].nunique()} пользователей")
        print(f"  Test: {len(test_data)} взаимодействий, {test_data['cookie'].nunique()} пользователей")

        return train_data, test_data, cutoff_date

    @staticmethod
    def limit_training_data(train_data, max_users=5000, max_items=10000):
        train_user_activity = train_data.groupby('cookie').size().sort_values(ascending=False)
        train_top_users = train_user_activity.head(max_users).index

        train_item_popularity = train_data.groupby('item').size().sort_values(ascending=False)
        train_top_items = train_item_popularity.head(max_items).index

        train_limited = train_data[
            train_data['cookie'].isin(train_top_users) &
            train_data['item'].isin(train_top_items)
            ]

        print(f"Ограниченный train:")
        print(f"  Пользователей: {train_limited['cookie'].nunique()}")
        print(f"  Товаров: {train_limited['item'].nunique()}")
        print(f"  Взаимодействий: {len(train_limited)}")

        return train_limited, train_top_users, train_top_items

    @staticmethod
    def filter_test_data(test_data, train_top_users, train_top_items):
        test_filtered = test_data[
            test_data['cookie'].isin(train_top_users) &
            test_data['item'].isin(train_top_items)
            ]

        print(f"Отфильтрованный test:")
        print(f"  Пользователей: {test_filtered['cookie'].nunique()}")
        print(f"  Товаров: {test_filtered['item'].nunique()}")
        print(f"  Взаимодействий: {len(test_filtered)}")

        return test_filtered

    @staticmethod
    def create_final_matrices(train_data, test_data):
        train_matrix = train_data.pivot_table(
            index='cookie', columns='item', values='event_weight',
            aggfunc='sum', fill_value=0
        )

        test_matrix = test_data.pivot_table(
            index='cookie', columns='item', values='event_weight',
            aggfunc='sum', fill_value=0
        )

        common_items = train_matrix.columns.intersection(test_matrix.columns)
        train_matrix = train_matrix[common_items]
        test_matrix = test_matrix[common_items]

        common_users = train_matrix.index.intersection(test_matrix.index)
        train_matrix = train_matrix.loc[common_users]
        test_matrix = test_matrix.loc[common_users]

        print(f"Итоговые матрицы:")
        print(f"  Train: {train_matrix.shape}")
        print(f"  Test: {test_matrix.shape}")
        print(f"  Общие пользователи: {len(common_users)}")
        print(f"  Общие товары: {len(common_items)}")

        return train_matrix, test_matrix, common_users

    @staticmethod
    def get_matrix(clickstream_processed):
        train_data_full, test_data_full, cutoff_date = DataMatrix.temporal_split_data(clickstream_processed)
        train_limited, train_top_users, train_top_items = DataMatrix.limit_training_data(train_data_full)

        test_filtered = DataMatrix.filter_test_data(test_data_full, train_top_users, train_top_items)
        train_matrix, test_matrix, common_users = DataMatrix.create_final_matrices(train_limited, test_filtered)

        return train_matrix, test_matrix, common_users