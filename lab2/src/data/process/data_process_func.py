import ast
from collections import Counter
import time

import pandas as pd

from lab2.src.cache.cache import Cache
from lab2.src.eda.eda import Eda


def process_all_data(eda: Eda, cache: Cache, use_cache: bool = True):
    def extract_params_features(df, sample_size=1000, max_attrs=10):
        start_time = time.time()

        sample_df = df['clean_params'].dropna().sample(min(sample_size, len(df)), random_state=42)

        attr_counter = Counter()
        for params_str in sample_df:
            try:
                params_list = ast.literal_eval(params_str)
                for param in params_list:
                    attr_counter[param['attr']] += 1
            except:
                continue

        top_attrs = [attr for attr, count in attr_counter.most_common(max_attrs)]
        print(f"Найдено {len(top_attrs)} самых частых атрибутов")

        for attr in top_attrs:
            df[f'attr_{attr}'] = 0

        from tqdm import tqdm
        tqdm.pandas(desc="Обработка clean_params")

        def check_attr_presence(params_str, target_attrs):
            if pd.isna(params_str):
                return [0] * len(target_attrs)
            try:
                params_list = ast.literal_eval(params_str)
                present_attrs = {p['attr'] for p in params_list}
                return [1 if attr in present_attrs else 0 for attr in target_attrs]
            except:
                return [0] * len(target_attrs)

        results = df['clean_params'].progress_apply(
            lambda x: check_attr_presence(x, top_attrs)
        )

        for i, attr in enumerate(top_attrs):
            df[f'attr_{attr}'] = results.apply(lambda x: x[i])

        print(f"Обработка clean_params заняла: {time.time() - start_time:.2f} сек")
        return df

    if use_cache:
        cat_features_processed = cache.get_dataframe("cat_features_processed")
    else:
        cat_features_processed = extract_params_features(eda.cat_features.copy())
        cache.set_dataframe("cat_features_processed", cat_features_processed)
        cache.save()

    text_features_processed = eda.text_features.copy()
    clickstream_processed = eda.clickstream.copy()

    clickstream_processed = clickstream_processed.merge(
        eda.events[['event', 'is_contact']],
        on='event',
        how='left'
    ).fillna({'is_contact': 0})

    clickstream_processed['event_weight'] = clickstream_processed['is_contact'].replace({1: 2.0, 0: 1.0})

    user_interaction_count = clickstream_processed.groupby('cookie').size()
    active_users = user_interaction_count[user_interaction_count >= 2].index

    clickstream_processed = clickstream_processed[
        clickstream_processed['cookie'].isin(active_users)
    ].copy()

    print(f"После фильтрации: {clickstream_processed.shape[0]} взаимодействий")
    print(f"Активных пользователей: {clickstream_processed['cookie'].nunique()}")

    return clickstream_processed, text_features_processed, cat_features_processed