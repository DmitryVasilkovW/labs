def generate_features(
        clickstream_processed,
        cat_features_processed,
        text_features_processed,
):
    print("Генерация фич пользователей...")
    user_features = clickstream_processed.groupby('cookie').agg({
        'item': 'count',
        'event': 'nunique',
        'is_contact': 'sum',
        'event_weight': 'sum',
        'platform': lambda x: x.mode()[0] if len(x) > 0 else -1,
        'surface': lambda x: x.mode()[0] if len(x) > 0 else -1
    }).rename(columns={
        'item': 'total_events',
        'event': 'unique_event_types',
        'is_contact': 'contact_events_count',
        'event_weight': 'weighted_activity',
        'platform': 'most_used_platform',
        'surface': 'most_used_surface'
    })

    user_temporal = clickstream_processed.groupby('cookie').agg({
        'event_date': ['min', 'max', 'count']
    })
    user_temporal.columns = ['first_activity', 'last_activity', 'activity_days']
    user_temporal['user_tenure_days'] = (user_temporal['last_activity'] - user_temporal['first_activity']).dt.days + 1

    user_features = user_features.merge(user_temporal, on='cookie')

    print("Генерация фич объявлений...")
    item_features = cat_features_processed.groupby('item').first().reset_index()

    item_popularity = clickstream_processed.groupby('item').agg({
        'cookie': 'count',
        'is_contact': 'sum',
        'event_weight': 'sum'
    }).rename(columns={
        'cookie': 'view_count',
        'is_contact': 'contact_count',
        'event_weight': 'weighted_popularity'
    })

    item_features = item_features.merge(item_popularity, on='item', how='left').fillna(0)

    item_features = item_features.merge(
        text_features_processed[['item', 'title_projection']],
        on='item',
        how='left'
    )

    return user_features, item_features