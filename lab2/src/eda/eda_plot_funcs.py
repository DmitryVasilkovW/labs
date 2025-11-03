from matplotlib import pyplot as plt

from lab2.src.eda.eda import Eda


def show_eda_plots(eda: Eda):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    event_counts = eda.clickstream['event'].value_counts().head(10)
    axes[0, 0].bar(range(len(event_counts)), event_counts.values)
    axes[0, 0].set_title('Топ-10 типов событий')
    axes[0, 0].set_xlabel('Тип события')
    axes[0, 0].set_ylabel('Количество')

    if 'event_date' in eda.clickstream.columns:
        daily_activity = eda.clickstream.groupby(eda.clickstream['event_date'].dt.date).size()
        axes[0, 1].plot(daily_activity.index, daily_activity.values)
        axes[0, 1].set_title('Активность пользователей по дням')
        axes[0, 1].set_xlabel('Дата')
        axes[0, 1].set_ylabel('Количество событий')
        axes[0, 1].tick_params(axis='x', rotation=45)

    platform_counts = eda.clickstream['platform'].value_counts()
    axes[0, 2].pie(platform_counts.values, labels=platform_counts.index, autopct='%1.1f%%')
    axes[0, 2].set_title('Распределение по платформам')

    if 'category' in eda.cat_features.columns:
        category_counts = eda.cat_features['category'].value_counts().head(10)
        axes[1, 0].bar(range(len(category_counts)), category_counts.values)
        axes[1, 0].set_title('Топ-10 категорий товаров')
        axes[1, 0].set_xlabel('Категория')
        axes[1, 0].set_ylabel('Количество')
        axes[1, 0].tick_params(axis='x', rotation=45)

    contact_stats = eda.clickstream.merge(eda.events, on='event').groupby('is_contact').size()
    axes[1, 1].pie(contact_stats.values, labels=['Неконтактные', 'Контактные'], autopct='%1.1f%%')
    axes[1, 1].set_title('Контактные vs Неконтактные события')

    user_activity = eda.clickstream.groupby('cookie').size()
    axes[1, 2].hist(user_activity, bins=50, edgecolor='black')
    axes[1, 2].set_title('Распределение активности пользователей')
    axes[1, 2].set_xlabel('Количество событий на пользователя')
    axes[1, 2].set_ylabel('Количество пользователей')

    plt.tight_layout()
    plt.show()

    print("\n6. СТАТИСТИКИ АКТИВНОСТИ ПОЛЬЗОВАТЕЛЕЙ:")
    print(f"Среднее событий на пользователя: {user_activity.mean():.2f}")
    print(f"Медиана событий на пользователя: {user_activity.median():.2f}")
    print(f"Максимум событий на пользователя: {user_activity.max()}")