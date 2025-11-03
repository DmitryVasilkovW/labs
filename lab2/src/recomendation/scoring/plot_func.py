import pandas as pd
from matplotlib import pyplot as plt


def show_scoring_plot(evaluation_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    models = list(evaluation_results.keys())
    metrics_to_plot = ['precision@k', 'recall@k', 'ndcg@k', 'coverage']

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]
        values = [evaluation_results[model][metric] for model in models]

        bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title(f'{metric.upper()} по моделям')
        ax.set_ylabel(metric)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(evaluation_results).T
    print("\nСравнительная таблица моделей:")
    print(results_df.round(4))
