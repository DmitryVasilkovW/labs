import pandas as pd


class DataInstaller:
    def __init__(self):
        self.text_features = pd.read_parquet('text_features.pq')
        self.cat_features = pd.read_parquet('cat_features.pq')
        self.clickstream = pd.read_parquet('clickstream.pq')
        self.events = pd.read_parquet('events.pq')
        self.test_users = pd.read_parquet('test_users.pq')

    def show_samples(self):
        print("ДАННЫЕ")
        print(f"Text features: {self.text_features.shape}")
        print(f"Category features: {self.cat_features.shape}")
        print(f"Clickstream: {self.clickstream.shape}")
        print(f"Events: {self.events.shape}")
        print(f"Test users: {self.test_users.shape}")

        print("\nПервые 5 строк каждого датасета:")
        print("\nText Features:")
        print(self.text_features.head())
        print("\nCategory Features:")
        print(self.cat_features.head())
        print("\nClickstream:")
        print(self.clickstream.head())
        print("\nEvents:")
        print(self.events.head())