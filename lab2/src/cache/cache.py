import json
import os
import pandas as pd
from typing import Any


class Cache:
    def __init__(self, file_name="cache.json"):
        self.file = file_name
        self.data = None
        self._ensure_cache_exists()

    def _ensure_cache_exists(self):
        if not os.path.exists(self.file):
            with open(self.file, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load_data(self):
        if self.data is None:
            try:
                with open(self.file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.data = {}

    def save(self):
        """Сохраняет текущие данные в файл"""
        with open(self.file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

    def set_string(self, key, value):
        self._load_data()
        self.data[key] = value
        self.save()

    def get_string(self, key):
        self._load_data()
        value = self.data.get(key)
        return str(value) if value is not None else None

    def set_int(self, key, value):
        self._load_data()
        self.data[key] = value
        self.save()

    def get_int(self, key):
        self._load_data()
        value = self.data.get(key)
        return int(value) if value is not None else None

    def set_object(self, key, value):
        self._load_data()

        if isinstance(value, pd.DataFrame):
            self.data[key] = self._dataframe_to_dict(value)
        elif hasattr(value, '__dict__'):
            self.data[key] = {
                '__class__': value.__class__.__name__,
                '__module__': value.__module__,
                'data': value.__dict__
            }
        else:
            self.data[key] = value

        self.save()

    def get_object(self, key, cls=None):
        self._load_data()
        data = self.data.get(key)

        if data is None:
            return None

        if isinstance(data, dict):
            if '__class__' in data and data['__class__'] == 'DataFrame':
                return self._dataframe_from_dict(data)
            elif '__class__' in data and cls:
                obj = cls()
                obj.__dict__.update(data['data'])
                return obj

        return data

    def _dataframe_to_dict(self, df: pd.DataFrame) -> dict:
        """Преобразует DataFrame в словарь для сериализации"""
        return {
            '__class__': 'DataFrame',
            'data': df.to_dict('records'),
            'columns': list(df.columns),
            'index': list(df.index) if df.index is not None else None,
            'dtypes': {col: str(df[col].dtype) for col in df.columns}
        }

    def _dataframe_from_dict(self, data: dict) -> pd.DataFrame:
        """Восстанавливает DataFrame из словаря"""
        df = pd.DataFrame(data['data'], columns=data['columns'])

        if data['index']:
            df.index = data['index']

        for col, dtype_str in data['dtypes'].items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype_str)
                except (ValueError, TypeError):
                    pass

        return df

    def set_dataframe(self, key, df: pd.DataFrame):
        self.set_object(key, df)

    def get_dataframe(self, key) -> pd.DataFrame:
        return self.get_object(key)