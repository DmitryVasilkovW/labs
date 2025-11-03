import json
import os


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
        if hasattr(value, '__dict__'):
            self.data[key] = value.__dict__
        else:
            self.data[key] = value
        self.save()

    def get_object(self, key, cls=None):
        self._load_data()
        data = self.data.get(key)
        if cls and data and hasattr(cls, '__dict__'):
            obj = cls()
            obj.__dict__.update(data)
            return obj
        return data