import json


def save_to_json(data, file_name: str = "saved_pages.json"):
    json_file = open(file_name, "w", encoding="utf-8")
    json.dump(data, json_file, ensure_ascii=False, indent=4)
