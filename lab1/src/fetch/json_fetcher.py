import json


def fetch_from_json(file_name: str = "saved_pages.json"):
    json_file = open(file_name, "r", encoding="utf-8")
    return json.load(json_file)
