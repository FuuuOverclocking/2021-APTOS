import codecs
from pathlib import Path
import csv
import json


def resolve_path(*paths):
    return str(Path(*paths).resolve())


def read_csv(*paths):
    path = resolve_path(*paths)
    row_list = None
    fieldnames = None

    with open(path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        row_list = [row for row in reader]

    return fieldnames, row_list


def read_json(*paths):
    with codecs.open(resolve_path(*paths), "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(obj, *paths, indent=None):
    with codecs.open(resolve_path(*paths), "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=indent)
