import os
import sys
import json


def write_to_file(content, file_name):
    with open(file_name, "w") as file:
        file.write(content)


def read_from_file(file_name):
    with open(file_name, "r") as file:
        return file.read()


def load_json(file):
    with open(file, "r") as j:
        return json.loads(j.read())


def dump_json(data, file_name):
    with open(file_name, "w") as j:
        json.dump(data, j, indent=2)


def load_jsonl(file):
    data = []
    with open(file, "r") as j:
        for line in j:
            data.append(json.loads(line))
    return data


def dump_jsonl(data, file_name):
    with open(file_name, "w") as j:
        for item in data:
            j.write(json.dumps(item) + "\n")


def load_java_file_paths(directory):
    java_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".java"):
                    java_files.append(os.path.join(root, file))
    except FileNotFoundError:
        print(f"Error: Directory {directory} not found")
        sys.exit(1)
    return java_files
