import sys
# import tiktoken
import os
from pathlib import Path



def str2file(str_content, filename):
    with open(filename, "w") as f:
        f.write(str_content)

def file2str(filename):
    res = ""
    with open(filename, "r") as f:
        for line in f.readlines():
            res = res + line
    return res

def extract_blank_prefix(string):
    string_stripped = string.strip()
    if len(string_stripped) > 0:
        return string.split(string_stripped)[0]
    else:
        return string




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


def read_input_paths(input_file):
    paths = []
    try:
        with open(input_file, "r") as f:
            for line in f:
                path = line.strip()
                if path and not path.startswith("#"):  # Skip empty lines and comments
                    paths.append(path)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)
    return paths


def file2str(filename):
    res = ""
    with open(filename, "r") as f:
        for line in f.readlines():
            res = res + line
    return res


def parse_code_from_reply(content):
    content = "a" + content
    extracted_str = content.split("```")[1]
    extracted_str = (
        extracted_str
        if not extracted_str.startswith("java")
        else extracted_str[len("java") :]
    )
    return extracted_str


# def count_str_token(string: str) -> int:
#     encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
#     num_tokens = len(encoding.encode(string))
#     return num_tokens


# def count_config_token(config) -> int:
#     sum = 0
#     for message in config["messages"]:
#         sum += count_str_token(message["content"])
#     return sum
