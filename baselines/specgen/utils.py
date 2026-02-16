import logging
import sys
from time import time
import tiktoken
from datetime import datetime
import os
from pathlib import Path
import json


def create_logger(_name, log_dir):
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    log_file_paths = os.path.join(log_dir, _name + ".log")
    fh = logging.FileHandler(log_file_paths)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


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


def count_str_token(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_config_token(config) -> int:
    sum = 0
    for message in config["messages"]:
        sum += count_str_token(message["content"])
    return sum
