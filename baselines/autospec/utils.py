import logging
import sys
import time
import os
import tiktoken



def create_logger(_name, log_dir):
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    log_file_paths=os.path.join(log_dir, _name + ".log")
    fh = logging.FileHandler(log_file_paths)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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



def read_file_as_str(filename):
    with open(filename) as f:
        return f.read()
    

def file2str(filename):
    res = ""
    with open(filename, "r") as f:
        for line in f.readlines():
            res = res + line
    return res

def read_file_as_list(filename):
    with open(filename) as f:
        return f.readlines()


def extract_blank_prefix(string):
    string_stripped = string.strip()
    if len(string_stripped) > 0:
        return string.split(string_stripped)[0]
    else:
        return string
    

# [check] this logger 
def print_while_logging(classname:str, content:str):
    current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    print(content)
    with open(os.path.abspath(".") + "/logs/log-{name}-{time_str}.txt".format(name=classname, time_str=current_time_str), 'a') as f:
        f.write(content + '\n')


def count_str_token(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def count_config_token(config) -> int:
    sum = 0
    for message in config["messages"]:
        sum += count_str_token(message["content"])
    return sum
