import json
from typing import Any


def write_to_file(content: str, file_name: str) -> None:
    try:
        with open(file_name, "w") as file:
            file.write(content)
    except IOError as e:
        raise IOError(f"Failed to write to file {file_name}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error writing to {file_name}: {e}")


def read_from_file(file_name: str) -> str:
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_name}")
    except IOError as e:
        raise IOError(f"Failed to read from file {file_name}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error reading from {file_name}: {e}")


def load_json(file: str) -> Any:
    try:
        with open(file, "r") as j:
            return json.loads(j.read())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSON file not found: {file}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file}: {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"Unexpected error loading JSON from {file}: {e}")


def dump_json(data: Any, file_name: str) -> None:
    try:
        with open(file_name, "w") as j:
            json.dump(data, j, indent=2)
    except TypeError as e:
        raise TypeError(f"Data is not JSON serializable for file {file_name}: {e}")
    except IOError as e:
        raise IOError(f"Failed to write JSON to file {file_name}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error dumping JSON to {file_name}: {e}")


def load_jsonl(file: str) -> list[Any]:
    data = []
    try:
        with open(file, "r") as j:
            for line_num, line in enumerate(j, 1):
                if line.strip():  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Line {line_num} in file {file}: {e.msg}", 
                            e.doc, 
                            e.pos
                        )
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSONL file not found: {file}")
    except json.JSONDecodeError:
        raise  # Re-raise with context already added
    except Exception as e:
        raise Exception(f"Unexpected error loading JSONL from {file}: {e}")


def dump_jsonl(data: list[Any], file_name: str) -> None:
    try:
        with open(file_name, "w") as j:
            for idx, item in enumerate(data):
                try:
                    j.write(json.dumps(item) + "\n")
                except TypeError as e:
                    raise TypeError(f"Item at index {idx} is not JSON serializable: {e}")
    except IOError as e:
        raise IOError(f"Failed to write JSONL to file {file_name}: {e}")
    except TypeError:
        raise  # Re-raise with context already added
    except Exception as e:
        raise Exception(f"Unexpected error dumping JSONL to {file_name}: {e}")



