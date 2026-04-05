"""Utilities for parsing type hints into JSON schemas. Derived from transformers."""

import inspect
import json
import re
import types
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin, get_type_hints


def get_imports(code: str) -> List[str]:
    code = re.sub(r"\s*try\s*:.*?except.*?:", "", code, flags=re.DOTALL)
    code = re.sub(r"if is_flash_attn[a-zA-Z0-9_]+available\(\):\s*(from flash_attn\s*.*\s*)+", "", code, flags=re.MULTILINE)
    imports = re.findall(r"^\s*import\s+(\S+?)(?:\s+as\s+\S+)?\s*$", code, flags=re.MULTILINE)
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", code, flags=re.MULTILINE)
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]
    return list(set(imports))


class TypeHintParsingException(Exception):
    pass

class DocstringParsingException(Exception):
    pass


def get_json_schema(func: Callable) -> Dict:
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(f"Cannot generate JSON schema for {func.__name__} because it has no docstring!")
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = _parse_google_format_docstring(doc)
    json_schema = _convert_type_hints_to_json_schema(func)
    if (return_dict := json_schema["properties"].pop("return", None)) is not None:
        if return_doc is not None:
            return_dict["description"] = return_doc
    for arg, schema in json_schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(f"Cannot generate JSON schema for {func.__name__} because the docstring has no description for the argument '{arg}'")
        desc = param_descriptions[arg]
        enum_choices = re.search(r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_choices:
            schema["enum"] = [c.strip() for c in json.loads(enum_choices.group(1))]
            desc = enum_choices.string[: enum_choices.start()].strip()
        schema["description"] = desc
    output = {"name": func.__name__, "description": main_doc, "parameters": json_schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


description_re = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
args_split_re = re.compile(
    r"(?:^|\n)\s*(\w+)\s*(?:\([^)]*?\))?:\s*(.*?)\s*(?=\n\s*\w+\s*(?:\([^)]*?\))?:|\Z)",
    re.DOTALL | re.VERBOSE,
)
returns_re = re.compile(r"\n\s*Returns:\n\s*(?:[^)]*?:\s*)?(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)


def _parse_google_format_docstring(docstring: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    description_match = description_re.search(docstring)
    args_match = args_re.search(docstring)
    returns_match = returns_re.search(docstring)
    description = description_match.group(1).strip() if description_match else None
    docstring_args = args_match.group(1).strip() if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None
    if docstring_args is not None:
        docstring_args = "\n".join([line for line in docstring_args.split("\n") if line.strip()])
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[0]: re.sub(r"\s*\n+\s*", " ", match[1].strip()) for match in matches}
    else:
        args_dict = {}
    return description, args_dict, returns


def _convert_type_hints_to_json_schema(func: Callable, error_on_missing_type_hints: bool = True) -> Dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)
    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty and error_on_missing_type_hints:
            raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param_name not in properties:
            properties[param_name] = {}
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["nullable"] = True
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parse_type_hint(hint: str) -> Dict:
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException("Couldn't parse this type hint: ", hint)
    elif origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        subtypes = [_parse_type_hint(t) for t in args if t is not type(None)]
        if len(subtypes) == 1:
            return_dict = subtypes[0]
        elif all(isinstance(subtype["type"], str) for subtype in subtypes):
            return_dict = {"type": sorted([subtype["type"] for subtype in subtypes])}
        else:
            return_dict = {"anyOf": subtypes}
        if type(None) in args:
            return_dict["nullable"] = True
        return return_dict
    elif origin is list:
        if not args:
            return {"type": "array"}
        return {"type": "array", "items": _parse_type_hint(args[0])}
    elif origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1:
            raise TypeHintParsingException(f"Single-element Tuple not supported: {hint}")
        if ... in args:
            raise TypeHintParsingException("Ellipsis in Tuple not supported. Use List[] instead.")
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}
    elif origin is dict:
        out = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out
    raise TypeHintParsingException("Couldn't parse this type hint: ", hint)


_BASE_TYPE_MAPPING = {
    int: {"type": "integer"}, float: {"type": "number"}, str: {"type": "string"},
    bool: {"type": "boolean"}, Any: {"type": "any"}, types.NoneType: {"type": "null"},
}

def _get_json_schema_type(param_type: str) -> Dict[str, str]:
    if param_type in _BASE_TYPE_MAPPING:
        return copy(_BASE_TYPE_MAPPING[param_type])
