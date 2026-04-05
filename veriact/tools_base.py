"""Tool base class and utilities for agent tool definitions."""

import ast
import inspect
import json
import logging
import sys
import types
from functools import wraps
from pathlib import Path
from typing import Dict, Union

from veriact._function_type_hints_utils import get_imports
from veriact.agent_types import handle_agent_input_types, handle_agent_output_types
from veriact.tool_validation import MethodChecker
from veriact.utility import get_source, instance_to_source, is_valid_name

logger = logging.getLogger(__name__)

AUTHORIZED_TYPES = ["string", "boolean", "integer", "number", "image", "audio", "array", "object", "any", "null"]
CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


def validate_after_init(cls):
    original_init = cls.__init__
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()
    cls.__init__ = new_init
    return cls


class Tool:
    """Base class for agent tools. Subclass and implement ``forward()``."""

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type, bool]]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required = {"description": str, "name": str, "inputs": dict, "output_type": str}
        for attr, expected in required.items():
            val = getattr(self, attr, None)
            if val is None:
                raise TypeError(f"You must set attribute {attr}.")
            if not isinstance(val, expected):
                raise TypeError(f"Attribute {attr} should be {expected.__name__}, got {type(val)}")
        if not is_valid_name(self.name):
            raise Exception(f"Invalid Tool name '{self.name}'")
        for iname, icontent in self.inputs.items():
            assert isinstance(icontent, dict), f"Input '{iname}' should be a dict."
            assert "type" in icontent and "description" in icontent
            if icontent["type"] not in AUTHORIZED_TYPES:
                raise Exception(f"Input '{iname}': type '{icontent['type']}' not in {AUTHORIZED_TYPES}")
        assert self.output_type in AUTHORIZED_TYPES

        if not (hasattr(self, "skip_forward_signature_validation") and self.skip_forward_signature_validation):
            sig = inspect.signature(self.forward)
            actual = set(k for k in sig.parameters if k != "self")
            expected_keys = set(self.inputs.keys())
            if actual != expected_keys:
                raise Exception(f"Tool '{self.name}': forward params {actual} != inputs {expected_keys}")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            if all(key in self.inputs for key in args[0]):
                args, kwargs = (), args[0]
        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        if sanitize_inputs_outputs:
            outputs = handle_agent_output_types(outputs, self.output_type)
        return outputs

    def setup(self):
        self.is_initialized = True

    def to_dict(self) -> dict:
        cls_name = self.__class__.__name__
        if cls_name == "SimpleTool":
            source_code = get_source(self.forward).replace("@tool", "")
            mc = MethodChecker(set())
            mc.visit(ast.parse(source_code))
            tool_code = "from typing import Any, Optional\n" + source_code
        else:
            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)
        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names}
        return {"name": self.name, "code": tool_code, "requirements": sorted(requirements)}

    def save(self, output_dir: str | Path, tool_file_name: str = "tool", make_gradio_app: bool = True):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / f"{tool_file_name}.py").write_text(self.to_dict()["code"], encoding="utf-8")

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        module = types.ModuleType("dynamic_tool")
        exec(tool_code, module.__dict__)
        tool_class = next((obj for _, obj in inspect.getmembers(module, inspect.isclass) if issubclass(obj, Tool) and obj is not Tool), None)
        if tool_class is None:
            raise ValueError("No Tool subclass found.")
        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)
        return tool_class(**kwargs)


