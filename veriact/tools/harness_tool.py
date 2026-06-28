"""
harness_tool.py
---------------
Self-contained module exposing ``evaluate_problem`` — a single entry-point
for evaluating one LLM-generated JML annotation against a benchmark Task.

All dependencies from harness.py, eval_spec.py, and eval_llm_response.py
are inlined here so this file can be used as a standalone tool.

Usage
-----
    from harness_tool import evaluate_problem, Task, TestCase

    task = Task(
        task_id="...",
        code="...",          # benchmark Solution.java source
        class_name="Solution",
        test_name="Test",
        javadoc="...",
        category="...",
        origin_id="...",
        test_code="...",     # benchmark Test.java source
        test_inputs=[TestCase(input="...", output="...")],
    )
    result = evaluate_problem(task, llm_code="...", openjml_path="openjml")
"""

from __future__ import annotations

import logging
import os
import re
import json
import subprocess
import textwrap
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from signal import signal
from typing import Any, Optional

import javalang
import javalang.tree as jtree

logger = logging.getLogger(__name__)



# Result types  (from harness.py)

class VerifyResult(Enum):
    OK = "ok"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class HarnessResult:
    metric: str
    total: int
    passed: int
    details: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.metric:25s}  score={self.score:.3f}"
            f"  ({self.passed}/{self.total})"
        )



# Test data structures  (from harness.py)

@dataclass
class TestPair:
    """One concrete input-output pair for postcondition evaluation."""

    inputs: dict[str, Any]  # {"paramName": pythonValue, …}
    output: Any  # expected return value
    label: str = ""


@dataclass
class InputCase:
    """One input case for precondition evaluation."""

    inputs: dict[str, Any]  # {"paramName": pythonValue, …}
    valid: bool  # True → T+  (must be admitted by precondition)
    # False → T-  (must be rejected by precondition)
    label: str = ""


@dataclass
class MethodSpec:
    """Holds the raw LLM-generated JML comment block, preserved as-is."""

    jml_block: str  # raw JML comment block (//@ or /*@ ... @*/) from LLM code
    auxiliaries: str = ""  # helper predicates / model methods (prepended to stub)

    @property
    def has_requires(self) -> bool:
        return bool(re.search(r"requires\b", self.jml_block))

    @property
    def has_ensures(self) -> bool:
        return bool(re.search(r"ensures\b", self.jml_block))

    # Back-compat aliases (deprecated)
    @property
    def precondition(self) -> Optional[str]:
        return self.jml_block if self.has_requires else None

    @property
    def postcondition(self) -> Optional[str]:
        return self.jml_block if self.has_ensures else None



# JType — javalang-based type descriptor  (from harness.py)

@dataclass
class JType:
    """
    Normalised Java type descriptor extracted from a javalang AST node.

    Examples
    --------
    int            → JType("int",     array_dims=0)
    int[]          → JType("int",     array_dims=1)
    int[][]        → JType("int",     array_dims=2)
    String         → JType("String",  array_dims=0)
    List<Integer>  → JType("List",    generic_args=["Integer"])
    Map<String,
        List<int>> → JType("Map",     generic_args=["String","List<int>"])
    """

    base: str
    array_dims: int = 0
    generic_args: list = field(default_factory=list)

    PRIMITIVES = frozenset(
        {"int", "long", "short", "byte", "double", "float", "boolean", "char"}
    )
    COLLECTIONS = frozenset(
        {
            "List",
            "ArrayList",
            "LinkedList",
            "Set",
            "HashSet",
            "TreeSet",
            "Map",
            "HashMap",
            "TreeMap",
        }
    )

    @property
    def is_primitive(self) -> bool:
        return self.base in self.PRIMITIVES

    @property
    def is_array(self) -> bool:
        return self.array_dims > 0

    @property
    def is_collection(self) -> bool:
        return self.base in self.COLLECTIONS

    def java_decl(self) -> str:
        """Full Java type declaration string, e.g. 'List<Integer>' or 'int[][]'."""
        if self.generic_args:
            args = ", ".join(self.generic_args)
            base = f"{self.base}<{args}>"
        else:
            base = self.base
        return base + "[]" * self.array_dims

    def element_type(self) -> "JType":
        """Return the element type for array types (strip one dimension)."""
        assert self.is_array, "element_type() called on non-array type"
        return JType(self.base, self.array_dims - 1, self.generic_args)



# Utility: parse a raw generic-arg string back to JType  (from harness.py)

def _split_generic_args(s: str) -> list[str]:
    """Split 'String, List<Integer>' respecting nested angle brackets."""
    depth, buf, result = 0, [], []
    for ch in s:
        if ch == "<":
            depth += 1
            buf.append(ch)
        elif ch == ">":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        result.append("".join(buf).strip())
    return result


def _parse_raw_jtype(raw: str) -> JType:
    """
    Minimal parser for strings like "Integer", "int[]", "List<String>",
    "Map<String, List<Integer>>".  Used when rendering nested generics.
    """
    raw = raw.strip()
    array_dims = 0
    while raw.endswith("[]"):
        array_dims += 1
        raw = raw[:-2]
    m = re.match(r"^(\w+)<(.+)>$", raw)
    if m:
        base = m.group(1)
        inner = _split_generic_args(m.group(2))
        return JType(base=base, array_dims=array_dims, generic_args=inner)
    return JType(base=raw, array_dims=array_dims)



# JavaMethodParser  (from harness.py)

class JavaMethodParser:
    """
    Parses a Java source string (class or bare method) using javalang
    and returns structured parameter / return type information.
    """

    def parse(self, source: str) -> dict:
        wrapped, class_name = self._ensure_class(source)
        try:
            tree = javalang.parse.parse(wrapped)
        except javalang.parser.JavaSyntaxError as e:
            raise ValueError(f"javalang parse error: {e}") from e

        for _, node in tree:
            if isinstance(node, jtree.MethodDeclaration):
                return {
                    "class_name": class_name,
                    "method_name": node.name,
                    "return_type": self._jtype(node.return_type),
                    "params": [
                        {"name": p.name, "jtype": self._jtype(p.type)}
                        for p in (node.parameters or [])
                    ],
                }
        raise ValueError("No MethodDeclaration found in the provided source.")

    @staticmethod
    def _ensure_class(source: str) -> tuple[str, str]:
        source = source.strip()
        m = re.search(r"\bclass\s+(\w+)", source)
        if m:
            return source, m.group(1)
        return f"class _Harness {{ {source} }}", "_Harness"

    def _jtype(self, node) -> JType:
        if node is None:
            return JType(base="void")
        if isinstance(node, jtree.BasicType):
            dims = len(node.dimensions) if node.dimensions else 0
            return JType(base=node.name, array_dims=dims)
        if isinstance(node, jtree.ReferenceType):
            dims = len(node.dimensions) if node.dimensions else 0
            generic_args = self._generic_args(node)
            return JType(base=node.name, array_dims=dims, generic_args=generic_args)
        return JType(base=str(node))

    def _generic_args(self, node: jtree.ReferenceType) -> list[str]:
        if not getattr(node, "arguments", None):
            return []
        result = []
        for arg in node.arguments:
            t = getattr(arg, "type", None)
            if t is not None:
                result.append(self._jtype(t).java_decl())
            elif hasattr(arg, "name"):
                result.append(arg.name)
        return result



# JavaLiteralRenderer  (from harness.py)

class JavaLiteralRenderer:
    """
    Converts a Python value to a syntactically valid Java literal /
    initialiser expression, guided by a JType descriptor.
    """

    def render(self, jtype: JType, value: Any) -> str:
        if value is None:
            return "null"
        if jtype.is_array:
            return self._array(jtype, value)
        if jtype.is_collection:
            return self._collection(jtype, value)
        return self._scalar(jtype.base, value)

    def _array(self, jtype: JType, value: Any) -> str:
        elem_type = jtype.element_type()
        if jtype.array_dims == 1:
            elems = ", ".join(self._scalar(jtype.base, v) for v in value)
            return f"new {jtype.base}[]{{{elems}}}"
        rows = ", ".join(self._array(elem_type, row) for row in value)
        inner = jtype.base + "[]" * (jtype.array_dims - 1)
        return f"new {inner}[]{{{rows}}}"

    def _collection(self, jtype: JType, value: Any) -> str:
        base = jtype.base
        if base in ("Map", "HashMap", "TreeMap"):
            return self._map(jtype, value)
        if base in ("Set", "HashSet", "TreeSet"):
            return self._set(jtype, value)
        return self._list(jtype, value)

    def _list(self, jtype: JType, value: list) -> str:
        elem = self._arg_type(jtype, 0)
        elems = ", ".join(self.render(elem, v) for v in value)
        concrete = "LinkedList" if jtype.base == "LinkedList" else "ArrayList"
        return f"new java.util.{concrete}<>(java.util.Arrays.asList({elems}))"

    def _set(self, jtype: JType, value) -> str:
        elem = self._arg_type(jtype, 0)
        elems = ", ".join(self.render(elem, v) for v in value)
        concrete = "TreeSet" if jtype.base == "TreeSet" else "HashSet"
        return f"new java.util.{concrete}<>(java.util.Arrays.asList({elems}))"

    def _map(self, jtype: JType, value: dict) -> str:
        k_type = self._arg_type(jtype, 0)
        v_type = self._arg_type(jtype, 1)
        concrete = "TreeMap" if jtype.base == "TreeMap" else "HashMap"
        entries = "; ".join(
            f"_m.put({self.render(k_type, k)}, {self.render(v_type, v)})"
            for k, v in value.items()
        )
        decl = jtype.java_decl()
        return f"new java.util.{concrete}<{decl}>() {{{{ {entries}; }}}}"

    def _scalar(self, base: str, value: Any) -> str:
        if base in ("int", "Integer", "short", "Short", "byte", "Byte"):
            return str(int(value))
        if base in ("long", "Long"):
            return f"{int(value)}L"
        if base in ("double", "Double"):
            return f"{float(value)}d"
        if base in ("float", "Float"):
            return f"{float(value)}f"
        if base in ("boolean", "Boolean"):
            return "true" if value else "false"
        if base in ("char", "Character"):
            return f"'{value}'"
        if base == "String":
            esc = str(value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{esc}"'
        return str(value)

    def _arg_type(self, jtype: JType, idx: int) -> JType:
        if jtype.generic_args and idx < len(jtype.generic_args):
            return _parse_raw_jtype(jtype.generic_args[idx])
        return JType(base="Object")



# StubBuilder  (from harness.py)

class StubBuilder:
    """
    Constructs Java harness stubs for each of the four Hoare triples.
    """

    def __init__(self):
        self._renderer = JavaLiteralRenderer()

    def post_correctness_stub(
        self, parsed: dict, spec: MethodSpec, pair: TestPair
    ) -> str:
        assumes = self._input_assumes(parsed, pair.inputs)
        ret = self._return_stmt(parsed, pair.output)
        return self._wrap_post(parsed, spec, assumes, ret)

    def post_completeness_stub(
        self, parsed: dict, spec: MethodSpec, pair: TestPair, mutated_output: Any
    ) -> str:
        assumes = self._input_assumes(parsed, pair.inputs)
        ret = self._return_stmt(parsed, mutated_output)
        return self._wrap_post(parsed, spec, assumes, ret)

    def pre_correctness_stub(
        self,
        parsed: dict,
        spec: MethodSpec,
        case: InputCase,
        original_source: str = "",
    ) -> str:
        if original_source:
            return self._wrap_caller(parsed, spec, original_source, case.inputs)
        assumes = self._input_assumes(parsed, case.inputs)
        return self._wrap_pre(parsed, spec, assumes)

    def pre_completeness_stub(
        self,
        parsed: dict,
        spec: MethodSpec,
        case: InputCase,
        original_source: str = "",
    ) -> str:
        if original_source:
            return self._wrap_caller(parsed, spec, original_source, case.inputs)
        assumes = self._input_assumes(parsed, case.inputs)
        return self._wrap_pre(parsed, spec, assumes)

    def _input_assumes(self, parsed: dict, inputs: dict[str, Any]) -> str:
        lines: list[str] = []
        for p in parsed["params"]:
            val = inputs[p["name"]]
            lines.extend(self._assume_for(p["name"], p["jtype"], val))
        return "\n".join(lines)

    def _assume_for(self, name: str, jtype: JType, value: Any) -> list[str]:
        I = "        "
        if value is None:
            return [f"{I}//@ assume {name} == null;"]
        if jtype.is_array:
            return self._assume_array(name, jtype, value)
        if jtype.is_collection:
            return self._assume_collection(name, jtype, value)
        if jtype.base == "String":
            lit = self._renderer._scalar("String", value)
            return [f"{I}//@ assume {name} != null && {name}.equals({lit});"]
        lit = self._renderer._scalar(jtype.base, value)
        return [f"{I}//@ assume {name} == {lit};"]

    def _assume_array(self, name: str, jtype: JType, value: list) -> list[str]:
        I = "        "
        lines = [
            f"{I}//@ assume {name} != null;",
            f"{I}//@ assume {name}.length == {len(value)};",
        ]
        if jtype.array_dims == 1:
            for i, v in enumerate(value):
                lit = self._renderer._scalar(jtype.base, v)
                lines.append(f"{I}//@ assume {name}[{i}] == {lit};")
        elif jtype.array_dims >= 2:
            elem = jtype.element_type()
            for i, row in enumerate(value):
                lines.append(f"{I}//@ assume {name}[{i}] != null;")
                lines.append(f"{I}//@ assume {name}[{i}].length == {len(row)};")
                if elem.array_dims == 0:
                    for j, v in enumerate(row):
                        lit = self._renderer._scalar(jtype.base, v)
                        lines.append(f"{I}//@ assume {name}[{i}][{j}] == {lit};")
        return lines

    def _assume_collection(
        self, name: str, jtype: JType, value: Any
    ) -> list[str]:
        I = "        "
        if value is None:
            return [f"{I}//@ assume {name} == null;"]
        lines = [f"{I}//@ assume {name} != null;"]
        if isinstance(value, dict):
            lines.append(f"{I}//@ assume {name}.size() == {len(value)};")
        elif isinstance(value, (list, set)):
            lines.append(f"{I}//@ assume {name}.size() == {len(value)};")
        return lines

    def _return_stmt(self, parsed: dict, value: Any) -> str:
        rt = parsed["return_type"]
        lit = self._renderer.render(rt, value)
        return f"        return {lit};"

    def _wrap_post(
        self, parsed: dict, spec: MethodSpec, assumes: str, ret_stmt: str
    ) -> str:
        params_decl = ", ".join(
            f"{p['jtype'].java_decl()} {p['name']}" for p in parsed["params"]
        )
        ret = parsed["return_type"].java_decl()
        mname = parsed["method_name"]
        cname = parsed["class_name"]
        aux = spec.auxiliaries or ""
        jml = self._indent_jml(spec.jml_block, "    ")
        return (
            f"{aux}\n"
            f"//@ nullable_by_default\n"
            f"public class {cname}Harness {{\n"
            f"{jml}\n"
            f"    public static {ret} {mname}({params_decl}) {{\n"
            f"{assumes}\n"
            f"{ret_stmt}\n"
            f"    }}\n"
            f"}}\n"
        )

    def _wrap_pre(self, parsed: dict, spec: MethodSpec, assumes: str) -> str:
        params_decl = ", ".join(
            f"{p['jtype'].java_decl()} {p['name']}" for p in parsed["params"]
        )
        mname = parsed["method_name"]
        cname = parsed["class_name"]
        aux = spec.auxiliaries or ""
        jml = self._indent_jml(spec.jml_block, "    ")
        return (
            f"{aux}\n"
            f"//@ nullable_by_default\n"
            f"public class {cname}Harness {{\n"
            f"{jml}\n"
            f"    public static void {mname}({params_decl}) {{\n"
            f"{assumes}\n"
            f"    }}\n"
            f"}}\n"
        )

    def _wrap_caller(
        self,
        parsed: dict,
        spec: MethodSpec,
        original_source: str,
        inputs: dict[str, Any],
    ) -> str:
        mname = parsed["method_name"]
        cname = parsed["class_name"]
        call_args = ", ".join(
            self._renderer.render(p["jtype"], inputs[p["name"]])
            for p in parsed["params"]
        )
        src = re.sub(r"\bpublic\s+class\b", "class", original_source, count=1)
        return (
            f"//@ nullable_by_default\n"
            f"{src}\n\n"
            f"public class {cname}Harness {{\n"
            f"    public static void check() {{\n"
            f"        {cname}.{mname}({call_args});\n"
            f"    }}\n"
            f"}}\n"
        )

    @staticmethod
    def _indent_jml(jml_block: str, indent: str) -> str:
        return "\n".join(
            indent + line.strip()
            for line in jml_block.splitlines()
            if line.strip()
        )



# OutputMutator  (from harness.py)

class OutputMutator:
    """
    Generates up to k output mutants for PostCompleteness.
    """

    def __init__(self, k: int = 5):
        self.k = k

    def mutate(self, jtype: JType, value: Any) -> list[Any]:
        if jtype.is_array:
            return self._mutate_array(list(value) if value else [])
        if jtype.is_collection:
            return self._mutate_collection(jtype, value)
        return self._mutate_scalar(jtype.base, value)

    def _mutate_scalar(self, base: str, v: Any) -> list:
        if base in ("int", "long", "short", "byte", "Integer", "Long"):
            v = int(v)
            return list(dict.fromkeys([v + 1, v - 1, v + 2, v - 2, 0]))[: self.k]
        if base in ("double", "float", "Double", "Float"):
            v = float(v)
            return [v + 1.0, v - 1.0, v * 2.0, -v][: self.k]
        if base in ("boolean", "Boolean"):
            return [not v]
        if base == "String":
            return [v + "_x", (v[:-1] if v else "x"), v.upper(), "", v[::-1]][: self.k]
        if base in ("char", "Character"):
            c = ord(v)
            return [chr(c + 1), chr(c - 1)][: self.k]
        return []

    def _mutate_array(self, v: list) -> list:
        mutants: list = []
        if v:
            mutants.append(v[:-1])
            last = v[-1]
            if not isinstance(last, list):
                mutants.append(v + [last + 1])
            if all(not isinstance(x, list) for x in v):
                mutants.append([x + 1 for x in v])
        mutants.append([])
        return [m for m in mutants if m != v][: self.k]

    def _mutate_collection(self, jtype: JType, v: Any) -> list:
        if jtype.base in ("Map", "HashMap", "TreeMap"):
            d = dict(v) if v else {}
            mutants: list = []
            if d:
                k0 = next(iter(d))
                mutants.append({k: val for k, val in d.items() if k != k0})
            mutants.append({})
            return mutants[: self.k]
        return self._mutate_array(list(v) if v else [])



# eval_spec helpers  (from eval_spec.py)

def _find_method_line(source: str) -> int | None:
    lines = source.splitlines()
    leading_blanks = 0
    for line in lines:
        if line.strip():
            break
        leading_blanks += 1

    stripped = source.strip()
    has_class = bool(re.search(r"\bclass\s+\w+", stripped))
    if has_class:
        to_parse = stripped
        offset = leading_blanks
    else:
        to_parse = f"class _Wrapper {{\n{stripped}\n}}"
        offset = leading_blanks - 1

    try:
        tree = javalang.parse.parse(to_parse)
    except javalang.parser.JavaSyntaxError:
        return None

    for _, node in tree:
        if isinstance(node, jtree.MethodDeclaration) and node.position:
            return node.position.line + offset
    return None


def extract_jml_spec(source: str) -> MethodSpec:
    """
    Extract the raw JML comment block from above the method declaration.
    """
    lines = source.splitlines()
    method_line = _find_method_line(source)

    if method_line is not None:
        end = method_line - 2
        while end >= 0:
            s = lines[end].strip()
            if (
                s.startswith("//@")
                or s.startswith("/*@")
                or s.endswith("@*/")
                or s.endswith("*/")
            ):
                break
            if not s or s.startswith("@"):
                end -= 1
            else:
                break

        start = end
        in_block_comment = False
        while start >= 0:
            s = lines[start].strip()
            if s.endswith("@*/") or s.endswith("*/"):
                in_block_comment = True
            if in_block_comment:
                if s.startswith("/*@") or s.startswith("/*"):
                    in_block_comment = False
                start -= 1
                continue
            if s.startswith("//@"):
                start -= 1
                continue
            break
        jml_block = "\n".join(lines[start + 1 : end + 1])
    else:
        jml_lines = [
            l
            for l in lines
            if l.strip().startswith("//@") or l.strip().startswith("/*@")
        ]
        jml_block = "\n".join(jml_lines)

    return MethodSpec(jml_block=jml_block)


@dataclass
class ReadOp:
    """One input-reading operation derived from Test.java."""

    name: str
    jtype: JType
    mode: str  # scalar | grouped_scalar | string | array_direct | array_sized |
    # array_2d | array_2d_rc | map | set | list_direct
    group_idx: int = -1


def _walk_ast(node):
    if node is None or not isinstance(node, jtree.Node):
        return
    yield node
    for attr_name in node.attrs:
        child = getattr(node, attr_name, None)
        if child is None:
            continue
        if isinstance(child, jtree.Node):
            yield from _walk_ast(child)
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, jtree.Node):
                    yield from _walk_ast(item)


def _is_string_array_decl(type_node, declarator) -> bool:
    if not isinstance(type_node, jtree.ReferenceType):
        return False
    if type_node.name != "String":
        return False
    if type_node.dimensions and len(type_node.dimensions) > 0:
        return True
    if declarator.dimensions and len(declarator.dimensions) > 0:
        return True
    return False


def _ast_has_method_call(node, method_name: str) -> bool:
    for n in _walk_ast(node):
        if isinstance(n, jtree.MethodInvocation) and n.member == method_name:
            return True
    return False


def _ast_find_member_ref(node) -> str | None:
    for n in _walk_ast(node):
        if isinstance(n, jtree.MemberReference) and not n.selectors:
            return n.member
    return None


def _ast_find_array_access(node) -> tuple[str, int] | None:
    for n in _walk_ast(node):
        if isinstance(n, jtree.MemberReference) and n.selectors:
            for sel in n.selectors:
                if isinstance(sel, jtree.ArraySelector):
                    idx_node = sel.index
                    if isinstance(idx_node, jtree.Literal):
                        try:
                            return (n.member, int(idx_node.value))
                        except (ValueError, TypeError):
                            pass
    return None


def _javalang_detect(
    test_src: str, all_param_names: set[str], params: list[dict]
) -> tuple[dict[str, int], bool, bool, dict[str, str]]:
    grouped_indices: dict[str, int] = {}
    has_2d_rc = False
    has_sized_1d = False
    input_order_map: dict[str, str] = {}

    parseable = re.sub(r"\bvar\s+(\w+)\s*=", r"Object \1 =", test_src)
    tree = javalang.parse.parse(parseable)

    main_body = None
    for _, node in tree:
        if isinstance(node, jtree.MethodDeclaration) and node.name == "main":
            main_body = node.body
            break
    if not main_body:
        return grouped_indices, has_2d_rc, has_sized_1d, input_order_map

    split_vars: set[str] = set()
    rc_vars: set[str] = set()

    for stmt in main_body:
        if not isinstance(stmt, jtree.LocalVariableDeclaration):
            continue
        for decl in stmt.declarators:
            vname = decl.name
            init = decl.initializer

            if (
                _is_string_array_decl(stmt.type, decl)
                and init is not None
                and _ast_has_method_call(init, "split")
            ):
                split_vars.add(vname)
                if vname.startswith(("_rc", "rc")):
                    rc_vars.add(vname)
                    has_2d_rc = True
                continue

            if vname in all_param_names and init is not None:
                arr_access = _ast_find_array_access(init)
                if arr_access is not None:
                    arr_name, idx = arr_access
                    if arr_name in split_vars and arr_name not in rc_vars:
                        grouped_indices[vname] = idx

    non_rc_splits = split_vars - rc_vars
    check_inline = bool(non_rc_splits) and not grouped_indices
    for stmt in main_body:
        for n in _walk_ast(stmt):
            if not (
                isinstance(n, jtree.MethodInvocation)
                and n.member == "solve"
                and n.arguments
            ):
                continue
            for arg_idx, arg in enumerate(n.arguments):
                if arg_idx >= len(params):
                    break
                sol_name = params[arg_idx]["name"]

                if check_inline:
                    arr_access = _ast_find_array_access(arg)
                    if arr_access is not None and arr_access[0] in non_rc_splits:
                        grouped_indices[sol_name] = arr_access[1]

                ref = _ast_find_member_ref(arg)
                if ref is not None and ref in all_param_names:
                    input_order_map[sol_name] = ref

    return grouped_indices, has_2d_rc, has_sized_1d, input_order_map


def _is_nested_collection(inner_type_str: str) -> bool:
    return inner_type_str.startswith(
        ("List", "ArrayList", "LinkedList")
    ) or inner_type_str.endswith("[]")


def _leaf_element_type(jtype: JType) -> str:
    if jtype.is_array:
        return jtype.base
    if jtype.is_collection and jtype.generic_args:
        inner = jtype.generic_args[0]
        m = re.match(r"^(\w+)<(.+)>$", inner)
        if m:
            inner_base = m.group(1)
            if inner_base in (
                "List",
                "ArrayList",
                "LinkedList",
                "Set",
                "HashSet",
                "TreeSet",
            ):
                return m.group(2).strip()
            return inner
        if inner.endswith("[]"):
            return inner[:-2]
        return inner
    return jtype.base


def _regex_fallback(
    test_src: str, all_param_names: set[str]
) -> tuple[dict[str, int], bool, bool]:
    grouped_indices: dict[str, int] = {}

    split_var_names = re.findall(
        r"String\[\]\s+(\w+)\s*=\s*scanner\."
        r"(?:hasNextLine\s*\(\)\s*\?\s*)?nextLine\(\).*?split\s*\(",
        test_src,
    )
    for sv in split_var_names:
        if re.match(r"_?rc|_?pr|_raw_", sv):
            continue
        for m in re.finditer(
            rf"(\w+)\s*=\s*\w+\.parse\w+\(\s*{re.escape(sv)}"
            rf"\s*\[\s*(\d+)\s*\]\s*\)",
            test_src,
        ):
            grouped_indices[m.group(1)] = int(m.group(2))
        for m in re.finditer(
            rf"(\w+)\s*=\s*{re.escape(sv)}"
            rf"\s*\[\s*(\d+)\s*\]\.charAt\s*\(\s*0\s*\)",
            test_src,
        ):
            grouped_indices[m.group(1)] = int(m.group(2))
        for m in re.finditer(
            rf"(?:String\s+)?(\w+)\s*=\s*{re.escape(sv)}" rf"\s*\[\s*(\d+)\s*\]\s*;",
            test_src,
        ):
            if m.group(1) in all_param_names:
                grouped_indices[m.group(1)] = int(m.group(2))

    has_2d_rc = bool(
        re.search(
            r"_?rc[\w]*\s*=\s*scanner\." r"(?:hasNextLine\s*\(\)\s*\?\s*)?nextLine",
            test_src,
        )
    )
    has_sized_1d = bool(
        re.search(r"int\s+n\d+\s*=\s*Integer\.parseInt\(scanner\.nextLine", test_src)
    )

    return grouped_indices, has_2d_rc, has_sized_1d


def detect_input_format(test_src: str, params: list[dict]) -> list[ReadOp]:
    """
    Analyse machine-generated Test.java to decide how each parameter is
    read from stdin.
    """
    ops: list[ReadOp] = []

    param_names = {p["name"] for p in params}
    positional_names = {f"p{i}" for i in range(len(params))}
    all_param_names = param_names | positional_names

    input_order_map: dict[str, str] = {}
    try:
        grouped_indices, has_2d_rc, has_sized_1d, input_order_map = _javalang_detect(
            test_src, all_param_names, params
        )
    except Exception:
        grouped_indices, has_2d_rc, has_sized_1d = _regex_fallback(
            test_src, all_param_names
        )

    ordered_params = list(params)
    if input_order_map:
        pn_to_sol: dict[str, dict] = {}
        for sol_name, pn_name in input_order_map.items():
            for p in params:
                if p["name"] == sol_name:
                    pn_to_sol[pn_name] = p
                    break
        if len(pn_to_sol) == len(params):
            ordered_params = [
                pn_to_sol[f"p{i}"] for i in range(len(params)) if f"p{i}" in pn_to_sol
            ]
            if len(ordered_params) != len(params):
                ordered_params = list(params)

    for pi, p in enumerate(ordered_params):
        jtype = p["jtype"]
        name = p["name"]
        tvar = f"p{pi}"

        if jtype.base in ("HashMap", "Map", "TreeMap"):
            ops.append(ReadOp(name, jtype, "map"))
            continue
        if jtype.base in ("HashSet", "Set", "TreeSet"):
            ops.append(ReadOp(name, jtype, "set"))
            continue
        if jtype.base in ("List", "ArrayList", "LinkedList"):
            if jtype.generic_args and _is_nested_collection(jtype.generic_args[0]):
                mode = "array_2d_rc" if has_2d_rc else "array_2d"
                ops.append(ReadOp(name, jtype, mode))
            else:
                ops.append(ReadOp(name, jtype, "list_direct"))
            continue
        if jtype.is_array and jtype.array_dims >= 2:
            mode = "array_2d_rc" if has_2d_rc else "array_2d"
            ops.append(ReadOp(name, jtype, mode))
            continue
        if jtype.is_array:
            mode = "array_sized" if has_sized_1d else "array_direct"
            ops.append(ReadOp(name, jtype, mode))
            continue
        if jtype.base == "String":
            ops.append(ReadOp(name, jtype, "string"))
            continue

        gidx = grouped_indices.get(name, grouped_indices.get(tvar, -1))
        if gidx >= 0:
            ops.append(ReadOp(name, jtype, "grouped_scalar", group_idx=gidx))
        else:
            ops.append(ReadOp(name, jtype, "scalar"))

    return ops


def _parse_scalar(type_name: str, s: str) -> Any:
    s = s.strip()
    if type_name in ("int", "Integer", "short", "Short", "byte", "Byte"):
        return int(s)
    if type_name in ("long", "Long"):
        return int(s)
    if type_name in ("double", "Double"):
        return float(s)
    if type_name in ("float", "Float"):
        return float(s)
    if type_name in ("boolean", "Boolean"):
        return s.lower() == "true"
    if type_name in ("char", "Character"):
        return s
    return s


def parse_input(input_str: str, ops: list[ReadOp]) -> dict[str, Any]:
    lines = input_str.split("\n")
    pos = 0
    result: dict[str, Any] = {}

    i = 0
    while i < len(ops):
        op = ops[i]

        if op.mode == "grouped_scalar":
            parts = lines[pos].strip().split()
            pos += 1
            j = i
            while j < len(ops) and ops[j].mode == "grouped_scalar":
                g = ops[j]
                result[g.name] = _parse_scalar(g.jtype.base, parts[g.group_idx])
                j += 1
            i = j
            continue

        if op.mode == "scalar":
            result[op.name] = _parse_scalar(op.jtype.base, lines[pos].strip())
            pos += 1
        elif op.mode == "string":
            result[op.name] = lines[pos]
            pos += 1
        elif op.mode == "array_direct":
            parts = lines[pos].strip().split()
            result[op.name] = [_parse_scalar(op.jtype.base, p) for p in parts]
            pos += 1
        elif op.mode == "array_sized":
            n = int(lines[pos].strip())
            pos += 1
            if n > 0:
                parts = lines[pos].strip().split()
                result[op.name] = [_parse_scalar(op.jtype.base, p) for p in parts]
                pos += 1
            else:
                result[op.name] = []
        elif op.mode == "array_2d":
            elem = _leaf_element_type(op.jtype)
            n_rows = int(lines[pos].strip())
            pos += 1
            mat = []
            for _ in range(n_rows):
                if pos < len(lines) and lines[pos].strip():
                    row = [_parse_scalar(elem, x) for x in lines[pos].strip().split()]
                    mat.append(row)
                    pos += 1
                else:
                    mat.append([])
            result[op.name] = mat
        elif op.mode == "array_2d_rc":
            elem = _leaf_element_type(op.jtype)
            rc = lines[pos].strip().split()
            n_rows, n_cols = int(rc[0]), int(rc[1])
            pos += 1
            mat = []
            if n_cols == 0:
                mat = [[] for _ in range(n_rows)]
            else:
                for _ in range(n_rows):
                    if pos < len(lines):
                        row = [
                            _parse_scalar(elem, x) for x in lines[pos].strip().split()
                        ]
                        mat.append(row)
                        pos += 1
                    else:
                        mat.append([])
            result[op.name] = mat
        elif op.mode == "map":
            n = int(lines[pos].strip())
            pos += 1
            d: dict = {}
            ktype = op.jtype.generic_args[0] if op.jtype.generic_args else "String"
            vtype = (
                op.jtype.generic_args[1]
                if len(op.jtype.generic_args) > 1
                else "Integer"
            )
            for _ in range(n):
                parts = lines[pos].split("\t")
                d[_parse_scalar(ktype, parts[0])] = _parse_scalar(
                    vtype, parts[1].strip()
                )
                pos += 1
            result[op.name] = d
        elif op.mode == "set":
            n = int(lines[pos].strip())
            pos += 1
            etype = op.jtype.generic_args[0] if op.jtype.generic_args else "Integer"
            parts = lines[pos].strip().split()
            result[op.name] = [_parse_scalar(etype, p) for p in parts]
            pos += 1
        elif op.mode == "list_direct":
            etype = op.jtype.generic_args[0] if op.jtype.generic_args else "Integer"
            line = lines[pos].strip()
            if line:
                parts = line.split()
                result[op.name] = [_parse_scalar(etype, p) for p in parts]
            else:
                result[op.name] = []
            pos += 1

        i += 1

    return result


def parse_output(output_str: str, return_type: JType) -> Any:
    output_str = output_str.rstrip("\n")

    if return_type.base == "void":
        return None

    if return_type.base in ("HashMap", "Map", "TreeMap"):
        lines = output_str.split("\n")
        count = int(lines[0].strip())
        ktype = return_type.generic_args[0] if return_type.generic_args else "String"
        vtype = (
            return_type.generic_args[1]
            if len(return_type.generic_args) > 1
            else "Integer"
        )
        d: dict = {}
        for j in range(1, count + 1):
            parts = lines[j].split("\t")
            d[_parse_scalar(ktype, parts[0])] = _parse_scalar(vtype, parts[1].strip())
        return d

    if return_type.is_array and return_type.array_dims >= 2:
        rows = []
        for line in output_str.split("\n"):
            line = line.strip()
            if line:
                row = [_parse_scalar(return_type.base, x) for x in line.split()]
                rows.append(row)
        return rows

    if return_type.is_array:
        tokens = output_str.strip().split()
        if not tokens or tokens == [""]:
            return []
        return [_parse_scalar(return_type.base, t) for t in tokens]

    if return_type.base in ("List", "ArrayList", "LinkedList"):
        tokens = output_str.strip().split()
        etype = return_type.generic_args[0] if return_type.generic_args else "Integer"
        if not tokens or tokens == [""]:
            return []
        return [_parse_scalar(etype, t) for t in tokens]

    if return_type.base in ("Set", "HashSet", "TreeSet"):
        tokens = output_str.strip().split()
        etype = return_type.generic_args[0] if return_type.generic_args else "Integer"
        if not tokens or tokens == [""]:
            return []
        return [_parse_scalar(etype, t) for t in tokens]

    return _parse_scalar(return_type.base, output_str.strip())


def generate_invalid_inputs(
    params: list[dict], valid_inputs: list[dict[str, Any]]
) -> list[InputCase]:
    """
    Auto-generate boundary / invalid input cases (T-) for PreCompleteness.
    """
    if not valid_inputs:
        return []

    base = dict(valid_inputs[0])
    invalid: list[InputCase] = []

    for p in params:
        jtype = p["jtype"]
        name = p["name"]

        is_ref = not jtype.is_primitive or jtype.is_array or jtype.is_collection
        if is_ref:
            case = dict(base)
            case[name] = None
            invalid.append(InputCase(case, False, f"null_{name}"))

        if jtype.is_array or jtype.is_collection:
            case = dict(base)
            case[name] = {} if jtype.base in ("HashMap", "Map", "TreeMap") else []
            invalid.append(InputCase(case, False, f"empty_{name}"))

        if (
            jtype.is_array
            and isinstance(base.get(name), list)
            and len(base.get(name, [])) > 1
        ):
            case = dict(base)
            case[name] = [base[name][0]]
            invalid.append(InputCase(case, False, f"single_{name}"))

        if (
            not jtype.is_array
            and not jtype.is_collection
            and jtype.base
            in ("int", "Integer", "long", "Long", "short", "Short", "byte", "Byte")
        ):
            case = dict(base)
            case[name] = -1
            invalid.append(InputCase(case, False, f"neg_{name}"))

    return invalid



# Task / TestCase data classes  (from eval_llm_response.py)

@dataclass
class TestCase:
    input: str
    output: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "TestCase":
        return cls(input=data["input"], output=data["output"])


@dataclass
class Task:
    task_id: str
    code: str
    class_name: str
    test_name: str
    javadoc: str
    category: str
    origin_id: str
    test_code: str = ""
    test_inputs: list[TestCase] = field(default_factory=list)
    generated_test_cases: list[TestCase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"],
            code=data["code"],
            class_name=data["class_name"],
            test_name=data["test_name"],
            javadoc=data.get("javadoc", ""),
            category=data.get("category", ""),
            origin_id=data.get("origin_id", ""),
            test_code=data.get("test_code", ""),
            test_inputs=[TestCase.from_dict(tc) for tc in data.get("test_inputs", [])],
            generated_test_cases=[
                TestCase.from_dict(tc)
                for tc in data.get("generated_test_cases", [])
            ],
        )



# OpenJMLRunnerPersistent  (from eval_llm_response.py)

class OpenJMLRunnerPersistent:
    """Runs OpenJML ESC, writing harness stubs to an actual directory."""

    def __init__(self, openjml_path: str, output_dir: str, timeout: int = 300):
        self.openjml_path = openjml_path
        self.output_dir = output_dir
        self.timeout = timeout
        os.makedirs(output_dir, exist_ok=True)

    def verify(
        self, java_source: str, class_name: str, label: str = ""
    ) -> tuple[VerifyResult, str]:
        harness_class = f"{class_name}Harness"
        fname = f"{harness_class}.java"
        if label:
            safe_label = re.sub(r"[^\w\-]", "_", label)
            sub_dir = os.path.join(self.output_dir, safe_label)
            os.makedirs(sub_dir, exist_ok=True)
            path = os.path.join(sub_dir, fname)
        else:
            path = os.path.join(self.output_dir, fname)
        with open(path, "w") as f:
            f.write(java_source)
        cmd = [
            "openjml",
            "--esc",
            "--esc-max-warnings",
            "1",
            "--prover=cvc4",
            "--nonnull-by-default",
            "--arithmetic-failure=quiet",
            "-nowarn",
            path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
            out = stdout + stderr
            return self._parse(proc.returncode, out), out
        except subprocess.TimeoutExpired:
            return VerifyResult.UNKNOWN, "timeout"
        except FileNotFoundError:
            raise RuntimeError(f"OpenJML binary not found at '{self.openjml_path}'.")
        except Exception as e:
            return VerifyResult.UNKNOWN, f"Error: {str(e)}"
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    def _parse(returncode: int, output: str) -> VerifyResult:
        low = output.strip().lower()
        if re.search(r"[1-9]\d* warning", low):
            return VerifyResult.FAIL
        if re.search(r"[1-9]\d* verification failure", low):
            return VerifyResult.FAIL
        if re.search(r"\berror\b", low):
            return VerifyResult.FAIL
        if returncode == 0 or "verified" in low or "0 warnings" in low or low == "":
            return VerifyResult.OK
        return VerifyResult.UNKNOWN



# PairResult  (from eval_llm_response.py)

@dataclass
class PairResult:
    """All 4 metric results for a single test pair."""

    pair_idx: int
    post_correct_detail: dict = field(default_factory=dict)
    post_complete_details: list[dict] = field(default_factory=list)
    pre_correct_detail: dict = field(default_factory=dict)
    pre_complete_details: list[dict] = field(default_factory=list)



# Inner thread helper  (from eval_llm_response.py)

def _log(
    verbose: bool,
    task_id: str,
    metric: str,
    label: str,
    verdict: VerifyResult,
    desired: bool,
    extra: str = "",
) -> None:
    if not verbose:
        return
    status = "pass" if desired else "fail"
    ex = f" [{extra}]" if extra else ""
    logger.debug(f"[{task_id}] [{metric}] {label}{ex} {verdict.value} {status}")


def _evaluate_one_pair(
    pair_idx: int,
    pair: TestPair,
    valid_input_case: InputCase,
    invalid_cases: list[InputCase],
    parsed: dict,
    spec: MethodSpec,
    rtype: JType,
    builder: StubBuilder,
    mutator: OutputMutator,
    runner: OpenJMLRunnerPersistent,
    cname: str,
    verbose: bool,
    task_id: str,
    java_source: str = "",
) -> PairResult:
    """Compute all 4 metrics for a single test pair in its own thread."""
    result = PairResult(pair_idx=pair_idx)

    # PostCorrectness
    if spec.postcondition is not None:
        stub = builder.post_correctness_stub(parsed, spec, pair)
        verdict, _ = runner.verify(stub, cname, f"post_correct_{pair.label}")
        ok = verdict == VerifyResult.OK
        result.post_correct_detail = {
            "label": pair.label,
            "verdict": verdict.value,
            "pass": ok,
            "stub": stub,
        }
        _log(verbose, task_id, "PostCorrectness", pair.label, verdict, ok)

    # PostCompleteness 
    if spec.postcondition is not None:
        for mut_out in mutator.mutate(rtype, pair.output):
            stub = builder.post_completeness_stub(parsed, spec, pair, mut_out)
            lbl = f"post_complete_{pair.label}_{str(mut_out)[:20]}"
            verdict, _ = runner.verify(stub, cname, lbl)
            killed = verdict == VerifyResult.FAIL
            result.post_complete_details.append(
                {
                    "label": pair.label,
                    "mutant": str(mut_out),
                    "verdict": verdict.value,
                    "killed": killed,
                    "stub": stub,
                }
            )
            _log(
                verbose,
                task_id,
                "PostCompleteness",
                pair.label,
                verdict,
                killed,
                f"mutant={mut_out}",
            )

    # PreCorrectness
    if spec.precondition is not None:
        stub = builder.pre_correctness_stub(
            parsed, spec, valid_input_case, original_source=java_source
        )
        verdict, _ = runner.verify(
            stub, cname, f"pre_correct_{valid_input_case.label}"
        )
        ok = verdict == VerifyResult.OK
        result.pre_correct_detail = {
            "label": valid_input_case.label,
            "verdict": verdict.value,
            "pass": ok,
            "stub": stub,
        }
        _log(verbose, task_id, "PreCorrectness", valid_input_case.label, verdict, ok)

    # PreCompleteness 
    if spec.precondition is not None:
        for case in invalid_cases:
            stub = builder.pre_completeness_stub(
                parsed, spec, case, original_source=java_source
            )
            verdict, _ = runner.verify(stub, cname, f"pre_complete_{case.label}")
            rejected = verdict == VerifyResult.FAIL
            result.pre_complete_details.append(
                {
                    "label": case.label,
                    "verdict": verdict.value,
                    "rejected": rejected,
                    "stub": stub,
                }
            )
            _log(verbose, task_id, "PreCompleteness", case.label, verdict, rejected)

    return result



# Main evaluation entry point

def evaluate_problem(
    task: Task,
    llm_code: str,
    openjml_path: str = "openjml",
    output_dir: str = "veriact_outputs",
    verbose: bool = False,
    max_pairs: int = 5,
    run_id: str = "",
) -> dict:
    """
    Evaluate one task against LLM-generated JML annotations.

    Each test pair gets its own thread to compute all 4 metrics in parallel:
      - PostCorrectness
      - PostCompleteness
      - PreCorrectness
      - PreCompleteness

    Parameters
    ----------
    task         : Task object with benchmark Solution.java, Test.java, and IO pairs
    llm_code     : LLM-generated Java source with JML annotations
    openjml_path : path to the OpenJML binary (default: "openjml")
    output_dir   : directory to store harness stub files
    verbose      : print per-case verdicts
    max_pairs    : max test pairs to use (0 = all)

    Returns
    -------
    dict with keys: task_id, post_correctness, post_completeness,
                    pre_correctness, pre_completeness
    Each metric value is a dict: {score, passed, total, details}
    """
    solution_src = task.code
    test_src = task.test_code
    io_pairs = [{"input": tc.input, "output": tc.output} for tc in task.test_inputs]
    io_pairs_gen = [
        {"input": tc.input, "output": tc.output} for tc in task.generated_test_cases
    ]

    if max_pairs > 0:
        io_pairs = io_pairs[:max_pairs]
        remaining = max_pairs - len(io_pairs)
        if remaining > 0:
            io_pairs.extend(io_pairs_gen[:remaining])

    # parse method signature from benchmark Solution.java 
    parser = JavaMethodParser()
    bench_parsed = parser.parse(solution_src)
    bench_params = bench_parsed["params"]
    return_type = bench_parsed["return_type"]

    # extract JML spec from LLM code 
    spec = extract_jml_spec(llm_code)
    if spec.postcondition is None:
        logger.warning(f"[{task.task_id}] no JML ensures found")

    # parse LLM method signature 
    try:
        llm_parsed = parser.parse(llm_code)
        llm_params = llm_parsed["params"]
    except (ValueError, Exception):
        llm_parsed = bench_parsed
        llm_params = bench_params

    # detect input format from Test.java 
    read_ops = detect_input_format(test_src, bench_params)

    # parse io_pairs into TestPair / InputCase objects 
    test_pairs: list[TestPair] = []
    valid_inputs: list[dict] = []

    for idx, pair in enumerate(io_pairs):
        try:
            inputs_bench = parse_input(pair["input"], read_ops)
            output = parse_output(pair["output"], return_type)
        except Exception as e:
            if verbose:
                logger.warning(f"[{task.task_id}] skipping case {idx}: {e}")
            continue

        inputs_llm: dict[str, Any] = {}
        for bp, lp in zip(bench_params, llm_params):
            inputs_llm[lp["name"]] = inputs_bench[bp["name"]]

        test_pairs.append(TestPair(inputs_llm, output, f"case_{idx}"))
        valid_inputs.append(inputs_llm)

    if not test_pairs:
        logger.error(f"[{task.task_id}] no test pairs could be parsed")
        return {}

    # build InputCases for Pre metrics
    valid_input_cases = [InputCase(tp.inputs, True, tp.label) for tp in test_pairs]
    invalid_input_cases = generate_invalid_inputs(llm_params, valid_inputs)

    # setup 
    builder = StubBuilder()
    mutator = OutputMutator(k=5)
    parsed = llm_parsed
    cname = parsed["class_name"]
    rtype = parsed["return_type"]

    safe_task_id = task.task_id.replace("/", "_").replace("\\", "_")
    if run_id:
        stubs_dir = os.path.join(output_dir, "stubs", safe_task_id, run_id)
    else:
        stubs_dir = os.path.join(output_dir, "stubs", safe_task_id)
    runner = OpenJMLRunnerPersistent(openjml_path, stubs_dir)

    # distribute invalid cases round-robin across test pairs
    invalid_per_pair: list[list[InputCase]] = [[] for _ in test_pairs]
    for i, ic in enumerate(invalid_input_cases):
        invalid_per_pair[i % len(test_pairs)].append(ic)

    # launch one thread per test pair
    n_threads = len(test_pairs)
    pair_results: list[PairResult] = [None] * n_threads  # type: ignore

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        future_to_idx = {
            pool.submit(
                _evaluate_one_pair,
                idx,
                pair,
                valid_input_cases[idx],
                invalid_per_pair[idx],
                parsed,
                spec,
                rtype,
                builder,
                mutator,
                runner,
                cname,
                verbose,
                task.task_id,
                llm_code,
            ): idx
            for idx, pair in enumerate(test_pairs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                pair_results[idx] = future.result()
            except Exception as e:
                logger.error(f"[{task.task_id}] pair {idx}: {e}")

    # aggregate PairResults into HarnessResults
    pc_details: list[dict] = []
    pcl_details: list[dict] = []
    prc_details: list[dict] = []
    prl_details: list[dict] = []

    for pr in pair_results:
        if pr is None:
            continue
        if pr.post_correct_detail:
            pc_details.append(pr.post_correct_detail)
        pcl_details.extend(pr.post_complete_details)
        if pr.pre_correct_detail:
            prc_details.append(pr.pre_correct_detail)
        prl_details.extend(pr.pre_complete_details)

    results: dict[str, HarnessResult] = {}

    hr = HarnessResult("PostCorrectness", len(pc_details), 0)
    hr.passed = sum(1 for d in pc_details if d.get("pass"))
    hr.details = pc_details
    results["post_correctness"] = hr

    hr = HarnessResult("PostCompleteness", len(pcl_details), 0)
    hr.passed = sum(1 for d in pcl_details if d.get("killed"))
    hr.details = pcl_details
    results["post_completeness"] = hr

    hr = HarnessResult("PreCorrectness", len(prc_details), 0)
    hr.passed = sum(1 for d in prc_details if d.get("pass"))
    hr.details = prc_details
    results["pre_correctness"] = hr

    hr = HarnessResult("PreCompleteness", len(prl_details), 0)
    hr.passed = sum(1 for d in prl_details if d.get("rejected"))
    hr.details = prl_details
    results["pre_completeness"] = hr

    # summary
    summary_lines = [f"Spec-Harness | task: {task.task_id} | method: {parsed['method_name']}"]
    for r in results.values():
        summary_lines.append(f"  {r}")
    logger.info("\n".join(summary_lines))

    # build output dict
    return {
        "task_id": task.task_id,
        "post_correctness": results["post_correctness"].score,
        "post_completeness": results["post_completeness"].score,
        "pre_correctness": results["pre_correctness"].score,
        "pre_completeness": results["pre_completeness"].score,
    }
