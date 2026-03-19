"""
Automated computation of four spec-harness metrics for LLM-generated JML annotations on Java methods.

Metrics
-------
  PostCorrectness  : |{(i,o)  in T   : verify {true} x:=i; y:=o  {phi(x,y)}}| / |T|
  PostCompleteness : |{(i,o') in T1  : ~verify{true} x:=i; y:=o' {phi(x,y)}}| / |T1|
  PreCorrectness   : |{i      in T+  : verify {true} x:=i         {psi(x)}}|   / |T+|
  PreCompleteness  : |{i'     in T-  : ~verify{true} x:=i'        {psi(x)}}|   / |T-|

All four reduce to the same operation:
  1. Parse Java method signature via javalang to extract parameter / return types
  2. Build a Java stub that replaces the method body with concrete assignments
  3. Invoke OpenJML (ESC mode); interpret verify / fail as desired outcome

Supported types
---------------
  Primitives  : int, long, short, byte, double, float, boolean, char
  Boxed       : Integer, Long, Double, Float, Boolean, Character, String
  Arrays      : T[], T[][], T[][][]  for any T above
  Collections : List<T>, ArrayList<T>, LinkedList<T>
                Set<T>,  HashSet<T>,  TreeSet<T>
                Map<K,V>, HashMap<K,V>, TreeMap<K,V>
                Nested generics: Map<String, List<Integer>>, List<int[]>
"""

from __future__ import annotations

import os
import re
import json
import textwrap
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import javalang
import javalang.tree as jtree


# ============================================================
# Result types
# ============================================================

class VerifyResult(Enum):
    OK      = "ok"
    FAIL    = "fail"
    UNKNOWN = "unknown"


@dataclass
class HarnessResult:
    metric:  str
    total:   int
    passed:  int
    details: list = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def __str__(self) -> str:
        return (f"{self.metric:25s}  score={self.score:.3f}"
                f"  ({self.passed}/{self.total})")


# ============================================================
# Test data structures
# ============================================================

@dataclass
class TestPair:
    """One concrete input-output pair for postcondition evaluation."""
    inputs: dict[str, Any]   # {"paramName": pythonValue, …}
    output: Any              # expected return value
    label:  str = ""


@dataclass
class InputCase:
    """One input case for precondition evaluation."""
    inputs: dict[str, Any]   # {"paramName": pythonValue, …}
    valid:  bool             # True  → T+  (must be admitted by precondition)
                             # False → T-  (must be rejected by precondition)
    label:  str = ""


@dataclass
class MethodSpec:
    """Holds the raw LLM-generated JML comment block, preserved as-is."""
    jml_block:     str             # raw JML comment block (//@ or /*@ ... @*/) from LLM code
    auxiliaries:   str = ""        # helper predicates / model methods (prepended to stub)

    @property
    def has_requires(self) -> bool:
        return bool(re.search(r'requires\b', self.jml_block))

    @property
    def has_ensures(self) -> bool:
        return bool(re.search(r'ensures\b', self.jml_block))

    # Back-compat aliases (deprecated)
    @property
    def precondition(self) -> Optional[str]:
        return self.jml_block if self.has_requires else None

    @property
    def postcondition(self) -> Optional[str]:
        return self.jml_block if self.has_ensures else None


# ============================================================
# javalang-based type descriptor
# ============================================================

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
    base:         str
    array_dims:   int  = 0
    generic_args: list = field(default_factory=list)

    # ---- convenience predicates ----------------------------------------

    PRIMITIVES = frozenset({
        "int","long","short","byte","double","float","boolean","char"
    })
    COLLECTIONS = frozenset({
        "List","ArrayList","LinkedList",
        "Set","HashSet","TreeSet",
        "Map","HashMap","TreeMap",
    })

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


# ============================================================
# javalang-based method signature parser
# ============================================================

class JavaMethodParser:
    """
    Parses a Java source string (class or bare method) using javalang
    and returns structured parameter / return type information.

    Returns
    -------
    {
      "class_name":   str,
      "method_name":  str,
      "return_type":  JType,
      "params":       list[dict{"name": str, "jtype": JType}],
    }
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
                    "class_name":  class_name,
                    "method_name": node.name,
                    "return_type": self._jtype(node.return_type),
                    "params": [
                        {"name": p.name, "jtype": self._jtype(p.type)}
                        for p in (node.parameters or [])
                    ],
                }
        raise ValueError("No MethodDeclaration found in the provided source.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_class(source: str) -> tuple[str, str]:
        source = source.strip()
        m = re.search(r'\bclass\s+(\w+)', source)
        if m:
            return source, m.group(1)
        # Bare method — wrap in a synthetic class
        return f"class _Harness {{ {source} }}", "_Harness"

    def _jtype(self, node) -> JType:
        """Convert a javalang type AST node to a JType."""
        if node is None:
            return JType(base="void")

        # ---- BasicType: int, boolean, double … -------------------------
        if isinstance(node, jtree.BasicType):
            dims = len(node.dimensions) if node.dimensions else 0
            return JType(base=node.name, array_dims=dims)

        # ---- ReferenceType: String, int[], List<Integer>, Map<K,V> … ---
        if isinstance(node, jtree.ReferenceType):
            dims         = len(node.dimensions) if node.dimensions else 0
            generic_args = self._generic_args(node)
            return JType(base=node.name, array_dims=dims,
                         generic_args=generic_args)

        return JType(base=str(node))

    def _generic_args(self, node: jtree.ReferenceType) -> list[str]:
        """Recursively flatten type arguments to string representations."""
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


# ============================================================
# Type-aware Java literal renderer
# ============================================================

class JavaLiteralRenderer:
    """
    Converts a Python value to a syntactically valid Java literal /
    initialiser expression, guided by a JType descriptor.

    Handles
    -------
    - All primitives and their boxed counterparts
    - 1-D, 2-D, 3-D (and deeper) arrays
    - List / ArrayList / LinkedList
    - Set / HashSet / TreeSet
    - Map / HashMap / TreeMap
    - Nested generics: Map<String, List<Integer>>, List<int[]>, …
    """

    def render(self, jtype: JType, value: Any) -> str:
        if value is None:
            return "null"
        if jtype.is_array:
            return self._array(jtype, value)
        if jtype.is_collection:
            return self._collection(jtype, value)
        return self._scalar(jtype.base, value)

    # ---- arrays --------------------------------------------------------

    def _array(self, jtype: JType, value: Any) -> str:
        """Recursively render N-dimensional arrays."""
        elem_type = jtype.element_type()
        if jtype.array_dims == 1:
            elems = ", ".join(self._scalar(jtype.base, v) for v in value)
            return f"new {jtype.base}[]{{{elems}}}"
        # 2D / 3D / … — recurse on each row
        rows = ", ".join(self._array(elem_type, row) for row in value)
        inner = jtype.base + "[]" * (jtype.array_dims - 1)
        return f"new {inner}[]{{{rows}}}"

    # ---- collections ---------------------------------------------------

    def _collection(self, jtype: JType, value: Any) -> str:
        base = jtype.base
        if base in ("Map", "HashMap", "TreeMap"):
            return self._map(jtype, value)
        if base in ("Set", "HashSet", "TreeSet"):
            return self._set(jtype, value)
        return self._list(jtype, value)  # List / ArrayList / LinkedList

    def _list(self, jtype: JType, value: list) -> str:
        elem = self._arg_type(jtype, 0)
        elems = ", ".join(self.render(elem, v) for v in value)
        concrete = ("LinkedList" if jtype.base == "LinkedList"
                    else "ArrayList")
        return f"new java.util.{concrete}<>(java.util.Arrays.asList({elems}))"

    def _set(self, jtype: JType, value) -> str:
        elem = self._arg_type(jtype, 0)
        elems = ", ".join(self.render(elem, v) for v in value)
        concrete = "TreeSet" if jtype.base == "TreeSet" else "HashSet"
        return (f"new java.util.{concrete}<>"
                f"(java.util.Arrays.asList({elems}))")

    def _map(self, jtype: JType, value: dict) -> str:
        k_type = self._arg_type(jtype, 0)
        v_type = self._arg_type(jtype, 1)
        concrete = "TreeMap" if jtype.base == "TreeMap" else "HashMap"
        entries = "; ".join(
            f"_m.put({self.render(k_type, k)}, {self.render(v_type, v)})"
            for k, v in value.items()
        )
        decl = jtype.java_decl()
        # anonymous initialiser block
        return (f"new java.util.{concrete}<{decl}>() "
                f"{{{{ {entries}; }}}}")

    # ---- scalars -------------------------------------------------------

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

    # ---- helper --------------------------------------------------------

    def _arg_type(self, jtype: JType, idx: int) -> JType:
        """Return the JType for the idx-th generic argument."""
        if jtype.generic_args and idx < len(jtype.generic_args):
            return _parse_raw_jtype(jtype.generic_args[idx])
        return JType(base="Object")


# ============================================================
# Stub builder
# ============================================================

class StubBuilder:
    """
    Constructs Java harness stubs for each of the four Hoare triples.

    Uses ``//@ assume`` to constrain inputs (OpenJML treats parameter
    reassignment differently from assumptions in postconditions) and
    ``return`` for outputs, or ``//@ assert`` for precondition checks.

      PostCorrectness  : //@ ensures phi;  body: assume inputs; return o;
      PostCompleteness : //@ ensures phi;  body: assume inputs; return o';
      PreCorrectness   : body: assume inputs;  //@ assert psi;
      PreCompleteness  : body: assume inputs'; //@ assert psi;
    """

    def __init__(self):
        self._renderer = JavaLiteralRenderer()

    # ---- public entry points -------------------------------------------

    def post_correctness_stub(self, parsed: dict,
                               spec: MethodSpec,
                               pair: TestPair) -> str:
        assumes = self._input_assumes(parsed, pair.inputs)
        ret     = self._return_stmt(parsed, pair.output)
        return self._wrap_post(parsed, spec, assumes, ret)

    def post_completeness_stub(self, parsed: dict,
                                spec: MethodSpec,
                                pair: TestPair,
                                mutated_output: Any) -> str:
        assumes = self._input_assumes(parsed, pair.inputs)
        ret     = self._return_stmt(parsed, mutated_output)
        return self._wrap_post(parsed, spec, assumes, ret)

    def pre_correctness_stub(self, parsed: dict,
                              spec: MethodSpec,
                              case: InputCase,
                              original_source: str = "") -> str:
        if original_source:
            return self._wrap_caller(parsed, spec, original_source,
                                     case.inputs)
        assumes = self._input_assumes(parsed, case.inputs)
        return self._wrap_pre(parsed, spec, assumes)

    def pre_completeness_stub(self, parsed: dict,
                               spec: MethodSpec,
                               case: InputCase,
                               original_source: str = "") -> str:
        if original_source:
            return self._wrap_caller(parsed, spec, original_source,
                                     case.inputs)
        assumes = self._input_assumes(parsed, case.inputs)
        return self._wrap_pre(parsed, spec, assumes)

    # ---- assume generation ---------------------------------------------

    def _input_assumes(self, parsed: dict,
                       inputs: dict[str, Any]) -> str:
        """Generate ``//@ assume`` statements for all input values."""
        lines: list[str] = []
        for p in parsed["params"]:
            val = inputs[p["name"]]
            lines.extend(self._assume_for(p["name"], p["jtype"], val))
        return "\n".join(lines)

    def _assume_for(self, name: str, jtype: JType,
                    value: Any) -> list[str]:
        """JML assume statement(s) constraining *name* to *value*."""
        I = "        "  # 8-space indent

        if value is None:
            return [f"{I}//@ assume {name} == null;"]

        # ---- arrays ----
        if jtype.is_array:
            return self._assume_array(name, jtype, value)

        # ---- collections ----
        if jtype.is_collection:
            return self._assume_collection(name, jtype, value)

        # ---- String ----
        if jtype.base == "String":
            lit = self._renderer._scalar("String", value)
            return [f"{I}//@ assume {name} != null && {name}.equals({lit});"]

        # ---- other scalars ----
        lit = self._renderer._scalar(jtype.base, value)
        return [f"{I}//@ assume {name} == {lit};"]

    def _assume_array(self, name: str, jtype: JType,
                      value: list) -> list[str]:
        """Element-wise assumes for 1-D / 2-D arrays."""
        I = "        "
        lines = [f"{I}//@ assume {name} != null;",
                 f"{I}//@ assume {name}.length == {len(value)};"]

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
                        lines.append(
                            f"{I}//@ assume {name}[{i}][{j}] == {lit};")
                # deeper arrays — skip per-element (would need recursion)
        return lines

    def _assume_collection(self, name: str, jtype: JType,
                           value: Any) -> list[str]:
        """Best-effort assume for collection inputs."""
        I = "        "
        if value is None:
            return [f"{I}//@ assume {name} == null;"]
        lines = [f"{I}//@ assume {name} != null;"]
        if isinstance(value, dict):
            lines.append(f"{I}//@ assume {name}.size() == {len(value)};")
        elif isinstance(value, (list, set)):
            lines.append(f"{I}//@ assume {name}.size() == {len(value)};")
        return lines

    # ---- return / assert generation ------------------------------------

    def _return_stmt(self, parsed: dict, value: Any) -> str:
        rt  = parsed["return_type"]
        lit = self._renderer.render(rt, value)
        return f"        return {lit};"

    # ---- wrapping ------------------------------------------------------

    def _wrap_post(self, parsed: dict, spec: MethodSpec,
                   assumes: str, ret_stmt: str) -> str:
        """Wrap for postcondition checking — raw JML block above method."""
        params_decl = ", ".join(
            f"{p['jtype'].java_decl()} {p['name']}"
            for p in parsed["params"]
        )
        ret   = parsed["return_type"].java_decl()
        mname = parsed["method_name"]
        cname = parsed["class_name"]
        aux   = spec.auxiliaries or ""
        jml   = self._indent_jml(spec.jml_block, "    ")

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

    def _wrap_pre(self, parsed: dict, spec: MethodSpec,
                  assumes: str) -> str:
        """Wrap for precondition checking — raw JML block above method."""
        params_decl = ", ".join(
            f"{p['jtype'].java_decl()} {p['name']}"
            for p in parsed["params"]
        )
        mname = parsed["method_name"]
        cname = parsed["class_name"]
        aux   = spec.auxiliaries or ""
        jml   = self._indent_jml(spec.jml_block, "    ")

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

    def _wrap_caller(self, parsed: dict, spec: MethodSpec,
                     original_source: str,
                     inputs: dict[str, Any]) -> str:
        """
        Precondition stubs: emits a *caller* that invokes the annotated method
        with concrete argument literals.

        OpenJML ESC evaluates the requires clause at the call site:
          - valid input   → no warning  (PreCorrectness: desired OK)
          - invalid input → warning     (PreCompleteness: desired FAIL)

        The original annotated source must be included so OpenJML sees the
        requires clause attached to the real method declaration.

        Generated shape
        ---------------
        class {C} {                  // ← NOT public (only one public class per file)
            //@ requires …;
            public static T {m}(…) { … }
        }

        public class {C}Harness {
            public static void check() {
                {C}.{m}(lit₁, lit₂, …);   // ← OpenJML checks requires HERE
            }
        }
        """
        mname = parsed["method_name"]
        cname = parsed["class_name"]

        # Build concrete argument literals in parameter order
        call_args = ", ".join(
            self._renderer.render(p["jtype"], inputs[p["name"]])
            for p in parsed["params"]
        )

        # Strip "public" from the original class declaration so Java allows
        # the Harness to be the single public class in the file.
        src = re.sub(r'\bpublic\s+class\b', 'class', original_source,
                     count=1)

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
        """Re-indent each line of the JML block to the given indent level."""
        return "\n".join(
            indent + line.strip() for line in jml_block.splitlines()
            if line.strip()
        )


# ============================================================
# Output mutator  (type-aware)
# ============================================================

class OutputMutator:
    """
    Generates up to k output mutants for PostCompleteness, using
    type-specific boundary perturbations guided by JType.
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
        if base in ("int","long","short","byte","Integer","Long"):
            v = int(v)
            return list(dict.fromkeys([v+1, v-1, v+2, v-2, 0]))[:self.k]
        if base in ("double","float","Double","Float"):
            v = float(v)
            return [v+1.0, v-1.0, v*2.0, -v][:self.k]
        if base in ("boolean","Boolean"):
            return [not v]
        if base == "String":
            return [v+"_x", (v[:-1] if v else "x"),
                    v.upper(), "", v[::-1]][:self.k]
        if base in ("char","Character"):
            c = ord(v)
            return [chr(c+1), chr(c-1)][:self.k]
        return []

    def _mutate_array(self, v: list) -> list:
        mutants: list = []
        if v:
            mutants.append(v[:-1])           # drop last element
            last = v[-1]
            if not isinstance(last, list):
                mutants.append(v + [last + 1])    # append perturbed element
            if all(not isinstance(x, list) for x in v):
                mutants.append([x+1 for x in v]) # shift all values
        mutants.append([])                   # empty array
        return [m for m in mutants if m != v][:self.k]

    def _mutate_collection(self, jtype: JType, v: Any) -> list:
        if jtype.base in ("Map","HashMap","TreeMap"):
            d = dict(v) if v else {}
            mutants: list = []
            if d:
                k0 = next(iter(d))
                mutants.append({k: val for k, val in d.items() if k != k0})
            mutants.append({})
            return mutants[:self.k]
        # List / Set — treat as sequence
        return self._mutate_array(list(v) if v else [])


# ============================================================
# OpenJML runner
# ============================================================

class OpenJMLRunner:

    def __init__(self, openjml_path: str = "openjml",
                 timeout: int = 30):
        self.openjml_path = openjml_path
        self.timeout      = timeout

    def verify(self, java_source: str,
               class_name: str) -> tuple[VerifyResult, str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"{class_name}Harness.java")
            with open(path, "w") as f:
                f.write(java_source)
            try:
                r = subprocess.run(
                    [self.openjml_path, "-esc", path],
                    capture_output=True, text=True,
                    timeout=self.timeout,
                )
                out = r.stdout + r.stderr
                return self._parse(out), out
            except subprocess.TimeoutExpired:
                return VerifyResult.UNKNOWN, "timeout"
            except FileNotFoundError:
                raise RuntimeError(
                    f"OpenJML binary not found at '{self.openjml_path}'. "
                    "Install OpenJML and pass its path via openjml_path=."
                )

    @staticmethod
    def _parse(output: str) -> VerifyResult:
        low = output.strip().lower()
        # OpenJML reports failures as:
        #   "N warning(s)"              (some versions)
        #   "N verification failure(s)" (newer versions)
        #   "error"                     (compilation / syntax errors)
        # Success: empty output, "verified", or "0 warnings".
        if re.search(r'[1-9]\d* warning', low):
            return VerifyResult.FAIL
        if re.search(r'[1-9]\d* verification failure', low):
            return VerifyResult.FAIL
        if re.search(r'\berror\b', low):
            return VerifyResult.FAIL
        if "verified" in low or "0 warnings" in low or low == "":
            return VerifyResult.OK
        return VerifyResult.UNKNOWN


# ============================================================
# SpecHarness  —  main API
# ============================================================

class SpecHarness:
    """
    Evaluates a LLM-generated JML spec against all four spec-harness metrics.

    Parameters
    ----------
    java_source   : Java class or method source string (parsed by javalang)
    spec          : MethodSpec holding raw JML precondition / postcondition
    openjml_path  : path to the OpenJML binary (default: "openjml")
    mutant_budget : max output mutants per test pair for PostCompleteness
    verbose       : print per-case verdicts to stdout

    Usage
    -----
    harness = SpecHarness(java_source, spec, openjml_path="/opt/openjml/openjml")
    results = harness.evaluate(test_pairs, input_cases)
    harness.save_results(results, "results.json")
    """

    def __init__(self, java_source: str, spec: MethodSpec,
                 openjml_path:  str  = "openjml",
                 mutant_budget: int  = 5,
                 verbose:       bool = False):

        self.spec    = spec
        self.verbose = verbose

        # Keep the original annotated source for call-site pre-condition stubs
        self._java_source = java_source

        # Parse the method signature once via javalang
        self._parsed = JavaMethodParser().parse(java_source)
        self._cname  = self._parsed["class_name"]
        self._rtype  = self._parsed["return_type"]

        self._builder = StubBuilder()
        self._runner  = OpenJMLRunner(openjml_path)
        self._mutator = OutputMutator(k=mutant_budget)

    # ------------------------------------------------------------------
    # Metric 1 — PostCorrectness
    # ------------------------------------------------------------------

    def post_correctness(self, pairs: list[TestPair]) -> HarnessResult:
        """
        PostCorrectness(phi, T) =
            |{ (i,o) in T : |= {true} x:=i; result:=o {phi(x,result)} }|
            ---------------------------------------------------------------
                                        |T|
        Desired outcome per case: VERIFY  (no false alarm on valid output)
        """
        res = HarnessResult("PostCorrectness", len(pairs), 0)
        for pair in pairs:
            stub = self._builder.post_correctness_stub(
                       self._parsed, self.spec, pair)
            verdict, log = self._runner.verify(stub, self._cname)
            ok = (verdict == VerifyResult.OK)
            if ok:
                res.passed += 1
            res.details.append({"label": pair.label,
                                 "verdict": verdict.value, "pass": ok})
            self._log("PostCorrectness", pair.label, verdict, ok)
        return res

    # ------------------------------------------------------------------
    # Metric 2 — PostCompleteness
    # ------------------------------------------------------------------

    def post_completeness(self, pairs: list[TestPair]) -> HarnessResult:
        """
        PostCompleteness(phi, T) =
            |{ (i,o') in T1 : not |= {true} x:=i; result:=o' {phi} }|
            -------------------------------------------------------------
                                        |T1|
        where T1 = union of output-mutant pairs for each (i,o) in T.
        Desired outcome per case: FAIL  (mutant correctly rejected)
        """
        all_mutants = [
            (pair, m)
            for pair in pairs
            for m in self._mutator.mutate(self._rtype, pair.output)
        ]
        res = HarnessResult("PostCompleteness", len(all_mutants), 0)
        for pair, mut_out in all_mutants:
            stub = self._builder.post_completeness_stub(
                       self._parsed, self.spec, pair, mut_out)
            verdict, log = self._runner.verify(stub, self._cname)
            killed = (verdict == VerifyResult.FAIL)
            if killed:
                res.passed += 1
            res.details.append({"label": pair.label, "mutant": str(mut_out),
                                 "verdict": verdict.value, "killed": killed})
            self._log("PostCompleteness", pair.label, verdict, killed,
                      extra=f"mutant={mut_out}")
        return res

    # ------------------------------------------------------------------
    # Metric 3 — PreCorrectness
    # ------------------------------------------------------------------

    def pre_correctness(self, cases: list[InputCase]) -> HarnessResult:
        """
        PreCorrectness(psi, T+) =
            |{ i in T+ : |= {true} x:=i {psi(x)} }|
            ------------------------------------------
                                |T+|
        Desired outcome per case: VERIFY  (valid input admitted, no false rejection)

        Uses a call-site stub: the harness *calls* the annotated method
        with concrete values so OpenJML evaluates the requires clause.
        """
        valid = [c for c in cases if c.valid]
        if not self.spec.has_requires:
            return HarnessResult("PreCorrectness", 0, 0)
        res   = HarnessResult("PreCorrectness", len(valid), 0)
        for case in valid:
            stub = self._builder.pre_correctness_stub(
                       self._parsed, self.spec, case,
                       original_source=self._java_source)
            verdict, log = self._runner.verify(
                stub, self._cname)
            ok = (verdict == VerifyResult.OK)
            if ok:
                res.passed += 1
            res.details.append({"label": case.label,
                                 "verdict": verdict.value, "pass": ok})
            self._log("PreCorrectness", case.label, verdict, ok)
        return res

    # ------------------------------------------------------------------
    # Metric 4 — PreCompleteness
    # ------------------------------------------------------------------

    def pre_completeness(self, cases: list[InputCase]) -> HarnessResult:
        """
        PreCompleteness(psi, T-) =
            |{ i' in T- : not |= {true} x:=i' {psi(x)} }|
            ------------------------------------------------
                                |T-|
        T- = known boundary/edge inputs that violate the method domain.
        Desired outcome per case: FAIL  (invalid input correctly rejected)

        Uses a call-site stub: the harness *calls* the annotated method
        with concrete values so OpenJML evaluates the requires clause.
        """
        invalid = [c for c in cases if not c.valid]
        if not self.spec.has_requires:
            return HarnessResult("PreCompleteness", 0, 0)
        res     = HarnessResult("PreCompleteness", len(invalid), 0)
        for case in invalid:
            stub = self._builder.pre_completeness_stub(
                       self._parsed, self.spec, case,
                       original_source=self._java_source)
            verdict, log = self._runner.verify(
                stub, self._cname)
            rejected = (verdict == VerifyResult.FAIL)
            if rejected:
                res.passed += 1
            res.details.append({"label": case.label,
                                 "verdict": verdict.value, "rejected": rejected})
            self._log("PreCompleteness", case.label, verdict, rejected)
        return res

    # ------------------------------------------------------------------
    # Convenience: run all four
    # ------------------------------------------------------------------

    def evaluate(self, test_pairs:  list[TestPair],
                 input_cases: list[InputCase]) -> dict[str, HarnessResult]:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Spec-Harness  |  method: {self._parsed['method_name']}")
        print(sep)
        results = {
            "post_correctness":  self.post_correctness(test_pairs),
            "post_completeness": self.post_completeness(test_pairs),
            "pre_correctness":   self.pre_correctness(input_cases),
            "pre_completeness":  self.pre_completeness(input_cases),
        }
        print(f"\n{sep}")
        for r in results.values():
            print(f"  {r}")
        print(sep)
        return results

    def save_results(self, results: dict[str, HarnessResult],
                     path: str) -> None:
        out = {
            k: {"score": v.score, "passed": v.passed,
                "total": v.total, "details": v.details}
            for k, v in results.items()
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved → {path}")

    def _log(self, metric: str, label: str,
             verdict: VerifyResult, desired: bool,
             extra: str = "") -> None:
        if not self.verbose:
            return
        status = "✓" if desired else "✗"
        ex = f"  [{extra}]" if extra else ""
        print(f"  [{metric:20s}] {label}{ex}  →  {verdict.value}  {status}")


# ============================================================
# Utility: parse a raw generic-arg string back to JType
# ============================================================

def _parse_raw_jtype(raw: str) -> JType:
    """
    Minimal parser for strings like "Integer", "int[]", "List<String>",
    "Map<String, List<Integer>>".  Used when rendering nested generics.
    """
    raw = raw.strip()
    # Count and strip trailing []
    array_dims = 0
    while raw.endswith("[]"):
        array_dims += 1
        raw = raw[:-2]
    # Generic  e.g. List<Integer> or Map<String, List<Integer>>
    m = re.match(r'^(\w+)<(.+)>$', raw)
    if m:
        base  = m.group(1)
        # Split top-level generic args (respecting nested <…>)
        inner = _split_generic_args(m.group(2))
        return JType(base=base, array_dims=array_dims, generic_args=inner)
    return JType(base=raw, array_dims=array_dims)


def _split_generic_args(s: str) -> list[str]:
    """Split 'String, List<Integer>' respecting nested angle brackets."""
    depth, buf, result = 0, [], []
    for ch in s:
        if ch == "<":
            depth += 1; buf.append(ch)
        elif ch == ">":
            depth -= 1; buf.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        result.append("".join(buf).strip())
    return result


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Example 1: twoSum — primitive array I/O
    # ------------------------------------------------------------------
    java_src_1 = """
    public class TwoSum {
        public static int[] twoSum(int[] nums, int target) { return null; }
    }
    """
    spec_1 = MethodSpec(
        jml_block = ("//@ requires nums != null && nums.length >= 2;\n"
                     "//@ ensures \\result != null && \\result.length == 2 && "
                     "nums[\\result[0]] + nums[\\result[1]] == target;"),
    )
    test_pairs_1 = [
        TestPair({"nums": [2,7,11,15], "target": 9},  [0,1],  "basic"),
        TestPair({"nums": [3,2,4],     "target": 6},  [1,2],  "mid"),
        TestPair({"nums": [3,3],        "target": 6},  [0,1],  "dup"),
    ]
    input_cases_1 = [
        InputCase({"nums": [2,7,11,15], "target": 9},  True,  "valid_basic"),
        InputCase({"nums": [3,2,4],     "target": 6},  True,  "valid_mid"),
        InputCase({"nums": None,          "target": 9},  False, "null_array"),
        InputCase({"nums": [1],            "target": 1},  False, "single_elem"),
        InputCase({"nums": [],             "target": 0},  False, "empty_array"),
    ]

    h1 = SpecHarness(java_src_1, spec_1, openjml_path="openjml", verbose=True)
    r1 = h1.evaluate(test_pairs_1, input_cases_1)
    h1.save_results(r1, "twosum_results.json")

    # ------------------------------------------------------------------
    # Example 2: matrix multiply — 2D array I/O
    # ------------------------------------------------------------------
    java_src_2 = """
    public class MatMul {
        public static int[][] multiply(int[][] a, int[][] b) { return null; }
    }
    """
    spec_2 = MethodSpec(
        jml_block = ("//@ requires a != null && b != null && a[0].length == b.length;\n"
                     "//@ ensures \\result != null && \\result.length == a.length;"),
    )
    test_pairs_2 = [
        TestPair(
            {"a": [[1,2],[3,4]], "b": [[5,6],[7,8]]},
            [[19,22],[43,50]], "2x2"
        ),
    ]
    input_cases_2 = [
        InputCase({"a": [[1,2],[3,4]], "b": [[5,6],[7,8]]}, True,  "valid_2x2"),
        InputCase({"a": None,           "b": [[1,2],[3,4]]}, False, "null_a"),
        InputCase({"a": [[1,2,3]],      "b": [[1,2],[3,4]]}, False, "dim_mismatch"),
    ]

    h2 = SpecHarness(java_src_2, spec_2, openjml_path="openjml", verbose=True)
    h2.evaluate(test_pairs_2, input_cases_2)

    # ------------------------------------------------------------------
    # Example 3: Map<String, List<Integer>> — nested collection I/O
    # ------------------------------------------------------------------
    java_src_3 = """
    import java.util.*;
    public class GroupBy {
        public static Map<String, List<Integer>> groupByLength(List<String> words) {
            return null;
        }
    }
    """
    spec_3 = MethodSpec(
        jml_block = ("//@ requires words != null;\n"
                     "//@ ensures \\result != null;"),
    )
    test_pairs_3 = [
        TestPair(
            {"words": ["hi","hey","yo"]},
            {"2": [0,2], "3": [1]},
            "basic_map"
        ),
    ]
    input_cases_3 = [
        InputCase({"words": ["hi","ho"]}, True,  "valid"),
        InputCase({"words": None},         False, "null_list"),
    ]

    h3 = SpecHarness(java_src_3, spec_3, openjml_path="openjml", verbose=True)
    h3.evaluate(test_pairs_3, input_cases_3)