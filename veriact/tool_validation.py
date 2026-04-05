"""AST-based validation for Tool classes."""

import ast
import builtins
from typing import Set

from veriact.utility import BASE_BUILTIN_MODULES

_BUILTIN_NAMES = set(vars(builtins))


class MethodChecker(ast.NodeVisitor):
    def __init__(self, class_attributes: Set[str], check_imports: bool = True):
        self.undefined_names = set()
        self.imports = {}
        self.from_imports = {}
        self.assigned_names = set()
        self.arg_names = set()
        self.class_attributes = class_attributes
        self.errors = []
        self.check_imports = check_imports
        self.typing_names = {"Any"}

    def visit_arguments(self, node):
        self.arg_names = {arg.arg for arg in node.args}
        if node.kwarg: self.arg_names.add(node.kwarg.arg)
        if node.vararg: self.arg_names.add(node.vararg.arg)

    def visit_Import(self, node):
        for name in node.names:
            self.imports[name.asname or name.name] = name.name

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for name in node.names:
            self.from_imports[name.asname or name.name] = (module, name.name)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
        self.visit(node.value)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.assigned_names.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node):
        target = node.target
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, ast.Tuple):
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators):
        for gen in generators:
            if isinstance(gen.target, ast.Name):
                self.assigned_names.add(gen.target.id)
            elif isinstance(gen.target, ast.Tuple):
                for elt in gen.target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_ListComp(self, node):
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)
    def visit_DictComp(self, node):
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)
    def visit_SetComp(self, node):
        self._handle_comprehension_generators(node.generators)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            self.generic_visit(node)

    def _is_defined(self, name):
        return (
            name in _BUILTIN_NAMES or name in BASE_BUILTIN_MODULES
            or name in self.arg_names or name == "self"
            or name in self.class_attributes or name in self.imports
            or name in self.from_imports or name in self.assigned_names
            or name in self.typing_names
        )

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and not self._is_defined(node.id):
            self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and not self._is_defined(node.func.id):
            self.errors.append(f"Name '{node.func.id}' is undefined.")
        self.generic_visit(node)


