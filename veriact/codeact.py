"""MultiStepAgent and CodeAgent — ReAct-style agentic loop with code execution."""

import ast
import builtins
import contextlib
import importlib
import inspect
import io
import json
import textwrap
import time
import traceback
from collections import deque
from logging import getLogger
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    TypedDict,
)

import yaml
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.rule import Rule
from rich.text import Text

from veriact.agent_types import AgentType, handle_agent_output_types
from veriact.default_tools import TaskCompletionTool
from veriact.memory import (
    ActionStep,
    AgentMemory,
    PlanningStep,
    SystemPromptStep,
    TaskCompleteStep,
    TaskStep,
    ToolCall,
)
from veriact.models import ChatMessage, MessageRole
from veriact.monitoring import GREEN_HEX, AgentLogger, LogLevel, Monitor
from veriact.tools_base import Tool
from veriact.utility import (
    CODEAGENT_RESPONSE_FORMAT,
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    extract_code_from_text,
    is_valid_name,
    make_json_serializable,
    truncate_content,
    find_tool_usage,
)

logger = getLogger(__name__)


# In-process execution guards


class _TaskCompleteSignal(Exception):
    """Raised by the injected task_complete() to cleanly signal run completion."""

    def __init__(self, result=None):
        self.result = result


# Top-level module names that agent code must not import
_BLOCKED_IMPORTS = frozenset(
    {
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "paramiko",
        "fabric",
        "multiprocessing",
        "ctypes",
        "cffi",
        "pty",
        "pexpect",
        "importlib",
    }
)

# os.* calls that could escape the process
_BLOCKED_OS_CALLS = frozenset(
    {
        "system",
        "popen",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "spawn",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "chmod",
        "chown",
        "setuid",
        "setgid",
    }
)

# Builtin calls blocked in agent code
_BLOCKED_BUILTINS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint"})


class _CodeSafetyChecker(ast.NodeVisitor):
    """AST visitor that collects dangerous code patterns before execution."""

    def __init__(self):
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in _BLOCKED_IMPORTS:
                self.violations.append(f"Blocked import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and node.module.split(".")[0] in _BLOCKED_IMPORTS:
            self.violations.append(f"Blocked import: 'from {node.module} import ...'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
            self.violations.append(f"Blocked builtin: '{node.func.id}()'")
        elif isinstance(node.func, ast.Attribute):
            obj, attr = node.func.value, node.func.attr
            if isinstance(obj, ast.Name):
                if obj.id == "os" and attr in _BLOCKED_OS_CALLS:
                    self.violations.append(f"Blocked OS call: 'os.{attr}()'")
                elif obj.id == "subprocess":
                    self.violations.append(f"Blocked call: 'subprocess.{attr}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Block common sandbox-escape paths via class hierarchy traversal
        if node.attr in {
            "__subclasses__",
            "__globals__",
            "__builtins__",
            "__code__",
            "__closure__",
        }:
            self.violations.append(f"Blocked attribute access: '.{node.attr}'")
        self.generic_visit(node)


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    return Template(template, undefined=StrictUndefined).render(**variables)


class PlanningPromptTemplate(TypedDict):
    initial_facts: str
    initial_plan: str
    update_facts_pre_messages: str
    update_facts_post_messages: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class TaskCompletePromptTemplate(TypedDict):
    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    system_prompt: str
    planning: PlanningPromptTemplate
    task_complete: TaskCompletePromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_facts="",
        initial_plan="",
        update_facts_pre_messages="",
        update_facts_post_messages="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    task_complete=TaskCompletePromptTemplate(pre_messages="", post_messages=""),
)


class MultiStepAgent:
    """ReAct-style agent: think → act → observe → repeat."""

    def __init__(
        self,
        tools,
        model,
        prompt_templates=None,
        max_steps=5,
        verbosity_level=LogLevel.INFO,
        grammar=None,
        step_callbacks=None,
        planning_interval=None,
        name=None,
        description=None,
        provide_run_summary=False,
        task_complete_checks=None,
        harness_threshold=0.5,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number = 0
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state = {}
        self.name = name if (name is None or is_valid_name(name)) else None
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.task_complete_checks = task_complete_checks
        self.harness_threshold = harness_threshold

        assert all(isinstance(t, Tool) for t in tools)
        self.tools = {t.name: t for t in tools}
        self.tools.setdefault("task_complete", TaskCompletionTool())

        self.task = None
        self.system_prompt = self.initialize_system_prompt()
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = list(step_callbacks or [])
        self.step_callbacks.append(self.monitor.update_metrics)

    def run(self, task, stream=False, reset=True, additional_args=None, max_steps=None):
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args:
            self.state.update(additional_args)
            self.task += f"\nAdditional args: {additional_args}"

        self.system_prompt = self.initialize_system_prompt(task=self.task)
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {getattr(self.model, 'model_id', '')}",
            level=LogLevel.INFO,
            title=getattr(self, "name", None),
        )
        self.memory.steps.append(TaskStep(task=self.task))

        if stream:
            return self._run(task=self.task, max_steps=max_steps)
        return deque(self._run(task=self.task, max_steps=max_steps), maxlen=1)[
            0
        ].task_complete

    def _run(self, task, max_steps) -> Generator[ActionStep | AgentType, None, None]:
        task_complete = None
        self.step_number = 1
        step_start_time = time.time()
        while task_complete is None and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)
            step_start_time = time.time()
            if self.planning_interval and (
                self.step_number == 1
                or (self.step_number - 1) % self.planning_interval == 0
            ):
                ps = self._create_planning_step(
                    task, is_first_step=(self.step_number == 1), step=self.step_number
                )
                self.memory.steps.append(ps)
                yield ps
            action_step = ActionStep(
                step_number=self.step_number, start_time=step_start_time
            )
            try:
                task_complete = self._execute_step(task, action_step)
            except AgentGenerationError:
                raise
            except AgentError as e:
                action_step.error = e
            finally:
                action_step.end_time = time.time()
                action_step.duration = action_step.end_time - step_start_time
                for cb in self.step_callbacks:
                    (
                        cb(action_step)
                        if len(inspect.signature(cb).parameters) == 1
                        else cb(action_step, agent=self)
                    )
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if task_complete is None and self.step_number == max_steps + 1:
            task_complete = self.provide_task_complete(task)
            final = ActionStep(
                step_number=self.step_number,
                error=AgentMaxStepsError("Reached max steps.", self.logger),
            )
            final.action_output = task_complete
            final.end_time = time.time()
            final.duration = final.end_time - step_start_time
            self.memory.steps.append(final)
            for cb in self.step_callbacks:
                (
                    cb(final)
                    if len(inspect.signature(cb).parameters) == 1
                    else cb(final, agent=self)
                )
            yield final
        yield TaskCompleteStep(task_complete=handle_agent_output_types(task_complete))

    def _execute_step(self, task, memory_step):
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        tc = self.step(memory_step)
        if tc is not None and self.task_complete_checks:
            for check in self.task_complete_checks:
                try:
                    assert check(tc, self.memory)
                except Exception as e:
                    raise AgentError(f"Check {check.__name__} failed: {e}", self.logger)
        return tc

    def _create_planning_step(self, task, is_first_step, step):
        if is_first_step:
            input_messages = [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools},
                            ),
                        }
                    ],
                }
            ]
            plan_message = self.model(input_messages, stop_sequences=["<end_plan>"])
            plan = f"Here are the facts and plan:\n```\n{plan_message.content}\n```"
        else:
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            pre = {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"][
                                "update_plan_pre_messages"
                            ],
                            variables={"task": task},
                        ),
                    }
                ],
            }
            post = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"][
                                "update_plan_post_messages"
                            ],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "remaining_steps": self.max_steps - step,
                            },
                        ),
                    }
                ],
            }
            input_messages = [pre] + memory_messages + [post]
            plan_message = self.model(input_messages, stop_sequences=["<end_plan>"])
            plan = f"Updated plan:\n```\n{plan_message.content}\n```"

        self.logger.log(
            Rule(
                f"[bold]{'Initial' if is_first_step else 'Updated'} plan",
                style="orange",
            ),
            Text(plan),
            level=LogLevel.INFO,
        )
        return PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=plan_message,
        )

    def initialize_system_prompt(self, task=None):
        if self.prompt_templates.get("system_prompt"):
            task_value = task if task is not None else getattr(self, "task", "")
            return populate_template(
                self.prompt_templates["system_prompt"],
                variables={
                    "tools": self.tools,
                    "task": task_value,
                    "harness_threshold": self.harness_threshold,
                },
            )
        return ""

    def write_memory_to_messages(self, summary_mode=False):
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for step in self.memory.steps:
            messages.extend(step.to_messages(summary_mode=summary_mode))
        return messages

    def provide_task_complete(self, task):
        pre = self.prompt_templates.get("task_complete", {}).get("pre_messages", "")
        messages = self.write_memory_to_messages(summary_mode=True)
        if pre:
            messages = [
                {
                    "role": MessageRole.SYSTEM,
                    "content": [{"type": "text", "text": pre}],
                }
            ] + messages
        try:
            return self.model(messages).content
        except Exception as e:
            return f"Error generating final output: {e}"

    def step(self, memory_step):
        pass

    def interrupt(self):
        self.interrupt_switch = True

    def visualize(self):
        self.logger.visualize_agent_tree(self)

    def to_dict(self):
        tool_dicts = [t.to_dict() for t in self.tools.values()]
        return {
            "provided_tools": [t["name"] for t in tool_dicts],
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
        }


class CodeAgent(MultiStepAgent):
    """Agent that generates Python code to call tools."""

    def __init__(
        self,
        tools,
        model,
        prompt_templates=None,
        grammar=None,
        planning_interval=None,
        code_block_tags=None,
        executor_kwargs=None,
        max_print_outputs_length=None,
        harness_threshold=0.5,
        **kwargs,
    ):
        self.max_print_outputs_length = max_print_outputs_length

        # Load default prompts if none provided
        if prompt_templates is None:
            try:
                prompt_templates = yaml.safe_load(
                    importlib.resources.files("veriact.prompts")
                    .joinpath("veriact_prompt.yaml")
                    .read_text()
                )
            except Exception:
                prompt_templates = EMPTY_PROMPT_TEMPLATES

        self.code_block_tags = (
            code_block_tags
            if isinstance(code_block_tags, tuple)
            else (
                ("```python", "```")
                if code_block_tags == "markdown"
                else ("<code>", "</code>")
            )
        )

        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            harness_threshold=harness_threshold,
            **kwargs,
        )

        self.executor_kwargs = executor_kwargs or {}
        self._init_execution_namespace()

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse the first JSON object from *text*, ignoring trailing content.

        Models like Claude may append extra text after the JSON object (e.g.
        "Calling tools: …").  ``json.loads`` rejects that with "Extra data",
        so we use ``json.JSONDecoder.raw_decode`` which stops after the first
        valid JSON value.  If the JSON doesn't start at position 0 we scan
        for the first ``{``.
        """
        text = text.strip()
        decoder = json.JSONDecoder()
        # Find the first '{' in case there's leading text
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response")
        obj, _ = decoder.raw_decode(text, start)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected a JSON object, got {type(obj).__name__}")
        return obj

    def _init_execution_namespace(self) -> None:
        """Set up a fresh shared namespace for in-process code execution.

        Tool instances are injected directly by name so agent code can call
        e.g. ``verify_with_openjml(...)`` without any import or serialisation.
        The namespace persists across all steps in a single run, so variables
        defined in step N are visible in step N+1.
        """
        # Per-step list that accumulates {tool_name, args, output} dicts.
        # Cleared at the start of each step in execute_code().
        self._current_tool_outputs: list[dict[str, Any]] = []

        ns: dict[str, Any] = {}
        for t in self.tools.values():
            if t.name == "task_complete":
                continue
            ns[t.name] = self._wrap_tool(t)

        # Wrap task_complete so calling it raises _TaskCompleteSignal, which
        # execute_code catches to set is_task_complete=True.
        _tc_tool = self.tools.get("task_complete")

        def _task_complete(summary: str = ""):
            result = _tc_tool(summary=summary) if _tc_tool else summary
            self._current_tool_outputs.append(
                {"tool_name": "task_complete", "args": {"summary": summary}, "output": result}
            )
            raise _TaskCompleteSignal(result)

        ns["task_complete"] = _task_complete

        # Defense-in-depth: strip the most dangerous builtins at runtime and
        # replace __import__ with a guarded version (AST check is the primary
        # guard; this catches anything that slips through at parse time).
        _orig_import = builtins.__import__

        def _guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
            if name.split(".")[0] in _BLOCKED_IMPORTS:
                raise ImportError(f"Import of '{name}' is not permitted in agent code.")
            return _orig_import(name, globals_, locals_, fromlist, level)

        safe_builtins = {
            k: v
            for k, v in vars(builtins).items()
            if k not in {"eval", "exec", "compile", "breakpoint"}
        }
        safe_builtins["__import__"] = _guarded_import
        ns["__builtins__"] = safe_builtins

        self.execution_namespace = ns

    def _wrap_tool(self, tool: Tool):
        """Return a wrapper that delegates to *tool* but records its output.

        Stdout is suppressed during the tool call so that internal noise
        (e.g. harness mutant-verification output) does not leak into the
        agent's observations.  Only the agent's own print() calls appear.
        """
        def wrapper(*args, **kwargs):
            # Suppress any stdout the tool produces internally so it
            # doesn't pollute the agent's observation buffer.
            with contextlib.redirect_stdout(io.StringIO()):
                result = tool(*args, **kwargs)
            self._current_tool_outputs.append(
                {"tool_name": tool.name, "args": {"args": args, "kwargs": kwargs}, "output": result}
            )
            return result
        # Preserve the original tool's attributes for introspection
        wrapper.__name__ = tool.name
        wrapper.__doc__ = getattr(tool, "description", None)
        return wrapper

    def _check_code_safety(self, code: str) -> list[str]:
        """Return a list of safety violations; empty means the code is safe."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]
        checker = _CodeSafetyChecker()
        checker.visit(tree)
        return checker.violations

    def execute_code(self, code_action: str) -> tuple[str, int, bool]:
        """Execute *code_action* in-process and return (output, error_code, is_task_complete)."""
        self._current_tool_outputs = []  # reset per step
        violations = self._check_code_safety(code_action)
        if violations:
            msg = "Code safety check failed:\n" + "\n".join(
                f"  - {v}" for v in violations
            )
            return msg, 1, False

        buf = io.StringIO()
        error_code = 0
        is_task_complete = False
        try:
            compiled = compile(code_action, "<agent_code>", "exec")
            with contextlib.redirect_stdout(buf):
                exec(compiled, self.execution_namespace)  # noqa: S102
        except _TaskCompleteSignal:
            is_task_complete = True
        except Exception:
            error_code = 1
            buf.write(traceback.format_exc())

        return buf.getvalue(), error_code, is_task_complete

    def run(self, task, stream=False, reset=True, additional_args=None, max_steps=None):
        if reset:
            self._init_execution_namespace()
        return super().run(
            task,
            stream=stream,
            reset=reset,
            additional_args=additional_args,
            max_steps=max_steps,
        )

    def step(self, memory_step):
        memory_messages = self.write_memory_to_messages()
        memory_step.model_input_messages = memory_messages.copy()
        _stop_sequences = ["observation:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            _stop_sequences.append(self.code_block_tags[1])

        try:
            additional_args: dict[str, Any] = {}
            if self.grammar:
                additional_args["grammar"] = self.grammar
            additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            chat_message = self.model(
                memory_messages, stop_sequences=_stop_sequences, **additional_args
            )
            memory_step.model_output_message = chat_message
            output_text = chat_message.content
            if output_text and not output_text.strip().endswith(
                self.code_block_tags[1]
            ):
                memory_step.model_output_message.content = (
                    output_text + self.code_block_tags[1]
                )
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(
                f"Error generating model output:\n{e}", self.logger
            ) from e

        self.logger.log_markdown(
            content=output_text, title="LLM Output:", level=LogLevel.DEBUG
        )

        try:
            parsed = self._parse_json_response(output_text)
            code_action = parsed["code"]
            code_action = (
                extract_code_from_text(code_action, self.code_block_tags) or code_action
            )
            memory_step.code_action = code_action
            memory_step.thought = parsed.get("thought", "")
        except Exception as e:
            raise AgentParsingError(f"Error parsing code:\n{e}\nRaw output:\n{output_text[:500]}", self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        self.logger.log_code(
            title="Executing:", content=code_action, level=LogLevel.INFO
        )

        output = ""
        error_code = -1
        is_task_complete = False
        execution_outputs_console = []
        try:
            output, error_code, is_task_complete = self.execute_code(code_action)
            if error_code == 0:
                memory_step.is_action_executed = True
            else:
                memory_step.is_action_executed = False
        except Exception as e:
            if error_code != 0:
                memory_step.observations = output
                memory_step.is_action_executed = False
            raise AgentExecutionError(str(e), self.logger)

        memory_step.observations = truncate_content(output)
        memory_step.tool_outputs = list(self._current_tool_outputs)

        if is_task_complete:
            self.logger.log(f"Task completed.", level=LogLevel.INFO)

        execution_outputs_console.append(
            Text(
                f"{'Task Completed - ' if is_task_complete else 'Out'}: \n{memory_step.observations}",
                style=f"bold {GREEN_HEX}" if is_task_complete else "",
            )
        )
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        return output if is_task_complete else None

    def get_last_jml_code(self) -> Optional[str]:
        """Extract the last JML-annotated Java code from the agent's execution namespace.

        Falls back to scanning action steps in reverse for code passed to
        verify_with_openjml or run_spec_harness.
        """
        # Primary: grab from the shared execution namespace
        code = self.execution_namespace.get("code")
        if code and isinstance(code, str) and "class" in code:
            return code

        # Fallback: scan steps in reverse for the last tool invocation with JML code
        import re

        action_steps = [
            s for s in self.memory.steps if isinstance(s, ActionStep) and s.code_action
        ]
        for step in reversed(action_steps):
            src = step.code_action
            if "verify_with_openjml" in src or "run_spec_harness" in src:
                # Try to extract a triple-quoted Java string
                match = re.search(
                    r"(?:r?'''(.*?)'''|r?\"\"\"(.*?)\"\"\")", src, re.DOTALL
                )
                if match:
                    return match.group(1) or match.group(2)
        return None

    def to_dict(self):
        d = super().to_dict()
        d["executor_kwargs"] = self.executor_kwargs
        d["trajectories"] = self._generate_trajectory()
        return d

    def _generate_trajectory(self):
        tool_names = [t.name for t in self.tools.values()]
        task_steps = [s for s in self.memory.steps if isinstance(s, TaskStep)]
        planning_steps = [s for s in self.memory.steps if isinstance(s, PlanningStep)]
        action_steps = [s for s in self.memory.steps if isinstance(s, ActionStep)]
        steps = []
        for step in action_steps:
            used = find_tool_usage(str(step.code_action), tool_names)
            steps.append(
                {
                    "step_no": step.step_number,
                    "thought": step.thought,
                    "code": step.code_action,
                    "observations": step.observations,
                    "tool_outputs": make_json_serializable(step.tool_outputs or []),
                    "invoked_tools": used,
                    "is_tool_executed": step.is_action_executed,
                }
            )
        return {
            "task": [t.task for t in task_steps if t.task],
            "plan": [p.plan for p in planning_steps if p.plan],
            "steps": steps,
        }
