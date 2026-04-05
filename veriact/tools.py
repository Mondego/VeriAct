"""VeriAct tool suite — Tool subclasses with stubs for forward()."""

import json
import logging
import os

from veriact.harness_tool import Task, evaluate_problem
from veriact.tools_base import Tool
from veriact.verifier_tool import (
    VerificationResult,
    classify_failures,
    extract_errors,
    verify_with_openjml as _verify_with_openjml,
)

logger = logging.getLogger(__name__)


REPAIR_HINTS: dict[str, str] = {
    "SyntaxError": "Fix JML syntax (missing semicolons, wrong keywords).",
    "PostconditionFailure": "The @ensures clause is logically incorrect.",
    "ExceptionalPostconditionFailure": "The @signals clause is incorrect.",
    "PreconditionFailure": "The @requires clause is too weak or wrong.",
    "LoopInvariantFailure": "The @maintaining clause doesn't hold.",
    "RankingFunctionFailure": "The @decreases expression is wrong.",
    "AssignableFailure": "The @assignable clause is too permissive or missing.",
    "ArrayIndex": "Missing array bounds check in @requires.",
    "NegativeSize": "Array size may be negative; add check in @requires.",
    "NullDeReference": "Missing null check in @requires.",
    "NullUnbox": "Potential null unboxing; add null check in @requires.",
    "DivideByZero": "Missing division-by-zero guard in @requires.",
    "ArithmeticOperationRange": "Integer overflow not guarded in @requires.",
    "ArithmeticCastRange": "Cast may overflow; add range guard in @requires.",
    "BadCast": "Unsafe cast; add type guard in @requires.",
    "BadArrayAssignment": "Incompatible array assignment; check element types.",
    "CalledMethodPrecondition": "Called method precondition not met; strengthen @requires.",
    "LargeShift": "Shift amount out of range; add bounds check in @requires.",
    "AssertFailure": "An @assert statement fails; check the invariant.",
    "UnknownVerificationFailure": "Unknown verification failure; review the full OpenJML log.",
}


class VerifyJMLTool(Tool):
    """Verify JML-annotated Java code using OpenJML (ESC mode)."""

    name = "verify_with_openjml"
    description = (
        "Run OpenJML Extended Static Checking on JML-annotated Java code. "
        "Returns verification status (verified/failed/unknown) and any error "
        "messages from the verifier. Error messages include line numbers and "
        "specific failure reasons."
    )
    inputs = {
        "jml_annotated_code": {
            "type": "string",
            "description": "Complete Java source with JML annotations.",
        }
    }
    output_type = "string"

    def __init__(self, openjml_path: str = "openjml", output_dir: str = "veriact_outputs"):
        super().__init__()
        self._openjml_path = openjml_path
        self._output_dir = os.path.join(output_dir, "tmp")

    def forward(self, jml_annotated_code: str) -> str:
        result: VerificationResult = _verify_with_openjml(
            jml_annotated_code,
            classname="Solution",
            output_dir=self._output_dir,
        )
        return json.dumps(
            {
                "verified": result.success,
                "return_code": result.return_code,
                "errors": result.classified_errors,
                "raw_output": result.error_log,
            }
        )


class AnalyzeErrorTool(Tool):
    """Classify failure modes from OpenJML output."""

    name = "analyze_openjml_errors"
    description = (
        "Analyze verification results to classify failures and produce repair hints."
    )
    inputs = {
        "openjml_log": {"type": "string", "description": "Raw OpenJML output."},
    }
    output_type = "string"

    def forward(self, openjml_log: str) -> str:
        errors = extract_errors(openjml_log)

        failure_modes: list[str] = []
        raw_errors: list[str] = []
        repair_hints: list[str] = []
        seen_hints: set[str] = set()

        for error_level, error_msg in errors:
            failure_type = classify_failures(error_level, error_msg)
            failure_modes.append(failure_type)
            raw_errors.append(error_msg[:500])
            hint = REPAIR_HINTS.get(failure_type)
            if hint and failure_type not in seen_hints:
                repair_hints.append(f"{failure_type}: {hint}")
                seen_hints.add(failure_type)

        if failure_modes:
            counts: dict[str, int] = {}
            for fm in failure_modes:
                counts[fm] = counts.get(fm, 0) + 1
            summary = ", ".join(
                f"{t} (x{c})" if c > 1 else t for t, c in counts.items()
            )
        else:
            summary = "No failures detected."

        return json.dumps(
            {
                "failure_modes": failure_modes,
                "raw_errors": raw_errors,
                "repair_hints": repair_hints,
                "summary": summary,
            }
        )


class RunHarnessTool(Tool):
    """Evaluate spec quality using the spec-harness 4-metric."""

    name = "run_spec_harness"
    description = (
        "Evaluate JML spec quality: PostCorrectness, PostCompleteness, "
        "PreCorrectness, PreCompleteness. Returns scores for each metric."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "Identifier for the task being evaluated.",
        },
        "jml_annotated_code": {
            "type": "string",
            "description": "LLM generated complete Java source with JML annotations.",
        },
    }
    output_type = "string"

    def __init__(self, openjml_path: str = "openjml", dataset_path: str = "", output_dir: str = "veriact_outputs"):
        super().__init__()
        self._openjml_path = openjml_path
        self._dataset_path = dataset_path
        self._output_dir = output_dir
        self._task_cache: dict[str, Task] = {}
        self._run_counter: dict[str, int] = {}  # per-task run counter

    def _load_task(self, task_id: str) -> Task:
        if task_id in self._task_cache:
            return self._task_cache[task_id]
        if not self._dataset_path:
            raise ValueError(
                "dataset_path not configured; cannot look up task by id. "
                "Pass dataset_path= when constructing RunHarnessTool."
            )
        with open(self._dataset_path) as fh:
            raw = fh.read().strip()
        if raw.startswith("["):
            records = json.loads(raw)
        else:
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for data in records:
            if data.get("task_id") == task_id:
                task = Task.from_dict(data)
                self._task_cache[task_id] = task
                return task
        raise KeyError(
            f"task_id '{task_id}' not found in dataset '{self._dataset_path}'"
        )

    def forward(self, task_id: str, jml_annotated_code: str) -> str:
        try:
            task = self._load_task(task_id)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            return json.dumps({"error": str(exc), "task_id": task_id})

        # Increment run counter for this task
        self._run_counter[task_id] = self._run_counter.get(task_id, 0) + 1
        run_id = f"run_{self._run_counter[task_id]}"

        scores = evaluate_problem(
            task,
            llm_code=jml_annotated_code,
            openjml_path=self._openjml_path,
            output_dir=self._output_dir,
            run_id=run_id,
        )

        if not scores:
            return json.dumps(
                {
                    "error": "No test pairs could be parsed for this task.",
                    "task_id": task_id,
                }
            )

        return json.dumps(
            {
                task_id: {
                    "post_correctness": scores.get("post_correctness", 0.0),
                    "post_completeness": scores.get("post_completeness", 0.0),
                    "pre_correctness": scores.get("pre_correctness", 0.0),
                    "pre_completeness": scores.get("pre_completeness", 0.0),
                }
            }
        )


def get_veriact_tools(
    openjml_path: str = "openjml",
    dataset_path: str = "",
    output_dir: str = "veriact_outputs",
) -> list[Tool]:
    return [
        VerifyJMLTool(openjml_path=openjml_path, output_dir=output_dir),
        AnalyzeErrorTool(),
        RunHarnessTool(openjml_path=openjml_path, dataset_path=dataset_path, output_dir=output_dir),
    ]


TOOL_NAMES = ["verify_with_openjml", "analyze_openjml_errors", "run_spec_harness"]
