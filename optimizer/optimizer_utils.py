import logging
import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("optimizer_utils")

# ============================================================================
# 1. TASK SCHEMA (from your benchmark)
# ============================================================================


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
            javadoc=data["javadoc"],
            category=data["category"],
            origin_id=data["origin_id"],
            test_code=data.get("test_code", ""),
            test_inputs=[TestCase.from_dict(tc) for tc in data.get("test_inputs", [])],
            generated_test_cases=[
                TestCase.from_dict(tc) for tc in data.get("generated_test_cases", [])
            ],
        )


# ============================================================================
# 2. ERROR CLASSIFICATION (from your error_utils.py)
# ============================================================================


def extract_errors(error_message: str) -> list[tuple[str, str]]:
    pattern = r"(/tmp/[^:]+:\d+: )(\w+):"
    matches = re.split(r"(?=/tmp/[^:]+:\d+: \w+:)", error_message)
    errors = []
    for match in matches:
        if match.strip():
            level_match = re.search(pattern, match)
            if level_match:
                error_level = level_match.group(2)
                if error_level != "warning":
                    if (
                        "Associated declaration:" in match
                        or "Associated method exit" in match
                    ):
                        if len(errors) == 0:
                            continue
                        errors[-1] = (errors[-1][0], errors[-1][1] + match)
                    else:
                        errors.append((error_level, match.strip()))
    return errors


def verification_failure_map(error_message: str) -> str:
    mapping = [
        ("LoopInvariantBeforeLoop", "LoopInvariantFailure"),
        ("ArithmeticOperationRange", "ArithmeticOperationRange"),
        ("Assignable", "AssignableFailure"),
        ("Postcondition:", "PostconditionFailure"),
        ("Assert)", "AssertFailure"),
        ("UndefinedNullDeReference", "NullDeReference"),
        ("PossiblyNullDeReference", "NullDeReference"),
        ("LoopInvariant)", "LoopInvariantFailure"),
        ("PossiblyNegativeIndex", "ArrayIndex"),
        ("PossiblyNegativeSize", "NegativeSize"),
        ("PossiblyTooLargeIndex", "ArrayIndex"),
        ("LoopDecreases", "RankingFunctionFailure"),
        ("PossiblyBadArrayAssignment", "BadArrayAssignment"),
        ("Precondition:", "PreconditionFailure"),
        ("Precondition conjunct is false:", "PreconditionFailure"),
        ("UndefinedTooLargeIndex", "ArrayIndex"),
        ("PossiblyDivideByZero", "DivideByZero"),
        ("PossiblyBadCast", "BadCast"),
        ("UndefinedDivideByZero", "DivideByZero"),
        ("ExceptionalPostcondition:", "ExceptionalPostconditionFailure"),
        ("UndefinedCalledMethodPrecondition:", "CalledMethodPrecondition"),
        ("UndefinedNegativeIndex", "ArrayIndex"),
        ("PossiblyNullUnbox", "NullUnbox"),
        ("LoopDecreasesNonNegative", "RankingFunctionFailure"),
        ("Postcondition)", "PostconditionFailure"),
        ("ArithmeticCastRange", "ArithmeticCastRange"),
        ("UndefinedNullUnbox", "NullUnbox"),
        ("PossiblyLargeShift", "LargeShift"),
    ]
    for pattern, failure_type in mapping:
        if pattern in error_message:
            return failure_type
    return "UnknownVerificationFailure"


def classify_failures(error_level: str, error_message: str) -> str:
    if error_level == "error":
        return "SyntaxError"
    elif error_level == "verify":
        return verification_failure_map(error_message)
    else:
        return "UnknownError"


# ============================================================================
# 3. OPENJML VERIFICATION
# ============================================================================


@dataclass
class VerificationResult:
    success: bool
    error_log: str
    return_code: int
    classified_errors: list[dict] = field(default_factory=list)


def write_to_file(content: str, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)


def verify_with_openjml(
    code_with_spec: str,
    classname: str,
    timeout: int = 300,
    output_dir: Optional[str] = None,
) -> VerificationResult:
    """Run OpenJML ESC verification and classify errors."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    tmp_filename = os.path.join(output_dir, f"{classname}.java")
    write_to_file(code_with_spec, tmp_filename)

    cmd = [
        "openjml",
        "--esc",
        "--esc-max-warnings",
        "1",
        "--prover=cvc4",
        "--nonnull-by-default",
        "--arithmetic-failure=quiet",
        "-nowarn",
        tmp_filename,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        raw_output = stdout + stderr
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        raw_output = "Timeout: OpenJML verification exceeded time limit"
        return_code = 1
    except Exception as e:
        raw_output = f"Error: {str(e)}"
        return_code = 1
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    # Classify errors
    classified_errors = []
    if return_code != 0 and raw_output.strip():
        try:
            errors = extract_errors(raw_output)
            for error_level, error_msg in errors:
                error_type = classify_failures(error_level, error_msg)
                classified_errors.append(
                    {
                        "type": error_type,
                        "raw": error_msg[:500],
                    }
                )
        except Exception:
            classified_errors.append(
                {
                    "type": "ParseError",
                    "raw": raw_output[:500],
                }
            )

    success = return_code == 0 and len(classified_errors) == 0

    return VerificationResult(
        success=success,
        error_log=raw_output,
        return_code=return_code,
        classified_errors=classified_errors,
    )


# ============================================================================
# 4. HELPERS
# ============================================================================


def clean_code_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = re.sub(r"```java\s*\n?", "", text)
    text = re.sub(r"```\s*$", "", text)
    return text.strip()


def format_error_feedback(result: VerificationResult, task_id: str) -> str:
    """Format classified errors as structured feedback for GEPA reflection."""
    if result.success:
        return f"[PASS] Task '{task_id}': Specification verified by OpenJML."

    error_counts: dict[str, int] = {}
    for err in result.classified_errors:
        t = err["type"]
        error_counts[t] = error_counts.get(t, 0) + 1

    error_summary = ", ".join(
        f"{t} (x{c})" if c > 1 else t for t, c in error_counts.items()
    )

    raw_log = result.error_log
    if len(raw_log) > 800:
        raw_log = raw_log[:800] + "\n... [truncated]"

    return (
        f"[FAIL] Task '{task_id}': {error_summary}\n"
        f"--- OpenJML Log ---\n{raw_log}\n"
        f"--- Fix Guidance ---\n"
        f"SyntaxError: Fix JML syntax (missing semicolons, wrong keywords).\n"
        f"PostconditionFailure: The @ensures clause is logically incorrect.\n"
        f"LoopInvariantFailure: The @maintaining clause doesn't hold at loop entry or is not preserved.\n"
        f"RankingFunctionFailure: The @decreases expression is wrong or not non-negative.\n"
        f"ArrayIndex: Missing array bounds check in @requires.\n"
        f"NullDeReference: Missing null check in @requires.\n"
        f"ArithmeticOperationRange: Integer overflow not guarded."
    )


# ============================================================================
# 5. GRADUATED SCORING
# ============================================================================


def compute_graduated_score(result: VerificationResult) -> float:
    """
    Graduated scoring based on error classification.
    Gives the optimizer a gradient to climb instead of sparse binary signal.

    Scoring:
        1.0  — Verification passes (correct spec)
        0.3  — 1 verification error (almost correct, e.g., one wrong postcondition)
        0.1  — 2+ verification errors (structurally valid JML, semantically wrong)
        0.0  — Syntax error or empty output (LLM didn't produce valid JML)
    """
    if result.success:
        return 1.0

    if not result.classified_errors:
        return 0.0

    # Separate syntax errors from verification failures
    syntax_errors = [e for e in result.classified_errors if e["type"] == "SyntaxError"]
    verification_errors = [
        e for e in result.classified_errors if e["type"] != "SyntaxError"
    ]

    # Any syntax error → 0.0 (spec is not even valid JML)
    if syntax_errors:
        return 0.0

    # Only verification failures from here
    if len(verification_errors) == 1:
        return 0.3  # almost correct
    else:
        return 0.1  # structurally ok, semantically wrong
