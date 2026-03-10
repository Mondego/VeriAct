import os
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from baselines.autospec import javalang
from baselines.utils.logger import create_logger
from baselines.utils.file_utility import dump_json, dump_jsonl, load_json
from baselines.utils.verifier import verify_with_openjml, validate_with_openjml
from baselines.autospec.prompts import get_fewshot_context, get_request_msg
from baselines.utils.models import (
    create_model_config,
    request_llm_engine,
    reset_token_usage,
    get_token_usage,
)


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


@dataclass
class InfillPoint:
    lineno: int
    spec_type: str


@dataclass
class SpecEntry:
    content: str
    lineno: int


class AutoSpecResult(TypedDict):
    status: str
    class_name: str
    verifier_calls: int
    final_code: str
    final_error: str
    verified: bool
    iterations: int
    input_tokens: int
    output_tokens: int


class _WorkerResultRequired(TypedDict):
    task_id: str
    status: str
    class_name: str


class WorkerResult(_WorkerResultRequired, total=False):
    config: dict[str, Any]
    verifier_calls: int
    iterations: int
    verified: bool
    log_file: str
    final_code: str
    final_error: str
    message: str
    input_tokens: int
    output_tokens: int


class AutoSpec:

    def __init__(
        self,
        model: str,
        temperature: float,
        max_iterations: int,
        output_dir: str,
        timeout: int,
        logger: logging.Logger,
        verbose: bool = False,
        prompt_type: str = "zero_shot",
        simplify: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose
        self.prompt_type = prompt_type
        self.simplify = simplify

    def _request_models(self, messages: list[dict[str, Any]]) -> Any:
        config = create_model_config(messages, self.model, self.temperature)
        self.logger.debug(
            f"Requesting LLM with model={self.model}, temperature={self.temperature}"
        )
        try:
            response = request_llm_engine(config)
            self.logger.debug("LLM response received successfully")
            return response
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}", exc_info=True)
            raise

    def _filter_validated_specs(
        self, specs: str, code: str, classname: str, lineno: int
    ) -> str:
        original_spec_lines = specs.split("\n")
        self.logger.debug(
            f"[{classname}] Filtering {len(original_spec_lines)} spec lines at line {lineno}"
        )
        self.logger.debug(f"[{classname}] Input specs:\n{specs}")

        instrumented_code = self._instrument_spec_into_code(
            code, [SpecEntry(content=specs, lineno=lineno)], unique=False
        )
        # [Note] Only Syntax validation, no verification
        err_info, returncode = validate_with_openjml(
            instrumented_code, classname, self.timeout, self.output_dir, self.logger
        )
        if self.verbose:
            self.logger.info(f"[{classname}] Validation result:\n{err_info}\n")
        if returncode == 0:
            self.logger.debug(
                f"[{classname}] All specs at line {lineno} passed validation"
            )
            return specs
        err_lineno_list = self._extract_lineno_from_err_info(err_info)
        self.logger.debug(f"[{classname}] Error lines detected: {err_lineno_list}")

        if not err_lineno_list:
            self.logger.warning(
                f"[{classname}] Validation failed at line {lineno} but no error lines could be "
                f"parsed from output. Dropping all specs to avoid passing bad specs through."
            )
            self.logger.debug(f"[{classname}] Unparseable validation error:\n{err_info}")
            return ""

        res = ""
        filtered_count = 0
        for index, spec in enumerate(original_spec_lines):
            if index + lineno in err_lineno_list:
                self.logger.debug(
                    f"[{classname}] Filtered out spec at line {index + lineno}: {spec.strip()}"
                )
                filtered_count += 1
                continue
            else:
                res = res + spec + "\n"

        remaining_lines = len(res.strip().split("\n")) if res.strip() else 0
        self.logger.info(
            f"[{classname}] Filtered {filtered_count}/{len(original_spec_lines)} specs at line {lineno}, {remaining_lines} remaining"
        )
        if self.verbose:
            self.logger.info(
                f"[{classname}] The remaining specs on this point are:\n```\n{res}```\n"
            )
        return res

    def _extract_blank_prefix(self, string: str) -> str:
        string_stripped = string.strip()
        if len(string_stripped) > 0:
            return string.split(string_stripped)[0]
        else:
            return string

    def _request_llm_for_spec_on_single_point(
        self, code: str, classname: str, lineno: int, spec_type: str
    ) -> str:
        code_for_request = ""
        for index, line in enumerate(code.split("\n")):
            if index + 1 == lineno:
                code_for_request = (
                    code_for_request
                    + self._extract_blank_prefix(line)
                    + "// >>>INFILL<<<\n"
                )
            code_for_request = code_for_request + line + "\n"
        context = get_fewshot_context(spec_type, self.prompt_type)
        request_msg = get_request_msg(code_for_request, spec_type)
        if self.verbose:
            self.logger.info(
                f"[{classname}] Requesting LLM for specs on line {lineno} of type {spec_type}..."
            )
        context.append(request_msg)
        if spec_type == "field":
            self.logger.debug(
                f"[{classname}] Using default spec for field at line {lineno}"
            )
            return "//@ spec_public"
        reply_msg = self._request_models(context)
        if self.verbose:
            self.logger.info(f"[{classname}] Received LLM response for line {lineno}\n")
        reply_content = reply_msg.choices[0].message.content.replace("```java", "```")
        reply_content = reply_content.replace("```jml", "```") # Also replace jml code block markers if any, just in case
        if reply_content.strip().startswith("//@"):
            reply_content = "```\n" + reply_content.strip() + "\n```"
        tmp_list = reply_content.split("```")
        if len(tmp_list) < 2:  # Check if LLM has returned any code block
            self.logger.warning(
                f"[{classname}] No code block found in LLM response for line {lineno}"
            )
            self.logger.debug(f"[{classname}] Response content: {reply_content}")
            specs_generated = ""
        else:
            specs_generated = tmp_list[1].strip()
        spec_list = specs_generated.split("\n")
        res = ""
        for line in spec_list:
            if line.strip() == "":
                continue
            res = res + line + "\n"
        final_result = res.strip()
        self.logger.debug(
            f"[{classname}] Generated {len(final_result.split(chr(10)) if final_result else [])} spec lines for line {lineno}"
        )
        return final_result

    def _obtain_infill_points(self, code: str) -> list[InfillPoint]:
        res: list[InfillPoint] = []
        try:
            tree = javalang.parse.parse(code)
        except Exception as e:
            self.logger.error(f"Failed to parse Java code: {e}")
            self.logger.debug(f"Code that failed to parse:\n{code}")
            raise

        for path, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                res.append(InfillPoint(lineno=node.position[0], spec_type="method"))
            elif isinstance(node, javalang.tree.FieldDeclaration):
                res.append(InfillPoint(lineno=node.position[0], spec_type="field"))
            elif isinstance(node, javalang.tree.WhileStatement) or isinstance(
                node, javalang.tree.ForStatement
            ):
                res.append(InfillPoint(lineno=node.position[0], spec_type="loop"))
        return res

    def _instrument_spec_into_code(
        self, code: str, specs_list: list[SpecEntry], unique: bool = True
    ) -> str:
        res = ""
        code_list = code.split("\n")
        tmp_spec_set: set[str] = set()
        for index, line in enumerate(code_list):
            if self._is_spec(line):
                tmp_spec_set.add(line.strip())
            for spec in specs_list:
                if index + 1 != spec.lineno:
                    continue
                for line_of_spec in spec.content.split("\n"):
                    if line_of_spec.strip() == "" or (
                        unique and self._is_in_set(line_of_spec.strip(), tmp_spec_set)
                    ):
                        continue
                    res = res + self._extract_blank_prefix(line) + line_of_spec + "\n"
            if not self._is_spec(line) and line.strip() != "":
                tmp_spec_set.clear()
            res = res + line + "\n"
        return res

    def _is_in_set(self, e: str, s: set[str]) -> bool:
        tmp_set = s
        old_len = len(tmp_set)
        tmp_set.add(e)
        return len(tmp_set) == old_len

    def _extract_lineno_from_err_info(self, err_info: str) -> list[int]:
        lineno_list: list[int] = []
        for line in err_info.split("\n"):
            if (
                line.find(self.output_dir) != -1
                and line.find(".java") != -1
                and line.find(":") != -1
            ):
                try:
                    lineno_list.append(int(line.split(":")[1]))
                except Exception:
                    continue
        return lineno_list

    def _is_spec(self, line: str) -> bool:
        return line.find("@") != -1 and (
            line.find("maintaining") != -1
            or line.find("loop_invariant") != -1
            or line.find("ensures") != -1
            or line.find("requires") != -1
            or line.find("decreases") != -1
            or line.find("invariant") != -1
            or line.find("spec_public") != -1
        )

    def _remove_errornous_and_redundant_spec(
        self, code_with_spec: str, err_info: str
    ) -> str:
        res = ""
        err_lineno_list = self._extract_lineno_from_err_info(err_info)
        tmp_spec_set: set[str] = set()
        for index, line in enumerate(code_with_spec.split("\n")):
            if self._is_spec(line) and (
                index + 1 in err_lineno_list or line.strip() in tmp_spec_set
            ):
                continue
            res = res + line + "\n"
            if self._is_spec(line):
                tmp_spec_set.add(line.strip())
            elif line.strip() != "":
                tmp_spec_set.clear()
        return res

    def _remove_spec_line(self, code: str, spec_line: str) -> str:
        """Remove the first occurrence of spec_line (matched by exact content) from code."""
        lines = code.split("\n")
        removed = False
        result = []
        for line in lines:
            if not removed and line == spec_line:
                removed = True
                continue
            result.append(line)
        return "\n".join(result)

    def _simplify_specs(
        self, code: str, class_name: str, verifier_calls: int
    ) -> tuple[str, int]:
        """
        Remove spec lines one-by-one and re-verify after each removal.
        A spec is redundant if the assertion still holds without it.
        Returns the minimal annotated code and the updated verifier call count.
        """
        spec_lines = [line for line in code.split("\n") if self._is_spec(line)]
        self.logger.info(
            f"[{class_name}] Simplification: trying to remove {len(spec_lines)} spec lines"
        )
        current_code = code
        removed_count = 0

        for spec_line in spec_lines:
            if spec_line not in current_code:
                # Already removed as a side-effect of a previous removal
                continue
            candidate = self._remove_spec_line(current_code, spec_line)
            err_info, returncode = verify_with_openjml(
                candidate, class_name, self.timeout, self.output_dir, self.logger
            )
            verifier_calls += 1
            if "Timeout:" in err_info or "timeout" in err_info.lower():
                self.logger.debug(
                    f"[{class_name}] Timeout while testing removal of: {spec_line.strip()} — keeping"
                )
                continue
            if returncode == 0:
                current_code = candidate
                removed_count += 1
                self.logger.debug(
                    f"[{class_name}] Removed redundant spec: {spec_line.strip()}"
                )
            else:
                self.logger.debug(
                    f"[{class_name}] Kept necessary spec: {spec_line.strip()}"
                )

        self.logger.info(
            f"[{class_name}] Simplification done: removed {removed_count}/{len(spec_lines)} redundant specs"
            f" ({verifier_calls} total verifier calls so far)"
        )
        return current_code, verifier_calls

    def run(self, code: str, class_name: str) -> AutoSpecResult | str:

        _verifier_calls_count: int = 0
        current_code: str = code.strip()
        verified_flag: bool = False
        num_iter: int = 0
        err_info: str = ""
        timed_out: bool = (
            False  # Flag to track if a timeout occurred during verification
        )
        status: str = (
            "unknown"  # haven't tried verification yet, or got an unexpected error during the process
        )

        while num_iter < self.max_iterations and not verified_flag:
            self.logger.info(
                f"[{class_name}] Starting iteration {num_iter + 1}/{self.max_iterations}"
            )
            current_code = current_code.strip()
            num_iter = num_iter + 1
            try:
                infill_points_list = self._obtain_infill_points(current_code)
                self.logger.debug(
                    f"[{class_name}] Found {len(infill_points_list)} infill points: {infill_points_list}"
                )
            except Exception as e:
                self.logger.error(
                    f"[{class_name}] Syntax error occurred when processing current code: {e}",
                    exc_info=True,
                )
                self.logger.debug(
                    f"[{class_name}] Code that caused syntax error:\n{current_code}"
                )
                return "Syntax error in input code or generated code!"
            specs_list: list[SpecEntry] = []
            for point in infill_points_list:
                """
                # Find if the point is already infilled
                found_flag = False
                for spec in specs_list:
                    if spec.lineno == point.lineno:
                        found_flag = True
                # If is, skip this point
                if found_flag:
                    continue
                """
                # Query the LLM for specs on this point
                try:
                    specs_generated = self._request_llm_for_spec_on_single_point(
                        current_code, class_name, point.lineno, point.spec_type
                    )
                except Exception as e:
                    self.logger.error(
                        f"[{class_name}] Failed to generate specs for line {point.lineno}: {e}",
                        exc_info=True,
                    )
                    specs_generated = ""

                # Validate generated specs and remove the ill-formed ones
                if specs_generated:
                    try:
                        specs_generated = self._filter_validated_specs(
                            specs_generated, current_code, class_name, point.lineno
                        )
                        _verifier_calls_count += 1
                    except Exception as e:
                        self.logger.error(
                            f"[{class_name}] Failed to filter specs for line {point.lineno}: {e}",
                            exc_info=True,
                        )
                        specs_generated = ""
                else:
                    self.logger.debug(
                        f"[{class_name}] No specs generated for line {point.lineno}"
                    )

                specs_list.append(
                    SpecEntry(content=specs_generated, lineno=point.lineno)
                )
            specs_list = sorted(specs_list, key=lambda x: x.lineno)
            # Verify the correctness of all generated code
            current_code = self._instrument_spec_into_code(current_code, specs_list)
            if self.verbose:
                self.logger.info(
                    f"Result of iteration {num_iter} is:\n{current_code}\n"
                )
            err_info, returncode = verify_with_openjml(
                current_code, class_name, self.timeout, self.output_dir, self.logger
            )
            _verifier_calls_count += 1
            if self.verbose:
                self.logger.info(
                    f"[{class_name}] OpenJML verification result for iteration {num_iter}:\n{err_info}\n"
                )
            if "Timeout:" in err_info or "timeout" in err_info.lower():  # timeout
                timed_out = True
                current_code = (
                    code.strip()
                )  # reset to original — at least the output is clean
                break

            if returncode == 0:  # verified
                verified_flag = True
                if self.simplify:
                    current_code, _verifier_calls_count = self._simplify_specs(
                        current_code, class_name, _verifier_calls_count
                    )
            else:
                current_code = self._remove_errornous_and_redundant_spec(
                    current_code, err_info
                )

        if verified_flag:
            status = "verified"
            timed_out = False  # Reset if eventually verified
        elif timed_out:
            status = "timed_out"
        else:
            status = "unverified"

        if verified_flag:
            self.logger.info(
                f"[{class_name}] ✓ Successfully verified after {num_iter} iterations and {_verifier_calls_count} verifier calls"
            )
        else:
            self.logger.warning(
                f"[{class_name}] ✗ Max iterations ({self.max_iterations}) reached without verification. "
                f"Final error: {err_info[:200]}..."
            )

        if self.verbose:
            self.logger.info(
                f"[{class_name}] Final result is:\n```\n{current_code}\n```\nVerifier called {_verifier_calls_count} times."
            )
        _input_tokens, _output_tokens = get_token_usage()
        return AutoSpecResult(
            status=status,
            class_name=class_name,
            verifier_calls=_verifier_calls_count,
            final_code=current_code,
            final_error=err_info,
            verified=verified_flag,
            iterations=num_iter,
            input_tokens=_input_tokens,
            output_tokens=_output_tokens,
        )


class AutoSpecRunner:

    def __init__(
        self,
        name: str,
        input: str,
        output: str,
        model: str,
        temperature: float,
        max_iterations: int,
        openjml_timeout: int,
        threads: int,
        verbose: bool,
        prompt_type: str,
        simplify: bool = False,
    ) -> None:
        self.name = name
        self.input_path: str = input
        self.output = output
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.openjml_timeout = openjml_timeout
        self.threads = threads
        self.verbose = verbose
        self.prompt_type = prompt_type
        self.simplify = simplify
        self.input_length: int = 0

    def _run_autospec(self, task: Task) -> WorkerResult:
        classname: str = task.class_name
        input_code: str = task.code
        task_id: str = task.task_id
        output_dir: str = os.path.abspath(self.output)
        run_config: dict[str, Any] = {
            "prompt_type": self.prompt_type,
            "model": self.model,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "simplify": self.simplify,
        }

        _thread_id: Optional[int] = threading.current_thread().ident
        log_file: str = "unknown"
        try:
            logger, log_file = create_logger(task_id, _thread_id, output_dir)
        except Exception as e:
            print(f"Failed to create logger for {task_id}: {e}")
            return WorkerResult(
                task_id=task_id,
                status="error",
                message=f"Logger creation failed: {str(e)}",
                class_name=classname,
                config=run_config,
                log_file="unknown",
            )

        reset_token_usage()
        logger.info(f"Starting AutoSpec for {task_id} (task id: {task_id})")

        _thread_autospec_artifacts: str = os.path.join(
            output_dir, f"{task_id}.Thread_{_thread_id}"
        )
        Path(_thread_autospec_artifacts).mkdir(parents=True, exist_ok=True)

        autospec = AutoSpec(
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            output_dir=_thread_autospec_artifacts,
            timeout=self.openjml_timeout,
            logger=logger,
            verbose=self.verbose,
            prompt_type=self.prompt_type,
            simplify=self.simplify,
        )
        try:
            _result = autospec.run(input_code, classname)

            if _result.get("verified", False):
                verified_status = "✓ VERIFIED"
            elif _result.get("timed_out", False):
                verified_status = "⏱ TIMED OUT"
            else:
                verified_status = "✗ UNVERIFIED"

            logger.info(
                f"{verified_status} - Completed {classname} with {_result['verifier_calls']} verifier calls "
                f"in {_result['iterations']} iterations "
                f"(tokens: {_result['input_tokens']} in / {_result['output_tokens']} out)"
            )

            return WorkerResult(
                task_id=task_id,
                status=_result["status"],
                class_name=classname,
                config=run_config,
                verifier_calls=_result["verifier_calls"],
                iterations=_result["iterations"],
                verified=_result.get("verified", False),
                log_file=log_file,
                final_code=_result["final_code"],
                final_error=_result["final_error"],
                input_tokens=_result["input_tokens"],
                output_tokens=_result["output_tokens"],
            )

        except Exception as e:
            logger.error(f"Error processing {classname}: {e}", exc_info=True)
            return WorkerResult(
                task_id=task_id,
                status="unknown",
                message=str(e),
                class_name=classname,
                config=run_config,
                log_file=log_file,
            )

    def _save_results(self, duration: float, results: list[WorkerResult]) -> None:
        """Save summary statistics to a JSON file"""
        verified = [r for r in results if r["status"] == "verified"]
        unverified = [r for r in results if r["status"] == "unverified"]
        timed_out = [r for r in results if r["status"] == "timed_out"]
        unknown = [r for r in results if r["status"] == "unknown"]

        completed = verified + unverified + timed_out
        total_verifier_calls = sum(r.get("verifier_calls", 0) for r in completed)
        avg_verifier_calls = total_verifier_calls / len(completed) if completed else 0
        total_input_tokens = sum(r.get("input_tokens", 0) for r in completed)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in completed)

        summary = {
            "total_files_processed": self.input_length,
            "verified": len(verified),
            "unverified": len(unverified),
            "timed_out": len(timed_out),
            "unknown": len(unknown),
            "verification_rate_percent": (
                round(len(verified) / self.input_length * 100, 1)
                if self.input_length > 0
                else 0
            ),
            "total_processing_time_seconds": round(duration, 2),
            "total_verifier_calls": total_verifier_calls,
            "average_verifier_calls_per_case": round(avg_verifier_calls, 1),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "threads_used": self.threads,
            "verified_cases": [
                {
                    "task_id": r["task_id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in verified
            ],
            "unverified_cases": [
                {
                    "task_id": r["task_id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in unverified
            ],
            "timed_out_cases": [
                {
                    "task_id": r["task_id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in timed_out
            ],
            "unknown_cases": [
                {
                    "task_id": r["task_id"],
                    "message": r["message"],
                    "class_name": r.get("class_name", "unknown"),
                    "log_file": r.get("log_file", "unknown"),
                }
                for r in unknown
            ],
        }

        print(f"\nProcessing completed in {duration:.2f} seconds")
        print(
            f"Verified: {len(verified)}/{self.input_length} ({len(verified)/self.input_length*100:.1f}%)"
        )
        print(
            f"Unverified: {len(unverified)}/{self.input_length} ({len(unverified)/self.input_length*100:.1f}%)"
        )
        print(
            f"Timed out: {len(timed_out)}/{self.input_length} ({len(timed_out)/self.input_length*100:.1f}%)"
        )
        print(f"Unknown (errors): {len(unknown)}/{self.input_length}")
        print(
            f"Total tokens used: {total_input_tokens} input / {total_output_tokens} output ({total_input_tokens + total_output_tokens} total)"
        )

        dump_jsonl(results, os.path.join(self.output, f"{self.name}_results_all.jsonl"))
        print(
            f"All results saved to: {os.path.join(self.output, f'{self.name}_results_all.jsonl')}"
        )
        dump_json(
            summary, os.path.join(self.output, f"{self.name}_results_summary.json")
        )
        print(
            f"Summary saved to: {os.path.join(self.output, f'{self.name}_results_summary.json')}"
        )

    def run_workers(self) -> None:

        _input_tasks: list[Task] = [
            Task.from_dict(t) for t in load_json(self.input_path)
        ]
        self.input_length = len(_input_tasks)
        print(f"Found {self.input_length} input tasks to process")

        output_dir: str = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        start_time: float = time.time()
        results: list[WorkerResult] = []
        completed_count: int = 0

        print(f"Starting processing with {self.threads} threads...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(self._run_autospec, task): task for task in _input_tasks
            }

            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if result["status"] == "verified":
                        print(
                            f"✓ [{completed_count}/{self.input_length}] {result['class_name']} - VERIFIED - {result['verifier_calls']} verifier calls"
                        )
                    elif result["status"] == "unverified":
                        print(
                            f"✗ [{completed_count}/{self.input_length}] {result['class_name']} - UNVERIFIED - {result['verifier_calls']} verifier calls"
                        )
                    elif result["status"] == "timed_out":
                        print(
                            f"⏱ [{completed_count}/{self.input_length}] {result['class_name']} - TIMED OUT - {result['verifier_calls']} verifier calls"
                        )
                    else:  # error
                        print(
                            f"✗ [{completed_count}/{self.input_length}] {task.class_name} - ERROR: {result.get('message', 'Unknown error')}"
                        )

                except Exception as exc:
                    results.append(
                        WorkerResult(
                            task_id=task.task_id,
                            status="unknown",
                            message=str(exc),
                            class_name=task.class_name,
                        )
                    )
                    print(
                        f"✗ [{completed_count}/{self.input_length}] {task.class_name} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time: float = time.time()
        duration: float = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
