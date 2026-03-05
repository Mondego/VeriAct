import os
import re
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from baselines.formalbench.failure_analysis import extract_errors, classify_failures
from baselines.formalbench.example import _GUIDANCE
from baselines.utils.logger import create_logger
from baselines.utils.models import create_model_config, request_llm_engine
from baselines.utils.file_utility import (
    load_json,
    dump_json,
    dump_jsonl
)
from baselines.utils.verifier import verify_with_openjml


FIX_SYS_MESSAGE = (
    "You are an expert on Java Modeling Language (JML). "
    "Your task is to fix the JML specifications annotated in the target Java code. "
    "You will be provided the error messages from the OpenJML tool and you need to "
    "fix the specifications accordingly."
)

_FIX_USER_TEMPLATE = (
    "The following Java code is annotated with JML specifications:\n"
    "```\n"
    "{curr_spec}\n"
    "```\n"
    "OpenJML Verification tool failed to verify the specifications given above, "
    "with error information as follows:\n\n"
    "### ERROR MESSAGE:\n"
    "```\n"
    "{curr_error}\n"
    "```\n\n"
    "### ERROR TYPES:\n"
    "{error_info}\n"
    "Please refine the specifications so that they can pass verification. "
    "Provide the specifications for the code and include the solution written "
    "between triple backticks, after `### FIXED SPECIFICATION`.\n"
)

_ERROR_INFO_TEMPLATE = (
    "- Error Type: {error_type}\n"
    "{error_description}\n"
    "{fix_instructions}\n\n"
)


class SpecFixer:

    def __init__(
        self,
        model,
        temperature,
        max_iters,
        output_dir,
        timeout,
        logger,
        verbose=False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_iters = max_iters
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _contains_annotations(self, java_code):
        lines = java_code.splitlines()
        inside_block_comment = False
        single_line_jml = re.compile(r"^\s*//@.*")
        block_comment_start = re.compile(r"^\s*/\*@")
        block_comment_end = re.compile(r"\s*\*/")

        for line in lines:
            if single_line_jml.match(line):
                return True
            if inside_block_comment:
                if block_comment_end.match(line):
                    inside_block_comment = False
            elif block_comment_start.match(line):
                inside_block_comment = True
                return True

        return False

    def _parse_spec_from_response(self, response):
        if "### SPECIFICATION" in response:
            response = response.split("### SPECIFICATION")[-1]
        if "### FIXED SPECIFICATION" in response:
            response = response.split("### FIXED SPECIFICATION")[-1]
        if "### RESPONSE" in response:
            response = response.split("### RESPONSE")[-1]

        if "```" not in response:
            return response.strip()

        if "```java" in response:
            pattern = r"```java(.*?)```"
        else:
            pattern = r"```(.*?)```"

        code_blocks = re.findall(pattern, response, re.DOTALL)
        return "\n// block\n".join(code_blocks)

    def _analyze_failures(self, err_info):
        """Classify OpenJML errors and return a formatted guidance string."""
        error_set = set()

        if "NOT IMPLEMENTED:" in err_info:
            if r"\sum" in err_info or r"\num_of" in err_info or r"\product" in err_info:
                error_set.add("UnsupportedSumNumOfProductQuantifierExpression")
            if r"\min" in err_info or r"\max" in err_info:
                error_set.add("UnsupportedMinMaxQuantifierExpression")
        else:
            errors = extract_errors(err_info)
            for level, error in errors:
                try:
                    failure_type = classify_failures(level, error)
                    if failure_type is not None:
                        error_set.add(failure_type)
                except ValueError:
                    self.logger.debug(f"Unknown failure type for error: {error[:100]}")

        error_info = ""
        for error in error_set:
            if error in _GUIDANCE:
                error_info += _ERROR_INFO_TEMPLATE.format(
                    error_type=error,
                    error_description=_GUIDANCE[error]["description"],
                    fix_instructions=_GUIDANCE[error]["guidance"],
                )
        return error_info

    def _build_fix_messages(self, curr_spec, err_info, error_info, history):
        """
        Build the full message list for a fix request.

        history is a flat list of prior {"role", "content"} dicts accumulated
        across iterations (user turns + assistant turns, no system message).
        The system message is always prepended here.
        """
        system_role = "user" if self.model == "o1-mini" else "system"

        new_user_msg = {
            "role": "user",
            "content": _FIX_USER_TEMPLATE.format(
                curr_spec=curr_spec,
                curr_error=err_info,
                error_info=error_info,
            ),
        }

        return (
            [{"role": system_role, "content": FIX_SYS_MESSAGE}]
            + history
            + [new_user_msg]
        )

    # ------------------------------------------------------------------
    # Main repair loop
    # ------------------------------------------------------------------

    def repair(self, input_spec, input_err_info, class_name):
        """
        Iteratively ask the LLM to fix a failing spec.

        Parameters
        ----------
        input_spec      : the annotated Java code that failed verification
        input_err_info  : the raw OpenJML error string from the first run
        class_name      : Java class name (used for tmp file naming and logs)

        Returns
        -------
        dict with keys: status, class_name, verifier_calls, iterations,
                        final_code, final_error, verified
        """
        self.logger.info(
            f"[{class_name}] Starting SpecFixer (max_iters={self.max_iters})..."
        )

        curr_spec = input_spec
        err_info = input_err_info
        verifier_calls = 0
        status = "unverified"
        history = []  # accumulated conversation turns (no system msg)

        for num_iter in range(1, self.max_iters + 1):
            self.logger.info(
                f"[{class_name}] Fix iteration {num_iter}/{self.max_iters}"
            )

            error_info = self._analyze_failures(err_info)
            if self.verbose:
                self.logger.debug(f"[{class_name}] Analyzed error types:\n{error_info[:300]}")

            messages = self._build_fix_messages(curr_spec, err_info, error_info, history)
            config = create_model_config(messages, self.model, self.temperature)
            ret = request_llm_engine(config)

            if ret is None:
                self.logger.error(f"[{class_name}] LLM returned no response on iter {num_iter}")
                status = "unknown"
                break

            raw_response = ret.choices[0].message.content
            if self.verbose:
                self.logger.debug(f"[{class_name}] LLM response: {raw_response[:500]}")

            # Append this turn (user + assistant) to history for next iteration
            history.append(messages[-1])  # the user turn we just sent
            history.append({"role": "assistant", "content": raw_response})

            new_spec = self._parse_spec_from_response(raw_response)

            if not new_spec:
                self.logger.warning(f"[{class_name}] Empty spec returned on iter {num_iter}")
                status = "empty_spec"
                break

            if not self._contains_annotations(new_spec):
                self.logger.warning(
                    f"[{class_name}] No JML annotations detected on iter {num_iter}"
                )
                status = "invalid_jml"
                break

            curr_spec = new_spec
            self.logger.info(f"[{class_name}] Verifying fixed spec...")
            err_info = verify_with_openjml(curr_spec, class_name, self.timeout, self.output_dir, self.logger)
            verifier_calls += 1

            if self.verbose:
                self.logger.debug(
                    f"[{class_name}] Verification output: {err_info[:500]}"
                )

            if "Timeout:" in err_info or "timeout" in err_info.lower():
                self.logger.warning(f"[{class_name}] Verification timed out")
                status = "timed_out"
                break

            if err_info == "":
                self.logger.info(
                    f"[{class_name}] Verified after {num_iter} fix iteration(s) "
                    f"and {verifier_calls} verifier call(s)"
                )
                status = "verified"
                break

            self.logger.debug(
                f"[{class_name}] Still failing after iter {num_iter}: {err_info[:200]}"
            )

        else:
            # Loop exhausted without breaking
            self.logger.warning(
                f"[{class_name}] Max fix iterations ({self.max_iters}) reached without verification"
            )
            status = "unverified"

        verified = status == "verified"
        return {
            "status": status,
            "class_name": class_name,
            "verifier_calls": verifier_calls,
            "iterations": num_iter,
            "final_code": curr_spec,
            "final_error": err_info,
            "verified": verified,
        }


class SpecFixerWorker:

    def __init__(self, output_dir, model, temperature, max_iters, timeout, verbose=False):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.max_iters = max_iters
        self.timeout = timeout
        self.verbose = verbose

    def run_spec_fixer(self, task: dict):
        class_name = task["class_name"]
        input_spec = task["spec"]
        input_err_info = task["err_info"]
        task_id = task["id"]

        thread_id = threading.current_thread().ident
        try:
            logger, log_file = create_logger(class_name, thread_id, self.output_dir)
        except Exception as e:
            print(f"Failed to create logger for {class_name}: {e}")
            return {
                "id": task_id,
                "status": "error",
                "message": f"Logger creation failed: {str(e)}",
                "class_name": class_name,
                "log_file": "unknown",
            }

        logger.info(f"Starting SpecFixer for {class_name} (task id: {task_id})")

        fixer = SpecFixer(
            model=self.model,
            temperature=self.temperature,
            max_iters=self.max_iters,
            output_dir=self.output_dir,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
        )

        try:
            _result = fixer.repair(input_spec, input_err_info, class_name)

            if _result.get("verified", False):
                verified_status = "VERIFIED"
            elif _result["status"] == "timed_out":
                verified_status = "TIMED OUT"
            else:
                verified_status = "UNVERIFIED"

            logger.info(
                f"{verified_status} - Completed {class_name} with "
                f"{_result['verifier_calls']} verifier call(s) "
                f"in {_result['iterations']} iteration(s)"
            )

            return {
                "id": task_id,
                "status": _result["status"],
                "class_name": class_name,
                "verifier_calls": _result["verifier_calls"],
                "iterations": _result["iterations"],
                "verified": _result.get("verified", False),
                "log_file": log_file,
                "final_code": _result["final_code"],
                "final_error": _result["final_error"],
            }

        except Exception as e:
            logger.error(f"[{class_name}] Unexpected error: {e}", exc_info=True)
            return {
                "id": task_id,
                "status": "unknown",
                "message": str(e),
                "class_name": class_name,
                "log_file": log_file,
            }


class SpecFixerRunner:

    def __init__(
        self,
        name,
        input,
        output,
        model,
        temperature,
        max_iters,
        openjml_timeout,
        threads,
        verbose,
    ):
        self.name = name
        self.input = input
        self.output = output
        self.model = model
        self.temperature = temperature
        self.max_iters = max_iters
        self.openjml_timeout = openjml_timeout
        self.threads = threads
        self.verbose = verbose

    def _save_results(self, duration, results):
        verified = [r for r in results if r["status"] == "verified"]
        unverified = [r for r in results if r["status"] == "unverified"]
        timed_out = [r for r in results if r["status"] == "timed_out"]
        invalid_jml = [r for r in results if r["status"] == "invalid_jml"]
        empty_spec = [r for r in results if r["status"] == "empty_spec"]
        unknown = [r for r in results if r["status"] in ("unknown", "error")]

        counted = verified + unverified + timed_out
        total_verifier_calls = sum(r.get("verifier_calls", 0) for r in counted)
        avg_verifier_calls = total_verifier_calls / len(counted) if counted else 0

        summary = {
            "total_files_processed": self.input_length,
            "verified": len(verified),
            "unverified": len(unverified),
            "timed_out": len(timed_out),
            "invalid_jml": len(invalid_jml),
            "empty_spec": len(empty_spec),
            "unknown": len(unknown),
            "verification_rate_percent": (
                round(len(verified) / self.input_length * 100, 1)
                if self.input_length > 0
                else 0
            ),
            "total_processing_time_seconds": round(duration, 2),
            "total_verifier_calls": total_verifier_calls,
            "average_verifier_calls_per_case": round(avg_verifier_calls, 1),
            "threads_used": self.threads,
            "max_iters": self.max_iters,
            "verified_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "iterations": r["iterations"],
                    "log_file": r["log_file"],
                }
                for r in verified
            ],
            "unverified_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "iterations": r["iterations"],
                    "log_file": r["log_file"],
                }
                for r in unverified
            ],
            "timed_out_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "iterations": r["iterations"],
                    "log_file": r["log_file"],
                }
                for r in timed_out
            ],
            "unknown_cases": [
                {
                    "id": r["id"],
                    "message": r.get("message", ""),
                    "class_name": r.get("class_name", "unknown"),
                    "log_file": r.get("log_file", "unknown"),
                }
                for r in unknown
            ],
        }

        dump_jsonl(results, os.path.join(self.output, f"{self.name}_results_all.jsonl"))
        print(f"All results saved to: {os.path.join(self.output, f'{self.name}_results_all.jsonl')}")

        dump_json(summary, os.path.join(self.output, f"{self.name}_results_summary.json"))

        print(f"\nProcessing completed in {duration:.2f} seconds")
        print(f"Verified:     {len(verified)}/{self.input_length} ({len(verified)/self.input_length*100:.1f}%)")
        print(f"Unverified:   {len(unverified)}/{self.input_length} ({len(unverified)/self.input_length*100:.1f}%)")
        print(f"Timed out:    {len(timed_out)}/{self.input_length} ({len(timed_out)/self.input_length*100:.1f}%)")
        print(f"Invalid JML:  {len(invalid_jml)}/{self.input_length}")
        print(f"Empty spec:   {len(empty_spec)}/{self.input_length}")
        print(f"Unknown:      {len(unknown)}/{self.input_length}")
        print(f"Summary saved to: {os.path.join(self.output, f'{self.name}_results_summary.json')}")

    def run_workers(self):
        _input_tasks = load_json(self.input)
        self.input_length = len(_input_tasks)
        print(f"Found {self.input_length} input tasks to process")

        output_dir = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        worker = SpecFixerWorker(
            output_dir=output_dir,
            model=self.model,
            temperature=self.temperature,
            max_iters=self.max_iters,
            timeout=self.openjml_timeout,
            verbose=self.verbose,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} thread(s)...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(worker.run_spec_fixer, task): task
                for task in _input_tasks
            }

            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    status = result["status"]
                    name = result["class_name"]
                    prefix = f"[{completed_count}/{self.input_length}]"

                    if status == "verified":
                        print(f"[VERIFIED]   {prefix} {name} - {result['verifier_calls']} verifier call(s), {result['iterations']} iter(s)")
                    elif status == "unverified":
                        print(f"[UNVERIFIED] {prefix} {name} - {result['verifier_calls']} verifier call(s), {result['iterations']} iter(s)")
                    elif status == "timed_out":
                        print(f"[TIMED OUT]  {prefix} {name}")
                    elif status == "invalid_jml":
                        print(f"[INVALID JML]{prefix} {name}")
                    elif status == "empty_spec":
                        print(f"[EMPTY SPEC] {prefix} {name}")
                    else:
                        print(f"[ERROR]      {prefix} {name} - {result.get('message', 'unknown error')}")

                except Exception as exc:
                    results.append(
                        {
                            "id": task["id"],
                            "status": "unknown",
                            "message": str(exc),
                            "class_name": task.get("class_name", "unknown"),
                        }
                    )
                    completed_count += 1
                    print(f"[ERROR]      [{completed_count}/{self.input_length}] {task.get('class_name', task['id'])} - {exc}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
