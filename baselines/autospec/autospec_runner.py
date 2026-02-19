import os
import time
import threading
import subprocess
from pathlib import Path

import javalang
from concurrent.futures import ThreadPoolExecutor, as_completed
from baselines.autospec.prompts import get_fewshot_context, get_request_msg
from baselines.utils.logger import create_logger
from baselines.utils.file_utility import dump_json, dump_jsonl, load_json, write_to_file
from baselines.utils.models import create_model_config, request_llm_engine


class AutoSpec:

    def __init__(
        self,
        model,
        temperature,
        max_iterations,
        output_dir,
        timeout,
        logger,
        verbose=False,
    ):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

    def _request_models(self, messages):
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

    def _verify_with_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.info(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.output_dir, "tmp")
        Path(tmp_dir).mkdir(exist_ok=True)

        tmp_filename = os.path.join(tmp_dir, f"{classname}.java")
        try:
            write_to_file(code_with_spec, tmp_filename)
            self.logger.debug(f"[{classname}] Wrote code to {tmp_filename}")
        except Exception as e:
            self.logger.error(f"[{classname}] Failed to write file: {e}", exc_info=True)
            raise

        cmd = f"openjml --esc --esc-max-warnings 1 --arithmetic-failure=quiet --nonnull-by-default --quiet -nowarn --prover=cvc4 {tmp_filename}"
        self.logger.debug(f"[{classname}] Running OpenJML verification command")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            res = result.stdout + result.stderr
            self.logger.debug(f"[{classname}] OpenJML return code: {result.returncode}")
            if result.stdout:
                self.logger.debug(
                    f"[{classname}] OpenJML stdout: {result.stdout[:500]}"
                )
            if result.stderr:
                self.logger.debug(
                    f"[{classname}] OpenJML stderr: {result.stderr[:500]}"
                )
            return res
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"[{classname}] OpenJML command timed out after 120 seconds"
            )
            return "Timeout: OpenJML verification exceeded time limit"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

    def _validate_with_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.info(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.output_dir, "tmp")
        Path(tmp_dir).mkdir(exist_ok=True)

        tmp_filename = os.path.join(tmp_dir, f"{classname}.java")
        try:
            write_to_file(code_with_spec, tmp_filename)
            self.logger.debug(f"[{classname}] Wrote code to {tmp_filename}")
        except Exception as e:
            self.logger.error(f"[{classname}] Failed to write file: {e}", exc_info=True)
            raise

        # [FIX ME] For validation this command will change
        cmd = f"openjml --esc --esc-max-warnings 1 --arithmetic-failure=quiet --nonnull-by-default --quiet -nowarn --prover=cvc4 {tmp_filename}"
        self.logger.debug(f"[{classname}] Running OpenJML verification command")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            res = result.stdout + result.stderr
            self.logger.debug(f"[{classname}] OpenJML return code: {result.returncode}")
            if result.stdout:
                self.logger.debug(
                    f"[{classname}] OpenJML stdout: {result.stdout[:500]}"
                )
            if result.stderr:
                self.logger.debug(
                    f"[{classname}] OpenJML stderr: {result.stderr[:500]}"
                )
            return res
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"[{classname}] OpenJML command timed out after 120 seconds"
            )
            return "Timeout: OpenJML verification exceeded time limit"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"
    

    def _filter_validated_specs(
        self, specs: str, code: str, classname: str, lineno: int
    ) -> str:
        original_spec_lines = specs.split("\n")
        self.logger.debug(
            f"[{classname}] Filtering {len(original_spec_lines)} spec lines at line {lineno}"
        )
        self.logger.debug(f"[{classname}] Input specs:\n{specs}")

        instrumented_code = self._instrument_spec_into_code(
            code, [{"content": specs, "lineno": lineno}], unique=False
        )
        err_info = self._validate_with_openjml(instrumented_code, classname)
        if self.verbose:
            self.logger.info(
                f"[{classname}] Validation result:\n{err_info}\n==============================\n"
            )
        err_lineno_list = self._extract_lineno_from_err_info(err_info)
        self.logger.debug(f"[{classname}] Error lines detected: {err_lineno_list}")

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
                f"[{classname}] The remaining specs on this point are:\n```\n{res}```\n===\n"
            )
        return res

    def _extract_blank_prefix(self, string):
        string_stripped = string.strip()
        if len(string_stripped) > 0:
            return string.split(string_stripped)[0]
        else:
            return string

    def _request_llm_for_spec_on_single_point(
        self, code: str, classname: str, lineno: int, type: str
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
        context = get_fewshot_context(type)
        request_msg = get_request_msg(code_for_request, type)
        if self.verbose:
            self.logger.info(
                f"[{classname}] Requesting LLM for specs on line {lineno} of type {type}..."
            )
        context.append(request_msg)
        if type == "field":
            self.logger.debug(
                f"[{classname}] Using default spec for field at line {lineno}"
            )
            reply_msg = {"role": "assistant", "content": "```\n//@ spec_public\n```"}
        else:
            reply_msg = self._request_models(context)
        if self.verbose:
            self.logger.info(
                f"[{classname}] Received LLM response for line {lineno}\n==============================\n"
            )
        reply_content = reply_msg.choices[0].message.content.replace("```java", "```")
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

    def _obtain_infill_points(self, code: str) -> list:
        res = []
        try:
            tree = javalang.parse.parse(code)
        except Exception as e:
            self.logger.error(f"Failed to parse Java code: {e}")
            self.logger.debug(f"Code that failed to parse:\n{code}")
            raise

        for path, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                res.append({"lineno": node.position[0], "type": "method"})
            elif isinstance(node, javalang.tree.FieldDeclaration):
                res.append({"lineno": node.position[0], "type": "field"})
            elif isinstance(node, javalang.tree.WhileStatement) or isinstance(
                node, javalang.tree.ForStatement
            ):
                res.append({"lineno": node.position[0], "type": "loop"})
        return res

    def _instrument_spec_into_code(
        self, code: str, specs_list: list, unique: bool = True
    ) -> str:
        res = ""
        code_list = code.split("\n")
        tmp_spec_set = set()
        for index, line in enumerate(code_list):
            if self._is_spec(line):
                tmp_spec_set.add(line.strip())
            for spec in specs_list:
                if index + 1 != spec["lineno"]:
                    continue
                for line_of_spec in spec["content"].split("\n"):
                    if line_of_spec.strip() == "" or (
                        unique and self._is_in_set(line_of_spec.strip(), tmp_spec_set)
                    ):
                        continue
                    res = res + self._extract_blank_prefix(line) + line_of_spec + "\n"
            if not self._is_spec(line) and line.strip() != "":
                tmp_spec_set.clear()
            res = res + line + "\n"
        return res

    def _is_in_set(self, e, s: set) -> bool:
        tmp_set = s
        old_len = len(tmp_set)
        tmp_set.add(e)
        return len(tmp_set) == old_len

    def _extract_lineno_from_err_info(self, err_info: str) -> list:
        lineno_list = []
        for line in err_info.split("\n"):
            if (
                line.find("/tmp/") != -1
                and line.find(".java") != -1
                and line.find(":") != -1
            ):
                try:
                    lineno_list.append(int(line.split(":")[1]))
                except Exception:
                    continue
        return lineno_list

    def _is_spec(self, line):
        return line.find("@") != -1 and (
            line.find("maintaining") != -1
            or line.find("loop_invariant") != -1
            or line.find("ensures") != -1
            or line.find("requires") != -1
            or line.find("decreases") != -1
            or line.find("invariant") != -1
            or line.find("spec_public") != -1
            or line.find("invariant") != -1
        )

    def _remove_errornous_and_redundant_spec(
        self, code_with_spec: str, err_info: str
    ) -> str:
        res = ""
        err_lineno_list = self._extract_lineno_from_err_info(err_info)
        tmp_spec_set = set()
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

    def _instrument_spec_into_code(
        self, code: str, specs_list: list, unique: bool = True
    ) -> str:
        res = ""
        code_list = code.split("\n")
        tmp_spec_set = set()
        for index, line in enumerate(code_list):
            if self._is_spec(line):
                tmp_spec_set.add(line.strip())
            for spec in specs_list:
                if index + 1 != spec["lineno"]:
                    continue
                for line_of_spec in spec["content"].split("\n"):
                    if line_of_spec.strip() == "" or (
                        unique and self._is_in_set(line_of_spec.strip(), tmp_spec_set)
                    ):
                        continue
                    res = res + self._extract_blank_prefix(line) + line_of_spec + "\n"
            if not self._is_spec(line) and line.strip() != "":
                tmp_spec_set.clear()
            res = res + line + "\n"
        return res

    def run(self, code, class_name):

        _verifier_calls_count = 0
        current_code = code.strip()
        verified_flag = False
        num_iter = 0
        err_info = ""
        timed_out = False  # Flag to track if a timeout occurred during verification
        status = "unknown"  # haven't tried verification yet, or got an unexpected error during the process

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
            specs_list = []
            for point in infill_points_list:
                """
                # Find if the point is already infilled
                found_flag = False
                for spec in specs_list:
                    if spec["lineno"] == point["lineno"]:
                        found_flag = True
                # If is, skip this point
                if found_flag:
                    continue
                """
                # Query the LLM for specs on this point
                try:
                    specs_generated = self._request_llm_for_spec_on_single_point(
                        current_code, class_name, point["lineno"], point["type"]
                    )
                except Exception as e:
                    self.logger.error(
                        f"[{class_name}] Failed to generate specs for line {point['lineno']}: {e}",
                        exc_info=True,
                    )
                    specs_generated = ""

                # Validate generated specs and remove the ill-formed ones
                if specs_generated:
                    try:
                        specs_generated = self._filter_validated_specs(
                            specs_generated, current_code, class_name, point["lineno"]
                        )
                        _verifier_calls_count += 1
                    except Exception as e:
                        self.logger.error(
                            f"[{class_name}] Failed to filter specs for line {point['lineno']}: {e}",
                            exc_info=True,
                        )
                        specs_generated = ""
                else:
                    self.logger.debug(
                        f"[{class_name}] No specs generated for line {point['lineno']}"
                    )

                specs_list.append(
                    {"content": specs_generated, "lineno": point["lineno"]}
                )
            specs_list = sorted(specs_list, key=lambda x: x["lineno"])
            # Verify the correctness of all generated code
            current_code = self._instrument_spec_into_code(current_code, specs_list)
            if self.verbose:
                self.logger.info(
                    f"Result of iteration {num_iter} is:\n{current_code}\n==============================\n"
                )
            err_info = self._verify_with_openjml(current_code, class_name)
            _verifier_calls_count += 1
            if self.verbose:
                self.logger.info(
                    f"[{class_name}] OpenJML verification result for iteration {num_iter}:\n{err_info}\n==============================\n"
                )
            if "Timeout:" in err_info or "timeout" in err_info.lower():  # timeout
                timed_out = True
                # break --- IGNORE ---

            if err_info.strip() == "":  # verified
                verified_flag = True
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
        return {
            "status": status,
            "class_name": class_name,
            "verifier_calls": _verifier_calls_count,
            "final_code": current_code,
            "final_error": err_info,
            "verified": verified_flag,
            "iterations": num_iter,
        }


class AutoSpecWorker:

    def __init__(
        self, output_dir, model, temperature, max_iterations, timeout, verbose=False
    ):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.timeout = timeout

    def run_autospec(self, task: dict):

        classname = task["class_name"]
        input_code = task["code"]
        task_id = task["id"]

        thread_id = threading.current_thread().ident

        try:
            logger, log_file = create_logger(classname, thread_id, self.output_dir)
        except Exception as e:
            print(f"Failed to create logger for {classname}: {e}")
            return {
                "id": task_id,
                "status": "error",
                "message": f"Logger creation failed: {str(e)}",
                "class_name": classname,
                "log_file": "unknown",
            }

        logger.info(f"Starting AutoSpec for {classname} (task id: {task_id})")

        self.autospec = AutoSpec(
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            output_dir=self.output_dir,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
        )
        try:
            _result = self.autospec.run(input_code, classname)
            

            if _result.get("verified", False):
                verified_status = "✓ VERIFIED"
            elif _result.get("timed_out", False):
                verified_status = "⏱ TIMED OUT"
            else:
                verified_status = "✗ UNVERIFIED"

            logger.info(
                f"{verified_status} - Completed {classname} with {_result['verifier_calls']} verifier calls "
                f"in {_result['iterations']} iterations"
            )

            return {
                "id": task_id,
                "status": _result["status"],
                "class_name": classname,
                "verifier_calls": _result["verifier_calls"],
                "iterations": _result["iterations"],
                "verified": _result.get("verified", False),
                "log_file": log_file,
                "final_code": _result["final_code"],
                "final_error": _result["final_error"],
            }

        except Exception as e:
            logger.error(f"Error processing {classname}: {e}", exc_info=True)
            return {
                "id": task_id,
                "status": "unknown",
                "message": str(e),
                "class_name": classname,
                "log_file": log_file,
            }


class AutoSpecRunner:

    def __init__(
        self,
        name,
        input,
        output,
        model,
        temperature,
        max_iterations,
        openjml_timeout,
        threads,
        verbose,
    ):
        self.name = name
        self.input = input
        self.output = output
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.openjml_timeout = openjml_timeout
        self.threads = threads
        self.verbose = verbose

    def _save_results(self, duration, results):
        """Save summary statistics to a JSON file"""
        verified = [r for r in results if r["status"] == "verified"]
        unverified = [r for r in results if r["status"] == "unverified"]
        timed_out = [r for r in results if r["status"] == "timed_out"]
        unknown = [r for r in results if r["status"] == "unknown"]

        total_verifier_calls = sum(
            r.get("verifier_calls", 0) for r in (verified + unverified + timed_out)
        )
        avg_verifier_calls = (
            total_verifier_calls / len(verified + unverified + timed_out)
            if (verified + unverified + timed_out)
            else 0
        )

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
            "threads_used": self.threads,
            "verified_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in verified
            ],
            "unverified_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in unverified
            ],
            "timed_out_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in timed_out
            ],
            "unknown_cases": [
                {
                    "id": r["id"],
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

    def run_workers(self):

        _input_tasks = load_json(self.input)
        self.input_length = len(_input_tasks)
        print(f"Found {self.input_length} input tasks to process")

        output_dir = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        worker = AutoSpecWorker(
            output_dir=output_dir,
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            timeout=self.openjml_timeout,
            verbose=self.verbose,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} threads...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(worker.run_autospec, task): task
                for task in _input_tasks
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
                            f"✗ [{completed_count}/{self.input_length}] {task['id']} - ERROR: {result.get('message', 'Unknown error')}"
                        )

                except Exception as exc:
                    results.append(
                        {"id": task["id"], "status": "unknown", "message": str(exc)}
                    )
                    print(
                        f"✗ [{completed_count}/{self.input_length}] {task['id']} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
