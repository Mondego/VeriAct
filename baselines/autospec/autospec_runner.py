import os
import time
import threading
import subprocess
import javalang

from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from baselines.autospec.prompts import get_fewshot_context, get_request_msg
from baselines.utils.logger import create_logger
from baselines.utils.file_utility import (
    dump_json,
    dump_jsonl,
    load_json,
)
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
        return request_llm_engine(config)

    def _verify_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.info(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.output_dir, "tmp")
        Path(tmp_dir).mkdir(exist_ok=True)

        tmp_filename = f"{tmp_dir}/{classname}.java"
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write(code_with_spec)

        cmd = f"openjml --esc --esc-max-warnings 1 --arithmetic-failure=quiet --nonnull-by-default --quiet -nowarn --prover=cvc4 {tmp_filename}"

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            res = result.stdout + result.stderr
            return res
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"[{classname}] OpenJML command timed out after 120 seconds"
            )
            return "Timeout: OpenJML verification exceeded time limit"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

    def _validate_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.info(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.output_dir, "tmp")
        Path(tmp_dir).mkdir(exist_ok=True)

        tmp_filename = f"{tmp_dir}/{classname}.java"
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write(code_with_spec)

        cmd = f"openjml --arithmetic-failure=quiet --quiet --prover=cvc4 {tmp_filename}"

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            res = result.stdout + result.stderr
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
        instrumented_code = self._instrument_spec_into_code(
            code, [{"content": specs, "lineno": lineno}], unique=False
        )
        err_info = self._validate_openjml(instrumented_code, classname)
        if self.verbose:
            self.logger.info(
                f"[{classname}] Validation result:\n{err_info}\n==============================\n"
            )
        err_lineno_list = self._extract_lineno_from_err_info(err_info)
        res = ""
        for index, spec in enumerate(specs.split("\n")):
            if index + lineno in err_lineno_list:
                continue
            else:
                res = res + spec + "\n"
        if self.verbose:
            self.logger.info(
                f"[{classname}] The remaining specs on this point are:\n```\n{res}```\n==============================\n"
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
            reply_msg = {"role": "assistant", "content": "```\n//@ spec_public\n```"}
        else:
            reply_msg = self._request_models(context)
        if self.verbose:
            self.logger.info(
                f"[{classname}] Received LLM response for line {lineno}:)\n==============================\n"
            )
        reply_content = reply_msg.choices[0].message.content.replace("```java", "```")
        if reply_content.strip().startswith("//@"):
            reply_content = "```\n" + reply_content.strip() + "\n```"
        tmp_list = reply_content.split("```")
        if len(tmp_list) < 2:  # Check if LLM has returned any code block
            specs_generated = ""
        else:
            specs_generated = tmp_list[1].strip()
        spec_list = specs_generated.split("\n")
        res = ""
        for line in spec_list:
            if line.strip() == "":
                continue
            res = res + line + "\n"
        return res.strip()

    def _obtain_infill_points(self, code: str) -> list:
        res = []
        tree = javalang.parse.parse(code)
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

        while num_iter < self.max_iterations and not verified_flag:
            current_code = current_code.strip()
            num_iter = num_iter + 1
            try:
                infill_points_list = self._obtain_infill_points(current_code)
            except Exception:
                if self.verbose:
                    self.logger.error(
                        "Syntax error occurred when processing current code!"
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
                specs_generated = self._request_llm_for_spec_on_single_point(
                    current_code, class_name, point["lineno"], point["type"]
                )
                # Validate generated specs and remove the ill-formed ones
                specs_generated = self._filter_validated_specs(
                    specs_generated, current_code, class_name, point["lineno"]
                )
                _verifier_calls_count += 1
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
            err_info = self._verify_openjml(current_code, class_name)
            _verifier_calls_count += 1
            if self.verbose:
                self.logger.info(
                    f"[{class_name}] OpenJML verification result for iteration {num_iter}:\n{err_info}\n==============================\n"
                )
            if err_info.strip() == "":  # Successfully verified
                verified_flag = True
            else:
                current_code = self._remove_errornous_and_redundant_spec(
                    current_code, err_info
                )
        if self.verbose:
            self.logger.info(
                f"[{class_name}] Final result is:\n```\n{current_code}\n```\nVerifier called {_verifier_calls_count} times."
            )
        return {
            "status": "success",
            "class_name": class_name,
            "verifier_calls": _verifier_calls_count,
            "final_code": current_code,
            "final_error": err_info,
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
        thread_id = threading.current_thread().ident
        logger, log_file = create_logger(classname, thread_id, self.output_dir)

        self.autospec = AutoSpec(
            output_dir=self.output_dir,
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
        )
        try:
            _result = self.autospec.run(input_code, classname)
            return {
                "id": task["id"],
                "status": "success",
                "class_name": classname,
                "verifier_calls": _result["verifier_calls"],
                "log_file": log_file,
                "final_code": _result["final_code"],
                "final_error": _result["final_error"],
            }

        except Exception as e:
            return {
                "id": task["id"],
                "status": "error",
                "message": str(e),
                "class_name": classname if "classname" in locals() else "unknown",
                "log_file": log_file if "log_file" in locals() else "unknown",
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
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        total_verifier_calls = sum(r.get("verifier_calls", 0) for r in successful)
        avg_verifier_calls = total_verifier_calls / len(successful) if successful else 0

        summary = {
            "total_files_processed": self.input_length,
            "successful": len(successful),
            "failed": len(failed),
            "success_rate_percent": (
                round(len(successful) / self.input_length * 100, 1)
                if self.input_length > 0
                else 0
            ),
            "total_processing_time_seconds": round(duration, 2),
            "total_verifier_calls": total_verifier_calls,
            "average_verifier_calls_per_successful_case": round(avg_verifier_calls, 1),
            "threads_used": self.threads,
            "successful_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in successful
            ],
            "failed_cases": [
                {
                    "id": r["id"],
                    "message": r["message"],
                    "class_name": r.get("class_name", "unknown"),
                    "log_file": r.get("log_file", "unknown"),
                }
                for r in failed
            ],
        }

        dump_jsonl(results, os.path.join(self.output, f"{self.name}_results_all.jsonl"))
        print(
            f"All results saved to: {os.path.join(self.output, f'{self.name}_results_all.jsonl')}"
        )

        dump_json(
            summary, os.path.join(self.output, f"{self.name}_results_summary.json")
        )

        print(f"\nProcessing completed in {duration:.2f} seconds")
        print(
            f"Success rate: {len(successful)}/{self.input_length} ({len(successful)/self.input_length*100:.1f}%)"
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

                    if result["status"] == "success":
                        print(
                            f"✓ [{completed_count}/{self.input_length}] {result['class_name']} - {result['verifier_calls']} verifier calls"
                        )
                    else:
                        print(
                            f"✗ [{completed_count}/{self.input_length}] {task['id']} - Error: {result.get('message', 'Unknown error')}"
                        )

                except Exception as exc:
                    results.append(
                        {"id": task["id"], "status": "error", "message": str(exc)}
                    )
                    print(
                        f"✗ [{completed_count}/{self.input_length}] {task['id']} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
