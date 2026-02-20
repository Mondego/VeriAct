import os
import re
import time
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import build_messages
from baselines.utils.logger import create_logger
from baselines.utils.models import create_model_config, request_llm_engine
from baselines.utils.file_utility import (
    load_json,
    dump_json,
    dump_jsonl,
    write_to_file,
)
from baselines.formalbench.example import JavaExample


VALID_PROMPT_TYPES = ["zero_shot", "zs_cot", "two_shot", "fs_cot", "fs_ltm"]


class FormalBench:

    def __init__(
        self,
        model,
        temperature,
        prompt_type,
        output_dir,
        timeout,
        logger,
        verbose=False,
    ):
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

        # Load few-shot examples based on prompt type
        if prompt_type == "fs_ltm":
            self.example_code1 = JavaExample.EXAMPLE_CODE2
            self.example_code2 = JavaExample.EXAMPLE_CODE3
            self.example_spec1 = JavaExample.EXAMPLE_LTM_RESPONSE2
            self.example_spec2 = JavaExample.EXAMPLE_LTM_RESPONSE3
        else:
            self.example_code1 = JavaExample.EXAMPLE_CODE1
            self.example_code2 = JavaExample.EXAMPLE_CODE2
            self.example_spec1 = JavaExample.EXAMPLE_SPEC1
            self.example_spec2 = JavaExample.EXAMPLE_SPEC2

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
                self.logger.debug(f"[{classname}] OpenJML stdout: {result.stdout[:500]}")
            if result.stderr:
                self.logger.debug(f"[{classname}] OpenJML stderr: {result.stderr[:500]}")
            return res
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"[{classname}] OpenJML command timed out after 120 seconds"
            )
            return "Timeout: OpenJML verification exceeded time limit"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

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

    def run(self, input_code, class_name):
        verifier_calls = 0
        status = "unknown"
        err_info = ""
        spec = None

        self.logger.info(f"[{class_name}] Generating specifications (prompt_type={self.prompt_type})...")

        messages = build_messages(
            prompt_type=self.prompt_type,
            model=self.model,
            code=input_code,
            example_code1=self.example_code1,
            example_code2=self.example_code2,
            example_spec1=self.example_spec1,
            example_spec2=self.example_spec2,
        )

        config = create_model_config(messages, self.model, self.temperature)
        ret = request_llm_engine(config)

        if ret is None:
            self.logger.error(f"[{class_name}] LLM returned no response")
            return {
                "status": "unknown",
                "class_name": class_name,
                "verifier_calls": verifier_calls,
                "final_code": None,
                "final_error": "LLM returned no response",
                "verified": False,
                "iterations": 1,
            }

        raw_response = ret.choices[0].message.content
        if self.verbose:
            self.logger.debug(f"[{class_name}] LLM response: {raw_response[:500]}")

        spec = self._parse_spec_from_response(raw_response)

        if not spec:
            self.logger.warning(f"[{class_name}] Empty specification returned by model")
            return {
                "status": "empty_spec",
                "class_name": class_name,
                "verifier_calls": verifier_calls,
                "final_code": None,
                "final_error": "Empty specification",
                "verified": False,
                "iterations": 1,
            }

        if not self._contains_annotations(spec):
            self.logger.warning(f"[{class_name}] No JML annotations detected in response")
            return {
                "status": "invalid_jml",
                "class_name": class_name,
                "verifier_calls": verifier_calls,
                "final_code": spec,
                "final_error": "No JML annotations detected",
                "verified": False,
                "iterations": 1,
            }

        self.logger.info(f"[{class_name}] Verifying with OpenJML...")
        err_info = self._verify_with_openjml(spec, class_name)
        verifier_calls += 1

        if self.verbose:
            self.logger.debug(f"[{class_name}] Verification output: {err_info[:500]}")

        if "Timeout:" in err_info or "timeout" in err_info.lower():
            status = "timed_out"
        elif err_info == "":
            status = "verified"
        else:
            status = "unverified"

        verified = status == "verified"

        if verified:
            self.logger.info(
                f"[{class_name}] Successfully verified with {verifier_calls} verifier call(s)"
            )
        else:
            self.logger.warning(
                f"[{class_name}] Verification {status}. Final error: {err_info[:200]}"
            )

        return {
            "status": status,
            "class_name": class_name,
            "verifier_calls": verifier_calls,
            "final_code": spec,
            "final_error": err_info,
            "verified": verified,
            "iterations": 1,
        }


class FormalBenchWorker:

    def __init__(self, output_dir, model, temperature, prompt_type, timeout, verbose=False):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.timeout = timeout
        self.verbose = verbose

    def run_formalbench(self, task: dict):
        class_name = task["class_name"]
        input_code = task["code"]
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

        logger.info(f"Starting FormalBench for {class_name} (task id: {task_id})")

        fb = FormalBench(
            model=self.model,
            temperature=self.temperature,
            prompt_type=self.prompt_type,
            output_dir=self.output_dir,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
        )

        try:
            _result = fb.run(input_code, class_name)

            if _result.get("verified", False):
                verified_status = "VERIFIED"
            elif _result["status"] == "timed_out":
                verified_status = "TIMED OUT"
            else:
                verified_status = "UNVERIFIED"

            logger.info(
                f"{verified_status} - Completed {class_name} with "
                f"{_result['verifier_calls']} verifier call(s)"
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


class FormalBenchRunner:

    def __init__(
        self,
        name,
        input,
        output,
        model,
        temperature,
        prompt_type,
        openjml_timeout,
        threads,
        verbose,
    ):
        self.name = name
        self.input = input
        self.output = output
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
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
            "prompt_type": self.prompt_type,
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

        worker = FormalBenchWorker(
            output_dir=output_dir,
            model=self.model,
            temperature=self.temperature,
            prompt_type=self.prompt_type,
            timeout=self.openjml_timeout,
            verbose=self.verbose,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} thread(s)...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(worker.run_formalbench, task): task
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
                        print(f"[VERIFIED]   {prefix} {name} - {result['verifier_calls']} verifier call(s)")
                    elif status == "unverified":
                        print(f"[UNVERIFIED] {prefix} {name} - {result['verifier_calls']} verifier call(s)")
                    elif status == "timed_out":
                        print(f"[TIMED OUT]  {prefix} {name} - {result['verifier_calls']} verifier call(s)")
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
