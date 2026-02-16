import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import List
from baselines.utils.file_utility import load_json, read_from_file, write_to_file
from baselines.utils.logger import create_logger
from concurrent.futures import ThreadPoolExecutor, as_completed


class Daikon:

    def __init__(
        self,
        run_id: str,
        code: str,
        test_code: str,
        test_inputs: List[str],
        class_name: str,
        out_dir: str,
        timeout: int,
        logger=None,
        verbose=False,
    ):
        self.run_id = run_id
        self.code = code
        self.test_code = test_code
        self.test_inputs = test_inputs
        self.class_name = class_name
        self.out_dir = out_dir
        self.timeout = timeout
        self.verbose = verbose
        self.logger = logger
        self._BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run_command_in_popen(self, _command, _wait_time=None):
        process = subprocess.Popen(
            _command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=isinstance(_command, str),
        )

        start_time = time.time()

        try:
            for line in process.stdout:
                if self.verbose:
                    self.logger.info(line.strip())
                else:
                    print(line, end="")
                if _wait_time is not None and (time.time() - start_time) > _wait_time:
                    self.logger.info(
                        f" Timeout reached. Terminating process... for {self.run_id}"
                    )
                    process.terminate()
                    break

            process.wait(timeout=5)

        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"Process did not terminate gracefully. Killing process for {self.run_id}..."
            )
            process.kill()
            return False

        except KeyboardInterrupt:
            self.logger.info(
                f"Interrupted by user. Killing process for {self.run_id}..."
            )
            process.kill()
            return False

        return process.returncode == 0

    def _run_command_in_subprocess(self, _command, _wait_time):
        try:
            result = subprocess.run(
                _command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=_wait_time,
            )
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with error: {e} at {self.run_id}")
            return False

    def _verify_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.info(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.out_dir, "tmp")
        Path(tmp_dir).mkdir(exist_ok=True)

        tmp_filename = f"{tmp_dir}/{classname}.java"
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write(code_with_spec)

        cmd = f"openjml --esc --esc-max-warnings 1 --arithmetic-failure=quiet --nonnull-by-default --quiet -nowarn --prover=cvc4 {tmp_filename}"

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=self.timeout
            )
            res = result.stdout + result.stderr
            return res
        except subprocess.TimeoutExpired:
            self.logger.error(
                f"[{classname}] OpenJML command timed out after {self.timeout} seconds"
            )
            return "Timeout: OpenJML verification exceeded time limit"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

    def _prepare_task_environment(self):
        try:
            write_to_file(
                os.path.join(self.out_dir, f"{self.class_name}.java"), self.code
            )
            write_to_file(
                os.path.join(self.out_dir, f"{self.class_name}Test.java"),
                self.test_code,
            )
            for index in range(0, len(self.test_inputs)):
                write_to_file(
                    os.path.join(self.out_dir, f"{index}.txt"),
                    self.test_inputs[index]["input"],
                )

            _script_path = os.path.join(self._BASE_DIR, "daikon")
            shutil.copy(os.path.join(_script_path, "daikon.sh"), self.out_dir)
            shutil.copy(os.path.join(_script_path, "inv_config.config"), self.out_dir)
            self._run_command_in_popen(
                ["chmod", "+x", os.path.join(_script_path, "daikon.sh")], self.timeout
            )

        except Exception as e:
            self.logger.error(f"Failed to prepare task environment: {e}")
            raise e

    def run(self):
        self._prepare_task_environment()
        test_count = len(self.test_inputs)
        os.chdir(self.out_dir)
        command = ["./daikon.sh", self.class_name, str(test_count)]
        self.logger.info(f"Running Daikon with command: {command} for {self.run_id}")
        success = self._run_command_in_popen(command, self.timeout * 2)
        if not success:
            self.logger.error(f"Daikon process failed or timed out for {self.run_id}")
            raise RuntimeError("Daikon execution failed or timed out")
        self.logger.info(f"Daikon execution completed successfully for {self.run_id}")
        os.chdir(self._BASE_DIR)  # change back to base directory after running daikon


class DaikonWorker:

    def __init__(
        self,
        out_dir: str,
        timeout: int,
        verbose=False,
    ):
        self.out_dir = out_dir
        self.timeout = timeout
        self.verbose = verbose

    def run_daikon(self, task: dict):

        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _thread_id = threading.current_thread().ident
        _logger, _log_file = create_logger(task["id"], _thread_id, self.out_dir)
        self._logger = _logger
        _logger.info(f"Starting Daikon for {task['id']} (Thread ID: {_thread_id})")

        _thread_daikon_artifacts = os.path.join(
            self.out_dir, f"{task['id']}_{_thread_id}"
        )
        Path.mkdir(_thread_daikon_artifacts, parents=True, exist_ok=True)

        daikon = Daikon(
            run_id=f"{task['id']}_{_thread_id}",
            code=task["code"],
            test_code=task["test_code"],
            test_inputs=task["test_inputs"],
            class_name=task["class_name"],
            out_dir=_thread_daikon_artifacts,
            timeout=self.timeout,
            logger=_logger,
            verbose=self.verbose,
        )

        jml_error_info = ""
        esc_error_info = ""

        try:
            daikon.run()
            _logger.info(f"Daikon run completed for task {task['id']}")
            # [READ] output from thread specific directory and return necessary info for summary
            # {class_name}.java-escannotated for ESC, {class_name}.java-jmlannotated for JML

            code_with_escspec = read_from_file(
                os.path.join(
                    _thread_daikon_artifacts, f"{task['class_name']}.java-escannotated"
                )
            )
            code_with_jmlspec = read_from_file(
                os.path.join(
                    _thread_daikon_artifacts, f"{task['class_name']}.java-jmlannotated"
                )
            )

            jml_error_info = daikon._verify_openjml(
                code_with_jmlspec, task["class_name"]
            )
            esc_error_info = daikon._verify_openjml(
                code_with_escspec, task["class_name"]
            )

            return {
                "id": task["id"],
                "status": "success",
                "class_name": task["class_name"],
                "verifier_calls": 0,  # placeholder for now, will be updated after parsing the annotated code
                "log_file": _log_file,
                "annotated_code": code_with_jmlspec,
                "final_error": jml_error_info if jml_error_info else esc_error_info,
            }

        except Exception as e:
            _logger.error(f"Error processing {task['id']}: {e}")
            return {
                "id": task["id"],
                "status": "error",
                "message": str(e),
                "class_name": task["class_name"] if "class_name" in task else "unknown",
                "log_file": _log_file if "log_file" in locals() else "unknown",
            }

        finally:
            _logger.info(f"Finished Daikon for {task['id']} (Thread ID: {_thread_id})")
            os.chdir(_BASE_DIR)  # change back to base directory after processing


class DaikonRunner:

    def __init__(self, name, input, output, timeout, threads, verbose=False):
        self.name = name
        self.input = input
        self.output = output
        self.timeout = timeout
        self.threads = threads
        self.verbose = verbose

    def _save_results(self, results):
        with open(
            os.path.join(self.output, f"{self.name}_results_all.jsonl"), "w"
        ) as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(
            f"All results saved to: {os.path.join(self.output, f'{self.name}_results_all.jsonl')}"
        )

    def _save_summary(self, duration, results):
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

        with open(
            os.path.join(self.output, f"{self.name}_results_summary.json"), "w"
        ) as f:
            json.dump(summary, f, indent=2)
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
        Path(output_dir).mkdir(parents=True, exist_ok=True)  # base output dir

        worker = DaikonWorker(
            out_dir=output_dir,
            timeout=self.timeout,
            verbose=self.verbose,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} threads...")
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_to_path = {
                # path will be replaced with data class containing all necessary info for the worker to run daikon
                executor.submit(worker.run_daikon, task): task
                for task in _input_tasks
                # for path, task in zip(input_paths, self._create_tasks(input_paths))
            }
            for future in as_completed(future_to_path):
                task = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if result["status"] == "success":
                        print(
                            f"✓ [{completed_count}/{len(_input_tasks)}] {result['class_name']} - {result['verifier_calls']} verifier calls"
                        )
                    else:
                        print(
                            f"✗ [{completed_count}/{len(_input_tasks)}] {task['id']} - Error: {result.get('message', 'Unknown error')}"
                        )

                except Exception as exc:
                    results.append(
                        {"id": task["id"], "status": "error", "message": str(exc)}
                    )
                    print(
                        f"✗ [{completed_count}/{len(_input_tasks)}] {task['id']} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(results)
        self._save_summary(duration, results)
