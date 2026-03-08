import os
import time
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed


from baselines.utils.file_utility import (
    load_json,
    read_from_file,
    write_to_file,
    dump_json,
    dump_jsonl,
)
from baselines.utils.logger import create_logger
from baselines.utils.verifier import verify_with_openjml


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
            test_code=data.get("test_code", ""),
            test_inputs=[TestCase.from_dict(tc) for tc in data.get("test_inputs", [])],
            generated_test_cases=[
                TestCase.from_dict(tc) for tc in data.get("generated_test_cases", [])
            ],
        )


class DaikonResult(TypedDict):
    status: str
    annotated_code: str
    verified: bool
    timed_out: bool
    final_error: str


class _WorkerResultRequired(TypedDict):
    task_id: str
    status: str
    class_name: str


class WorkerResult(_WorkerResultRequired, total=False):
    verifier_calls: int
    log_file: str
    annotated_code: str
    verified: bool
    final_error: str
    message: str


class Daikon:

    def __init__(
        self,
        run_id: str,
        code: str,
        test_code: str,
        test_inputs: list[TestCase],
        class_name: str,
        test_name: str,
        out_dir: str,
        timeout: int,
        logger: logging.Logger,
        verbose: bool = False,
    ) -> None:
        self.run_id = run_id
        self.code = code
        self.test_code = test_code
        self.test_inputs = test_inputs
        self.class_name = class_name
        self.test_name = test_name
        self.out_dir = out_dir
        self.timeout = timeout
        self.verbose = verbose
        self.logger = logger
        self._BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run_command_in_popen(
        self, _command: list[str] | str, _wait_time: Optional[float] = None
    ) -> bool:
        process = subprocess.Popen(
            _command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=isinstance(_command, str),
        )

        start_time: float = time.time()

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

    def _prepare_task_environment(self) -> None:
        try:
            write_to_file(
                self.code, os.path.join(self.out_dir, f"{self.class_name}.java")
            )
            write_to_file(
                self.test_code,
                os.path.join(self.out_dir, f"{self.test_name}.java"),
            )
            for index in range(0, len(self.test_inputs)):
                write_to_file(
                    self.test_inputs[index].input,
                    os.path.join(self.out_dir, f"{index}.txt"),
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

    def run(self) -> DaikonResult:
        self._prepare_task_environment()
        test_count: int = len(self.test_inputs)
        os.chdir(self.out_dir)
        command: list[str] = [
            "./daikon.sh",
            self.class_name,
            self.test_name,
            str(test_count),
        ]
        self.logger.info(f"Running Daikon with command: {command} for {self.run_id}")
        success: bool = self._run_command_in_popen(command, self.timeout * 2)
        if not success:
            self.logger.error(f"Daikon process failed or timed out for {self.run_id}")
            raise RuntimeError("Daikon execution failed or timed out")
        self.logger.info(f"Daikon execution completed successfully for {self.run_id}")
        os.chdir(self._BASE_DIR)  # change back to base directory after running daikon

        escspec_path = os.path.join(
            self.out_dir, f"{self.class_name}.java-escannotated"
        )
        jmlspec_path = os.path.join(
            self.out_dir, f"{self.class_name}.java-jmlannotated"
        )

        if not os.path.exists(escspec_path) and not os.path.exists(jmlspec_path):
            self.logger.error(
                f"Daikon produced no annotated output files for {self.run_id}"
            )
            return DaikonResult(
                status="unverified",
                annotated_code="",
                verified=False,
                timed_out=False,
                final_error="Daikon produced no annotated output files.",
            )

        code_with_escspec: str = (
            read_from_file(escspec_path) if os.path.exists(escspec_path) else ""
        )
        code_with_jmlspec: str = (
            read_from_file(jmlspec_path) if os.path.exists(jmlspec_path) else ""
        )
        jml_error_info: str = verify_with_openjml(
            code_with_jmlspec, self.class_name, self.timeout, self.out_dir, self.logger
        )
        esc_error_info: str = verify_with_openjml(
            code_with_escspec, self.class_name, self.timeout, self.out_dir, self.logger
        )
        self.logger.info(f"Daikon run completed for task {self.run_id}")

        final_error: str = jml_error_info if jml_error_info else esc_error_info
        timed_out: bool = "Timeout:" in final_error or "timeout" in final_error.lower()
        verified_flag: bool = final_error.strip() == ""

        if verified_flag:
            status = "verified"
            timed_out = False
        elif timed_out:
            status = "timed_out"
        else:
            status = "unverified"

        return DaikonResult(
            status=status,
            annotated_code=code_with_jmlspec,
            verified=verified_flag,
            timed_out=timed_out,
            final_error=final_error,
        )


class DaikonRunner:

    def __init__(
        self,
        name: str,
        input: str,
        output: str,
        timeout: int,
        threads: int,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.input_path: str = input
        self.output = output
        self.timeout = timeout
        self.threads = threads
        self.verbose = verbose
        self.input_length: int = 0

    def _run_daikon(self, task: Task) -> WorkerResult:
        class_name: str = task.class_name
        input_code: str = task.code
        task_id: str = task.task_id
        test_code: str = task.test_code
        test_inputs: list[TestCase] = task.test_inputs

        _BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir: str = os.path.abspath(self.output)
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
                class_name=class_name,
                log_file="unknown",
            )
        logger.info(f"Starting Daikon for {task_id} (Thread ID: {_thread_id})")

        _thread_daikon_artifacts: str = os.path.join(
            output_dir, f"{task_id}_{_thread_id}"
        )
        Path(_thread_daikon_artifacts).mkdir(parents=True, exist_ok=True)

        daikon = Daikon(
            run_id=f"{task_id}_{_thread_id}",
            code=input_code,
            test_code=test_code,
            test_inputs=test_inputs,
            class_name=class_name,
            test_name=task.test_name,
            out_dir=_thread_daikon_artifacts,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
        )

        try:
            _result = daikon.run()

            if _result.get("verified", False):
                verified_status = "✓ VERIFIED"
            elif _result.get("timed_out", False):
                verified_status = "⏱ TIMED OUT"
            else:
                verified_status = "✗ UNVERIFIED"

            logger.info(
                f"{verified_status} - Completed {class_name} with 2 verifier calls"
            )

            return WorkerResult(
                task_id=task_id,
                status=_result["status"],
                class_name=class_name,
                verifier_calls=2,
                log_file=log_file,
                annotated_code=_result["annotated_code"],
                verified=_result.get("verified", False),
                final_error=_result.get("final_error", ""),
            )

        except Exception as e:
            logger.error(f"Error processing {task_id}: {e}", exc_info=True)
            return WorkerResult(
                task_id=task_id,
                status="unknown",
                message=str(e),
                class_name=class_name,
                log_file=log_file,
            )

        finally:
            logger.info(f"Finished Daikon for {task_id} (Thread ID: {_thread_id})")
            os.chdir(_BASE_DIR)  # change back to base directory after processing

    def _save_results(self, duration: float, results: list[WorkerResult]) -> None:
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

        dump_jsonl(results, os.path.join(self.output, f"{self.name}_results_all.jsonl"))
        print(
            f"All results saved to: {os.path.join(self.output, f'{self.name}_results_all.jsonl')}"
        )

        dump_json(
            summary, os.path.join(self.output, f"{self.name}_results_summary.json")
        )

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
            f"Summary saved to: {os.path.join(self.output, f'{self.name}_results_summary.json')}"
        )

    def run_workers(self) -> None:
        _input_tasks: list[Task] = [
            Task.from_dict(t) for t in load_json(self.input_path)
        ]
        self.input_length = len(_input_tasks)
        print(f"Found {self.input_length} input tasks to process")

        output_dir: str = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)  # base output dir

        start_time: float = time.time()
        results: list[WorkerResult] = []
        completed_count: int = 0

        print(f"Starting processing with {self.threads} threads...")
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_to_path = {
                # path will be replaced with data class containing all necessary info for the worker to run daikon
                executor.submit(self._run_daikon, task): task
                for task in _input_tasks
                # for path, task in zip(input_paths, self._create_tasks(input_paths))
            }
            for future in as_completed(future_to_path):
                task = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if result["status"] == "verified":
                        print(
                            f"✓ [{completed_count}/{len(_input_tasks)}] {result['class_name']} - VERIFIED - {result['verifier_calls']} verifier calls"
                        )
                    elif result["status"] == "unverified":
                        print(
                            f"✗ [{completed_count}/{len(_input_tasks)}] {result['class_name']} - UNVERIFIED - {result['verifier_calls']} verifier calls"
                        )
                    elif result["status"] == "timed_out":
                        print(
                            f"⏱ [{completed_count}/{len(_input_tasks)}] {result['class_name']} - TIMED OUT - {result['verifier_calls']} verifier calls"
                        )
                    else:  # unknown
                        print(
                            f"✗ [{completed_count}/{len(_input_tasks)}] {result['class_name']} - ERROR: {result.get('message', 'Unknown error')}"
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
                        f"✗ [{completed_count}/{len(_input_tasks)}] {task.task_id} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time: float = time.time()
        duration: float = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
