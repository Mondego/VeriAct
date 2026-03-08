import os
import time
import logging
import threading
import subprocess

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from baselines.utils.logger import create_logger
from baselines.utils.verifier import verify_with_openjml
from baselines.utils.file_utility import write_to_file, load_json, dump_json, dump_jsonl


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


@dataclass
class Annotation:
    lineno: int
    content: str


@dataclass
class MergedLine:
    is_annotation: bool
    content: str


class HoudiniResult(TypedDict):
    status: str
    annotated_code: str
    verified: bool
    timed_out: bool
    final_error: str
    verifier_calls: int


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


class Houdini:

    def __init__(
        self,
        code: str,
        class_name: str,
        esc_tool_path: str,
        output_dir: str,
        timeout: int,
        logger: logging.Logger,
        verbose: bool = False,
    ) -> None:
        self.code = code
        self.class_name = class_name
        self.esc_tool_path = esc_tool_path
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

    def _extract_blank_prefix(self, _code: str) -> str:
        string_stripped = _code.strip()
        if len(string_stripped) > 0:
            return _code.split(string_stripped)[0]
        else:
            return _code

    def _gen_annotation(self, code: str, classname: str) -> str:

        tmp_filename = os.path.join(self.output_dir, "tmp", f"{classname}.java")
        write_to_file(code, tmp_filename)

        outdir = os.path.join(self.output_dir, "tmp", "houdini_output")
        Path(outdir).mkdir(parents=True, exist_ok=True)

        cmd = (
            self.esc_tool_path
            + "/Houdini/annotationGen -outdir "
            + outdir
            + " "
            + tmp_filename
            + " > "
            + outdir
            + "/tmp.log"
        )

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=self.timeout
            )
            res = result.stdout + result.stderr
            return res
        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"[{classname}] OpenJML command timed out after {self.timeout} seconds"
            )
            return f"Timeout: OpenJML verification exceeded time limit {self.timeout} seconds"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

    def _read_annotations_instr(self) -> list[Annotation]:

        annotations_path = os.path.join(
            self.output_dir, "tmp", "houdini_output", "log", "annotations.instr"
        )

        if not os.path.exists(annotations_path):
            self.logger.error("Error: Failed to generate candidate annotation set\n")
            return []

        res_list: list[Annotation] = []
        with open(annotations_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                tmp_list = line.split("'")
                # >>> added
                # check tmp_list length to avoid index error when there is no annotation content
                if len(tmp_list) < 3:
                    continue
                # >>> end
                content = tmp_list[1]
                for part in tmp_list[2 : len(tmp_list) - 1]:
                    content = content + "'" + part
                tmp_list = tmp_list[0].split()
                lineno = int(tmp_list[-3])
                if (
                    content.find("Explicating default constructor") != -1
                    or content.find("*/final/*") != -1
                    or content.find("requires false;") != -1
                ):
                    continue
                res_list.append(Annotation(lineno=lineno, content=content))
        return res_list

    def _merge_annotation_into_code(
        self, annotation_list: list[Annotation], code: str
    ) -> list[MergedLine]:
        code_list = code.split("\n")
        res_code_list: list[MergedLine] = []
        i, j = 0, 0
        while i < len(annotation_list) and j < len(code_list):
            if annotation_list[i].lineno <= j + 1:
                res_code_list.append(
                    MergedLine(
                        is_annotation=True,
                        content=self._extract_blank_prefix(code_list[j])
                        + "//@ "
                        + annotation_list[i].content,
                    )
                )
                i = i + 1
            else:
                res_code_list.append(
                    MergedLine(is_annotation=False, content=code_list[j])
                )
                j = j + 1
        while i < len(annotation_list):
            res_code_list.append(
                MergedLine(
                    is_annotation=True,
                    content=self._extract_blank_prefix(code_list[j])
                    + annotation_list[i].content,
                )
            )
            i = i + 1
        while j < len(code_list):
            res_code_list.append(MergedLine(is_annotation=False, content=code_list[j]))
            j = j + 1
        return res_code_list

    def _extract_lineno_from_err_info(self, err_info: str) -> list[int]:
        temp_list: list[str] = []
        err_list: list[list[str]] = []
        err_info_list = err_info.split("\n")
        for line in err_info_list:
            if line.strip() == "^":
                err_list.append(temp_list)
                temp_list = []
            else:
                temp_list.append(line)
        lineno_list: list[int] = []
        for err in err_list:
            lineno_list.append(int(err[0].split(":")[1]))
        return lineno_list

    def run(self) -> HoudiniResult:
        # Annotation generation and merging
        self.logger.info(
            f"Generating annotations for {self.class_name} in thread {self.output_dir}"
        )
        self._gen_annotation(self.code, self.class_name)
        annotation_list: list[Annotation] = self._read_annotations_instr()
        merged_list: list[MergedLine] = self._merge_annotation_into_code(
            annotation_list, self.code
        )
        err_info: str = "anything"
        merged_code: str = ""
        _verifier_calls_count: int = 0

        # Main loop of houdini algorithm
        max_iterations: int = (
            1000  # Add hardcoded max iterations to avoid `while True` infinite loop
        )
        iteration_count: int = 0
        while iteration_count < max_iterations:
            merged_code = ""
            for line in merged_list:
                merged_code = merged_code + line.content + "\n"
            self.logger.info(
                f"Writing merged code for {self.class_name} in thread {self.output_dir}"
            )
            self.logger.debug(merged_code + "\n")
            err_info = verify_with_openjml(
                merged_code, self.class_name, self.timeout, self.output_dir, self.logger
            )
            _verifier_calls_count = _verifier_calls_count + 1
            self.logger.debug(f"Error info: {err_info}")
            if err_info == "":
                break
            else:
                flag: bool = False
                refuted_lineno_list = self._extract_lineno_from_err_info(err_info)
                for lineno in refuted_lineno_list:
                    if merged_list[lineno - 1].is_annotation == True:
                        merged_list.pop(lineno - 1)
                        flag = True
                        break
                if not flag:
                    break
            iteration_count += 1

        if iteration_count >= max_iterations:
            self.logger.warning(
                f"Max iterations ({max_iterations}) reached for {self.class_name}"
            )

        # end of main loop, output results
        self.logger.info(
            f"Merged code for {self.class_name} in thread {self.output_dir}"
        )
        self.logger.debug(merged_code)

        timed_out: bool = "Timeout:" in err_info or "timeout" in err_info.lower()
        verified_flag: bool = err_info.strip() == ""

        if verified_flag:
            status = "verified"
            timed_out = False
        elif timed_out:
            status = "timed_out"
        else:
            status = "unverified"

        return HoudiniResult(
            status=status,
            annotated_code=merged_code,
            verified=verified_flag,
            timed_out=timed_out,
            final_error=err_info,
            verifier_calls=_verifier_calls_count,
        )


class HoudiniRunner:

    def __init__(
        self,
        name: str,
        input: str,
        output: str,
        openjml_timeout: int,
        threads: int,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.input_path: str = input
        self.output = output
        self.openjml_timeout = openjml_timeout
        self.threads = threads
        self.verbose = verbose
        self.input_length: int = 0
        self.esc_tool_path: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ESCTools2",  # Hardcoded path to ESCTools2, will be modified to be configurable if needed
        )

    def _run_houdini(self, task: Task) -> WorkerResult:
        class_name: str = task.class_name
        code: str = task.code
        task_id: str = task.task_id

        output_dir: str = os.path.abspath(self.output)
        _thread_id: Optional[int] = threading.current_thread().ident
        log_file: str = "unknown"
        try:
            _logger, log_file = create_logger(task_id, _thread_id, output_dir)
        except Exception as e:
            print(f"Failed to create logger for {task_id}: {e}")
            return WorkerResult(
                task_id=task_id,
                status="error",
                message=f"Logger creation failed: {str(e)}",
                class_name=class_name,
                log_file="unknown",
            )

        _logger.info(f"Starting Houdini for {task_id} (Thread ID: {_thread_id})")

        _thread_output: str = os.path.join(
            output_dir, f"{task_id}.Thread_{_thread_id}"
        )  # thread specific output directory
        Path(_thread_output).mkdir(parents=True, exist_ok=True)
        _thread_houdini_logs: str = os.path.join(
            _thread_output, "houdini_logs"
        )  # thread specific houdini logs directory
        Path(_thread_houdini_logs).mkdir(parents=True, exist_ok=True)
        _thread_tmp_dir: str = os.path.join(
            _thread_output, "tmp"
        )  # thread specific tmp directory for houdini intermediate files
        Path(_thread_tmp_dir).mkdir(parents=True, exist_ok=True)

        houdini = Houdini(
            code=code,
            class_name=class_name,
            esc_tool_path=self.esc_tool_path,
            output_dir=_thread_output,  # adding thread specific subdirectory for output
            timeout=self.openjml_timeout,
            logger=_logger,  # pass the logger to Houdini instance
            verbose=self.verbose,
        )

        try:
            _result = houdini.run()

            if _result.get("verified", False):
                verified_status = "✓ VERIFIED"
            elif _result.get("timed_out", False):
                verified_status = "⏱ TIMED OUT"
            else:
                verified_status = "✗ UNVERIFIED"

            _logger.info(
                f"{verified_status} - Completed {class_name} with {_result['verifier_calls']} verifier calls"
            )

            return WorkerResult(
                task_id=task_id,
                status=_result["status"],
                class_name=class_name,
                verifier_calls=_result["verifier_calls"],
                log_file=log_file,
                annotated_code=_result["annotated_code"],
                verified=_result.get("verified", False),
                final_error=_result["final_error"],
            )

        except Exception as e:
            _logger.error(
                f"[{class_name}] Error during Houdini execution: {e}", exc_info=True
            )
            return WorkerResult(
                task_id=task_id,
                status="unknown",
                message=str(e),
                class_name=class_name,
                log_file=log_file,
            )

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
        print(f"Found {self.input_length} input files to process")

        output_dir: str = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        start_time: float = time.time()
        results: list[WorkerResult] = []
        completed_count: int = 0

        print(f"Starting processing with {self.threads} threads...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_to_path = {
                executor.submit(self._run_houdini, task): task for task in _input_tasks
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
