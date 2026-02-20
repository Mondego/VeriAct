import os
import time
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from baselines.utils.logger import create_logger
from baselines.utils.file_utility import write_to_file, load_json, dump_json, dump_jsonl
from baselines.utils.verifier import verify_with_openjml

class Houdini:

    def __init__(
        self,
        code,
        class_name,
        esc_tool_path,
        output_dir,
        timeout,
        logger,
        verbose=False,
    ):
        self.code = code
        self.class_name = class_name
        self.esc_tool_path = esc_tool_path
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

    def _extract_blank_prefix(self, _code):
        string_stripped = _code.strip()
        if len(string_stripped) > 0:
            return _code.split(string_stripped)[0]
        else:
            return _code

    def _gen_annotation(self, code, classname):

        tmp_filename = os.path.join(self.output_dir, "tmp", f"{classname}.java")
        write_to_file(code, tmp_filename)

        outdir = os.path.join(self.output_dir, "tmp", "houdini_output")
        Path.mkdir(outdir, parents=True, exist_ok=True)

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

    def _read_annotations_instr(self):

        annotations_path = os.path.join(
            self.output_dir, "tmp", "houdini_output", "log", "annotations.instr"
        )

        if not os.path.exists(annotations_path):
            self.logger.error("Error: Failed to generate candidate annotation set\n")
            return []

        res_list = []
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
                tmp_dict = {"lineno": lineno, "content": content}
                res_list.append(tmp_dict)
        return res_list

    def _merge_annotation_into_code(self, annotation_list, code):
        code_list = code.split("\n")
        res_code_list = []
        i, j = 0, 0
        while i < len(annotation_list) and j < len(code_list):
            if annotation_list[i]["lineno"] <= j + 1:
                res_code_list.append(
                    {
                        "is_annotation": True,
                        "content": self._extract_blank_prefix(code_list[j])
                        + "//@ "
                        + annotation_list[i]["content"],
                    }
                )
                i = i + 1
            else:
                res_code_list.append({"is_annotation": False, "content": code_list[j]})
                j = j + 1
        while i < len(annotation_list):
            res_code_list.append(
                {
                    "is_annotation": True,
                    "content": self._extract_blank_prefix(code_list[j])
                    + annotation_list[i]["content"],
                }
            )
            i = i + 1
        while j < len(code_list):
            res_code_list.append({"is_annotation": False, "content": code_list[j]})
            j = j + 1
        return res_code_list

    def _extract_lineno_from_err_info(self, err_info):
        temp_list = []
        err_list = []
        err_info_list = err_info.split("\n")
        for line in err_info_list:
            if line.strip() == "^":
                err_list.append(temp_list)
                temp_list = []
            else:
                temp_list.append(line)
        lineno_list = []
        for err in err_list:
            lineno_list.append(int(err[0].split(":")[1]))
        return lineno_list

    def run(self):
        # Annotation generation and merging
        self.logger.info(
            f"Generating annotations for {self.class_name} in thread {self.output_dir}"
        )
        self._gen_annotation(self.code, self.class_name)
        annotation_list = self._read_annotations_instr()
        merged_list = self._merge_annotation_into_code(annotation_list, self.code)
        err_info = "anything"
        merged_code = ""
        _verifier_calls_count = 0

        # Main loop of houdini algorithm
        max_iterations = (
            1000  # Add hardcoded max iterations to avoid `while True` infinite loop
        )
        iteration_count = 0
        while iteration_count < max_iterations:
            merged_code = ""
            for line in merged_list:
                merged_code = merged_code + line["content"] + "\n"
            self.logger.info(
                f"Writing merged code for {self.class_name} in thread {self.output_dir}"
            )
            self.logger.debug(merged_code + "\n")
            err_info = verify_with_openjml(merged_code, self.class_name, self.timeout, self.output_dir, self.logger)
            _verifier_calls_count = _verifier_calls_count + 1
            self.logger.debug(f"Error info: {err_info}")
            if err_info == "":
                break
            else:
                flag = False
                refuted_lineno_list = self._extract_lineno_from_err_info(err_info)
                for lineno in refuted_lineno_list:
                    if merged_list[lineno - 1]["is_annotation"] == True:
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

        status = "success" if err_info == "" else "failure"
        return {
            "annotated_code": merged_code,
            "final_error": err_info,
            "verifier_calls": _verifier_calls_count,
            "status": status,
        }


class HoudiniWorker:

    def __init__(self, output, timeout, verbose=False):
        self.output = output
        self.timeout = timeout
        self.verbose = verbose
        self.esc_tool_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ESCTools2",  # Hardcoded path to ESCTools2, will be modified to be configurable if needed
        )

    def run_houdini(self, task: dict):

        class_name = task["class_name"]
        code = task["code"]

        _thread_id = threading.current_thread().ident
        _logger, _log_file = create_logger(class_name, _thread_id, self.output)

        _logger.info(f"Starting Houdini for {class_name} (Thread ID: {_thread_id})")

        _thread_output = os.path.join(
            self.output, f"{class_name}_{_thread_id}"
        )  # thread specific output directory
        Path.mkdir(_thread_output, parents=True, exist_ok=True)
        _thread_houdini_logs = os.path.join(
            _thread_output, "houdini_logs"
        )  # thread specific houdini logs directory
        Path.mkdir(_thread_houdini_logs, parents=True, exist_ok=True)
        _thread_tmp_dir = os.path.join(
            _thread_output, "tmp"
        )  # thread specific tmp directory for houdini intermediate files
        Path.mkdir(_thread_tmp_dir, parents=True, exist_ok=True)

        houdini = Houdini(
            code=code,
            class_name=class_name,
            esc_tool_path=self.esc_tool_path,
            output_dir=_thread_output,  # adding thread specific subdirectory for output
            timeout=self.timeout,
            logger=_logger,  # pass the logger to Houdini instance
            verbose=self.verbose,
        )

        try:

            _results = houdini.run()

            return {
                "id": task["id"],
                "status": _results["status"],
                "class_name": class_name,
                "verifier_calls": _results["verifier_calls"],
                "log_file": _log_file,
                "annotated_code": _results["annotated_code"],
                "final_error": _results["final_error"],
            }

        except Exception as e:
            _logger.error(f"[{class_name}] Error during Houdini execution: {e}")
            return {
                "id": task["id"],
                "status": "error",
                "message": str(e),
                "class_name": class_name,
                "log_file": _log_file,
            }


class HoudiniRunner:

    def __init__(self, name, input, output, openjml_timeout, threads, verbose=False):
        self.name = name
        self.input = input
        self.output = output
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
        print(f"Found {self.input_length} input files to process")

        output_dir = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        worker = HoudiniWorker(
            output=output_dir,
            timeout=self.openjml_timeout,
            verbose=self.verbose,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} threads...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_to_path = {
                executor.submit(worker.run_houdini, task): task for task in _input_tasks
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
        self._save_results(duration, results)
