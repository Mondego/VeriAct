import os
import json
import time
import threading
import subprocess
from anyio import Path
from logger import create_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import file2str, load_java_file_paths, str2file, extract_blank_prefix


class Houdini:

    def __init__(self, esc_tool_path, output_dir, timeout, logger, verbose=False):
        self.esc_tool_path = esc_tool_path
        self.output_dir = output_dir
        self.timeout = timeout
        self.verbose = verbose
        self.logger = logger

    def validate_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.debug(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.output_dir, "tmp")
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)

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
            self.logger.warning(
                f"[{classname}] OpenJML command timed out after {self.timeout} seconds"
            )
            return f"Timeout: OpenJML verification exceeded time limit {self.timeout} seconds"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

    def gen_annotation(self, code, classname):
        tmp_filename = self.output_dir + "/tmp/{filename}.java".format(
            filename=classname
        )
        str2file(code, tmp_filename)
        outdir = self.output_dir + "/tmp/houdini_output"

        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

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

    def read_annotations_instr(self):
        annotations_path = self.output_dir + "/tmp/houdini_output/log/annotations.instr"
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

    def merge_annotation_into_code(self, annotation_list, code):
        code_list = code.split("\n")
        res_code_list = []
        i, j = 0, 0
        while i < len(annotation_list) and j < len(code_list):
            if annotation_list[i]["lineno"] <= j + 1:
                res_code_list.append(
                    {
                        "is_annotation": True,
                        "content": extract_blank_prefix(code_list[j])
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
                    "content": extract_blank_prefix(code_list[j])
                    + annotation_list[i]["content"],
                }
            )
            i = i + 1
        while j < len(code_list):
            res_code_list.append({"is_annotation": False, "content": code_list[j]})
            j = j + 1
        return res_code_list

    def extract_lineno_from_err_info(self, err_info):
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


class HoudiniWorker:

    def __init__(self, output, timeout, verbose=False):
        self.output = output
        self.timeout = timeout
        self.verbose = verbose
        self.esc_tool_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "ESCTools2",  # Hardcoded path to ESCTools2, can be modified to be configurable if needed
        )

    def run_houdini(self, input_path):

        classname = input_path.split("/")[-1].split(".")[0]
        code = file2str(input_path)

        _thread_id = threading.current_thread().ident
        _logger, _log_file = create_logger(classname, _thread_id, self.output)

        _logger.info(f"Starting Houdini for {classname} (Thread ID: {_thread_id})")

        _thread_output = os.path.join(
            self.output, f"{classname}_{_thread_id}"
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
            esc_tool_path=self.esc_tool_path,
            output_dir=_thread_output,  # adding thread specific subdirectory for output
            timeout=self.timeout,
            logger=_logger,  # pass the logger to Houdini instance
            verbose=self.verbose,
        )

        # Annotation generation and merging
        _logger.info(f"Generating annotations for {classname} in thread {_thread_id}")
        try:
            houdini.gen_annotation(code, classname)
            annotation_list = houdini.read_annotations_instr()
            merged_list = houdini.merge_annotation_into_code(annotation_list, code)
            err_info = "anything"
            merged_code = ""
            _verifier_calls_count = 0

            # Main loop of houdini algorithm
            # [FIX ME] : Add max_iteration to avoid infinite loop)
            while True:
                merged_code = ""
                for line in merged_list:
                    merged_code = merged_code + line["content"] + "\n"
                _logger.info(
                    f"Writing merged code for {classname} in thread {_thread_id}"
                )
                _logger.debug(merged_code + "\n")
                err_info = houdini.validate_openjml(merged_code, classname)
                _verifier_calls_count = _verifier_calls_count + 1
                _logger.debug(f"Error info: {err_info}")
                if err_info == "":
                    break
                else:
                    flag = False
                    refuted_lineno_list = houdini.extract_lineno_from_err_info(err_info)
                    for lineno in refuted_lineno_list:
                        if merged_list[lineno - 1]["is_annotation"] == True:
                            merged_list.pop(lineno - 1)
                            flag = True
                            break
                    if not flag:
                        break

            # end of main loop, output results
            _logger.info(f"Merged code for {classname} in thread {_thread_id}:")
            _logger.debug(merged_code)
            status = "success" if err_info == "" else "failure"

            return {
                "path": input_path,
                "status": status,
                "classname": classname,
                "verifier_calls": _verifier_calls_count,
                "log_file": _log_file,
                "annotated_code": merged_code,
                "final_error": err_info,
            }

        except Exception as e:
            _logger.error(f"[{classname}] Error during Houdini execution: {e}")
            return {
                "path": input_path,
                "status": "error",
                "message": str(e),
                "classname": classname if "classname" in locals() else "unknown",
                "log_file": _log_file if "log_file" in locals() else "unknown",
            }


class HoudiniRunner:

    def __init__(self, name, input, output, openjml_timeout, threads, verbose=False):
        self.name = name
        self.input = input
        self.output = output
        self.openjml_timeout = openjml_timeout
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
                    "classname": r["classname"],
                    "verifier_calls": r["verifier_calls"],
                    "log_file": r["log_file"],
                }
                for r in successful
            ],
            "failed_cases": [
                {
                    "path": r["path"],
                    "message": r["message"],
                    "classname": r.get("classname", "unknown"),
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
        input_paths = load_java_file_paths(self.input)
        self.input_length = len(input_paths)
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
                executor.submit(worker.run_houdini, path): path for path in input_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if result["status"] == "success":
                        print(
                            f"✓ [{completed_count}/{len(input_paths)}] {result['classname']} - {result['verifier_calls']} verifier calls"
                        )
                    else:
                        print(
                            f"✗ [{completed_count}/{len(input_paths)}] {path} - Error: {result.get('message', 'Unknown error')}"
                        )

                except Exception as exc:
                    results.append(
                        {"path": path, "status": "error", "message": str(exc)}
                    )
                    print(
                        f"✗ [{completed_count}/{len(input_paths)}] {path} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(results)
        self._save_summary(duration, results)
