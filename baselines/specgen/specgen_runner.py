import os
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts import (
    FORMAT_REFINE_PROMPT,
    GenerationPrompt,
    RefinementPrompt,
)
from baselines.utils.logger import create_logger
from baselines.utils.models import request_llm_engine
from baselines.utils.verifier import verify_with_openjml
from baselines.utils.file_utility import (
    load_json,
    dump_json,
    dump_jsonl,
)


class SpecGen:

    def __init__(
        self,
        model,
        temperature,
        max_iterations,
        output_dir,
        timeout,
        logger,
        verbose=False,
        prompt_type="zero_shot",
    ):
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.timeout = timeout
        self.verbose = verbose
        self.logger = logger
        self.prompt_type = prompt_type
        self.generation_prompt = GenerationPrompt(self.prompt_type)
        self.refinement_prompt = RefinementPrompt()

    def _parse_code_from_model_response(self, content):
        content = "a" + content
        extracted_str = content.split("```")[1]
        extracted_str = (
            extracted_str
            if not extracted_str.startswith("java")
            else extracted_str[len("java") :]
        )
        return extracted_str

    def _mutate_token_list_random(self, token_list, has_forall, dont_mutate_logical):
        res_list = []
        token_variant_list = []
        if len(token_list) == 0:
            return [[""]]
        if token_list[0].find("\\forall") != -1 or token_list[0].find("\\exists") != -1:
            dont_mutate_logical = True
            tmp_str = token_list[0]
            token_variant_list.append(tmp_str.replace("forall", "exists"))
            token_variant_list.append(tmp_str.replace("exists", "forall"))
        elif token_list[0] == "&&" or token_list[0] == "||":
            if dont_mutate_logical:
                token_variant_list = [token_list[0]]
            else:
                token_variant_list = ["&&", "||"]
            if has_forall:
                dont_mutate_logical = False
        elif token_list[0] == "<=":
            token_variant_list = ["<", "<=", "- 1 <="]
        elif token_list[0] == ">=":
            token_variant_list = [">", ">=", "+ 1 >="]
        elif token_list[0] == "<":
            token_variant_list = ["<", "<="]
        elif token_list[0] == ">":
            token_variant_list = [">", ">="]
        elif not has_forall and (token_list[0] == "+" or token_list[0] == "-"):
            token_variant_list = ["+", "-"]
        else:
            token_variant_list = [token_list[0]]
        for variant in token_variant_list:
            for res in self._mutate_token_list_random(
                token_list[1:], has_forall, dont_mutate_logical
            ):
                tmp_list = [variant]
                tmp_list.extend(res)
                res_list.append(tmp_list)
        return res_list

    def _spec_mutator_random(self, line):
        res_list = []
        has_forall = line.find("forall") != -1 or line.find("exists") != -1
        res_token_list_list = self._mutate_token_list_random(
            line.split(" "), has_forall, True
        )
        for token_list in res_token_list_list:
            tmp_str = ""
            for token in token_list:
                tmp_str = tmp_str + token + " "
            res_list.append(tmp_str)
        return res_list

    def _mutate_token_list_prior(self, token_list, current_index):
        res_list = []
        token_variant_list = []
        if current_index >= len(token_list):
            return [[""]]
        if token_list[current_index] == "<=":
            token_variant_list = ["<", "<=", "- 1 <="]
        elif token_list[current_index] == ">=":
            token_variant_list = [">", ">=", "+ 1 >="]
        elif token_list[current_index] == "<":
            token_variant_list = ["<", "<="]
        elif token_list[current_index] == ">":
            token_variant_list = [">", ">="]
        else:
            token_variant_list = [token_list[current_index]]
        for variant in token_variant_list:
            for res in self._mutate_token_list_prior(token_list, current_index + 1):
                tmp_list = [variant]
                tmp_list.extend(res)
                res_list.append(tmp_list)
        return res_list

    def _spec_mutator_heuristic(self, line):
        res_list = []
        has_forall = line.find("forall") != -1 or line.find("exists") != -1
        token_list = line.split(" ")
        res_token_list_list = self._mutate_token_list_prior(token_list, 0)
        for token_list in res_token_list_list:
            tmp_str = ""
            for token in token_list:
                tmp_str = tmp_str + token + " "
            res_list.append(tmp_str)

        res_list_random = self._spec_mutator_random(line)
        res_list_random_filtered = []
        for str1 in res_list_random:
            flag = False
            for str2 in res_list:
                if str1 == str2:
                    flag = True
                    break
            if not flag:
                res_list_random_filtered.append(str1)
        res_list.extend(res_list_random_filtered)
        return res_list

    def _is_invariant_or_postcondition(self, line):
        return line.find("@") != -1 and (
            line.find("invariant") != -1
            or line.find("maintaining") != -1
            or line.find("ensures") != -1
            or line.find("decreases") != -1
            or line.find("increases") != -1
        )

    def _is_assert(self, line):
        return line.find("@") != -1 and line.find("assert") != -1

    def _config_to_str(self, config):
        res = ""
        for message in config["messages"]:
            res += "{role}: {content}\n".format(
                role=message["role"], content=message["content"]
            )
        return res

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

    def run(self, input_code, class_name):
        ### [Long function as it is from the Authors implementation, and refactoring may introduce bugs]]

        # Specification Generation Phase
        _verifier_calls_count = 0
        self.logger.info(f"[{class_name}] Starting Generation Phase...")
        done_flag = False
        config = {}
        current_code = input_code
        verified_flag = False
        err_info = ""
        err_types = []
        timed_out = False  # Flag to track timeout
        status = "unknown"  # Initial status

        for num_iter in range(1, self.max_iterations + 1):
            self.logger.info(
                f"[{class_name}] Starting iteration {num_iter}/{self.max_iterations}"
            )

            if num_iter == 1:
                config = self.generation_prompt.create_generation_prompt_config(
                    current_code, class_name, self.model, self.temperature
                )
                if self.verbose:
                    self.logger.debug(self._config_to_str(config))

                ret = request_llm_engine(config)
                if self.verbose:
                    self.logger.debug(ret.choices[0].message.content)
                current_code = self._parse_code_from_model_response(
                    ret.choices[0].message.content
                )
                current_code = current_code.strip()
                config["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"```\n{current_code}\n```",
                    }
                )
            else:
                err_types = self.refinement_prompt.extract_err_type(err_info)
                if len(err_types) != 0:
                    tmp_config = (
                        self.refinement_prompt.create_specialized_patcher_prompt_config(
                            current_code, err_info, self.model, self.temperature
                        )
                    )
                    if self.verbose:
                        self.logger.debug(self._config_to_str(tmp_config))
                    ret = request_llm_engine(tmp_config)
                    if self.verbose:
                        self.logger.debug(
                            f"[{class_name}] assistant: {ret.choices[0].message.content}"
                        )
                    current_code = self._parse_code_from_model_response(
                        ret.choices[0].message.content
                    )
                    current_code = current_code.strip()
                    config["messages"][-1]["content"] = f"```\n{current_code}\n```"
                elif (
                    err_info.find("LoopInvariant") == -1
                    and err_info.find("Postcondition") == -1
                ):
                    refine_msg = {
                        "role": "user",
                        "content": FORMAT_REFINE_PROMPT.format(err_info=err_info),
                    }
                    refine_msg["content"] += self.refinement_prompt.gen_extra_guidance(
                        err_info
                    )
                    config["messages"].append(refine_msg)
                    if self.verbose:
                        self.logger.debug(refine_msg["content"])

                    ret = request_llm_engine(config)
                    if self.verbose:
                        self.logger.debug(
                            f"[{class_name}] assistant: {ret.choices[0].message.content}"
                        )
                    current_code = self._parse_code_from_model_response(
                        ret.choices[0].message.content
                    )
                    current_code = current_code.strip()
                    config["messages"].append(
                        {
                            "role": "assistant",
                            "content": "```\n{code}\n```".format(code=current_code),
                        }
                    )
                else:
                    done_flag = True
                    break

            self.logger.info(f"[{class_name}] {current_code}")

            err_info = verify_with_openjml(
                current_code, class_name, self.timeout, self.output_dir, self.logger
            )
            _verifier_calls_count += 1

            if self.verbose:
                self.logger.debug(f"[{class_name}] {err_info}")

            if "Timeout:" in err_info or "timeout" in err_info.lower():
                timed_out = True
                # break --- IGNORE ---

            err_types = self.refinement_prompt.extract_err_type(err_info)
            self.logger.debug(f"[{class_name}] {err_info}")
            if err_info == "" or done_flag:
                break

        # Mutation Phase
        self.logger.info(f"[{class_name}] Starting Mutation Phase...")
        current_code_list = current_code.split("\n")
        mutated_spec_list = []

        # Generate mutated spec set
        for index in range(len(current_code_list)):
            if self._is_invariant_or_postcondition(current_code_list[index]):
                for mutated_spec in self._spec_mutator_heuristic(
                    current_code_list[index]
                ):
                    mutated_spec_list.append({"content": mutated_spec, "index": index})

        # Mutation loop with safety limit
        mutation_iterations = 0
        MAX_MUTATION_ITERATIONS = 1000  # Safety limit to avoid infinite loop

        while True:
            if err_info == "":
                verified_flag = True
                break
            mutation_iterations += 1
            if mutation_iterations >= MAX_MUTATION_ITERATIONS:
                self.logger.warning(
                    f"[{class_name}] Max mutation iterations ({MAX_MUTATION_ITERATIONS}) reached"
                )
                break

            if not self._is_invariant_or_postcondition(err_info):
                self.logger.debug(
                    f"[{class_name}] Unexpected verification error. Aborted."
                )
                break

            refuted_lineno_list = self._extract_lineno_from_err_info(err_info)
            # replace each error spec with mutated spec
            for lineno in refuted_lineno_list:
                index = lineno - 1
                if self._is_assert(current_code_list[index]):
                    current_code_list[index] = " "
                    continue
                if not self._is_invariant_or_postcondition(current_code_list[index]):
                    continue
                # Find mutated spec with same lineno
                found_flag = False
                for item in mutated_spec_list:
                    if item["index"] == index:
                        # replace error spec with mutated spec
                        current_code_list[index] = item["content"]
                        mutated_spec_list.remove(item)
                        found_flag = True
                        break
                if not found_flag:
                    current_code_list[index] = " "

            current_code = ""
            for line in current_code_list:
                current_code = current_code + line + "\n"

            if self.verbose:
                self.logger.debug(f"[{class_name}] {current_code}")
            self.logger.debug(current_code + "\n")
            err_info = verify_with_openjml(
                current_code, class_name, self.timeout, self.logger, self.output_dir
            )
            _verifier_calls_count += 1
            if self.verbose:
                self.logger.debug(f"[{class_name}] {err_info}")

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


class SpecGenWorker:

    def __init__(
        self,
        output_dir,
        model,
        temperature,
        max_iterations,
        timeout,
        verbose=False,
        prompt_type="zero_shot",
    ):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.timeout = timeout
        self.prompt_type = prompt_type

    def run_specgen(self, task: dict):
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

        logger.info(f"Starting SpecGen for {class_name} (task id: {task_id})")

        specgen = SpecGen(
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            output_dir=self.output_dir,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
            prompt_type=self.prompt_type,
        )

        try:
            _result = specgen.run(input_code, class_name)

            if _result.get("verified", False):
                verified_status = "✓ VERIFIED"
            elif _result.get("timed_out", False):
                verified_status = "⏱ TIMED OUT"
            else:
                verified_status = "✗ UNVERIFIED"

            logger.info(
                f"{verified_status} - Completed {class_name} with {_result['verifier_calls']} verifier calls "
                f"in {_result['iterations']} iterations"
            )

            return {
                "id": task_id,
                "status": _result["status"],
                "class_name": class_name,
                "prompt_type": self.prompt_type,
                "verifier_calls": _result["verifier_calls"],
                "iterations": _result["iterations"],
                "verified": _result.get("verified", False),
                "log_file": log_file,
                "final_code": _result["final_code"],
                "final_error": _result["final_error"],
            }

        except Exception as e:
            return {
                "id": task_id,
                "prompt_type": self.prompt_type,
                "status": "unknown",
                "message": str(e),
                "class_name": class_name,
                "log_file": log_file,
            }


class SpecGenRunner:

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
        prompt_type,
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
        self.prompt_type = prompt_type

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

    def run_workers(self):

        _input_tasks = load_json(self.input)
        self.input_length = len(_input_tasks)
        print(f"Found {self.input_length} input tasks to process")

        output_dir = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        worker = SpecGenWorker(
            output_dir=output_dir,
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            timeout=self.openjml_timeout,
            verbose=self.verbose,
            prompt_type=self.prompt_type,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} threads...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(worker.run_specgen, task): task for task in _input_tasks
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
                            f"✗ [{completed_count}/{self.input_length}] {result['class_name']} - ERROR: {result.get('message', 'Unknown error')}"
                        )

                except Exception as exc:
                    results.append(
                        {
                            "id": task["id"],
                            "status": "unknown",
                            "message": str(exc),
                            "class_name": task.get("class_name", "unknown"),
                        }
                    )
                    print(
                        f"✗ [{completed_count}/{self.input_length}] {task.get('class_name', task['id'])} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
