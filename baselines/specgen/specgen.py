import os
from pathlib import Path
from tabnanny import verbose
import time
import threading
import subprocess
import json
from models import request_llm_engine, token_limit_fitter
from utils import file2str, load_java_file_paths, parse_code_from_reply
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import (
    FORMAT_REFINE_PROMPT,
    GenerationPrompt,
    RefinementPrompt,
)
from logger import create_logger


class SpecGen:

    def __init__(self, output_dir, timeout, logger, verbose=False):
        self.output_dir = output_dir
        self.timeout = timeout
        self.verbose = verbose
        self.logger = logger

    def _verify_with_openjml(self, code_with_spec, classname):
        if self.verbose:
            self.logger.debug(f"[{classname}] Validating with OpenJML...")

        tmp_dir = os.path.join(self.output_dir, "tmp")
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
            return f"Timeout: OpenJML verification exceeded time limit {self.timeout} seconds"
        except Exception as e:
            self.logger.error(f"[{classname}] Error running OpenJML: {e}")
            return f"Error: {str(e)}"

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

    def _print_config(self, config):
        print(self._config_to_str(config))

    def _print_messages(self, message):
        print("{r}:{c}".format(r=message["role"], c=message["content"]))

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


class SpecGenWorker:

    def __init__(
        self, output_dir, model, temperature, max_iterations, timeout, verbose=False
    ):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.timeout = timeout
        self.generation_prompt = GenerationPrompt()
        self.refinement_prompt = RefinementPrompt()

    def run_specgen(self, input_path):
        try:
            classname = input_path.split("/")[-1].split(".")[0]
            input_code = file2str(input_path)
            thread_id = threading.current_thread().ident
            logger, log_file = create_logger(classname, thread_id, self.output_dir)

            self.specgen = SpecGen(
                output_dir=self.output_dir,
                timeout=self.timeout,
                logger=logger,
                verbose=self.verbose,
            )

            # Specification Generation Phase
            _verifier_calls_count = 0
            logger.info(f"[{classname}] Starting Generation Phase...")
            done_flag = False
            config = {}
            current_code = input_code
            err_info = ""
            err_types = []

            for i in range(1, self.max_iterations + 1):
                if self.verbose:
                    # print(f"[{classname}] Iteration {i}")
                    logger.debug(f"[{classname}] Iteration {i}")

                if i == 1:
                    # add config invokation
                    config = self.generation_prompt.create_generation_prompt_config(
                        input_code, classname, self.model, self.temperature
                    )
                    if self.verbose:
                        logger.debug(self.specgen._config_to_str(config))
                        # self.specgen._print_config(config)

                    # add model invokation
                    ret = request_llm_engine(config)
                    if self.verbose:
                        logger.debug(ret.choices[0].message.content)
                    current_code = parse_code_from_reply(ret.choices[0].message.content)
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
                        # add config invokation
                        tmp_config = self.refinement_prompt.create_specialized_patcher_prompt_config(
                            current_code, err_info, self.model, self.temperature
                        )
                        if self.verbose:
                            # self.specgen._print_config(tmp_config)
                            logger.debug(self.specgen._config_to_str(tmp_config))
                        # add model invokation
                        ret = request_llm_engine(tmp_config)
                        if self.verbose:
                            logger.debug(
                                f"[{classname}] assistant: {ret.choices[0].message.content}"
                            )
                        current_code = parse_code_from_reply(
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
                        refine_msg[
                            "content"
                        ] += self.refinement_prompt.gen_extra_guidance(err_info)
                        config["messages"].append(refine_msg)
                        if self.verbose:
                            self.specgen._print_messages(refine_msg)
                        token_limit_fitter(config, 3000)
                        # add model invokation
                        ret = request_llm_engine(config)
                        if self.verbose:
                            logger.debug(
                                f"[{classname}] assistant: {ret.choices[0].message.content}"
                            )
                        current_code = parse_code_from_reply(
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

                # logger.write(current_code + "\n")
                logger.info(f"[{classname}] {current_code}")
                # add OpenJML invokation
                err_info = self.specgen._verify_with_openjml(current_code, classname)
                _verifier_calls_count += 1
                if self.verbose:
                    logger.debug(f"[{classname}] {err_info}")
                err_types = self.refinement_prompt.extract_err_type(err_info)
                logger.debug(f"[{classname}] {err_info}")
                if err_info == "" or done_flag:
                    break

            # Mutation Phase
            logger.info(f"[{classname}] Starting Mutation Phase...")
            current_code_list = current_code.split("\n")
            mutated_spec_list = []

            # Generate mutated spec set
            for index in range(len(current_code_list)):
                if self.specgen._is_invariant_or_postcondition(
                    current_code_list[index]
                ):
                    for mutated_spec in self.specgen._spec_mutator_heuristic(
                        current_code_list[index]
                    ):
                        mutated_spec_list.append(
                            {"content": mutated_spec, "index": index}
                        )

            while True:
                if err_info == "":
                    break
                if not self.specgen._is_invariant_or_postcondition(err_info):
                    logger.debug(
                        f"[{classname}] Unexpected verification error. Aborted."
                    )
                    break

                refuted_lineno_list = self.specgen._extract_lineno_from_err_info(
                    err_info
                )
                # replace each error spec with mutated spec
                for lineno in refuted_lineno_list:
                    index = lineno - 1
                    if self.specgen._is_assert(current_code_list[index]):
                        current_code_list[index] = " "
                        continue
                    if not self.specgen._is_invariant_or_postcondition(
                        current_code_list[index]
                    ):
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
                    logger.debug(f"[{classname}] {current_code}")
                logger.debug(current_code + "\n")
                # add OpenJML invokation
                err_info = self.specgen._verify_with_openjml(current_code, classname)
                _verifier_calls_count += 1
                if self.verbose:
                    logger.debug(f"[{classname}] {err_info}")

            logger.info(f"[{classname}] Finished. Verifier invoked {_verifier_calls_count} times")

            return {
                "path": input_path,
                "status": "success",
                "classname": classname,
                "verifier_calls": _verifier_calls_count,
                "log_file": log_file,
                "final_code": current_code,
                "final_error": err_info,
            }

        except Exception as e:
            return {
                "path": input_path,
                "status": "error",
                "message": str(e),
                "classname": classname if "classname" in locals() else "unknown",
                "log_file": log_file if "log_file" in locals() else "unknown",
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

        worker = SpecGenWorker(
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
            future_to_path = {
                executor.submit(worker.run_specgen, path): path for path in input_paths
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
