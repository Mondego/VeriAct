import os
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed


from baselines.specgen.prompts import (
    FORMAT_REFINE_PROMPT,
    GenerationPrompt,
    RefinementPrompt,
)
from baselines.utils.logger import create_logger
from baselines.utils.verifier import verify_with_openjml
from baselines.utils.models import (
    request_llm_engine,
    reset_token_usage,
    get_token_usage,
)
from baselines.utils.file_utility import (
    load_json,
    dump_json,
    dump_jsonl,
)


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
    origin_id: str
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
            origin_id=data["origin_id"],
            test_code=data.get("test_code", ""),
            test_inputs=[TestCase.from_dict(tc) for tc in data.get("test_inputs", [])],
            generated_test_cases=[
                TestCase.from_dict(tc) for tc in data.get("generated_test_cases", [])
            ],
        )


@dataclass
class MutatedSpec:
    content: str
    index: int


class SpecGenResult(TypedDict):
    status: str
    class_name: str
    verifier_calls: int
    final_code: str
    final_error: str
    verified: bool
    iterations: int
    input_tokens: int
    output_tokens: int


class _WorkerResultRequired(TypedDict):
    task_id: str
    status: str
    class_name: str


class WorkerResult(_WorkerResultRequired, total=False):
    config: dict[str, Any]
    verifier_calls: int
    iterations: int
    verified: bool
    log_file: str
    final_code: str
    final_error: str
    message: str
    input_tokens: int
    output_tokens: int


class SpecGen:

    def __init__(
        self,
        model: str,
        temperature: float,
        max_iterations: int,
        output_dir: str,
        timeout: int,
        logger: logging.Logger,
        verbose: bool = False,
        prompt_type: str = "zero_shot",
    ) -> None:
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

    def _parse_code_from_model_response(self, content: str) -> str:
        content = "a" + content
        extracted_str = content.split("```")[1]
        extracted_str = (
            extracted_str
            if not extracted_str.startswith("java")
            else extracted_str[len("java") :]
        )
        return extracted_str

    def _mutate_token_list_random(
        self, token_list: list[str], has_forall: bool, dont_mutate_logical: bool
    ) -> list[list[str]]:
        res_list: list[list[str]] = []
        token_variant_list: list[str] = []
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

    def _spec_mutator_random(self, line: str) -> list[str]:
        res_list: list[str] = []
        has_forall = line.find("forall") != -1 or line.find("exists") != -1
        res_token_list_list = self._mutate_token_list_random(
            line.split(" "), has_forall, True
        )
        for token_list in res_token_list_list:
            res_list.append(" ".join(token_list) + " ")
        return res_list

    def _mutate_token_list_prior(
        self, token_list: list[str], current_index: int
    ) -> list[list[str]]:
        res_list: list[list[str]] = []
        token_variant_list: list[str] = []
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

    def _spec_mutator_heuristic(self, line: str) -> list[str]:
        res_list: list[str] = []
        has_forall = line.find("forall") != -1 or line.find("exists") != -1
        token_list = line.split(" ")
        res_token_list_list = self._mutate_token_list_prior(token_list, 0)
        for tokens in res_token_list_list:
            res_list.append(" ".join(tokens) + " ")

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

    def _is_invariant_or_postcondition(self, line: str) -> bool:
        return line.find("@") != -1 and (
            line.find("invariant") != -1
            or line.find("maintaining") != -1
            or line.find("ensures") != -1
            or line.find("decreases") != -1
            or line.find("increases") != -1
        )

    def _is_assert(self, line: str) -> bool:
        return line.find("@") != -1 and line.find("assert") != -1

    def _config_to_str(self, config: dict[str, Any]) -> str:
        res = ""
        for message in config["messages"]:
            res += "{role}: {content}\n".format(
                role=message["role"], content=message["content"]
            )
        return res

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

    def run(self, input_code: str, class_name: str) -> SpecGenResult:
        ### [Long function as it is from the Authors implementation, and refactoring may introduce bugs]]

        # Specification Generation Phase
        _verifier_calls_count: int = 0
        self.logger.info(f"[{class_name}] Starting Generation Phase...")
        done_flag: bool = False
        config: dict[str, Any] = {}
        current_code: str = input_code
        verified_flag: bool = False
        err_info: str = ""
        err_types: list[str] = []
        timed_out: bool = False  # Flag to track timeout
        status: str = "unknown"  # Initial status

        num_iter: int = 0
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
                break

            self.logger.debug(f"[{class_name}] {err_info}")
            if err_info == "" or done_flag:
                break

        # Mutation Phase
        self.logger.info(f"[{class_name}] Starting Mutation Phase...")
        current_code_list: list[str] = current_code.split("\n")
        mutated_spec_list: list[MutatedSpec] = []

        # Generate mutated spec set
        for index in range(len(current_code_list)):
            if self._is_invariant_or_postcondition(current_code_list[index]):
                for mutated_spec in self._spec_mutator_heuristic(
                    current_code_list[index]
                ):
                    mutated_spec_list.append(
                        MutatedSpec(content=mutated_spec, index=index)
                    )

        # Mutation loop with safety limit
        mutation_iterations: int = 0
        MAX_MUTATION_ITERATIONS: int = 1000  # Safety limit to avoid infinite loop

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
                    if item.index == index:
                        # replace error spec with mutated spec
                        current_code_list[index] = item.content
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
                current_code, class_name, self.timeout, self.output_dir, self.logger
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
        _input_tokens, _output_tokens = get_token_usage()
        return SpecGenResult(
            status=status,
            class_name=class_name,
            verifier_calls=_verifier_calls_count,
            final_code=current_code,
            final_error=err_info,
            verified=verified_flag,
            iterations=num_iter,
            input_tokens=_input_tokens,
            output_tokens=_output_tokens,
        )


class SpecGenRunner:

    def __init__(
        self,
        name: str,
        input: str,
        output: str,
        model: str,
        temperature: float,
        max_iterations: int,
        openjml_timeout: int,
        threads: int,
        verbose: bool,
        prompt_type: str,
    ) -> None:
        self.name = name
        self.input_path: str = input
        self.output = output
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.openjml_timeout = openjml_timeout
        self.threads = threads
        self.verbose = verbose
        self.prompt_type = prompt_type
        self.input_length: int = 0

    def _run_specgen(self, task: Task) -> WorkerResult:
        class_name: str = task.class_name
        input_code: str = task.code
        task_id: str = task.task_id
        output_dir: str = os.path.abspath(self.output)
        run_config: dict[str, Any] = {
            "prompt_type": self.prompt_type,
            "model": self.model,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
        }

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
                config=run_config,
                log_file="unknown",
            )

        reset_token_usage()
        logger.info(f"Starting SpecGen for {class_name} (task id: {task_id})")

        _thread_specgen_artifacts: str = os.path.join(
            output_dir, f"{task_id}.Thread_{_thread_id}"
        )
        Path(_thread_specgen_artifacts).mkdir(parents=True, exist_ok=True)

        specgen = SpecGen(
            model=self.model,
            temperature=self.temperature,
            max_iterations=self.max_iterations,
            output_dir=_thread_specgen_artifacts,
            timeout=self.openjml_timeout,
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
                f"in {_result['iterations']} iterations "
                f"(tokens: {_result['input_tokens']} in / {_result['output_tokens']} out)"
            )

            return WorkerResult(
                task_id=task_id,
                status=_result["status"],
                class_name=class_name,
                config=run_config,
                verifier_calls=_result["verifier_calls"],
                iterations=_result["iterations"],
                verified=_result.get("verified", False),
                log_file=log_file,
                final_code=_result["final_code"],
                final_error=_result["final_error"],
                input_tokens=_result["input_tokens"],
                output_tokens=_result["output_tokens"],
            )

        except Exception as e:
            return WorkerResult(
                task_id=task_id,
                status="unknown",
                message=str(e),
                class_name=class_name,
                config=run_config,
                log_file=log_file,
            )

    def _save_results(self, duration: float, results: list[WorkerResult]) -> None:
        """Save summary statistics to a JSON file"""
        verified = [r for r in results if r["status"] == "verified"]
        unverified = [r for r in results if r["status"] == "unverified"]
        timed_out = [r for r in results if r["status"] == "timed_out"]
        unknown = [r for r in results if r["status"] == "unknown"]

        completed = verified + unverified + timed_out
        total_verifier_calls = sum(r.get("verifier_calls", 0) for r in completed)
        avg_verifier_calls = total_verifier_calls / len(completed) if completed else 0
        total_input_tokens = sum(r.get("input_tokens", 0) for r in completed)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in completed)

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
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
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
            f"Total tokens used: {total_input_tokens} input / {total_output_tokens} output ({total_input_tokens + total_output_tokens} total)"
        )
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
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        start_time: float = time.time()
        results: list[WorkerResult] = []
        completed_count: int = 0

        print(f"Starting processing with {self.threads} threads...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(self._run_specgen, task): task for task in _input_tasks
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
                        WorkerResult(
                            task_id=task.task_id,
                            status="unknown",
                            message=str(exc),
                            class_name=task.class_name,
                        )
                    )
                    print(
                        f"✗ [{completed_count}/{self.input_length}] {task.class_name} - Exception: {exc}"
                    )
                    completed_count += 1

        end_time: float = time.time()
        duration: float = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
