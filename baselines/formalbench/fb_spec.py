import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from baselines.formalbench.formalbench_runner import FormalBench, VALID_PROMPT_TYPES
from baselines.formalbench.spec_fixer import SpecFixer
from baselines.formalbench.prompts import build_messages
from baselines.utils.logger import create_logger
from baselines.utils.models import create_model_config, request_llm_engine
from baselines.utils.file_utility import load_json, dump_json, dump_jsonl
from baselines.utils.verifier import verify_with_openjml


class FBSpec:
    """
    Combined generation + repair pipeline in a single unified loop.

    Each iteration is one LLM call.  The phase is implicit:
      - curr_spec is None  →  generation attempt
      - curr_spec is set   →  fix attempt (using error from last verification)

    If a generation call returns an empty or invalid spec the loop retries
    generation on the next iteration instead of stopping.  Once a valid spec
    exists the loop switches to fixing and continues up to max_iters total
    iterations.
    """

    def __init__(
        self,
        model,
        temperature,
        prompt_type,
        max_iters,
        output_dir,
        timeout,
        logger,
        verbose=False,
    ):
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.max_iters = max_iters
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

        # Used for utility methods and example loading only — not called as
        # black-box runners here.
        self._generator = FormalBench(
            model=model,
            temperature=temperature,
            prompt_type=prompt_type,
            output_dir=output_dir,
            timeout=timeout,
            logger=logger,
            verbose=verbose,
        )

        self._fixer = SpecFixer(
            model=model,
            temperature=temperature,
            max_iters=max_iters,
            output_dir=output_dir,
            timeout=timeout,
            logger=logger,
            verbose=verbose,
        )

    def run(self, input_code, class_name):
        curr_spec = None        # None  →  still need a valid generation
        curr_err = None
        fix_history = []        # growing conversation history for fix turns
        verifier_calls = 0
        gen_verifier_calls = 0
        fix_verifier_calls = 0
        status = "unverified"
        num_iter = 0

        for num_iter in range(1, self.max_iters + 1):
            in_gen = curr_spec is None

            # ----------------------------------------------------------
            # LLM call
            # ----------------------------------------------------------
            if in_gen:
                self.logger.info(
                    f"[{class_name}] Iter {num_iter}/{self.max_iters}: generating..."
                )
                messages = build_messages(
                    prompt_type=self.prompt_type,
                    model=self.model,
                    code=input_code,
                    example_code1=self._generator.example_code1,
                    example_code2=self._generator.example_code2,
                    example_spec1=self._generator.example_spec1,
                    example_spec2=self._generator.example_spec2,
                )
            else:
                self.logger.info(
                    f"[{class_name}] Iter {num_iter}/{self.max_iters}: fixing..."
                )
                error_info = self._fixer._analyze_failures(curr_err)
                messages = self._fixer._build_fix_messages(
                    curr_spec, curr_err, error_info, fix_history
                )

            config = create_model_config(messages, self.model, self.temperature)
            ret = request_llm_engine(config)

            if ret is None:
                self.logger.error(
                    f"[{class_name}] LLM returned no response on iter {num_iter}"
                )
                status = "unknown"
                break

            raw = ret.choices[0].message.content
            if self.verbose:
                self.logger.debug(f"[{class_name}] LLM response: {raw[:500]}")

            new_spec = self._generator._parse_spec_from_response(raw)

            # ----------------------------------------------------------
            # Validate parsed spec — retry same phase on bad output
            # ----------------------------------------------------------
            if not new_spec or not self._generator._contains_annotations(new_spec):
                self.logger.warning(
                    f"[{class_name}] Iter {num_iter}: empty or invalid spec — retrying..."
                )
                if not in_gen:
                    # record the rejected fix attempt in history so the LLM
                    # has context on the next iteration
                    fix_history.append(messages[-1])
                    fix_history.append({"role": "assistant", "content": raw})
                continue

            # valid spec — commit to history if in fix phase
            if not in_gen:
                fix_history.append(messages[-1])
                fix_history.append({"role": "assistant", "content": raw})

            curr_spec = new_spec

            self.logger.info(f"[{class_name}] Iter {num_iter}: verifying...")
            curr_err = verify_with_openjml(curr_spec, class_name, self.timeout, self.output_dir, self.logger)
            verifier_calls += 1
            if in_gen:
                gen_verifier_calls += 1
            else:
                fix_verifier_calls += 1

            if self.verbose:
                self.logger.debug(
                    f"[{class_name}] Verification output: {curr_err[:500]}"
                )

            if "Timeout:" in curr_err or "timeout" in curr_err.lower():
                self.logger.warning(
                    f"[{class_name}] Verification timed out at iter {num_iter}"
                )
                status = "timed_out"
                break

            if curr_err == "":
                self.logger.info(
                    f"[{class_name}] Verified at iter {num_iter} "
                    f"({verifier_calls} verifier call(s))"
                )
                status = "verified"
                break

            self.logger.debug(
                f"[{class_name}] Iter {num_iter} still failing: {curr_err[:200]}"
            )

        else:
            self.logger.warning(
                f"[{class_name}] Max iterations ({self.max_iters}) reached "
                "without verification"
            )
            status = "unverified"

        # fix_iterations = number of fix-phase LLM calls made
        fix_iters = len(fix_history) // 2

        return {
            "status": status,
            "class_name": class_name,
            "verifier_calls": verifier_calls,
            "gen_phase_verifier_calls": gen_verifier_calls,
            "fix_phase_verifier_calls": fix_verifier_calls,
            "fix_iterations": fix_iters,
            "iterations": num_iter,
            "final_code": curr_spec,
            "final_error": curr_err or "",
            "verified": status == "verified",
        }


class FBSpecWorker:

    def __init__(
        self, output_dir, model, temperature, prompt_type, max_iters, timeout, verbose=False
    ):
        self.output_dir = output_dir
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.max_iters = max_iters
        self.timeout = timeout
        self.verbose = verbose

    def run_fb_spec(self, task: dict):
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

        logger.info(f"Starting FBSpec for {class_name} (task id: {task_id})")

        fb_spec = FBSpec(
            model=self.model,
            temperature=self.temperature,
            prompt_type=self.prompt_type,
            max_iters=self.max_iters,
            output_dir=self.output_dir,
            timeout=self.timeout,
            logger=logger,
            verbose=self.verbose,
        )

        try:
            _result = fb_spec.run(input_code, class_name)

            if _result["verified"]:
                phase = "gen" if _result["fix_iterations"] == 0 else "fix"
                logger.info(
                    f"VERIFIED - {class_name} verified in {phase} phase, "
                    f"{_result['verifier_calls']} total verifier call(s), "
                    f"{_result['iterations']} total iteration(s)"
                )
            else:
                logger.warning(
                    f"UNVERIFIED - {class_name} status={_result['status']}, "
                    f"{_result['verifier_calls']} total verifier call(s)"
                )

            return {
                "id": task_id,
                "status": _result["status"],
                "class_name": class_name,
                "verifier_calls": _result["verifier_calls"],
                "gen_phase_verifier_calls": _result["gen_phase_verifier_calls"],
                "fix_phase_verifier_calls": _result["fix_phase_verifier_calls"],
                "fix_iterations": _result["fix_iterations"],
                "iterations": _result["iterations"],
                "verified": _result["verified"],
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


class FBSpecRunner:

    def __init__(
        self,
        name,
        input,
        output,
        model,
        temperature,
        prompt_type,
        max_iters,
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
        self.max_iters = max_iters
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

        # Verified in generation phase (no repair needed)
        verified_in_gen = [r for r in verified if r.get("fix_iterations", 0) == 0]
        # Verified only after repair
        verified_in_fix = [r for r in verified if r.get("fix_iterations", 0) > 0]

        counted = verified + unverified + timed_out
        total_verifier_calls = sum(r.get("verifier_calls", 0) for r in counted)
        avg_verifier_calls = total_verifier_calls / len(counted) if counted else 0

        summary = {
            "total_files_processed": self.input_length,
            "verified": len(verified),
            "verified_in_gen_phase": len(verified_in_gen),
            "verified_in_fix_phase": len(verified_in_fix),
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
            "max_iters": self.max_iters,
            "verified_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "gen_phase_verifier_calls": r["gen_phase_verifier_calls"],
                    "fix_phase_verifier_calls": r["fix_phase_verifier_calls"],
                    "fix_iterations": r["fix_iterations"],
                    "log_file": r["log_file"],
                }
                for r in verified
            ],
            "unverified_cases": [
                {
                    "id": r["id"],
                    "class_name": r["class_name"],
                    "verifier_calls": r["verifier_calls"],
                    "fix_iterations": r["fix_iterations"],
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
        print(f"Verified:          {len(verified)}/{self.input_length} ({len(verified)/self.input_length*100:.1f}%)")
        print(f"  - in gen phase:  {len(verified_in_gen)}/{self.input_length}")
        print(f"  - in fix phase:  {len(verified_in_fix)}/{self.input_length}")
        print(f"Unverified:        {len(unverified)}/{self.input_length} ({len(unverified)/self.input_length*100:.1f}%)")
        print(f"Timed out:         {len(timed_out)}/{self.input_length} ({len(timed_out)/self.input_length*100:.1f}%)")
        print(f"Invalid JML:       {len(invalid_jml)}/{self.input_length}")
        print(f"Empty spec:        {len(empty_spec)}/{self.input_length}")
        print(f"Unknown:           {len(unknown)}/{self.input_length}")
        print(f"Summary saved to: {os.path.join(self.output, f'{self.name}_results_summary.json')}")

    def run_workers(self):
        _input_tasks = load_json(self.input)
        self.input_length = len(_input_tasks)
        print(f"Found {self.input_length} input tasks to process")

        output_dir = os.path.abspath(self.output)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        worker = FBSpecWorker(
            output_dir=output_dir,
            model=self.model,
            temperature=self.temperature,
            prompt_type=self.prompt_type,
            max_iters=self.max_iters,
            timeout=self.openjml_timeout,
            verbose=self.verbose,
        )

        start_time = time.time()
        results = []
        completed_count = 0

        print(f"Starting processing with {self.threads} thread(s)...")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            future_tasks = {
                executor.submit(worker.run_fb_spec, task): task
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
                        phase = "gen" if result.get("fix_iterations", 0) == 0 else "fix"
                        print(
                            f"[VERIFIED/{phase}] {prefix} {name} - "
                            f"{result['verifier_calls']} verifier call(s), "
                            f"{result['iterations']} iter(s)"
                        )
                    elif status == "unverified":
                        print(
                            f"[UNVERIFIED]  {prefix} {name} - "
                            f"{result['verifier_calls']} verifier call(s), "
                            f"{result.get('fix_iterations', 0)} fix iter(s)"
                        )
                    elif status == "timed_out":
                        print(f"[TIMED OUT]   {prefix} {name}")
                    elif status == "invalid_jml":
                        print(f"[INVALID JML] {prefix} {name}")
                    elif status == "empty_spec":
                        print(f"[EMPTY SPEC]  {prefix} {name}")
                    else:
                        print(f"[ERROR]       {prefix} {name} - {result.get('message', 'unknown error')}")

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
                    print(f"[ERROR]       [{completed_count}/{self.input_length}] {task.get('class_name', task['id'])} - {exc}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nAll tasks completed in {duration:.2f} seconds")
        self._save_results(duration, results)
