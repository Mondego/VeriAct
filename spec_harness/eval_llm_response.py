"""
eval_spec_with_model_benchmark_response.py
-------------------------------------------
Batch-evaluate LLM-generated JML annotations from a JSONL response file
against a benchmark JSON, with two levels of threading:

  --threads    : outer parallelism — how many tasks to evaluate concurrently
  --max-pairs  : number of test pairs from benchmark test_inputs to use;
                 each test pair gets its own thread to compute all 4 metrics

Only responses with "status": "verified" are evaluated.

Usage
-----
  python eval_spec_with_model_benchmark_response.py \\
      --benchmark_path benchmarks/benchmark.json \\
      --llm_response_path responses.jsonl \\
      --openjml /path/to/openjml \\
      --output results/ \\
      --threads 4 \\
      --max-pairs 10 \\
      [-v]
"""

from __future__ import annotations

from asyncio.log import logger
import os
import re
import json
import argparse
from signal import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from harness import (
    MethodSpec,
    TestPair,
    InputCase,
    JavaMethodParser,
    JType,
    HarnessResult,
    VerifyResult,
    StubBuilder,
    OutputMutator,
)
from eval_spec import (
    extract_jml_spec,
    detect_input_format,
    parse_input,
    parse_output,
    generate_invalid_inputs,
)


# ============================================================
# Data classes
# ============================================================


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
            javadoc=data.get("javadoc", ""),
            category=data.get("category", ""),
            origin_id=data.get("origin_id", ""),
            test_code=data.get("test_code", ""),
            test_inputs=[TestCase.from_dict(tc) for tc in data.get("test_inputs", [])],
            generated_test_cases=[
                TestCase.from_dict(tc) for tc in data.get("generated_test_cases", [])
            ],
        )


@dataclass
class LLMResponse:
    """One line from the JSONL response file."""

    task_id: str
    status: str
    final_code: str
    class_name: str
    config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMResponse":
        return cls(
            task_id=data["task_id"],
            status=data.get("status", ""),
            final_code=data.get("final_code", ""),
            class_name=data.get("class_name", "Solution"),
            config=data.get("config", {}),
        )


# ============================================================
# Loaders
# ============================================================


def load_benchmark(benchmark_path: str) -> dict[str, Task]:
    """Load benchmark JSON once and return a task_id -> Task lookup dict."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    return {entry["task_id"]: Task.from_dict(entry) for entry in benchmark}


def load_verified_responses(jsonl_path: str) -> list[LLMResponse]:
    """Read JSONL, return only responses with status == 'verified'."""
    responses: list[LLMResponse] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("status") == "verified":
                responses.append(LLMResponse.from_dict(obj))
    return responses


# ============================================================
# OpenJML runner — writes stubs to a persistent directory
# ============================================================


class OpenJMLRunnerPersistent:
    """Runs OpenJML ESC, writing harness stubs to an actual directory."""

    def __init__(self, openjml_path: str, output_dir: str, timeout: int = 300):
        self.openjml_path = openjml_path
        self.output_dir = output_dir
        self.timeout = timeout
        os.makedirs(output_dir, exist_ok=True)

    def verify(
        self, java_source: str, class_name: str, label: str = ""
    ) -> tuple[VerifyResult, str]:
        # Filename MUST match the public class name inside the stub
        # (OpenJML / javac requires this).  Use a per-label subdirectory
        # so that parallel invocations don't overwrite each other.
        harness_class = f"{class_name}Harness"
        fname = f"{harness_class}.java"
        if label:
            safe_label = re.sub(r"[^\w\-]", "_", label)
            sub_dir = os.path.join(self.output_dir, safe_label)
            os.makedirs(sub_dir, exist_ok=True)
            path = os.path.join(sub_dir, fname)
        else:
            path = os.path.join(self.output_dir, fname)
        with open(path, "w") as f:
            f.write(java_source)
        cmd = [
            "openjml",
            "--esc",
            "--esc-max-warnings",
            "1",
            "--prover=cvc4",
            "--nonnull-by-default",
            "--arithmetic-failure=quiet",
            "-nowarn",
            path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            returncode = proc.returncode
            stdout, stderr = proc.communicate(timeout=self.timeout)
            out = stdout + stderr
            return self._parse(returncode, out), out
        except subprocess.TimeoutExpired:
            return VerifyResult.UNKNOWN, "timeout"
        except Exception as e:
            return VerifyResult.UNKNOWN, f"Error: {str(e)}"
        except FileNotFoundError:
            raise RuntimeError(f"OpenJML binary not found at '{self.openjml_path}'.")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    def _parse(returncode: int, output: str) -> VerifyResult:
        low = output.strip().lower()
        if re.search(r"[1-9]\d* warning", low):
            return VerifyResult.FAIL
        if re.search(r"[1-9]\d* verification failure", low):
            return VerifyResult.FAIL
        if re.search(r"\berror\b", low):
            return VerifyResult.FAIL
        if returncode == 0 or "verified" in low or "0 warnings" in low or low == "":
            return VerifyResult.OK
        return VerifyResult.UNKNOWN


# ============================================================
# Per-pair result — returned by each inner thread
# ============================================================


@dataclass
class PairResult:
    """All 4 metric results for a single test pair."""

    pair_idx: int
    # PostCorrectness
    post_correct_detail: dict = field(default_factory=dict)
    # PostCompleteness (multiple mutants)
    post_complete_details: list[dict] = field(default_factory=list)
    # PreCorrectness (valid input for this pair)
    pre_correct_detail: dict = field(default_factory=dict)
    # PreCompleteness (invalid inputs assigned to this pair)
    pre_complete_details: list[dict] = field(default_factory=list)


# ============================================================
# Inner thread: compute all 4 metrics for one test pair
# ============================================================


def _evaluate_one_pair(
    pair_idx: int,
    pair: TestPair,
    valid_input_case: InputCase,
    invalid_cases: list[InputCase],
    parsed: dict,
    spec: MethodSpec,
    rtype: JType,
    builder: StubBuilder,
    mutator: OutputMutator,
    runner: OpenJMLRunnerPersistent,
    cname: str,
    verbose: bool,
    task_id: str,
    java_source: str = "",
) -> PairResult:
    """Compute all 4 metrics for a single test pair in its own thread."""
    result = PairResult(pair_idx=pair_idx)

    # ---- PostCorrectness ---------------------------------------------------
    if spec.postcondition is not None:
        stub = builder.post_correctness_stub(parsed, spec, pair)
        verdict, _ = runner.verify(stub, cname, f"post_correct_{pair.label}")
        ok = verdict == VerifyResult.OK
        result.post_correct_detail = {
            "label": pair.label,
            "verdict": verdict.value,
            "pass": ok,
            "stub": stub,
        }
        _log(verbose, task_id, "PostCorrectness", pair.label, verdict, ok)

    # ---- PostCompleteness --------------------------------------------------
    if spec.postcondition is not None:
        for mut_out in mutator.mutate(rtype, pair.output):
            stub = builder.post_completeness_stub(parsed, spec, pair, mut_out)
            lbl = f"post_complete_{pair.label}_{str(mut_out)[:20]}"
            verdict, _ = runner.verify(stub, cname, lbl)
            killed = verdict == VerifyResult.FAIL
            result.post_complete_details.append(
                {
                    "label": pair.label,
                    "mutant": str(mut_out),
                    "verdict": verdict.value,
                    "killed": killed,
                    "stub": stub,
                }
            )
            _log(
                verbose,
                task_id,
                "PostCompleteness",
                pair.label,
                verdict,
                killed,
                f"mutant={mut_out}",
            )

    # ---- PreCorrectness ----------------------------------------------------
    if spec.precondition is not None:
        stub = builder.pre_correctness_stub(
            parsed, spec, valid_input_case, original_source=java_source
        )
        verdict, _ = runner.verify(stub, cname, f"pre_correct_{valid_input_case.label}")
        ok = verdict == VerifyResult.OK
        result.pre_correct_detail = {
            "label": valid_input_case.label,
            "verdict": verdict.value,
            "pass": ok,
            "stub": stub,
        }
        _log(verbose, task_id, "PreCorrectness", valid_input_case.label, verdict, ok)

    # ---- PreCompleteness ---------------------------------------------------
    if spec.precondition is not None:
        for case in invalid_cases:
            stub = builder.pre_completeness_stub(
                parsed, spec, case, original_source=java_source
            )
            verdict, _ = runner.verify(stub, cname, f"pre_complete_{case.label}")
            rejected = verdict == VerifyResult.FAIL
            result.pre_complete_details.append(
                {
                    "label": case.label,
                    "verdict": verdict.value,
                    "rejected": rejected,
                    "stub": stub,
                }
            )
            _log(verbose, task_id, "PreCompleteness", case.label, verdict, rejected)

    return result


# ============================================================
# Main evaluation — one thread per test pair
# ============================================================


def evaluate_problem(
    task: Task,
    llm_code: str,
    openjml_path: str = "openjml",
    output_dir: str = ".",
    verbose: bool = False,
    max_pairs: int = 0,
) -> dict:
    """
    Evaluate one task. Each test pair gets its own thread to compute
    all 4 metrics in parallel.
    """
    solution_src = task.code
    test_src = task.test_code
    io_pairs = [{"input": tc.input, "output": tc.output} for tc in task.test_inputs]
    _io_pairs_len = len(io_pairs)
    if max_pairs > 0:
        if _io_pairs_len < max_pairs:
            max_pairs = _io_pairs_len
        io_pairs = io_pairs[:max_pairs]

    # ---- parse method signature from benchmark Solution.java ---------------
    parser = JavaMethodParser()
    bench_parsed = parser.parse(solution_src)
    bench_params = bench_parsed["params"]
    return_type = bench_parsed["return_type"]

    # ---- extract JML spec from LLM code ------------------------------------
    spec = extract_jml_spec(llm_code)
    if spec.precondition is None and spec.postcondition is None:
        print(
            f"WARNING [{task.task_id}]: no JML requires/ensures found", file=sys.stderr
        )

    # ---- parse LLM method signature ----------------------------------------
    try:
        llm_parsed = parser.parse(llm_code)
        llm_params = llm_parsed["params"]
    except (ValueError, Exception):
        llm_parsed = bench_parsed
        llm_params = bench_params

    # ---- detect input format from Test.java --------------------------------
    read_ops = detect_input_format(test_src, bench_params)

    # ---- parse io_pairs into TestPair / InputCase objects -------------------
    test_pairs: list[TestPair] = []
    valid_inputs: list[dict] = []

    for idx, pair in enumerate(io_pairs):
        try:
            inputs_bench = parse_input(pair["input"], read_ops)
            output = parse_output(pair["output"], return_type)
        except Exception as e:
            if verbose:
                print(
                    f"WARNING [{task.task_id}]: skipping case {idx}: {e}",
                    file=sys.stderr,
                )
            continue

        inputs_llm: dict[str, Any] = {}
        for bp, lp in zip(bench_params, llm_params):
            inputs_llm[lp["name"]] = inputs_bench[bp["name"]]

        test_pairs.append(TestPair(inputs_llm, output, f"case_{idx}"))
        valid_inputs.append(inputs_llm)

    if not test_pairs:
        print(f"ERROR [{task.task_id}]: no test pairs could be parsed", file=sys.stderr)
        return {}

    # ---- build InputCases for Pre metrics ----------------------------------
    valid_input_cases = [InputCase(tp.inputs, True, tp.label) for tp in test_pairs]
    invalid_input_cases = generate_invalid_inputs(llm_params, valid_inputs)

    # ---- setup -------------------------------------------------------------
    builder = StubBuilder()
    mutator = OutputMutator(k=5)
    parsed = llm_parsed
    cname = parsed["class_name"]
    rtype = parsed["return_type"]

    safe_task_id = task.task_id.replace("/", "_").replace("\\", "_")
    stubs_dir = os.path.join(output_dir, "stubs", safe_task_id)
    runner = OpenJMLRunnerPersistent(openjml_path, stubs_dir)

    # ---- distribute invalid cases round-robin across test pairs ------------
    invalid_per_pair: list[list[InputCase]] = [[] for _ in test_pairs]
    for i, ic in enumerate(invalid_input_cases):
        invalid_per_pair[i % len(test_pairs)].append(ic)

    # ---- launch one thread per test pair -----------------------------------
    n_threads = len(test_pairs)
    pair_results: list[PairResult] = [None] * n_threads  # type: ignore

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        future_to_idx = {
            pool.submit(
                _evaluate_one_pair,
                idx,
                pair,
                valid_input_cases[idx],
                invalid_per_pair[idx],
                parsed,
                spec,
                rtype,
                builder,
                mutator,
                runner,
                cname,
                verbose,
                task.task_id,
                llm_code,
            ): idx
            for idx, pair in enumerate(test_pairs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                pair_results[idx] = future.result()
            except Exception as e:
                print(f"ERROR [{task.task_id}] pair {idx}: {e}", file=sys.stderr)

    # ---- aggregate PairResults into HarnessResults -------------------------
    pc_details: list[dict] = []
    pcl_details: list[dict] = []
    prc_details: list[dict] = []
    prl_details: list[dict] = []

    for pr in pair_results:
        if pr is None:
            continue
        if pr.post_correct_detail:
            pc_details.append(pr.post_correct_detail)
        pcl_details.extend(pr.post_complete_details)
        if pr.pre_correct_detail:
            prc_details.append(pr.pre_correct_detail)
        prl_details.extend(pr.pre_complete_details)

    results: dict[str, HarnessResult] = {}

    # PostCorrectness
    hr = HarnessResult("PostCorrectness", len(pc_details), 0)
    hr.passed = sum(1 for d in pc_details if d.get("pass"))
    hr.details = pc_details
    results["post_correctness"] = hr

    # PostCompleteness
    hr = HarnessResult("PostCompleteness", len(pcl_details), 0)
    hr.passed = sum(1 for d in pcl_details if d.get("killed"))
    hr.details = pcl_details
    results["post_completeness"] = hr

    # PreCorrectness
    hr = HarnessResult("PreCorrectness", len(prc_details), 0)
    hr.passed = sum(1 for d in prc_details if d.get("pass"))
    hr.details = prc_details
    results["pre_correctness"] = hr

    # PreCompleteness
    hr = HarnessResult("PreCompleteness", len(prl_details), 0)
    hr.passed = sum(1 for d in prl_details if d.get("rejected"))
    hr.details = prl_details
    results["pre_completeness"] = hr

    # ---- summary -----------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}")
    print(
        f"  Spec-Harness  |  task: {task.task_id}"
        f"  |  method: {parsed['method_name']}"
    )
    print(sep)
    for r in results.values():
        print(f"  {r}")
    print(sep)

    # ---- build output dict -------------------------------------------------
    output: dict[str, Any] = {"task_id": task.task_id}
    for k, v in results.items():
        output[k] = {
            "score": v.score,
            "passed": v.passed,
            "total": v.total,
            "details": v.details,
        }
    return output


def _log(
    verbose: bool,
    task_id: str,
    metric: str,
    label: str,
    verdict: VerifyResult,
    desired: bool,
    extra: str = "",
) -> None:
    if not verbose:
        return
    status = "✓" if desired else "✗"
    ex = f"  [{extra}]" if extra else ""
    print(f"  [{task_id}] [{metric:20s}] {label}{ex}" f"  →  {verdict.value}  {status}")


# ============================================================
# Outer-level batch processing with threading
# ============================================================


def process_one(
    task: Task,
    resp: LLMResponse,
    openjml_path: str,
    output_dir: str,
    verbose: bool,
    max_pairs: int,
) -> dict | None:
    """Evaluate a single (task, response) pair. Returns result dict or None."""
    try:
        result = evaluate_problem(
            task=task,
            llm_code=resp.final_code,
            openjml_path=openjml_path,
            output_dir=output_dir,
            verbose=verbose,
            max_pairs=max_pairs,
        )
        if result:
            safe_id = task.task_id.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(output_dir, f"{safe_id}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        return result
    except Exception as e:
        print(f"ERROR [{resp.task_id}]: {e}", file=sys.stderr)
        return None


def run_batch(
    benchmark_path: str,
    llm_response_path: str,
    openjml_path: str,
    output_dir: str,
    threads: int = 1,
    verbose: bool = False,
    max_pairs: int = 0,
) -> list[dict]:
    """
    Load benchmark + verified LLM responses, evaluate all.

    Parameters
    ----------
    threads    : outer parallelism — number of tasks evaluated concurrently
    max_pairs  : number of test pairs to use per task (0 = all);
                 also the number of inner threads per task
    """
    os.makedirs(output_dir, exist_ok=True)

    task_lookup = load_benchmark(benchmark_path)
    responses = load_verified_responses(llm_response_path)

    print(
        f"Loaded {len(task_lookup)} benchmark tasks, "
        f"{len(responses)} verified responses"
    )

    # Match responses to tasks
    work: list[tuple[Task, LLMResponse]] = []
    for resp in responses:
        task = task_lookup.get(resp.task_id)
        if task is None:
            print(
                f"WARNING: task_id '{resp.task_id}' not in benchmark, " f"skipping",
                file=sys.stderr,
            )
            continue
        work.append((task, resp))

    print(
        f"Evaluating {len(work)} tasks  "
        f"(outer threads={threads}, "
        f"inner threads per task=max_pairs={max_pairs or 'all'})"
    )

    all_results: list[dict] = []

    if threads <= 1:
        for task, resp in work:
            r = process_one(task, resp, openjml_path, output_dir, verbose, max_pairs)
            if r:
                all_results.append(r)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            future_to_tid = {
                pool.submit(
                    process_one,
                    task,
                    resp,
                    openjml_path,
                    output_dir,
                    verbose,
                    max_pairs,
                ): resp.task_id
                for task, resp in work
            }
            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    r = future.result()
                    if r:
                        all_results.append(r)
                except Exception as e:
                    print(f"ERROR [{tid}]: {e}", file=sys.stderr)

    # ---- write summary -----------------------------------------------------
    summary = {
        "total_evaluated": len(all_results),
        "total_verified_responses": len(responses),
        "tasks": [],
    }
    for r in all_results:
        summary["tasks"].append(
            {
                "task_id": r["task_id"],
                "post_correctness": r["post_correctness"]["score"],
                "post_completeness": r["post_completeness"]["score"],
                "pre_correctness": r["pre_correctness"]["score"],
                "pre_completeness": r["pre_completeness"]["score"],
            }
        )
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Batch complete: {len(all_results)}/{len(work)} tasks evaluated")
    print(f"  Summary -> {summary_path}")
    print(f"{'=' * 60}")

    return all_results


# ============================================================
# CLI
# ============================================================


def main():
    ap = argparse.ArgumentParser(
        description="Batch-evaluate LLM JML annotations from a JSONL "
        "response file against a benchmark JSON."
    )
    ap.add_argument(
        "--benchmark_path", required=True, help="Path to benchmark JSON file"
    )
    ap.add_argument(
        "--llm_response_path",
        required=True,
        help="Path to JSONL file with LLM responses",
    )
    ap.add_argument("--openjml", default="openjml", help="Path to OpenJML binary")
    ap.add_argument(
        "--output",
        default="results",
        help="Directory to store per-task results and summary",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of tasks to evaluate concurrently " "(default: 1)",
    )
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=1,
        help="Number of test pairs from benchmark test_inputs "
        "to use (0 = all); each pair gets its own thread "
        "for computing all 4 metrics",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    run_batch(
        benchmark_path=args.benchmark_path,
        llm_response_path=args.llm_response_path,
        openjml_path=args.openjml,
        output_dir=args.output,
        threads=args.threads,
        verbose=args.verbose,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":

    main()
