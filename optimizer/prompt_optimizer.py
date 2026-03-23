"""
Prompt Optimization Comparison for Formal Specification Synthesis
======================================================================

Compares COPRO vs MIPROv2 vs GEPA on optimizing the best-performing
instruction prompt from baselines.

Design:
    - Optimize on FormalBench train (100 tasks, stratified by category)
    - Validate on FormalBench val (50 tasks)
    - Evaluate on FormalBench test (512 tasks) + SpecGenBench (120 tasks)
    - Graduated scoring during optimization, binary for final reporting

Setup:
    pip install dspy python-dotenv --break-system-packages

Usage:
    python prompt_optimizer.py \
        --formalbench_path ./data/formalbench_tasks.json \
        --specgenbench_path ./data/specgenbench_tasks.json \
        --best_seed formalbench_ltm \
        --model openai/gpt-4o \
        --reflection_model openai/gpt-4o
"""

import json
import os
import random
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import dspy
from dotenv import load_dotenv

from optimizer.optimizer_utils import (
    Task,
    VerificationResult,
    verify_with_openjml,
    compute_graduated_score,
    clean_code_fences,
    format_error_feedback,
)
import baselines.utils.config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prompt_optimizer")

DEFAULT_NUM_THREADS = 8
OPENJML_OUTPUT_DIR: str | None = None


# ============================================================================
# 1. SEED PROMPTS — 4 unique instructions from SpecGen + FormalBench
# ============================================================================
# Extracted from the actual prompt JSON files.
# System message + user template guidance merged into one instruction.
# Set --best_seed to whichever performed best accross the baselinse

SEED_PROMPTS: dict[str, str] = {
    "zero": (
        "You are an JML specification generator for java programs. "
        "Please generate JML specifications for the Java program given below."
    ),
    "cot": (
        "You are an expert in Java Modeling Language (JML). You will be provided "
        "with Java code snippets. Your task is to generate JML specifications for "
        "the given Java code. The specifications should be written as annotations "
        "within the Java code and must be compatible with the OpenJML tool for "
        "verification. Ensure the specifications include detailed preconditions, "
        "postconditions, necessary loop invariants, invariants, assertions, and "
        "any relevant assumptions. Think step by step."
    ),
    "ltm": (
        "You are an expert in Java Modeling Language (JML). You will be provided "
        "with Java code snippets. Your task is to generate JML specifications for "
        "the given Java code. The specifications should be written as annotations "
        "within the Java code and must be compatible with the OpenJML tool for "
        "verification. Ensure the specifications include detailed preconditions, "
        "postconditions, necessary loop invariants, invariants, assertions, and "
        "any relevant assumptions. Break down the problem: First, identify the "
        "weakest preconditions including nullness and arithmetic bounds. Second, "
        "determine the strongest postconditions. Third, derive necessary loop "
        "invariants, assertions, assumptions, and ranking functions."
    ),
}


# ============================================================================
# 2. FIXED 2-SHOT DEMOS — from SpecGenBench with verified ground-truth specs
# ============================================================================
# These 2 tasks are excluded from SpecGenBench evaluation to prevent leakage.
# Demo 1: Smallest element (sequential + loop)
# Demo 2: Binary search (branched + loop)

# Used to exclude demo tasks from SpecGenBench test set.
DEMO_TASK_IDS = {"SGB_Smallest", "SGB_BinarySearch"}

DEMO_EXAMPLES = [
    # Demo 1: Find smallest element in array (sequential + loop)
    dspy.Example(
        java_program=(
            "public class Solution {\n"
            "    static public int solve(int[] a) {\n"
            "        if (a.length == 0) return -1;\n"
            "\n"
            "        int index = 0;\n"
            "        int smallest = 0;\n"
            "        while (a.length - index > 0) {\n"
            "            if (a[index] < a[smallest]) {\n"
            "                smallest = index;\n"
            "            }\n"
            "            index = index + 1;\n"
            "        }\n"
            "        return smallest;\n"
            "    }\n"
            "}"
        ),
        jml_specification=(
            "public class Solution {\n"
            "    //@ ensures \\result == -1 <==> a.length == 0;\n"
            "    //@ ensures -1 < \\result ==> (\\forall int i; 0 <= i && i < a.length; a[\\result] <= a[i]);\n"
            "    static public int solve(int[] a) {\n"
            "        if (a.length == 0) return -1;\n"
            "\n"
            "        int index = 0;\n"
            "        int smallest = 0;\n"
            "        //@ maintaining 0 <= index && index <= a.length;\n"
            "        //@ maintaining 0 <= smallest && smallest < a.length;\n"
            "        //@ maintaining (\\forall int i; 0 <= i && i < index; a[smallest] <= a[i]);\n"
            "        //@ decreases a.length - index;\n"
            "        while (a.length - index > 0) {\n"
            "            if (a[index] < a[smallest]) {\n"
            "                smallest = index;\n"
            "            }\n"
            "            index = index + 1;\n"
            "        }\n"
            "        return smallest;\n"
            "    }\n"
            "}"
        ),
    ).with_inputs("java_program"),
    # Demo 2: Binary search (branched + loop)
    dspy.Example(
        java_program=(
            "public class Solution {\n"
            "    public static int solve(int[] arr, int key) {\n"
            "        if (arr.length == 0) {\n"
            "            return -1;\n"
            "        } else {\n"
            "            int low = 0;\n"
            "            int high = arr.length;\n"
            "            int mid =  high / 2;\n"
            "            while (low < high && arr[mid] != key) {\n"
            "                if (arr[mid] < key) {\n"
            "                    low = mid + 1;\n"
            "                } else {\n"
            "                    high = mid;\n"
            "                }\n"
            "                mid = low + (high - low) / 2;\n"
            "            }\n"
            "            if (low >= high) {\n"
            "                return -1;\n"
            "            }\n"
            "            return mid;\n"
            "        }\n"
            "    }\n"
            "}"
        ),
        jml_specification=(
            "public class Solution {\n"
            "    //@ requires \\forall int j; 0 <= j && j < arr.length; \\forall int i; 0 <= i && i < j ;arr[i] <= arr[j];\n"
            "    //@ ensures \\result == -1 <==> (\\forall int i; 0 <= i && i < arr.length; arr[i] != key) || arr.length == 0;\n"
            "    //@ ensures 0 <= \\result && \\result < arr.length ==> arr[\\result] == key;\n"
            "    public static int solve(int[] arr, int key) {\n"
            "        if (arr.length == 0) {\n"
            "            return -1;\n"
            "        } else {\n"
            "            int low = 0;\n"
            "            int high = arr.length;\n"
            "            int mid =  high / 2;\n"
            "            //@ maintaining 0 <= low && low <= high  && high <= arr.length && mid == low + (high - low) / 2;\n"
            "            //@ maintaining (\\forall int i; 0 <= i && i < low; arr[i] < key);\n"
            "            //@ maintaining (\\forall int i; high <= i && i < arr.length; key < arr[i]);\n"
            "            //@ decreases high - low;\n"
            "            while (low < high && arr[mid] != key) {\n"
            "                if (arr[mid] < key) {\n"
            "                    low = mid + 1;\n"
            "                } else {\n"
            "                    high = mid;\n"
            "                }\n"
            "                mid = low + (high - low) / 2;\n"
            "            }\n"
            "            if (low >= high) {\n"
            "                return -1;\n"
            "            }\n"
            "            return mid;\n"
            "        }\n"
            "    }\n"
            "}"
        ),
    ).with_inputs("java_program"),
]


# ============================================================================
# 3. DSPy SIGNATURE & MODULE
# ============================================================================


class GenerateJMLSpec(dspy.Signature):
    """Generate JML specifications for a Java program.
    Return ONLY the complete annotated Java code with JML comments."""

    java_program: str = dspy.InputField(
        desc="Java source code to annotate with JML specifications"
    )
    jml_specification: str = dspy.OutputField(
        desc="Complete Java program annotated with JML @requires, @ensures, "
        "@maintaining, and @decreases clauses"
    )


class SpecSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateJMLSpec)

    def forward(self, java_program: str) -> dspy.Prediction:
        return self.generate(java_program=java_program)


# ============================================================================
# 4. BENCHMARK LOADING
# ============================================================================


def load_tasks(path: str) -> list[Task]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", data.get("data", []))
    return [Task.from_dict(t) for t in data]


def tasks_to_dspy_examples(tasks: list[Task]) -> list[dspy.Example]:
    examples = []
    for task in tasks:
        ex = dspy.Example(
            java_program=task.code,
            task_id=task.task_id,
            class_name=task.class_name,
            category=task.category,
        ).with_inputs("java_program")
        examples.append(ex)
    return examples


def stratified_split(
    tasks: list[Task],
    train_size: int,
    val_size: int,
    seed: int = 42,
) -> tuple[list[Task], list[Task], list[Task]]:
    """
    Stratified split by category into train / val / test.
    train_size and val_size are targets; remaining goes to test.
    """
    rng = random.Random(seed)

    # Group by category
    by_category: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_category[task.category].append(task)

    # Shuffle within each category
    for cat in by_category:
        rng.shuffle(by_category[cat])

    total = len(tasks)
    train_frac = train_size / total
    val_frac = val_size / total

    train, val, test = [], [], []

    for cat, cat_tasks in by_category.items():
        n = len(cat_tasks)
        n_train = max(1, round(n * train_frac))
        n_val = max(1, round(n * val_frac))

        train.extend(cat_tasks[:n_train])
        val.extend(cat_tasks[n_train : n_train + n_val])
        test.extend(cat_tasks[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ============================================================================
# 5. METRICS
# ============================================================================


def spec_metric_graduated(example, prediction, trace=None) -> float:
    """Graduated metric for COPRO and MIPROv2 optimization."""
    class_name = example.class_name
    generated = getattr(prediction, "jml_specification", "")
    generated = clean_code_fences(generated)

    if not generated.strip():
        return 0.0

    result = verify_with_openjml(
        code_with_spec=generated,
        classname=class_name,
        output_dir=OPENJML_OUTPUT_DIR,
    )
    return compute_graduated_score(result)


def spec_metric_with_feedback(
    example, prediction, trace=None, pred_name=None, pred_trace=None
):
    """Graduated metric + rich feedback for GEPA reflection."""
    class_name = example.class_name
    task_id = getattr(example, "task_id", class_name)
    generated = getattr(prediction, "jml_specification", "")
    generated = clean_code_fences(generated)

    if not generated.strip():
        feedback = (
            f"[FAIL] Task '{task_id}': LLM returned empty or unparseable output. "
            f"The instruction must clearly ask for annotated Java code with "
            f"JML comments in //@ format. Do not include natural language "
            f"explanations - only the Java code with JML annotations."
        )
        return dspy.Prediction(score=0.0, feedback=feedback)

    result = verify_with_openjml(
        code_with_spec=generated,
        classname=class_name,
        output_dir=OPENJML_OUTPUT_DIR,
    )

    score = compute_graduated_score(result)
    feedback = format_error_feedback(result, task_id)

    # Enrich feedback with score explanation for GEPA
    if score == 0.3:
        feedback += (
            f"\n--- Score: 0.3 (1 verification error) ---\n"
            f"The specification is almost correct. Focus on fixing the single "
            f"failing clause identified above."
        )
    elif score == 0.1:
        feedback += (
            f"\n--- Score: 0.1 ({len(result.classified_errors)} errors) ---\n"
            f"JML syntax is valid but multiple clauses are semantically wrong. "
            f"Reconsider the overall specification strategy."
        )
    elif score == 0.0 and result.classified_errors:
        feedback += (
            f"\n--- Score: 0.0 (syntax errors) ---\n"
            f"Output contains JML syntax errors. Ensure correct use of "
            f"@requires, @ensures, @maintaining, @decreases with proper "
            f"JML operators (\\result, \\old, \\forall, \\exists)."
        )

    return dspy.Prediction(score=score, feedback=feedback)


def spec_metric_binary(example, prediction, trace=None) -> float:
    """Binary pass/fail for final evaluation. This goes in the paper."""
    class_name = example.class_name
    generated = getattr(prediction, "jml_specification", "")
    generated = clean_code_fences(generated)

    if not generated.strip():
        return 0.0

    result = verify_with_openjml(
        code_with_spec=generated,
        classname=class_name,
        output_dir=OPENJML_OUTPUT_DIR,
    )
    return 1.0 if result.success else 0.0


# ============================================================================
# 6. OPTIMIZER FACTORY
# ============================================================================


def create_optimizer(
    name: str,
    prompt_model: dspy.LM,
    task_model: dspy.LM,
    log_dir: str,
) -> tuple:
    """Create a DSPy optimizer. Returns (optimizer, compile_kwargs)."""

    if name == "copro":
        optimizer = dspy.COPRO(
            prompt_model=prompt_model,
            metric=spec_metric_graduated,
            depth=3,
            breadth=5,
            init_temperature=0.7,
        )
        compile_kwargs = {
            "eval_kwargs": {
                "num_threads": DEFAULT_NUM_THREADS,
                "display_progress": True,
            },
        }
        return optimizer, compile_kwargs

    elif name == "miprov2":
        optimizer = dspy.MIPROv2(
            metric=spec_metric_graduated,
            prompt_model=prompt_model,
            task_model=task_model,
            auto="light",
            init_temperature=0.7,
            num_threads=DEFAULT_NUM_THREADS,
            log_dir=os.path.join(log_dir, "miprov2"),
        )
        compile_kwargs = {"max_bootstrapped_demos": 0, "max_labeled_demos": 0}
        return optimizer, compile_kwargs

    elif name == "gepa":
        optimizer = dspy.GEPA(
            metric=spec_metric_with_feedback,
            # max_metric_calls=10,  # for debugging; increase for final runs
            auto="medium",
            reflection_lm=prompt_model,
            num_threads=DEFAULT_NUM_THREADS,
            log_dir=os.path.join(log_dir, "gepa"),
            track_stats=True,
        )
        compile_kwargs = {}
        return optimizer, compile_kwargs

    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ============================================================================
# 7. SINGLE OPTIMIZATION RUN
# ============================================================================


def build_student(seed_instruction: str) -> SpecSynthesizer:
    """Create a fresh student module with seed instruction and fixed demos."""
    student = SpecSynthesizer()
    student.generate.signature = student.generate.signature.with_instructions(
        seed_instruction
    )
    student.generate.demos = list(DEMO_EXAMPLES)
    return student


def run_optimization(
    optimizer_name: str,
    seed_instruction: str,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    prompt_model: dspy.LM,
    task_model: dspy.LM,
    log_dir: str,
) -> dict:
    """Run one optimizer. Returns metadata + optimized module."""

    logger.info(f"  Running {optimizer_name.upper()}...")

    student = build_student(seed_instruction)
    optimizer, compile_kwargs = create_optimizer(
        optimizer_name, prompt_model, task_model, log_dir
    )

    try:
        # GEPA and MIPROv2 accept valset; COPRO does not
        if optimizer_name in ("gepa", "miprov2"):
            optimized = optimizer.compile(
                student, trainset=trainset, valset=valset, **compile_kwargs
            )
        else:
            optimized = optimizer.compile(student, trainset=trainset, **compile_kwargs)

        optimized_instruction = optimized.generate.signature.instructions
        logger.info(f"  {optimizer_name.upper()} done.")
        logger.info(f"  Instruction: {optimized_instruction[:150]}...")

        return {
            "optimizer": optimizer_name,
            "optimized_instruction": optimized_instruction,
            "optimized_module": optimized,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"  {optimizer_name.upper()} failed: {e}", exc_info=True)
        return {
            "optimizer": optimizer_name,
            "optimized_instruction": seed_instruction,
            "optimized_module": build_student(seed_instruction),
            "status": f"failed: {str(e)}",
        }


# ============================================================================
# 8. EVALUATION
# ============================================================================


def evaluate_module(
    module: SpecSynthesizer,
    testset: list[dspy.Example],
    label: str,
    num_threads: int = DEFAULT_NUM_THREADS,
) -> dict:
    """Evaluate a module on a test set using binary pass/fail."""
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=spec_metric_binary,
        num_threads=num_threads,
        display_progress=True,
        provide_traceback=True,
    )
    result = evaluator(module)
    score = getattr(result, "score", None)
    if score is None:
        score = getattr(result, "avg", None)
    if score is None:
        score = getattr(result, "average", None)
    if score is None and isinstance(result, (int, float)):
        score = float(result)
    if score is None:
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            score = metrics.get("score") or metrics.get("avg") or metrics.get("average")
    if score is None:
        raise TypeError(f"Unexpected evaluation result type: {type(result).__name__}")

    logger.info(f"  [{label:<45}] {score:.2%}")
    return {"success_rate": score}


# ============================================================================
# 9. MAIN EXPERIMENT
# ============================================================================


def run_experiment(args):
    global OPENJML_OUTPUT_DIR
    # --- Load API keys ---
    load_dotenv(dotenv_path=args.env_path)

    OPENJML_OUTPUT_DIR = args.openjml_output_dir

    # --- Configure DSPy ---
    task_lm = dspy.LM(model=args.model, temperature=0.7, max_tokens=8192)
    prompt_lm = dspy.LM(model=args.reflection_model, temperature=0.7, max_tokens=8192)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    dspy.configure(lm=task_lm)

    # --- Validate seed ---
    if args.best_seed not in SEED_PROMPTS:
        raise ValueError(
            f"Unknown seed '{args.best_seed}'. Options: {list(SEED_PROMPTS.keys())}"
        )
    seed_instruction = SEED_PROMPTS[args.best_seed]
    logger.info(f"Best seed: {args.best_seed}")

    # --- Load benchmarks ---
    fb_tasks = load_tasks(args.formalbench_path)
    sgb_tasks_raw = load_tasks(args.specgenbench_path)

    # Exclude demo tasks from SpecGenBench to prevent leakage
    sgb_tasks = [t for t in sgb_tasks_raw if t.task_id not in DEMO_TASK_IDS]
    n_excluded = len(sgb_tasks_raw) - len(sgb_tasks)
    assert n_excluded == 2, (
        f"Expected to exclude 2 demo tasks, but excluded {n_excluded}. "
        f"Check that DEMO_TASK_IDS match your SpecGenBench task_ids."
    )
    logger.info(f"FormalBench: {len(fb_tasks)} tasks")
    logger.info(
        f"SpecGenBench: {len(sgb_tasks)} tasks ({n_excluded} demo tasks excluded)"
    )

    # --- Stratified split of FormalBench ---
    fb_train, fb_val, fb_test = stratified_split(
        fb_tasks,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    trainset = tasks_to_dspy_examples(fb_train)
    valset = tasks_to_dspy_examples(fb_val)
    fb_testset = tasks_to_dspy_examples(fb_test)
    sgb_testset = tasks_to_dspy_examples(sgb_tasks)

    logger.info(f"  FormalBench train: {len(trainset)}")
    logger.info(f"  FormalBench val:   {len(valset)}")
    logger.info(f"  FormalBench test:  {len(fb_testset)}")
    logger.info(f"  SpecGenBench test: {len(sgb_testset)}")

    # --- Log category distribution ---
    for split_name, split in [("train", fb_train), ("val", fb_val), ("test", fb_test)]:
        cats = defaultdict(int)
        for t in split:
            cats[t.category] += 1
        logger.info(f"  {split_name} categories: {dict(cats)}")

    # --- Run 3 optimizers from best seed ---
    optimizer_names = [o.strip() for o in args.optimizers.split(",")]
    for o in optimizer_names:
        if o not in ("copro", "miprov2", "gepa"):
            raise ValueError(f"Unknown optimizer '{o}'. Use copro/miprov2/gepa.")
    results = {}

    print(f"\n{'='*65}")
    print(f" Optimizer Comparison (seed={args.best_seed})")
    print(f"{'='*65}\n")

    for opt_name in optimizer_names:
        results[opt_name] = run_optimization(
            optimizer_name=opt_name,
            seed_instruction=seed_instruction,
            trainset=trainset,
            valset=valset,
            prompt_model=prompt_lm,
            task_model=task_lm,
            log_dir=os.path.join(args.log_dir, opt_name),
        )

    # --- Evaluate on held-out test sets ---
    print(f"\n{'='*65}")
    print(f"  Final Evaluation (binary pass/fail)")
    print(f"{'='*65}\n")

    eval_results = {}

    # 1. Baseline (original prompt, no optimization)
    baseline_module = build_student(seed_instruction)

    print(f"  --- FormalBench Test ({len(fb_testset)} tasks) ---")
    eval_results["baseline_fb"] = evaluate_module(
        baseline_module, fb_testset, f"Baseline ({args.best_seed})"
    )
    for opt_name, r in results.items():
        eval_results[f"{opt_name}_fb"] = evaluate_module(
            r["optimized_module"], fb_testset, f"{opt_name.upper()} optimized"
        )

    print(f"\n  --- SpecGenBench ({len(sgb_testset)} tasks) ---")
    eval_results["baseline_sgb"] = evaluate_module(
        baseline_module, sgb_testset, f"Baseline ({args.best_seed})"
    )
    for opt_name, r in results.items():
        eval_results[f"{opt_name}_sgb"] = evaluate_module(
            r["optimized_module"], sgb_testset, f"{opt_name.upper()} optimized"
        )

    # --- Save results ---
    output = {
        "seed": args.best_seed,
        "seed_instruction": seed_instruction,
        "optimizers": {
            opt_name: {
                "optimized_instruction": r["optimized_instruction"],
                "status": r["status"],
                "formalbench_test": eval_results[f"{opt_name}_fb"],
                "specgenbench_test": eval_results[f"{opt_name}_sgb"],
            }
            for opt_name, r in results.items()
        },
        "baseline": {
            "formalbench_test": eval_results["baseline_fb"],
            "specgenbench_test": eval_results["baseline_sgb"],
        },
        "config": {
            "model": args.model,
            "reflection_model": args.reflection_model,
            "train_size": len(trainset),
            "val_size": len(valset),
            "formalbench_test_size": len(fb_testset),
            "specgenbench_test_size": len(sgb_testset),
            "random_seed": args.seed,
        },
    }

    output_path = Path(args.output_dir) / "optimizer_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary table ---
    print(f"\n{'='*65}")
    print(f" Results Summary (seed={args.best_seed})")
    print(f"{'='*65}")
    print(f"\n  {'Approach':<25} {'FormalBench':>15} {'SpecGenBench':>15}")
    print(f"  {'-'*55}")
    print(
        f"  {'Baseline':<25} "
        f"{eval_results['baseline_fb']['success_rate']:>14.2%} "
        f"{eval_results['baseline_sgb']['success_rate']:>14.2%}"
    )
    for opt_name in optimizer_names:
        print(
            f"  {'+ ' + opt_name.upper():<25} "
            f"{eval_results[f'{opt_name}_fb']['success_rate']:>14.2%} "
            f"{eval_results[f'{opt_name}_sgb']['success_rate']:>14.2%}"
        )

    print(f"\n  Results saved to: {output_path}\n")

    # --- Save optimized modules for reuse ---
    for opt_name, r in results.items():
        if r["status"] == "success":
            module_path = Path(args.output_dir) / f"optimized_{opt_name}.json"
            r["optimized_module"].save(str(module_path))
            logger.info(f"  Saved {opt_name} module to {module_path}")


# ============================================================================
# 10. CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimizer Comparison for Formal Spec Synthesis"
    )
    parser.add_argument(
        "--formalbench_path",
        type=str,
        required=True,
        help="Path to FormalBench tasks JSON",
    )
    parser.add_argument(
        "--specgenbench_path",
        type=str,
        required=True,
        help="Path to SpecGenBench tasks JSON (fully held out)",
    )
    parser.add_argument(
        "--best_seed",
        type=str,
        default="formalbench_ltm",
        help="Seed prompt name from (zero/cot/ltm)",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="copro,miprov2,gepa",
        help="Comma-separated optimizers to run (copro,miprov2,gepa)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Target LLM for spec generation",
    )
    parser.add_argument(
        "--reflection_model",
        type=str,
        default="openai/gpt-4o",
        help="Strong LLM for GEPA reflection / MIPROv2 proposals",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=100,
        help="Number of FormalBench tasks for optimization training",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=50,
        help="Number of FormalBench tasks for optimization validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--env_path",
        type=str,
        default="config/.env",
        help="Path to .env file with API keys",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./optimizer_logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./optimizer_results",
    )
    parser.add_argument(
        "--openjml_output_dir",
        type=str,
        default=None,
        help="Directory for OpenJML temp files (defaults to per-run temp dir)",
    )
    args = parser.parse_args()
    run_experiment(args)
