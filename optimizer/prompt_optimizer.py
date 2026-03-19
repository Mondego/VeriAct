"""
Prompt Optimization for Formal Specification Synthesis
============================================================

Compares three DSPy optimizers (COPRO, MIPROv2, GEPA) on improving
the instruction prompt for JML specification generation.

Uses ONLY the `dspy` package — no standalone `gepa` package.

Setup:
    pip install dspy-ai --break-system-packages

Usage:
    # Run all 3 optimizers from 3 representative seeds
    python rq3_prompt_optimization.py \
        --benchmark_path ./benchmark/tasks.json \
        --model openai/gpt-4o \
        --reflection_model openai/gpt-4o

    # Run a single optimizer from a single seed
    python rq3_prompt_optimization.py \
        --benchmark_path ./benchmark/tasks.json \
        --optimizers gepa \
        --seeds specgen_4shot
"""

import json
import os
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import dspy
from dotenv import load_dotenv

from specgent.run.optimizer_utils import (
    Task,
    compute_graduated_score,
    verify_with_openjml,
    clean_code_fences,
    format_error_feedback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prompt_optimizer")


# ============================================================================
# 1. SEED PROMPTS — 11 instruction-only texts from 3 approaches
# ============================================================================
# TODO: Replace with the EXACT instruction text from each approach.

SEED_PROMPTS: dict[str, str] = {
    # --- SpecGen (3) ---
    "specgen_0shot": (
        "You are a JML specification generator for Java programs. "
        "Given a Java program, generate complete JML specifications including "
        "preconditions (requires), postconditions (ensures), and loop invariants "
        "(maintaining). The specifications must be syntactically valid JML and "
        "semantically consistent with the program behavior."
    ),
    "specgen_2shot": (
        "You are a JML specification generator for Java programs. "
        "Generate JML specifications for the given Java program. Include "
        "preconditions (@requires), postconditions (@ensures), and loop "
        "invariants (@maintaining) where applicable. Ensure the generated "
        "specifications are verifiable by OpenJML."
    ),
    "specgen_4shot": (
        "You are a JML specification generator for Java programs. "
        "Generate comprehensive JML specifications for the given Java program. "
        "Include @requires clauses for preconditions, @ensures clauses for "
        "postconditions, and @maintaining clauses for loop invariants. "
        "Pay attention to boundary conditions, null checks, and array bounds. "
        "The specifications must pass OpenJML verification."
    ),
    # --- AutoSpec (3) ---
    "autospec_0shot": (
        "You are an expert in formal verification. Given a Java method, "
        "generate JML annotations that formally specify the method's behavior. "
        "Decompose the method into logical components and specify each component "
        "before combining them into the overall specification."
    ),
    "autospec_2shot": (
        "You are an expert in formal verification and JML specification. "
        "Analyze the given Java method by decomposing it into components. "
        "For each component, generate corresponding JML specifications. "
        "Combine all component specifications to produce the complete "
        "method-level specification with @requires, @ensures, and loop invariants."
    ),
    "autospec_4shot": (
        "You are an expert in formal verification and JML specification. "
        "For the given Java method: (1) identify all logical components, "
        "(2) generate JML specifications for each component including "
        "preconditions and postconditions, (3) synthesize loop invariants "
        "for any loops, (4) combine into a complete verifiable specification. "
        "Ensure consistency across all specification clauses."
    ),
    # --- FormalBench (5) ---
    "formalbench_0shot": (
        "Generate JML specifications for the following Java program. "
        "Include preconditions, postconditions, and loop invariants."
    ),
    "formalbench_fewshot": (
        "Generate JML (Java Modeling Language) specifications for the "
        "following Java program. JML specifications are written as special "
        "comments starting with //@. Use @requires for preconditions, "
        "@ensures for postconditions, @maintaining for loop invariants, "
        "and @decreases for termination metrics."
    ),
    "formalbench_0shot_cot": (
        "Generate JML specifications for the following Java program. "
        "Think step by step: first analyze the method signature and parameters, "
        "then identify preconditions that must hold before execution, "
        "then determine postconditions that hold after execution, "
        "and finally derive loop invariants for any loops."
    ),
    "formalbench_fewshot_cot": (
        "Generate JML specifications for the following Java program. "
        "Follow this reasoning process: (1) Analyze inputs and their valid ranges, "
        "(2) Trace the control flow and identify key program states, "
        "(3) Write @requires clauses for input constraints, "
        "(4) Write @ensures clauses capturing the output relationship, "
        "(5) For each loop, derive an @maintaining invariant and @decreases variant."
    ),
    "formalbench_fewshot_ltm": (
        "Generate JML specifications for the following Java program. "
        "Break the problem into subproblems: First, what are the valid inputs? "
        "Second, what does the method compute or return? Third, what properties "
        "are preserved in each loop iteration? Solve each subproblem, then "
        "combine the answers into complete JML annotations with @requires, "
        "@ensures, @maintaining, and @decreases clauses."
    ),
}


# ============================================================================
# 2. FIXED 2-SHOT DEMOS — neutral, independent of all approaches
# ============================================================================
# TODO: Replace with your actual selected examples.

DEMO_EXAMPLES = [
    dspy.Example(
        java_program=(
            "public class Max {\n"
            "    public static int max(int a, int b) {\n"
            "        if (a >= b) { return a; }\n"
            "        else { return b; }\n"
            "    }\n"
            "}"
        ),
        jml_specification=(
            "public class Max {\n"
            "    //@ ensures \\\\result >= a && \\\\result >= b;\n"
            "    //@ ensures \\\\result == a || \\\\result == b;\n"
            "    public static int max(int a, int b) {\n"
            "        if (a >= b) { return a; }\n"
            "        else { return b; }\n"
            "    }\n"
            "}"
        ),
    ).with_inputs("java_program"),
    dspy.Example(
        java_program=(
            "public class Sum {\n"
            "    public static int sum(int[] arr) {\n"
            "        int s = 0;\n"
            "        for (int i = 0; i < arr.length; i++) { s += arr[i]; }\n"
            "        return s;\n"
            "    }\n"
            "}"
        ),
        jml_specification=(
            "public class Sum {\n"
            "    //@ requires arr != null;\n"
            "    //@ ensures \\\\result == (\\\\sum int j; 0 <= j && j < arr.length; arr[j]);\n"
            "    public static int sum(int[] arr) {\n"
            "        int s = 0;\n"
            "        //@ maintaining 0 <= i && i <= arr.length;\n"
            "        //@ maintaining s == (\\\\sum int j; 0 <= j && j < i; arr[j]);\n"
            "        //@ decreases arr.length - i;\n"
            "        for (int i = 0; i < arr.length; i++) { s += arr[i]; }\n"
            "        return s;\n"
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
    Return ONLY the annotated Java code with JML comments, no explanation."""

    java_program: str = dspy.InputField(
        desc="Java source code to annotate with JML specifications"
    )
    jml_specification: str = dspy.OutputField(
        desc="The Java program annotated with JML @requires, @ensures, "
        "@maintaining, and @decreases clauses"
    )


class SpecSynthesizer(dspy.Module):
    """DSPy module for JML specification generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateJMLSpec)

    def forward(self, java_program: str) -> dspy.Prediction:
        return self.generate(java_program=java_program)


# ============================================================================
# 4. BENCHMARK LOADER
# ============================================================================


def load_tasks(benchmark_path: str) -> list[Task]:
    """Load tasks from a JSON file using the Task dataclass."""
    with open(benchmark_path, "r") as f:
        data = json.load(f)
    # Support both list of tasks and dict with "tasks" key
    if isinstance(data, dict):
        data = data.get("tasks", data.get("data", []))
    return [Task.from_dict(t) for t in data]


def tasks_to_dspy_examples(tasks: list[Task]) -> list[dspy.Example]:
    """Convert Task objects to DSPy Examples."""
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


def stratified_split_tasks(
    tasks: list[Task], seed: int
) -> tuple[list[Task], list[Task], list[Task]]:
    """
    Stratified split by category for the 662-task distribution.
    Returns (train_tasks, val_tasks, test_tasks).
    """
    random.seed(seed)
    by_category: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_category[task.category].append(task)

    # Per-category counts for 100/50/512 split (train/val/test).
    # Total: train=100, val=50, test=512
    counts = {
        "branch": (12, 6, 61),
        "multi_path_loop": (36, 18, 183),
        "nested": (15, 8, 80),
        "sequential": (16, 8, 82),
        "single_path_loop": (21, 10, 106),
    }

    train_tasks: list[Task] = []
    val_tasks: list[Task] = []
    test_tasks: list[Task] = []

    for category, items in by_category.items():
        if category not in counts:
            raise ValueError(f"Unknown category '{category}' for stratified split.")

        random.shuffle(items)
        train_n, val_n, test_n = counts[category]
        total_needed = train_n + val_n + test_n
        if len(items) < total_needed:
            raise ValueError(
                f"Category '{category}' has {len(items)} tasks,"
                f" but {total_needed} are required for the split."
            )

        train_tasks.extend(items[:train_n])
        val_tasks.extend(items[train_n : train_n + val_n])
        test_tasks.extend(items[train_n + val_n : train_n + val_n + test_n])

    random.shuffle(train_tasks)
    random.shuffle(val_tasks)
    random.shuffle(test_tasks)

    return train_tasks, val_tasks, test_tasks


# ============================================================================
# 5. METRICS
# ============================================================================


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
    )
    return 1.0 if result.success else 0.0


def spec_metric(example, prediction, trace=None) -> float:
    """
    Graduated metric for COPRO and MIPROv2.
    Returns partial credit based on error classification.
    """
    class_name = example.class_name
    generated = getattr(prediction, "jml_specification", "")
    generated = clean_code_fences(generated)

    if not generated.strip():
        return 0.0

    result = verify_with_openjml(
        code_with_spec=generated,
        classname=class_name,
    )
    return compute_graduated_score(result)


def spec_metric_with_feedback(
    example, prediction, trace=None, pred_name=None, pred_trace=None
):
    """
    Graduated metric + rich textual feedback for GEPA.
    GEPA uses both the score AND the feedback for reflective mutation.
    """
    class_name = example.class_name
    task_id = getattr(example, "task_id", class_name)
    generated = getattr(prediction, "jml_specification", "")
    generated = clean_code_fences(generated)

    if not generated.strip():
        feedback = (
            f"[FAIL] Task '{task_id}': LLM returned empty or unparseable output. "
            f"The instruction must clearly ask for annotated Java code with "
            f"JML comments in //@ format. Do not include natural language "
            f"explanations — only the Java code with JML annotations."
        )
        return dspy.Prediction(score=0.0, feedback=feedback)

    result = verify_with_openjml(
        code_with_spec=generated,
        classname=class_name,
    )

    score = compute_graduated_score(result)
    feedback = format_error_feedback(result, task_id)

    # Enrich feedback with score explanation so GEPA understands the gradient
    if score == 0.3:
        feedback += (
            f"\n--- Score: 0.3 (1 verification error) ---\n"
            f"The specification is almost correct. Focus on fixing the single "
            f"failing clause identified above."
        )
    elif score == 0.1:
        feedback += (
            f"\n--- Score: 0.1 ({len(result.classified_errors)} verification errors) ---\n"
            f"The JML syntax is valid but multiple clauses are semantically wrong. "
            f"Reconsider the overall specification strategy."
        )
    elif score == 0.0 and result.classified_errors:
        feedback += (
            f"\n--- Score: 0.0 (syntax errors) ---\n"
            f"The output contains JML syntax errors. Ensure correct use of "
            f"@requires, @ensures, @maintaining, @decreases with proper "
            f"JML operators (\\result, \\old, \\forall, \\exists)."
        )

    return dspy.Prediction(score=score, feedback=feedback)


# ============================================================================
# 6. OPTIMIZER FACTORY
# ============================================================================


def create_optimizer(
    name: str,
    task_model: dspy.LM,
    prompt_model: dspy.LM,
    log_dir: str = "./rq3_logs",
) -> tuple:
    """
    Create a DSPy optimizer by name.

    Returns: (optimizer, compile_kwargs)
    """
    if name == "copro":
        # COPRO: coordinate ascent hill-climbing on instructions only
        optimizer = dspy.COPRO(
            prompt_model=prompt_model,
            metric=spec_metric,
            depth=3,  # 3 rounds of iterative refinement
            breadth=5,  # 5 candidate instructions per round
            init_temperature=1.0,
        )
        # COPRO.compile requires eval_kwargs (dict passed to dspy.Evaluate)
        compile_kwargs = {
            "eval_kwargs": {
                "num_threads": 4,
                "display_progress": True,
            },
        }
        return optimizer, compile_kwargs

    elif name == "miprov2":
        # MIPROv2: Bayesian optimization over instructions + few-shot
        optimizer = dspy.MIPROv2(
            metric=spec_metric,
            prompt_model=prompt_model,
            task_model=task_model,
            auto="medium",
            init_temperature=1.0,
            log_dir=os.path.join(log_dir, "miprov2"),
        )
        compile_kwargs = {
            "max_bootstrapped_demos": 0,  # use our fixed demos only
            "max_labeled_demos": 2,
            "requires_permission_to_run": False,
        }
        return optimizer, compile_kwargs

    elif name == "gepa":
        # GEPA: evolutionary + LLM reflection on error traces
        optimizer = dspy.GEPA(
            metric=spec_metric_with_feedback,
            auto="medium",
            reflection_lm=prompt_model,
            log_dir=os.path.join(log_dir, "gepa"),
            track_stats=True,
        )
        compile_kwargs = {}
        return optimizer, compile_kwargs

    else:
        raise ValueError(f"Unknown optimizer: {name}. Use copro/miprov2/gepa.")


# ============================================================================
# 7. SINGLE OPTIMIZATION RUN
# ============================================================================


def run_single_optimization(
    optimizer_name: str,
    seed_name: str,
    seed_instruction: str,
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    task_model: dspy.LM,
    prompt_model: dspy.LM,
    log_dir: str,
) -> dict:
    """Run one optimizer from one seed. Returns the optimized module + metadata."""

    logger.info(f"{'='*60}")
    logger.info(f"  Optimizer: {optimizer_name} | Seed: {seed_name}")
    logger.info(f"{'='*60}")

    # Build fresh student module with the seed instruction
    student = SpecSynthesizer()
    student.generate.signature = student.generate.signature.with_instructions(
        seed_instruction
    )
    student.generate.demos = list(DEMO_EXAMPLES)  # fixed 2-shot demos

    # Create optimizer
    optimizer, compile_kwargs = create_optimizer(
        optimizer_name, task_model, prompt_model, log_dir
    )

    # Run optimization
    try:
        compile_args = {"trainset": trainset, **compile_kwargs}
        if optimizer_name == "gepa":
            compile_args["valset"] = valset

        optimized = optimizer.compile(student, **compile_args)

        # Extract the optimized instruction
        optimized_instruction = optimized.generate.signature.instructions
        logger.info(
            f"  Done. Optimized instruction:\n    {optimized_instruction[:200]}..."
        )

        return {
            "optimizer": optimizer_name,
            "seed_name": seed_name,
            "seed_instruction": seed_instruction,
            "optimized_instruction": optimized_instruction,
            "optimized_module": optimized,
            "status": "success",
        }

    except Exception as e:
        logger.error(f"  Optimization failed: {e}", exc_info=True)
        return {
            "optimizer": optimizer_name,
            "seed_name": seed_name,
            "seed_instruction": seed_instruction,
            "optimized_instruction": seed_instruction,  # fallback
            "optimized_module": student,
            "status": f"failed: {str(e)}",
        }


# ============================================================================
# 8. EVALUATION
# ============================================================================


def evaluate_instruction(
    instruction: str,
    testset: list[dspy.Example],
    label: str,
) -> dict:
    """Evaluate a single instruction on the test set."""
    module = SpecSynthesizer()
    module.generate.signature = module.generate.signature.with_instructions(instruction)
    module.generate.demos = list(DEMO_EXAMPLES)

    passed = 0
    total = len(testset)
    per_category: dict[str, dict] = {}

    for ex in testset:
        category = getattr(ex, "category", "unknown")
        if category not in per_category:
            per_category[category] = {"passed": 0, "total": 0}
        per_category[category]["total"] += 1

        try:
            pred = module(java_program=ex.java_program)
            score = spec_metric_binary(ex, pred)
            if score >= 1.0:
                passed += 1
                per_category[category]["passed"] += 1
        except Exception as e:
            logger.debug(f"  Eval error for {getattr(ex, 'task_id', '?')}: {e}")

    rate = passed / total if total > 0 else 0.0
    logger.info(f"  [{label:<45}] {passed:>4}/{total} = {rate:.2%}")

    return {
        "passed": passed,
        "total": total,
        "success_rate": rate,
        "per_category": {
            cat: {**v, "rate": v["passed"] / v["total"] if v["total"] > 0 else 0.0}
            for cat, v in per_category.items()
        },
    }


# ============================================================================
# 9. MAIN EXPERIMENT
# ============================================================================


def run_experiment(args):

    # --- Configure DSPy ---
    task_lm = dspy.LM(
        model=args.model,
        temperature=0.7,
        max_tokens=2048,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    prompt_lm = dspy.LM(
        model=args.reflection_model,
        temperature=1.0,
        max_tokens=4096,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    dspy.configure(lm=task_lm)

    # --- Load benchmark ---
    tasks = load_tasks(args.benchmark_path)
    logger.info(f"Loaded {len(tasks)} tasks from {args.benchmark_path}")

    # --- Train / test split ---
    if len(tasks) == 662:
        train_tasks, val_tasks, test_tasks = stratified_split_tasks(
            tasks, seed=args.seed
        )
    else:
        random.seed(args.seed)
        random.shuffle(tasks)

        split_idx = int(len(tasks) * 0.7)
        train_tasks = tasks[:split_idx]
        test_tasks = tasks[split_idx:]
        val_tasks = []

    trainset = tasks_to_dspy_examples(train_tasks)
    valset = tasks_to_dspy_examples(val_tasks)
    testset = tasks_to_dspy_examples(test_tasks)

    logger.info(f"  Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")

    # --- Determine which seeds to run ---
    seed_names = [s.strip() for s in args.seeds.split(",")]
    for s in seed_names:
        if s not in SEED_PROMPTS:
            raise ValueError(
                f"Unknown seed '{s}'. Options: {list(SEED_PROMPTS.keys())}"
            )

    optimizer_names = [o.strip() for o in args.optimizers.split(",")]
    for o in optimizer_names:
        if o not in ("copro", "miprov2", "gepa"):
            raise ValueError(f"Unknown optimizer '{o}'. Use copro/miprov2/gepa.")

    # --- Run optimizations ---
    all_results = []
    for opt_name in optimizer_names:
        for seed_name in seed_names:
            result = run_single_optimization(
                optimizer_name=opt_name,
                seed_name=seed_name,
                seed_instruction=SEED_PROMPTS[seed_name],
                trainset=trainset,
                valset=valset,
                task_model=task_lm,
                prompt_model=prompt_lm,
                log_dir=str(Path(args.log_dir) / opt_name / seed_name),
            )
            all_results.append(result)

    # --- Evaluate on held-out test set ---
    logger.info(f"\n{'='*60}")
    logger.info(f"  Test Set Evaluation ({len(testset)} programs)")
    logger.info(f"{'='*60}")

    # Evaluate optimized prompts
    optimized_eval = {}
    for r in all_results:
        label = f">> {r['optimizer'].upper()} (seed={r['seed_name']})"
        key = f"{r['optimizer']}_{r['seed_name']}"
        optimized_eval[key] = evaluate_instruction(
            r["optimized_instruction"], testset, label
        )
        optimized_eval[key]["optimized_instruction"] = r["optimized_instruction"]
        optimized_eval[key]["status"] = r["status"]

    logger.info(f"  {'-'*60}")

    # Evaluate all 11 original baselines
    baseline_eval = {}
    for name, instruction in SEED_PROMPTS.items():
        baseline_eval[name] = evaluate_instruction(instruction, testset, name)

    # --- Save results ---
    output = {
        "optimized": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_category"}
            for k, v in optimized_eval.items()
        },
        "optimized_per_category": {
            k: v.get("per_category", {}) for k, v in optimized_eval.items()
        },
        "baselines": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_category"}
            for k, v in baseline_eval.items()
        },
        "baselines_per_category": {
            k: v.get("per_category", {}) for k, v in baseline_eval.items()
        },
        "config": {
            "model": args.model,
            "reflection_model": args.reflection_model,
            "optimizers": optimizer_names,
            "seeds": seed_names,
            "random_seed": args.seed,
            "train_size": len(trainset),
            "val_size": len(valset),
            "test_size": len(testset),
        },
    }

    output_path = Path(args.output_dir) / "prompt_optimizer_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # --- Print summary table ---
    print(f"\n{'='*70}")
    print(f"  Optimization Summary")
    print(f"{'='*70}")
    print(f"\n  {'Approach':<45} {'Passed':>7} {'Total':>7} {'Rate':>9}")
    print(f"  {'-'*68}")

    # Optimized results (sorted by rate)
    sorted_opt = sorted(
        optimized_eval.items(), key=lambda x: x[1]["success_rate"], reverse=True
    )
    for key, r in sorted_opt:
        print(
            f"  >> {key:<41} {r['passed']:>7} {r['total']:>7} {r['success_rate']:>9.2%}"
        )

    print(f"  {'-'*68}")

    # Baselines (sorted by rate)
    sorted_base = sorted(
        baseline_eval.items(), key=lambda x: x[1]["success_rate"], reverse=True
    )
    for name, r in sorted_base:
        print(
            f"  {name:<45} {r['passed']:>7} {r['total']:>7} {r['success_rate']:>9.2%}"
        )

    print(f"\n  Results saved to: {output_path}\n")


# ============================================================================
# 10. CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt Optimization Comparison (COPRO vs MIPROv2 vs GEPA)"
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        required=True,
        help="Path to benchmark JSON file (list of Task dicts)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Target LLM for spec generation (DSPy model string)",
    )
    parser.add_argument(
        "--reflection_model",
        type=str,
        default="openai/gpt-4o",
        help="Strong LLM for GEPA reflection / MIPROv2 proposals",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="copro,miprov2,gepa",
        help="Comma-separated optimizer names to run",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="specgen_4shot,autospec_2shot,formalbench_0shot",
        help="Comma-separated seed prompt names (best,mid,worst from RQ1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./prompt_optimizer_logs",
        help="Directory for optimizer logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prompt_optimizer_results",
        help="Directory for final results JSON",
    )
    args = parser.parse_args()
    run_experiment(args)
