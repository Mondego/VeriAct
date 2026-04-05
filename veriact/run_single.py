"""
run_single.py — Run VeriAct sequentially on all tasks in a benchmark dataset.

Usage:
    python run_single.py \
        --benchmark benchmarks/specgenbench/sgb.json \
        --model gpt-4o \
        --output-dir veriact_outputs \
        --openjml-path openjml \
        --max-steps 15 \
        --planning_interval 3
"""

import argparse
import json
import logging
import os

from veriact import (
    AnthropicModel,
    GeminiModel,
    OpenAIServerModel,
    VeriActAgent,
    VLLMModel,
)
from veriact.data_types import Task
from veriact.file_utility import dump_json, load_json, load_jsonl
import veriact.config

logger = logging.getLogger(__name__)

MODEL_PREFIX_TO_PROVIDER = {
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude": "anthropic",
    "gemini": "gemini",
}


def detect_provider(model_id: str) -> str:
    model_lower = model_id.lower()
    for prefix, provider in MODEL_PREFIX_TO_PROVIDER.items():
        if model_lower.startswith(prefix):
            return provider
    raise ValueError(
        f"Cannot detect provider from model name '{model_id}'. "
        f"Known prefixes: {list(MODEL_PREFIX_TO_PROVIDER.keys())}. "
        f"For vLLM models, set VLLM_API_BASE env var."
    )


def create_model(model_id: str):
    vllm_base = os.getenv("VLLM_API_BASE")
    if vllm_base:
        return VLLMModel(model_id=model_id, api_base=vllm_base)

    provider = detect_provider(model_id)
    if provider == "openai":
        return OpenAIServerModel(model_id=model_id)
    elif provider == "anthropic":
        return AnthropicModel(model_id=model_id)
    elif provider == "gemini":
        return GeminiModel(model_id=model_id)


def load_benchmark(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return load_jsonl(path)
    return load_json(path)


def main():
    parser = argparse.ArgumentParser(description="Run VeriAct on a single task.")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark dataset (.json or .jsonl)")
    parser.add_argument("--model", default="gpt-4o", help="Model ID (default: gpt-4o)")
    parser.add_argument("--output-dir", default="veriact_outputs", help="Output directory (default: veriact_outputs)")
    parser.add_argument("--openjml-path", default="openjml", help="Path to OpenJML binary (default: openjml)")
    parser.add_argument("--max-steps", type=int, default=15, help="Max agent steps per task (default: 15)")
    parser.add_argument("--planning_interval", type=int, default=5, help="Planning interval (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    raw_tasks = load_benchmark(args.benchmark)
    tasks = [Task.from_dict(t) for t in raw_tasks]
    logger.info(f"Loaded {len(tasks)} tasks from {args.benchmark}")

    os.makedirs(args.output_dir, exist_ok=True)
    trajectory_jsonl_path = os.path.join(args.output_dir, "trajectories.jsonl")

    completed = []
    failed = []

    for task in tasks:
        print(f"\n{'='*55}")
        print(f"  Task: {task.task_id}  ({task.class_name})")
        print(f"{'='*55}\n")

        try:
            model = create_model(args.model)
            agent = VeriActAgent(
                model=model,
                openjml_path=args.openjml_path,
                dataset_path=args.benchmark,
                output_dir=args.output_dir,
                max_steps=args.max_steps,
                planning_interval=args.planning_interval,
            )
            result = agent.run(task)
            completed.append(task.task_id)
            with open(trajectory_jsonl_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"  Success:    {result['success']}")
            print(f"  Iterations: {result['iterations']}")
            if result.get("best_spec"):
                print(f"  requires  {result['best_spec'].get('precondition')}")
                print(f"  ensures   {result['best_spec'].get('postcondition')}")
        except Exception:
            import traceback
            tb = traceback.format_exc()
            failed.append({"task_id": task.task_id, "error": tb})
            logger.error(f"[FAIL] {task.task_id}\n{tb}")

    if failed:
        dump_json(failed, os.path.join(args.output_dir, "failed_tasks.json"))
        logger.warning(f"{len(failed)} task(s) failed. See {args.output_dir}/failed_tasks.json")

    logger.info(f"Complete: {len(completed)} succeeded, {len(failed)} failed out of {len(tasks)} total")


if __name__ == "__main__":
    main()
