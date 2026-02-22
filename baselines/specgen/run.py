import os
import sys
import argparse

from baselines.specgen.prompts import VALID_PROMPT_TYPES
from baselines.specgen.specgen_runner import SpecGenRunner


def _retrive_input_arguments():
    parser = argparse.ArgumentParser(description="Multi-threaded SpecGen processor")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Dataset file paths (e.g, metadata.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save results (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--openjml_timeout",
        type=int,
        default=300,
        help="Timeout for OpenJML validation in seconds (default: 300)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="Number of worker threads (default: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for SpecGen (e.g., gpt, gemini, claude, vllm)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature for the model (e.g., 0.7)",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="zero_shot",
        choices=VALID_PROMPT_TYPES,
        help=f"Prompt strategy to use. One of: {VALID_PROMPT_TYPES} (default: zero_shot)",
    )
    
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=10,
        help="Maximum iterations per input (default: 10)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def _validate_arguments(args):
    if not args.name:
        print("Error: Experiment name is required.")
        sys.exit(1)
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    if args.threads < 1:
        print("Error: Number of threads must be at least 1.")
        sys.exit(1)
    if args.openjml_timeout < 0:
        print("Error: OpenJML timeout must be a non-negative integer.")
        sys.exit(1)
    if args.temperature < 0 or args.temperature > 2:
        print("Error: Temperature must be between 0 and 2.")
        sys.exit(1)
    if args.max_iterations < 1:
        print("Error: Maximum iterations must be at least 1.")
        sys.exit(1)
    valid_models = ["gpt", "gemini", "claude", "vllm"]
    model_prefixes = [model for model in valid_models if args.model.startswith(model)]
    if not model_prefixes:
        print(f"Error: Model must start with one of {valid_models}.")
        sys.exit(1)


def main():
    _args = _retrive_input_arguments()
    _validate_arguments(_args)

    _specgen_runner = SpecGenRunner(
        name=_args.name,
        input=_args.input,
        output=_args.output,
        model=_args.model,
        temperature=_args.temperature,
        max_iterations=_args.max_iterations,
        prompt_type=_args.prompt_type,
        openjml_timeout=_args.openjml_timeout,
        threads=_args.threads,
        verbose=_args.verbose,
    )
    _specgen_runner.run_workers()


if __name__ == "__main__":
    main()
