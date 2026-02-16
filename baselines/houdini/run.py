import os
import sys
import argparse

from houdini.houdini import HoudiniRunner


def _retrive_input_arguments():
    parser = argparse.ArgumentParser(description="Multi-threaded Houdini processor")
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
        help="Directory containing input Java files to process",
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
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def _validate_arguments(args):
    if not os.path.isdir(args.input):
        print(f"Error: Input path '{args.input}' is not a valid directory.")
        sys.exit(1)
    if args.threads < 1:
        print("Error: Number of threads must be at least 1.")
        sys.exit(1)
    if args.openjml_timeout < 0:
        print("Error: OpenJML timeout must be a non-negative integer.")
        sys.exit(1)


def main():
    args = _retrive_input_arguments()
    _validate_arguments(args)
    _hudini_worker = HoudiniRunner(
        name=args.name,
        input_dir=args.input,
        output_dir=args.output,
        openjml_timeout=args.openjml_timeout,
        threads=args.threads,
        verbose=args.verbose,
    )
    _hudini_worker.run_workers()


if __name__ == "__main__":
    main()
