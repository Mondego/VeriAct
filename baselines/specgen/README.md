# SpecGen
[SpecGen: Automated Generation of Formal Program Specifications via Large Language Models
](https://dl.acm.org/doi/10.1109/ICSE55347.2025.00129)



## changes 

- Refactored monolithic script into modular classes (SpecGen, SpecGenRunner) with typed dataclasses and TypedDicts
- Added multi-threaded task execution via ThreadPoolExecutor with per-thread artifact isolation
- Replaced print() statements with structured logging.Logger with per-task log files
- Added configurable timeout handling for OpenJML verification using subprocess.Popen (original used blocking os.popen with no timeout)
- Added explicit timeout detection in both the conversation and mutation loops
- Replaced empty-string verification check (err_info == "") with process return code check (returncode == 0)
- Added a tight upper bound on mutation loop iterations to guarantee termination
- Added process group management for reliable cleanup of verifier subprocesses
- Added token usage tracking (input/output tokens per task)
- Added structured result collection with summary statistics and per-case breakdowns, exported as JSONL and JSON
- Parameterized few-shot prompting with configurable prompt type (zero_shot, two_shot, four_shot)
- Preserved the core algorithm (conversation-driven generation, three-way refinement branching, mutation operators, and candidate selection) unchanged from the original artifact