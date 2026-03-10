# AutoSpec
[Enchanting program specification synthesis by large language models using static analysis and program verification](https://dl.acm.org/doi/10.1007/978-3-031-65630-9_16)


# Changes and Improvements

- Ported from C/Frama-C to Java/OpenJML, adapting the verification pipeline to JML specification language and OpenJML tooling
Fixed verification success condition: replaced fragile empty-output check (err_info.strip() == "") with process exit code (returncode == 0) for reliable verification detection
- Fixed timeout misclassification: the original silently treated a timed-out verification as successful; timeouts are now explicitly detected and reported as a distinct timed_out outcome, with the code reset to the clean original
- Fixed thread-unsafe global verifier call counter with a local, per-run counter, enabling correct multi-task parallel execution
Fixed AttributeError crash when processing field declarations: replaced deferred dict access with an early return of the hardcoded //@ spec_public annotation
- Fixed silent pass-through of invalid specs in filter_validated_specs: when the verifier reports failure but no error lines can be parsed, all candidate specs are dropped rather than passed through unfiltered
- Replaced Linux-only SIGALRM-based timeout mechanism with a portable, thread-safe timeout handled inside the verifier utility
- Implemented the paper's optional specification simplification step after successful verification, redundant specs are removed one-by-one and re-verified to produce a minimal annotation set; exposed as an opt-in --simplify flag
- Added structured logging with severity levels (DEBUG, INFO, WARNING, ERROR) and per-task log files, replacing bare print statements
- Added parallel task execution via ThreadPoolExecutor with a configurable thread count
- Added LLM token usage tracking (input/output tokens) per task and in aggregate
- Added configurable prompt strategy (--prompt_type) to support zero-shot and few-shot prompting modes without code changes
- Added jml code fence stripping alongside java fences to handle varied LLM response formats
- Removed duplicate invariant keyword check in the spec line detector
- Added structured JSON summary output with per-task results, verification rates, verifier call statistics, and token usage