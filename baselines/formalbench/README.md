# FormalBench

[Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference](https://aclanthology.org/2025.acl-long.1068.pdf)


## Changes adopted

- Replaced LangGraph with direct loop — Eliminated the StateGraph/MemorySaver framework dependency; the generation and repair pipeline runs as a plain Python for loop.

- Unified iteration budget — Generation and fixing share a single max_iters budget instead of operating as two independent workflows.

- Functional fix loop — The original graph routes verification failures to END (effectively one fix attempt). This implementation loops fix attempts up to max_iters.

- Resilient retry on bad output — Empty or invalid specs trigger a retry on the next iteration instead of immediate failure. In the fix phase, rejected attempts are preserved in conversation history for context. A --strict flag restores the original single-shot behavior.

- Graceful failure classification — Unknown failure types in classify_failures are caught and logged instead of raising unhandled exceptions.

- Direct OpenJML invocation — Replaced the Docker-based verifier with a direct subprocess call using explicit flags (--esc-max-warnings 1, --prover=cvc4, --nonnull-by-default, --arithmetic-failure=quiet, -nowarn).

- Token usage tracking — Per-task input/output token accounting via reset_token_usage() / get_token_usage().

- Simplified response parsing — Removed local-model-specific parsing paths (e.g., <|im_start|>, DeepSeek-R1 think tags, CodeLlama [/INST]) that are irrelevant for API-based inference.

- Threaded parallel execution — Added FBSpecRunner with ThreadPoolExecutor for concurrent task processing, per-thread logging, and isolated artifact directories.