import re
import logging
from typing import Any, TypedDict

from baselines.utils.verifier import verify_with_openjml
from baselines.utils.models import create_model_config, request_llm_engine

from baselines.formalbench.example import _GUIDANCE
from baselines.formalbench.failure_analysis import extract_errors, classify_failures


FIX_SYS_MESSAGE = (
    "You are an expert on Java Modeling Language (JML). "
    "Your task is to fix the JML specifications annotated in the target Java code. "
    "You will be provided the error messages from the OpenJML tool and you need to "
    "fix the specifications accordingly."
)

_FIX_USER_TEMPLATE = (
    "The following Java code is annotated with JML specifications:\n"
    "```\n"
    "{curr_spec}\n"
    "```\n"
    "OpenJML Verification tool failed to verify the specifications given above, "
    "with error information as follows:\n\n"
    "### ERROR MESSAGE:\n"
    "```\n"
    "{curr_error}\n"
    "```\n\n"
    "### ERROR TYPES:\n"
    "{error_info}\n"
    "Please refine the specifications so that they can pass verification. "
    "Provide the specifications for the code and include the solution written "
    "between triple backticks, after `### FIXED SPECIFICATION`.\n"
)

_ERROR_INFO_TEMPLATE = (
    "- Error Type: {error_type}\n" "{error_description}\n" "{fix_instructions}\n\n"
)


class SpecFixResult(TypedDict):
    status: str
    class_name: str
    verifier_calls: int
    iterations: int
    final_code: str
    final_error: str
    verified: bool


class SpecFixer:

    def __init__(
        self,
        model: str,
        temperature: float,
        max_iters: int,
        output_dir: str,
        timeout: int,
        logger: logging.Logger,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_iters = max_iters
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _contains_annotations(self, java_code: str) -> bool:
        lines = java_code.splitlines()
        inside_block_comment = False
        single_line_jml = re.compile(r"^\s*//@.*")
        block_comment_start = re.compile(r"^\s*/\*@")
        block_comment_end = re.compile(r"\s*\*/")

        for line in lines:
            if single_line_jml.match(line):
                return True
            if inside_block_comment:
                if block_comment_end.match(line):
                    inside_block_comment = False
            elif block_comment_start.match(line):
                inside_block_comment = True
                return True

        return False

    def _parse_spec_from_response(self, response: str) -> str:
        if "### SPECIFICATION" in response:
            response = response.split("### SPECIFICATION")[-1]
        if "### FIXED SPECIFICATION" in response:
            response = response.split("### FIXED SPECIFICATION")[-1]
        if "### RESPONSE" in response:
            response = response.split("### RESPONSE")[-1]

        if "```" not in response:
            return response.strip()

        if "```java" in response:
            pattern = r"```java(.*?)```"
        else:
            pattern = r"```(.*?)```"

        code_blocks = re.findall(pattern, response, re.DOTALL)
        return "\n// block\n".join(code_blocks)

    def _analyze_failures(self, err_info: str) -> str:
        """Classify OpenJML errors and return a formatted guidance string."""
        error_set: set[str] = set()

        if "NOT IMPLEMENTED:" in err_info:
            if r"\sum" in err_info or r"\num_of" in err_info or r"\product" in err_info:
                error_set.add("UnsupportedSumNumOfProductQuantifierExpression")
            if r"\min" in err_info or r"\max" in err_info:
                error_set.add("UnsupportedMinMaxQuantifierExpression")
        else:
            errors = extract_errors(err_info)
            for level, error in errors:
                try:
                    failure_type = classify_failures(level, error)
                    if failure_type is not None:
                        error_set.add(failure_type)
                except ValueError:
                    self.logger.debug(f"Unknown failure type for error: {error[:100]}")

        error_info: str = ""
        for error in error_set:
            if error in _GUIDANCE:
                error_info += _ERROR_INFO_TEMPLATE.format(
                    error_type=error,
                    error_description=_GUIDANCE[error]["description"],
                    fix_instructions=_GUIDANCE[error]["guidance"],
                )
        return error_info

    def _build_fix_messages(
        self,
        curr_spec: str,
        err_info: str,
        error_info: str,
    ) -> list[dict[str, Any]]:
        """
        Build the message list for a fix request.

        Following the original FormalBench implementation, each fix attempt
        is stateless: the LLM sees only the system message and the current
        error — no prior conversation history is included in the prompt.
        History is tracked externally for logging/counting purposes only.
        """
        system_role: str = "user" if self.model == "o1-mini" else "system"

        new_user_msg: dict[str, Any] = {
            "role": "user",
            "content": _FIX_USER_TEMPLATE.format(
                curr_spec=curr_spec,
                curr_error=err_info,
                error_info=error_info,
            ),
        }

        return [
            {"role": system_role, "content": FIX_SYS_MESSAGE},
            new_user_msg,
        ]

    # ------------------------------------------------------------------
    # Main repair loop
    # ------------------------------------------------------------------

    def repair(
        self, input_spec: str, input_err_info: str, class_name: str
    ) -> SpecFixResult:
        """
        Iteratively ask the LLM to fix a failing spec.

        Parameters
        ----------
        input_spec      : the annotated Java code that failed verification
        input_err_info  : the raw OpenJML error string from the first run
        class_name      : Java class name (used for tmp file naming and logs)

        Returns
        -------
        dict with keys: status, class_name, verifier_calls, iterations,
                        final_code, final_error, verified
        """
        self.logger.info(
            f"[{class_name}] Starting SpecFixer (max_iters={self.max_iters})..."
        )

        curr_spec: str = input_spec
        err_info: str = input_err_info
        verifier_calls: int = 0
        status: str = "unverified"
        history: list[dict[str, Any]] = (
            []
        )  # accumulated conversation turns (no system msg)
        num_iter: int = 0

        for num_iter in range(1, self.max_iters + 1):
            self.logger.info(
                f"[{class_name}] Fix iteration {num_iter}/{self.max_iters}"
            )

            error_info: str = self._analyze_failures(err_info)
            if self.verbose:
                self.logger.debug(
                    f"[{class_name}] Analyzed error types:\n{error_info[:300]}"
                )

            messages: list[dict[str, Any]] = self._build_fix_messages(
                curr_spec, err_info, error_info
            )
            config: dict[str, Any] = create_model_config(
                messages, self.model, self.temperature
            )
            ret = request_llm_engine(config)

            if ret is None:
                self.logger.error(
                    f"[{class_name}] LLM returned no response on iter {num_iter}"
                )
                status = "unknown"
                break

            raw_response: str = ret.choices[0].message.content
            if self.verbose:
                self.logger.debug(f"[{class_name}] LLM response: {raw_response[:500]}")

            # Append this turn (user + assistant) to history for next iteration
            history.append(messages[-1])  # the user turn we just sent
            history.append({"role": "assistant", "content": raw_response})

            new_spec: str = self._parse_spec_from_response(raw_response)

            if not new_spec:
                self.logger.warning(
                    f"[{class_name}] Empty spec returned on iter {num_iter}"
                )
                status = "empty_spec"
                break

            if not self._contains_annotations(new_spec):
                self.logger.warning(
                    f"[{class_name}] No JML annotations detected on iter {num_iter}"
                )
                status = "invalid_jml"
                break

            curr_spec = new_spec
            self.logger.info(f"[{class_name}] Verifying fixed spec...")
            err_info, returncode = verify_with_openjml(
                curr_spec, class_name, self.timeout, self.output_dir, self.logger
            )
            verifier_calls += 1

            if self.verbose:
                self.logger.debug(
                    f"[{class_name}] Verification output: {err_info[:500]}"
                )

            if "Timeout:" in err_info or "timeout" in err_info.lower():
                self.logger.warning(f"[{class_name}] Verification timed out")
                status = "timed_out"
                break

            if returncode == 0:
                self.logger.info(
                    f"[{class_name}] Verified after {num_iter} fix iteration(s) "
                    f"and {verifier_calls} verifier call(s)"
                )
                status = "verified"
                break

            self.logger.debug(
                f"[{class_name}] Still failing after iter {num_iter}: {err_info[:200]}"
            )

        else:
            # Loop exhausted without breaking
            self.logger.warning(
                f"[{class_name}] Max fix iterations ({self.max_iters}) reached without verification"
            )
            status = "unverified"

        verified: bool = status == "verified"
        return SpecFixResult(
            status=status,
            class_name=class_name,
            verifier_calls=verifier_calls,
            iterations=num_iter,
            final_code=curr_spec,
            final_error=err_info,
            verified=verified,
        )
