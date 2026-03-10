import re
import logging
from typing import TypedDict

from baselines.formalbench.prompts import build_messages
from baselines.formalbench.example import JavaExample
from baselines.utils.verifier import verify_with_openjml
from baselines.utils.models import create_model_config, request_llm_engine


VALID_PROMPT_TYPES = ["zero_shot", "zs_cot", "two_shot", "fs_cot", "fs_ltm"]


class FBInferResult(TypedDict):
    status: str
    class_name: str
    prompt_type: str
    verifier_calls: int
    final_code: str | None
    final_error: str
    verified: bool
    iterations: int


class FormalBench:

    def __init__(
        self,
        model: str,
        temperature: float,
        prompt_type: str,
        output_dir: str,
        timeout: int,
        logger: logging.Logger,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.output_dir = output_dir
        self.timeout = timeout
        self.logger = logger
        self.verbose = verbose

        # Load few-shot examples based on prompt type
        if prompt_type == "fs_ltm":
            self.example_code1 = JavaExample.EXAMPLE_CODE2
            self.example_code2 = JavaExample.EXAMPLE_CODE3
            self.example_spec1 = JavaExample.EXAMPLE_LTM_RESPONSE2
            self.example_spec2 = JavaExample.EXAMPLE_LTM_RESPONSE3
        else:
            self.example_code1 = JavaExample.EXAMPLE_CODE1
            self.example_code2 = JavaExample.EXAMPLE_CODE2
            self.example_spec1 = JavaExample.EXAMPLE_SPEC1
            self.example_spec2 = JavaExample.EXAMPLE_SPEC2

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

    def run(self, input_code: str, class_name: str) -> FBInferResult:
        verifier_calls: int = 0
        status: str = "unknown"
        err_info: str = ""
        spec: str | None = None

        self.logger.info(
            f"[{class_name}] Generating specifications (prompt_type={self.prompt_type})..."
        )

        messages = build_messages(
            prompt_type=self.prompt_type,
            model=self.model,
            code=input_code,
            example_code1=self.example_code1,
            example_code2=self.example_code2,
            example_spec1=self.example_spec1,
            example_spec2=self.example_spec2,
        )

        config = create_model_config(messages, self.model, self.temperature)
        ret = request_llm_engine(config)

        if ret is None:
            self.logger.error(f"[{class_name}] LLM returned no response")
            return FBInferResult(
                status="unknown",
                class_name=class_name,
                prompt_type=self.prompt_type,
                verifier_calls=verifier_calls,
                final_code=None,
                final_error="LLM returned no response",
                verified=False,
                iterations=1,
            )

        raw_response: str = ret.choices[0].message.content
        if self.verbose:
            self.logger.debug(f"[{class_name}] LLM response: {raw_response[:500]}")

        spec = self._parse_spec_from_response(raw_response)

        if not spec:
            self.logger.warning(f"[{class_name}] Empty specification returned by model")
            return FBInferResult(
                status="empty_spec",
                class_name=class_name,
                prompt_type=self.prompt_type,
                verifier_calls=verifier_calls,
                final_code=None,
                final_error="Empty specification",
                verified=False,
                iterations=1,
            )

        if not self._contains_annotations(spec):
            self.logger.warning(
                f"[{class_name}] No JML annotations detected in response"
            )
            return FBInferResult(
                status="invalid_jml",
                class_name=class_name,
                prompt_type=self.prompt_type,
                verifier_calls=verifier_calls,
                final_code=spec,
                final_error="No JML annotations detected",
                verified=False,
                iterations=1,
            )

        self.logger.info(f"[{class_name}] Verifying with OpenJML...")
        err_info, returncode = verify_with_openjml(
            spec, class_name, self.timeout, self.output_dir, self.logger
        )
        verifier_calls += 1

        if self.verbose:
            self.logger.debug(f"[{class_name}] Verification output: {err_info[:500]}")

        if "Timeout:" in err_info or "timeout" in err_info.lower():
            status = "timed_out"
        elif returncode == 0:
            status = "verified"
        else:
            status = "unverified"

        verified: bool = status == "verified"

        if verified:
            self.logger.info(
                f"[{class_name}] Successfully verified with {verifier_calls} verifier call(s)"
            )
        else:
            self.logger.warning(
                f"[{class_name}] Verification {status}. Final error: {err_info[:200]}"
            )

        return FBInferResult(
            status=status,
            class_name=class_name,
            prompt_type=self.prompt_type,
            verifier_calls=verifier_calls,
            final_code=spec,
            final_error=err_info,
            verified=verified,
            iterations=1,
        )
