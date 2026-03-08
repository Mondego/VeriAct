import os
import logging
import subprocess
from pathlib import Path
from baselines.utils.file_utility import write_to_file


def verify_with_openjml(
    code_with_spec: str,
    classname: str,
    _timeout: int,
    output_dir: str,
    logger: logging.Logger,
) -> str:
    logger.info(f"[{classname}] Validating with OpenJML...")
    # tmp_dir = os.path.join(output_dir, "tmp")
    # Path(tmp_dir).mkdir(exist_ok=True)

    tmp_filename = os.path.join(output_dir, f"{classname}.java")
    try:
        write_to_file(code_with_spec, tmp_filename)
        logger.debug(f"[{classname}] Wrote code to {tmp_filename}")
    except Exception as e:
        logger.error(f"[{classname}] Failed to write file: {e}", exc_info=True)
        raise

    cmd = f"openjml --esc --esc-max-warnings 1 --arithmetic-failure=quiet --nonnull-by-default --quiet -nowarn --prover=cvc4 {tmp_filename}"
    logger.debug(f"[{classname}] Running OpenJML verification command")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=_timeout
        )
        res = result.stdout + result.stderr
        logger.debug(f"[{classname}] OpenJML return code: {result.returncode}")
        if result.stdout:
            logger.debug(f"[{classname}] OpenJML stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.debug(f"[{classname}] OpenJML stderr: {result.stderr[:500]}")
        return res
    except subprocess.TimeoutExpired:
        logger.error(
            f"[{classname}] OpenJML command timed out after {_timeout} seconds"
        )
        return "Timeout: OpenJML verification exceeded time limit"
    except Exception as e:
        logger.error(f"[{classname}] Error running OpenJML: {e}")
        return f"Error: {str(e)}"


def validate_with_openjml(
    code_with_spec: str,
    classname: str,
    _timeout: int,
    output_dir: str,
    logger: logging.Logger,
) -> str:
    logger.info(f"[{classname}] Validating with OpenJML...")
    # tmp_dir = os.path.join(output_dir, "tmp")
    # Path(tmp_dir).mkdir(exist_ok=True)

    tmp_filename = os.path.join(output_dir, f"{classname}.java")
    try:
        write_to_file(code_with_spec, tmp_filename)
        logger.debug(f"[{classname}] Wrote code to {tmp_filename}")
    except Exception as e:
        logger.error(f"[{classname}] Failed to write file: {e}", exc_info=True)
        raise

    # [FIX ME] For validation this command will change
    # cmd = f"openjml --esc --esc-max-warnings 1 --arithmetic-failure=quiet --nonnull-by-default --quiet -nowarn --prover=cvc4 {tmp_filename}"
    cmd = f"openjml --check {tmp_filename}"
    logger.debug(f"[{classname}] Running OpenJML verification command")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=_timeout
        )
        res = result.stdout + result.stderr
        logger.debug(f"[{classname}] OpenJML return code: {result.returncode}")
        if result.stdout:
            logger.debug(f"[{classname}] OpenJML stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.debug(f"[{classname}] OpenJML stderr: {result.stderr[:500]}")
        return res
    except subprocess.TimeoutExpired:
        logger.error(
            f"[{classname}] OpenJML command timed out after {_timeout} seconds"
        )
        return "Timeout: OpenJML verification exceeded time limit"
    except Exception as e:
        logger.error(f"[{classname}] Error running OpenJML: {e}")
        return f"Error: {str(e)}"
