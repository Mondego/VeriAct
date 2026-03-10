import os
import signal
import shlex
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
    tmp_filename = os.path.join(output_dir, f"{classname}.java")
    try:
        write_to_file(code_with_spec, tmp_filename)
        logger.debug(f"[{classname}] Wrote code to {tmp_filename}")
    except Exception as e:
        logger.error(f"[{classname}] Failed to write file: {e}", exc_info=True)
        raise

    cmd = [
        "openjml",
        "--esc",
        "--esc-max-warnings",
        "1",
        "--prover=cvc4",
        "--nonnull-by-default",
        "--arithmetic-failure=quiet",
        "-nowarn",
        tmp_filename,
    ]
    logger.debug(f"[{classname}] Running OpenJML verification command")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        stdout, stderr = proc.communicate(timeout=_timeout)
        res = stdout + stderr
        logger.debug(f"[{classname}] OpenJML return code: {proc.returncode}")
        if stdout:
            logger.debug(f"[{classname}] OpenJML stdout: {stdout[:500]}")
        if stderr:
            logger.debug(f"[{classname}] OpenJML stderr: {stderr[:500]}")
        return res, proc.returncode
    except subprocess.TimeoutExpired:
        logger.error(
            f"[{classname}] OpenJML command timed out after {_timeout} seconds"
        )
        return "Timeout: OpenJML verification exceeded time limit", 1
    except Exception as e:
        logger.error(f"[{classname}] Error running OpenJML: {e}")
        return f"Error: {str(e)}", 1
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def validate_with_openjml(
    code_with_spec: str,
    classname: str,
    _timeout: int,
    output_dir: str,
    logger: logging.Logger,
) -> str:
    logger.info(f"[{classname}] Validating with OpenJML...")

    tmp_filename = os.path.join(output_dir, f"{classname}.java")
    try:
        write_to_file(code_with_spec, tmp_filename)
        logger.debug(f"[{classname}] Wrote code to {tmp_filename}")
    except Exception as e:
        logger.error(f"[{classname}] Failed to write file: {e}", exc_info=True)
        raise

    # [CHECK] Only JML syntax and type checking, without full verification
    cmd = ["openjml", "--check", tmp_filename]
    logger.debug(f"[{classname}] Running OpenJML verification command")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        stdout, stderr = proc.communicate(timeout=_timeout)
        res = stdout + stderr
        logger.debug(f"[{classname}] OpenJML return code: {proc.returncode}")
        if stdout:
            logger.debug(f"[{classname}] OpenJML stdout: {stdout[:500]}")
        if stderr:
            logger.debug(f"[{classname}] OpenJML stderr: {stderr[:500]}")
        return res, proc.returncode
    except subprocess.TimeoutExpired:
        logger.error(
            f"[{classname}] OpenJML command timed out after {_timeout} seconds"
        )
        return "Timeout: OpenJML verification exceeded time limit", 1
    except Exception as e:
        logger.error(f"[{classname}] Error running OpenJML: {e}")
        return f"Error: {str(e)}", 1
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
