"""Unified LLM client supporting multiple providers via OpenAI-compatible APIs."""

import os
import time
import threading
from openai import APIConnectionError, APIError, OpenAI, OpenAIError, RateLimitError
import baselines.utils.config

# ---------------------------------------------------------------------------
# Provider registry: prefix → (base_url, env_var_for_api_key)
#   - base_url=None means use the default OpenAI endpoint
#   - api_key value of None means read from the env var
# ---------------------------------------------------------------------------


VALID_MODEL_PREFIXES = [
    "gpt",
    "o1",
    "o3",
    "gemini",
    "claude",
    "qwen",
    "deepseek-ai",
    "deepseek",
    "codellama",
    "mistral",
]


_PROVIDERS: dict[str, dict] = {
    "gpt": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "o1": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "o3": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "claude": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "qwen": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
    },
    "codellama": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
    },
    "deepseek-ai": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
}

MAX_RETRIES = 5

# Thread-local token accumulator — each worker thread tracks its own totals
_token_usage = threading.local()


def reset_token_usage() -> None:
    """Reset token counters for the current thread. Call at the start of each task."""
    _token_usage.input_tokens = 0
    _token_usage.output_tokens = 0


def get_token_usage() -> tuple[int, int]:
    """Return (input_tokens, output_tokens) accumulated for the current thread."""
    return (
        getattr(_token_usage, "input_tokens", 0),
        getattr(_token_usage, "output_tokens", 0),
    )


# ---------------------------------------------------------------------------
# Public API  (unchanged — all existing callers keep working)
# ---------------------------------------------------------------------------
def create_model_config(messages: list[dict], model: str, temperature: float) -> dict:
    """Build a provider-agnostic config dict."""
    _resolve_provider(model)  # fail fast if unsupported
    return {"model": model, "temperature": temperature, "messages": messages}


def request_llm_engine(_config: dict):
    """Send a chat completion request to the appropriate provider."""
    provider = _resolve_provider(_config["model"])
    client = _build_client(provider)
    return _request_with_retries(client, _config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_provider(model: str) -> dict:
    """Return the provider config for *model*, or raise ValueError."""
    for prefix in sorted(_PROVIDERS, key=len, reverse=True):
        if model.lower().startswith(prefix):
            return _PROVIDERS[prefix]
    raise ValueError(
        f"Unsupported model: {model!r}. "
        f"Supported prefixes: {list(_PROVIDERS.keys())}"
    )


def _build_client(provider: dict) -> OpenAI:
    """Instantiate an OpenAI client configured for *provider*."""
    api_key = provider.get("api_key_literal") or os.getenv(
        provider.get("api_key_env", ""), ""
    )
    kwargs: dict = {"api_key": api_key}
    if provider["base_url"] is not None:
        kwargs["base_url"] = provider["base_url"]
    return OpenAI(**kwargs)


def _request_with_retries(client: OpenAI, _config: dict):
    """Execute a chat completion with retries on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**_config)
            if response.usage:
                _token_usage.input_tokens = (
                    getattr(_token_usage, "input_tokens", 0)
                    + response.usage.prompt_tokens
                )
                _token_usage.output_tokens = (
                    getattr(_token_usage, "output_tokens", 0)
                    + response.usage.completion_tokens
                )
            return response
        except APIError as e:
            print(f"API error: {e}")
            if e.code == "invalid_request_error":
                break
        except RateLimitError as e:
            print(f"Rate limit exceeded. Waiting... {e}")
            time.sleep(5)
        except APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except OpenAIError as e:
            print(f"Unknown error on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                raise
    return None
