import os
import time
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, OpenAIError
from utils import count_config_token
import config


def token_limit_fitter(_config, token_limit=4090):
    res = _config
    while count_config_token(res) > token_limit:
        res["messages"] = res["messages"][3 : len(res["messages"])]
        res["messages"].insert(0, _config["messages"][0])
    return res


def create_model_config(messages, model, temperature):
    if model.startswith("gpt"):
        return create_chatgpt_config(messages, model, temperature)
    elif model.startswith("vllm"):
        return create_vllm_config(messages, model, temperature)
    # will add more models in the future


def request_llm_engine(_config):
    if _config["model"].startswith("gpt"):
        return request_openai_engine(_config)
    elif _config["model"].startswith("vllm"):
        return request_vllm_engine(_config)
    # will add more models in the future


def create_chatgpt_config(messages, model, temperature):
    _config = {"model": model, "temperature": temperature, "messages": []}
    _config["messages"] = messages
    return _config


def request_openai_engine(_config):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    max_retries = 5
    for attempt in range(max_retries):
        try:
            ret = client.chat.completions.create(**_config)
            return ret
        except APIError as e:
            print(f"API error: {e}")
            if e.code == "invalid_request_error":
                break  # Don't retry invalid requests
        except RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            time.sleep(5)
        except APIConnectionError as e:
            print("API connection error. Waiting...")
            time.sleep(5)
        except OpenAIError as e:
            print(f"Unknown error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise e
    return None


def create_vllm_config(messages, model, temperature):
    """
    Create a configuration for vLLM API request using OpenAI SDK.
    """
    _config = {"model": model, "temperature": temperature, "messages": []}
    _config["messages"] = messages
    return _config


def request_vllm_engine(_config):
    vllm_base_url = "http://localhost:8000/v1"
    client = OpenAI(base_url=vllm_base_url, api_key="spec")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            ret = client.chat.completions.create(**_config)
            return ret
        except APIError as e:
            print(f"API error: {e}")
            if e.code == "invalid_request_error":
                break  # Don't retry invalid requests
        except RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            time.sleep(5)
        except APIConnectionError as e:
            print("API connection error. Waiting...")
            time.sleep(5)
        except OpenAIError as e:
            print(f"Unknown error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise e
    return None
