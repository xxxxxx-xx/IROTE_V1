"""
LLM Interface for Closed-Source Models
======================================
Simplified interface for API-based models only.
Supports: GPT-4o, MiMo, and any OpenAI-compatible API.
"""

from abc import abstractmethod, ABC
import os
import json
from time import sleep
import threading
import asyncio
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI, AsyncOpenAI

# Import config
from config import (
    GPT_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_KEY,
    GENERIC_API_MODELS,
)


class LLM(ABC):
    """Base class for LLM models."""

    @classmethod
    @abstractmethod
    def process(cls, message, max_tokens=2000, temperature=None):
        pass

    @classmethod
    def reset_status(cls):
        pass


# ============================================================
# Azure OpenAI Models (GPT-4o)
# ============================================================

try:
    url = GPT_ENDPOINT
    azure_model_map = {
        "GPT-4o-Mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o"
    }

    if url and (AZURE_OPENAI_API_KEY or OPENAI_API_KEY):
        from openai import AsyncAzureOpenAI, AzureOpenAI

        async_aoiclient = AsyncAzureOpenAI(
            azure_endpoint=url,
            api_version="2024-12-01-preview",
            api_key=AZURE_OPENAI_API_KEY or OPENAI_API_KEY,
            max_retries=5
        )

        aoiclient = AzureOpenAI(
            azure_endpoint=url,
            api_version="2024-12-01-preview",
            api_key=OPENAI_API_KEY or AZURE_OPENAI_API_KEY,
            max_retries=5
        )
        print(f"AzureOpenAI registered: endpoint={url}")
    else:
        aoiclient = None
        async_aoiclient = None
        print("AzureOpenAI not configured (no endpoint or API key)")

except Exception as e:
    print(f"AzureOpenAI not registered: {e}")
    aoiclient = None
    async_aoiclient = None


class InAzureModel(LLM):
    """Azure OpenAI model base class."""
    token_count = 0
    lock = threading.Lock()

    @classmethod
    def process(cls, message, max_tokens=2000, retry=2, temperature=1):
        global aoiclient
        if aoiclient is None:
            raise RuntimeError("AzureOpenAI not configured")

        model = azure_model_map.get(cls.model_name, cls.model_name)
        try:
            response = aoiclient.chat.completions.create(
                model=model,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(f"{cls.model_name} error: {e}")
            sleep(5)
            if retry > 0:
                return cls.process(message, max_tokens, retry - 1, temperature)
            raise

    @classmethod
    async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
        global async_aoiclient
        if async_aoiclient is None:
            raise RuntimeError("AzureOpenAI not configured")

        model = azure_model_map.get(cls.model_name, cls.model_name)
        try:
            response = await async_aoiclient.chat.completions.create(
                model=model,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            with cls.lock:
                cls.token_count += response.usage.total_tokens
            return text
        except Exception as e:
            print(f"{cls.model_name} error: {e}")
            if retry > 0:
                await asyncio.sleep(5)
                return await cls.async_process(message, max_tokens, retry - 1, temperature)
            raise


class GPT4OMini(InAzureModel):
    model_name = "GPT-4o-Mini"
    token_count = 0
    lock = threading.Lock()


class GPT4O(InAzureModel):
    model_name = "GPT-4o"
    token_count = 0
    lock = threading.Lock()


# ============================================================
# Generic API Model (for MiMo, DeepSeek, etc.)
# ============================================================

class GenericAPIModel(LLM):
    """
    Flexible model class for any OpenAI-compatible API endpoint.
    Configured via config.GENERIC_API_MODELS dict.
    """
    token_count = 0
    lock = threading.Lock()
    model_name = ""
    _base_url = ""
    _api_key = ""

    @classmethod
    def get_setting(cls, async_=False):
        api_key = cls._api_key
        if async_:
            return AsyncOpenAI(api_key=api_key, base_url=cls._base_url)
        else:
            return OpenAI(api_key=api_key, base_url=cls._base_url)

    @classmethod
    def process(cls, message, max_tokens=2000, retry=2, temperature=1.0):
        client = cls.get_setting()
        try:
            response = client.chat.completions.create(
                model=cls.model_name,
                messages=message,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            with cls.lock:
                cls.token_count += getattr(response.usage, 'total_tokens', 0) if response.usage else 0
            return text
        except Exception as e:
            print(f"{cls.model_name} error: {e}")
            sleep(4)
            if retry > 0:
                return cls.process(message, max_tokens, retry - 1, temperature)
            raise

    @classmethod
    async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1.0):
        client = cls.get_setting(async_=True)
        try:
            async with client:
                response = await client.chat.completions.create(
                    model=cls.model_name,
                    messages=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            text = response.choices[0].message.content or ""
            with cls.lock:
                cls.token_count += getattr(response.usage, 'total_tokens', 0) if response.usage else 0
            return text
        except Exception as e:
            print(f"{cls.model_name} error: {e}")
            if retry > 0:
                await asyncio.sleep(4)
                return await cls.async_process(message, max_tokens, retry - 1, temperature)
            raise


def _register_generic_models():
    """Dynamically register models from config.GENERIC_API_MODELS."""
    for display_name, model_cfg in GENERIC_API_MODELS.items():
        if display_name in PRODUCT_MAP:
            continue

        api_key = model_cfg.get("auth_header", "") or os.getenv(model_cfg.get("api_key_env", ""), "")
        base_url = model_cfg.get("base_url", "")
        model_id = model_cfg.get("model_id", display_name)

        new_class = type(
            f"Generic_{display_name.replace('-', '_').replace('.', '_')}",
            (GenericAPIModel,),
            {
                "model_name": model_id,
                "_base_url": base_url,
                "_api_key": api_key,
                "token_count": 0,
                "lock": threading.Lock(),
            }
        )
        PRODUCT_MAP[display_name] = new_class
        chat_models.append(display_name)
        print(f"Registered generic API model: {display_name} -> {model_id}")


# ============================================================
# Factory
# ============================================================

PRODUCT_MAP = {
    "GPT-4o-Mini": GPT4OMini,
    "GPT-4o": GPT4O,
}
chat_models = ['GPT-4o-Mini', 'GPT-4o']
api_models = chat_models.copy()

# Register generic models from config
_register_generic_models()


class LLMFactory:
    """Factory for creating and managing LLM instances."""

    @classmethod
    def process(cls, message, model_name, max_tokens=1000, temperature=1.0, **kwargs):
        if model_name in PRODUCT_MAP:
            product = PRODUCT_MAP[model_name]
            return product.process(message=message, max_tokens=max_tokens, temperature=temperature, **kwargs)
        raise NotImplementedError(f"{model_name} not registered")

    @classmethod
    async def async_process(cls, message, model_name, max_tokens=1000, temperature=1.0, **kwargs):
        if model_name in PRODUCT_MAP:
            product = PRODUCT_MAP[model_name]
            return await product.async_process(message=message, max_tokens=max_tokens, temperature=temperature, **kwargs)
        raise NotImplementedError(f"{model_name} not registered")

    @classmethod
    async def gather_multiple_async_messages(cls, messages, model_name, **kwargs):
        return await asyncio.gather(*[cls.async_process(msg, model_name, **kwargs) for msg in messages])

    @classmethod
    def gather_multiple_messages(cls, messages, model_name, **kwargs):
        return asyncio.run(cls.gather_multiple_async_messages(messages, model_name, **kwargs))

    @classmethod
    def print_token_count(cls, model_name):
        if model_name in PRODUCT_MAP:
            return PRODUCT_MAP[model_name].token_count / 1e6
        return 0.0

    @classmethod
    def print_all_token_count(cls):
        res = ""
        for name, product in PRODUCT_MAP.items():
            res += f"{name}: {product.token_count / 1e6:.2f}M\n"
        return res
