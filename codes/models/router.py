"""
Model Router for Closed-Source API Models
==========================================
[FIX #1] Added request_llm_with_logprobs for p_e(y|x) estimation.
"""

from typing import List, Dict, Optional
from .llm_interface import LLMFactory, chat_models, api_models


class ModelRouter:
    """Route requests to API-based models."""

    def __init__(
        self,
        model_names: List[str],
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_model_len: int = 2048,
        **kwargs,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_model_len = max_model_len

        self.api_models = []
        for model_name in model_names:
            if model_name in api_models or model_name in chat_models:
                if model_name not in self.api_models:
                    self.api_models.append(model_name)
            else:
                raise ValueError(
                    f"Model '{model_name}' is not registered. "
                    f"Available: {api_models + chat_models}"
                )

    def request_llm(
        self,
        conversations: List[List[dict]],
        model: str,
        max_length: int = 10,
        temperature: float = None,
    ) -> List[str]:
        """Request completions from model."""
        if model not in self.api_models:
            raise ValueError(f"Model '{model}' not in router's model list")

        temp = temperature if temperature is not None else self.temperature
        return LLMFactory.gather_multiple_messages(
            conversations,
            model_name=model,
            max_tokens=max_length,
            temperature=temp,
        )

    def request_llm_with_logprobs(
        self,
        conversations: List[List[dict]],
        model: str,
        max_length: int = 10,
        temperature: float = None,
    ) -> Dict:
        """
        [FIX #1] Request completion with logprobs for p_e(y|x) estimation.
        Returns: {"text": str, "logprob": float} or str if logprobs unavailable.
        """
        if model not in self.api_models:
            raise ValueError(f"Model '{model}' not in router's model list")

        temp = temperature if temperature is not None else self.temperature

        try:
            result = LLMFactory.gather_messages_with_logprobs(
                conversations,
                model_name=model,
                max_tokens=max_length,
                temperature=temp,
            )
            return result
        except (AttributeError, Exception):
            # Fallback: standard generation
            responses = self.request_llm(conversations, model, max_length, temperature)
            return {"text": responses[0] if responses else "", "logprob": 0.0}
