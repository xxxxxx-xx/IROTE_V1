"""
IROTE Configuration File
========================
Centralized configuration for model API keys, endpoints, and paths.

Conda Environment: cottonagent (E:\anaconda\envs\cottonagent)

Usage:
    E:\\anaconda\\envs\\cottonagent\\python.exe run_irote.py --model_name MiMo-v2.5-Pro
"""

import os

# ============================================================
# 1. Path Configuration
# ============================================================
# SimCSE model path (only needed if USE_SIMCSE=True)
PATH_TO_SIMCSE_MODEL = os.getenv(
    "IROTE_SIMCSE_PATH",
    os.path.join(os.path.dirname(__file__), "models", "sup-simcse-roberta-large")
)

# Whether to use SimCSE for text similarity (requires ~1.4GB download)
# If False, uses lightweight TF-IDF + cosine similarity instead (no extra download needed)
USE_SIMCSE = False

# Path to a classifier model (optional, only needed for HuggingfaceEvaluator)
CLASSIFIER_PATH = os.getenv("IROTE_CLASSIFIER_PATH", "")

# ============================================================
# 2. Azure OpenAI Configuration (for GPT-4o / GPT-4o-Mini)
# ============================================================
GPT_ENDPOINT = os.getenv("GPT_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ============================================================
# 3. vLLM Placeholder Constants (unused for closed-source setup)
# ============================================================
MISTRAL_VLLM_NAME = os.getenv("MISTRAL_VLLM_NAME", "Mistral-7B-Instruct-v0.3")
MISTRAL_VLLM_PORT = int(os.getenv("MISTRAL_VLLM_PORT", "8000"))
QWEN_VLLM_NAME = os.getenv("QWEN_VLLM_NAME", "Qwen2.5-7B-Instruct")
QWEN_VLLM_PORT = int(os.getenv("QWEN_VLLM_PORT", "8001"))

# ============================================================
# 4. Generic API Model Registry
# ============================================================
# To add a new closed-source model, add an entry here.
# Format: "display_name": { "type": "openai_compatible", "base_url": ..., "api_key_env": ..., "model_id": ... }
GENERIC_API_MODELS = {
    "MiMo-v2.5-Pro": {
        "type": "anthropic_compatible",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "api_key_env": "MIMO_AUTH_TOKEN",
        "model_id": "mimo-v2.5-pro",
        "auth_header": "xxxxx",
    },
}
