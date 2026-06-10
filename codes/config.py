"""
IROTE Configuration File
========================
Centralized configuration for model API keys, endpoints, and paths.

Conda Environment: cottonagent (E:\anaconda\envs\cottonagent)

Usage:
    # Set environment variable first:
    # $env:MIMO_AUTH_TOKEN="your-token-here"
    # Then run:
    E:\anaconda\envs\cottonagent\python.exe optimization_irote.py --model_name MiMo-v2.5-Pro
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(_env_path)

# ============================================================
# 1. Path Configuration
# ============================================================
PATH_TO_SIMCSE_MODEL = os.getenv(
    "IROTE_SIMCSE_PATH",
    os.path.join(os.path.dirname(__file__), "models", "sup-simcse-roberta-large")
)

USE_SIMCSE = False

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
# Set MIMO_AUTH_TOKEN environment variable before running:
#   PowerShell: $env:MIMO_AUTH_TOKEN="tp-xxxxxxxxxxxx"
#   CMD:        set MIMO_AUTH_TOKEN=tp-xxxxxxxxxxxx
GENERIC_API_MODELS = {
    "MiMo-v2.5-Pro": {
        "type": "openai_compatible",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "api_key_env": "MIMO_AUTH_TOKEN",
        "model_id": "mimo-v2.5-pro",
        "auth_header": os.getenv("MIMO_AUTH_TOKEN", ""),
    },
}
