from typing import List
from .llm_interface import LLMFactory, chat_models, api_models
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

try:
    from .vllm_wrapper import VLLMWrapper, VLLM_AVAILABLE
except ImportError:
    VLLM_AVAILABLE = False
    VLLMWrapper = None

class ModelRouter:
    """Incoporate api request models and local vllm models"""
    def __init__(self, 
                 model_names, 
                 model_dir = None, 
                 temperature: float = 1.0, 
                 top_p: float = 0.95, 
                 max_model_len:int = 2048,
                 gpu_memory_utilization:float = 0.9,
                 tensor_parallel_size:int = 1,
                    pipeline_parallel_size:int = 1,
                 ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_model_len = max_model_len

        vllm_models, api_models_list = {}, []
        for model_name in model_names:
            if model_name in api_models or model_name in chat_models:
                api_models_list.append(model_name)
            else:
                if not VLLM_AVAILABLE:
                    raise ValueError(
                        f"Model '{model_name}' is not a registered API model and vllm is not installed. "
                        f"Registered API models: {api_models + chat_models}. "
                        f"To use local models, install vllm. Otherwise, use a closed-source API model."
                    )
                model_path = model_name if model_dir is None else str(os.path.join(model_dir, model_name))
                print(f"Loading model {model_name} from {model_path}")
                vllm_models[model_name] = VLLMWrapper(model_path,
                                                       gpu_memory_utilization=gpu_memory_utilization,
                                                       temperature=temperature,
                                                       top_p=top_p,
                                                       max_model_len=max_model_len,
                                                       tensor_parallel_size=tensor_parallel_size,
                                                       pipeline_parallel_size=pipeline_parallel_size)
        self.vllm_models = vllm_models
        self.api_models = api_models_list
    
    def request_llm(self, 
                    conversations: List[List[dict]], 
                    model: str,
                    max_length:int = 10, 
                    temperature=None):
        """Request single turn generation from model"""
        if model in self.api_models:
            if temperature is not None:
                res = LLMFactory.gather_multiple_messages(conversations, model_name=model, max_tokens=max_length, temperature=temperature)
            else:
                res = LLMFactory.gather_multiple_messages(conversations, model_name=model, max_tokens=max_length, temperature=self.temperature)
        elif model in self.vllm_models:
            res = self.vllm_models[model].generate_multi_turn(conversations, use_tqdm=False, temperature=temperature) 
        else:
            raise ValueError(f"Invalid model name: {model}, not in ModelRouter model list")
        return res
        
    def request_llm_n(self, 
                        conversations: List[List[dict]], 
                        model: str,
                        max_length:int = 10, 
                        temperature=None,
                        num_generations:int=1):
        res = []
        if model in self.api_models:
            res = LLMFactory.gather_multiple_messages_n(conversations, model_name=model, max_tokens=max_length, n=num_generations, temperature=temperature)
        elif model in self.vllm_models:
            res = self.vllm_models[model].generate_multi_turn_n(conversations, use_tqdm=False, temperature=temperature, num_generations=num_generations)
        else:
            raise ValueError(f"Invalid model name: {model}, not in ModelRouter model list")
        return res
    