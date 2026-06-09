from typing import List
import random
from tqdm import tqdm
random.seed(42)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("[Warning] vllm not installed. Local model inference is disabled. Use closed-source API models instead.")

try:
    from transformers import AutoTokenizer
except ImportError:
    pass

class VLLMWrapper:
    def __init__(self, model_path,
                 pipeline_parallel_size=1,
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.9,
                 max_model_len=8192,
                 temperature=0,
                 top_p=0.95,
                 max_tokens=2048,
                 ):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm is not installed. Cannot use local model inference. "
                "Please use closed-source API models (e.g., GPT-4o) instead, "
                "or install vllm: pip install vllm"
            )
        self.model_path = model_path
        self.model_name = self.model_path.split("/")[-1]

        self.model = LLM(model=model_path,
                         pipeline_parallel_size=pipeline_parallel_size,
                         tensor_parallel_size=tensor_parallel_size,
                         gpu_memory_utilization=gpu_memory_utilization,
                         max_model_len=max_model_len,
                         trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)


    def generate_multi_turn_n(self, conversations: List[List[dict]], 
                            use_tqdm:bool =True,
                            num_generations:int=1,
                            temperature=None):
        """
        Generating multiple responses for single-turn conversation
        Output: List of List of responses [[response1_gen1, response1_gen2, ...], [response2_gen1, response2_gen2, ...], ...]
        """
        tmp_sampling_params = self.sampling_params.clone()
        if temperature is not None:
            tmp_sampling_params.temperature = temperature

        total_responses = []
        if not use_tqdm:
            genarator = range(num_generations)
        else:
            genarator = tqdm(range(num_generations))

        for _ in genarator:
            outputs = self.llm.chat(conversations,  sampling_params=tmp_sampling_params, use_tqdm=False)
            responses = [output.outputs[0].text for output in outputs]
            total_responses.append(responses)
        return [[total_responses[j][i] for j in range(num_generations)] for i in range(len(conversations))]


    def generate_multi_turn(self, 
                            conversations: List[List[dict]], 
                            use_tqdm:bool =True,
                            temperature=None):
        """
        Generating responses for multi-turn conversations
        Output: List of responses [response1, response2, ...]
        """
        # print('in generate_multi_turn')
        tmp_sampling_params = self.sampling_params.clone()
        if temperature is not None:
            tmp_sampling_params.temperature = temperature
        outputs = self.model.chat(conversations,  sampling_params=tmp_sampling_params, use_tqdm=use_tqdm)
        # print('outputs', [output.outputs[0].text for output in outputs])
        return [output.outputs[0].text for output in outputs]
    
    def generate_raw_prompts(self, prompts: List[str],
                             use_tqdm:bool =True,
                             temperature=None):
        tmp_sampling_params = self.sampling_params.clone()
        if temperature is not None:
            tmp_sampling_params.temperature = temperature
        responses = self.model.generate(prompts, sampling_params=tmp_sampling_params, use_tqdm=use_tqdm)
        return [response.outputs[0].text for response in responses]