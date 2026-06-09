"""Reconstruct the LLM interface for the LLMs"""
from abc import abstractmethod, ABC
import os
import openai
import json
from time import sleep
from os import path
import requests
import threading
import asyncio
from queue import Queue
import aiohttp
try:
	from mistralai import Mistral
except Exception as e:
	print("Mistral not registered", e)
try:
	import google.generativeai as genai
except Exception as e:
	print("Google AI not registered", e)

from openai import OpenAI, AsyncOpenAI

# Import centralized config for model names, ports, and API keys
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
	GPT_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_KEY,
	MISTRAL_VLLM_NAME, MISTRAL_VLLM_PORT,
	QWEN_VLLM_NAME, QWEN_VLLM_PORT,
	GENERIC_API_MODELS,
)

class LLM(ABC):
	# product
	
	@classmethod
	@abstractmethod
	def process(cls, message, max_tokens=2000, temperature=None):
		pass
	
	@classmethod
	def reset_status(cls):
		openai.api_key = None
		openai.api_base = "https://api.openai.com/v1"
		openai.api_type = "open_ai"
		openai.api_version = None


try:
	url = GPT_ENDPOINT
	azure_model_map = {
			"GPT-4o-Mini": "gpt-4o-mini",
			'GPT-4o': "gpt-4o"
		}
	from openai import AsyncAzureOpenAI, AzureOpenAI

	async_aoiclient = AsyncAzureOpenAI(
					azure_endpoint=url,
					api_version="2024-12-01-preview",
					api_key=AZURE_OPENAI_API_KEY,
					max_retries=5
	)

	aoiclient = AzureOpenAI(
					azure_endpoint=url,
					api_version="2024-12-01-preview",
					api_key=OPENAI_API_KEY,
					max_retries=5
	)
	print(f"AzureOpenAI registered: endpoint={url}")


except Exception as e:
	print("AzureOpenAI not registered", e)

class InAzureModel(LLM):
	token_count = 0
	lock = threading.Lock()
	
	@classmethod
	def process(cls, message, max_tokens=2000, retry=2, temperature=1):
		global aoiclient, azure_model_map
		model = azure_model_map[cls.model_name]
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
			print(cls.model_name, "has error")
			print(message, e)
			sleep(5)
			if retry > 0:
				text = cls.process(message, max_tokens, retry-1)
				return text
			else:
				raise e
			
	@classmethod
	async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
		global azure_model_map, async_aoiclient
		model = azure_model_map[cls.model_name]
		
		try:
			response = await async_aoiclient.chat.completions.create(
								model=model,
								messages=message,
								max_tokens=max_tokens,
								temperature=temperature,
				)
			text = response.choices[0].message.content
			# print('Success!', text)
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return text
		except Exception as e:  
			print(f"Model name: {model}")
			print(f"Error type: {type(e)}")
			print(f"Error message: {str(e)}")
			if retry > 0:  
				await asyncio.sleep(5)
				return await cls.async_process(message, max_tokens=max_tokens, retry=retry-1, temperature=temperature)
			else:
				raise e
	
	@classmethod
	async def async_process_n(cls, message, max_tokens=2000, n=1, temperature=1, retry=2):
		global azure_model_map
		model = azure_model_map[cls.model_name]
		
		try:
			response = await async_aoiclient.chat.completions.create(
							model=model,
							messages=message,
							max_tokens=max_tokens,
							temperature=temperature,
							n=n
			)
			texts = [response.choices[i].message.content for i in range(n)]
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return texts
		except Exception as e:
			print(cls.model_name, "has error", model)
			print(message, e)
			if retry > 0:
				await asyncio.sleep(10)
				return await cls.async_process_n(message, max_tokens=max_tokens, n=n, temperature=temperature, retry=retry-1)
			else:
				raise e

class GPT4OMini(InAzureModel):
	model_name = "GPT-4o-Mini"
	token_count = 0
	lock = threading.Lock()

class GPT4O(InAzureModel):
	model_name = "GPT-4o"
	token_count = 0
	lock = threading.Lock()

class AzureModel(LLM):
	token_count = 0
	lock = threading.Lock()
	@classmethod
	def async_azure_setting(cls):
		from azure.ai.inference import ChatCompletionsClient
		from azure.core.credentials import AzureKeyCredential
		api_key = cls.api_key
		endpoint = cls.endpoint  
		client = ChatCompletionsClient(
		endpoint=endpoint,
		credential=AzureKeyCredential(api_key),
		)
		return client
	
	@classmethod
	def azure_setting(cls):
		from azure.ai.inference.aio import ChatCompletionsClient
		from azure.core.credentials import AzureKeyCredential
		api_key = cls.api_key
		endpoint = cls.endpoint
		client = ChatCompletionsClient(
			endpoint=endpoint,
			credential=AzureKeyCredential(api_key),
		)
		return client

	@classmethod
	def process(cls, message, max_tokens=2000, retry=2, temperature=0.7):
		client = cls.azure_setting()
		
		from azure.ai.inference.models import AssistantMessage, UserMessage, SystemMessage
		input_messages = []
		for m in message:
			if m['role'] == 'user':
				input_messages.append(UserMessage(content=m['content']))
			elif m['role'] == 'system':
				input_messages.append(SystemMessage(content=m['content']))
			else:
				input_messages.append(AssistantMessage(content=m['content']))

		try:
			response = client.complete(
				messages=input_messages,
				model=cls.model_name,
			)
			text = response.choices[0].message.content
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			
			if "</think>" in text:
				text = text.split("</think>")[1]
			return text
		except Exception as e:
			print(message, e)
			print(e)
			exit()
	
	async def transform(message):
		from azure.ai.inference.models import AssistantMessage, UserMessage, SystemMessage
		input_messages = []
		for m in message:
			if m['role'] == 'user':
				input_messages.append(UserMessage(content=m['content']))
			elif m['role'] == 'system':
				input_messages.append(SystemMessage(content=m['content']))
			else:
				input_messages.append(AssistantMessage(content=m['content']))
		return input_messages

	@classmethod
	async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
		from azure.ai.inference.aio import ChatCompletionsClient
		from azure.core.credentials import AzureKeyCredential
		input_messages = await cls.transform(message)

		try:
			async with ChatCompletionsClient(
				endpoint=cls.endpoint,
				credential=AzureKeyCredential(cls.api_key),
			) as client:
				response = await client.complete(
					messages=input_messages,
					model=cls.model_name,
					
				)
				text = response.choices[0].message.content
				if "</think>" in text:
					text = text.split("</think>")[1]
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			
			return text
		except Exception as e:
			print(message, e)
			print(e)
			if retry > 0:
				await asyncio.sleep(4)
				return await cls.async_process(message, max_tokens, retry=retry-1)
			exit()
			
class CustomOpenAIModel(LLM):
	token_count = 0
	lock = threading.Lock()
	@classmethod
	def get_setting(cls):
		pass

	@classmethod
	def process(cls, message, max_tokens=2000, retry=2, temperature=1):
		client = cls.get_setting()
		try:  
			response = client.chat.completions.create(
				model=cls.model_name,
				top_p=1, n=1, max_tokens=max_tokens,
				temperature=temperature,
				messages=message,
				)
			text = response.choices[0].message.content
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return text
		except Exception as e:
			print(cls.model_name, "has error")
			print(message, e)
			sleep(4)
			if retry > 0:
				text = cls.process(message, max_tokens, retry-1)
				return text
			else:
				raise e

	@classmethod
	async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
		client = cls.get_setting(async_=True)
		try:
			async with client:
				response = await client.chat.completions.create(
					model=cls.model_name,
					top_p=1, n=1, max_tokens=max_tokens,
					temperature=temperature,
					messages=message,
					)
			text = response.choices[0].message.content
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			asyncio.sleep(2)
			return text
		except Exception as e:
			print(cls.model_name, "has error")
			print(e)
			if retry > 0:
				await asyncio.sleep(4)
				text = await cls.async_process(message, max_tokens, retry=retry-1)
				return text
			else:
				raise e
	
	@classmethod
	async def async_process_n(cls, message, max_tokens=2000, n=1, temperature=1, retry=2):
		client = cls.get_setting(async_=True)
		try:
			async with client:
				response = await client.chat.completions.create(
					model=cls.model_name,
					top_p=1, n=n, max_tokens=max_tokens,
					temperature=temperature,
					messages=message,
					)
			texts = [response.choices[i].message.content for i in range(n)]
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return texts
		except Exception as e:
			print(cls.model_name, "has error")
			print(message, e)
			if retry > 0:
				await asyncio.sleep(4)
				return await cls.async_process_n(message, max_tokens=max_tokens, n=n, temperature=temperature, retry=retry-1)
			else:
				raise e

class VLLMOpenAIModel(CustomOpenAIModel):
	token_count = 0
	lock = threading.Lock()
	@classmethod
	def get_setting(cls):
		pass

	@classmethod
	def process(cls, message, max_tokens=2000, retry=2, temperature=1):
		# check and filter the system message
		# message = [m for m in message if m['role'] != 'system']
		client = cls.get_setting()
		try:  
			response = client.chat.completions.create(
				model=cls.model_name,
				top_p=1, n=1, max_tokens=max_tokens,
				temperature=temperature,
				messages=message,
				)
			text = response.choices[0].message.content
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return text
		except Exception as e:
			print(cls.model_name, "has error")
			print(message, e)
			sleep(4)
			if retry > 0:
				text = cls.process(message, max_tokens, retry-1, temperature=temperature)
				return text
			else:
				raise e

	@classmethod
	async def async_process(cls, message, max_tokens=2000, retry=2, temperature=1):
		# check and filter the system message
		message = [m for m in message if m['role'] != 'system']
		
		client = cls.get_setting(async_=True)
		try:
			async with client:
				response = await client.chat.completions.create(
					model=cls.model_name,
					top_p=1, n=1, max_tokens=max_tokens,
					temperature=temperature,
					messages=message,
					)
			text = response.choices[0].message.content
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return text
		except Exception as e:
			print(cls.model_name, "has error")            
			print(message, e)
			if retry > 0:
				await asyncio.sleep(4)
				text = await cls.async_process(message, max_tokens, retry=retry-1, temperature=temperature)
				return text
			else:
				raise e
	
	@classmethod
	async def async_process_n(cls, message, max_tokens=2000, n=1, temperature=1, retry=2):
		message = [m for m in message if m['role'] != 'system']
		client = cls.get_setting(async_=True)
		try:
			async with client:
				response = await client.chat.completions.create(
					model=cls.model_name,
					top_p=1, n=n, max_tokens=max_tokens,
					temperature=temperature,
					messages=message,
					)
			texts = [response.choices[i].message.content for i in range(n)]
			with cls.lock:
				cls.token_count += response.usage.total_tokens
			return texts
		except Exception as e:
			print(cls.model_name, "has error")
			print(message, e)
			if retry > 0:
				await asyncio.sleep(4)
				return await cls.async_process_n(message, max_tokens, n=n, temperature=temperature, retry=retry-1)
			else:
				raise e

class Mistral7B(VLLMOpenAIModel):
	token_count = 0
	lock = threading.Lock()
	model_name = MISTRAL_VLLM_NAME
	port = MISTRAL_VLLM_PORT
	@classmethod
	def get_setting(cls, async_=False):
		api_key = "123"
		api_base = f"http://localhost:{cls.port}/v1"
		if async_:
			client = AsyncOpenAI(api_key=api_key, base_url=api_base)
		else:
			client = OpenAI(api_key=api_key, base_url=api_base)
		return client

class Qwen7B(VLLMOpenAIModel):
	token_count = 0
	lock = threading.Lock()
	model_name = QWEN_VLLM_NAME
	port = QWEN_VLLM_PORT
	@classmethod
	def get_setting(cls, async_=False):
		api_key = "123"
		api_base = f"http://localhost:{cls.port}/v1"
		if async_:
			client = AsyncOpenAI(api_key=api_key, base_url=api_base)
		else:
			client = OpenAI(api_key=api_key, base_url=api_base)
		return client


class GenericAPIModel(CustomOpenAIModel):
	"""
	Flexible model class for any OpenAI-compatible API endpoint.
	Configured via config.GENERIC_API_MODELS dict.
	Supports custom auth headers (e.g., for MiMo API).
	"""
	token_count = 0
	lock = threading.Lock()
	# These will be set dynamically during registration
	model_name = ""
	_base_url = ""
	_api_key = ""
	_auth_header = ""  # custom auth header value (e.g., for MiMo)

	@classmethod
	def get_setting(cls, async_=False):
		# Use auth_header as api_key if available (sent as Authorization: Bearer <key>)
		api_key = cls._auth_header if cls._auth_header else cls._api_key
		if async_:
			client = AsyncOpenAI(api_key=api_key, base_url=cls._base_url)
		else:
			client = OpenAI(api_key=api_key, base_url=cls._base_url)
		return client


def _register_generic_models():
	"""
	Dynamically register models from config.GENERIC_API_MODELS into PRODUCT_MAP.
	This allows adding new closed-source models without modifying this file.
	"""
	for display_name, model_cfg in GENERIC_API_MODELS.items():
		if display_name in PRODUCT_MAP:
			continue  # skip if already registered
		api_key = os.getenv(model_cfg.get("api_key_env", ""), model_cfg.get("api_key_env", ""))
		base_url = model_cfg.get("base_url", "")
		model_id = model_cfg.get("model_id", display_name)
		auth_header = model_cfg.get("auth_header", "")

		# Create a new class dynamically
		new_class = type(
			f"Generic_{display_name.replace('-', '_').replace('.', '_')}",
			(GenericAPIModel,),
			{
				"model_name": model_id,
				"_base_url": base_url,
				"_api_key": api_key,
				"_auth_header": auth_header,
				"token_count": 0,
				"lock": threading.Lock(),
			}
		)
		PRODUCT_MAP[display_name] = new_class
		chat_models.append(display_name)
		print(f"Registered generic API model: {display_name} -> {model_id}")


"""
Factory Method
"""
PRODUCT_MAP = {
	"GPT-4o-Mini": GPT4OMini,
	"GPT-4o": GPT4O,
	'Mistral-7B-Instruct-v0.3': Mistral7B,
	'Qwen2.5-7B-Instruct': Qwen7B
}
chat_models = ['GPT-4o-Mini', 'GPT-4o', "Qwen2.5-7B-Instruct", 'Mistral-7B-Instruct-v0.3']
api_models = chat_models

# Register any generic API models defined in config
_register_generic_models()

class LLMFactory:

	@classmethod
	def completion(cls, prompt, model_name, max_tokens=1000, temperature=1.0, **kwargs):
		if model_name in PRODUCT_MAP:
			product = PRODUCT_MAP[model_name]
			if temperature is not None:
				return product.process([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, **kwargs)
			else:
				return product.process([{"role": "user", "content": prompt}], max_tokens=max_tokens, **kwargs)
		else:
			raise NotImplementedError(f"{model_name} Not implemented yet")
	@classmethod
	def process(cls, message, model_name, max_tokens=1000, temperature=1.0, **kwargs):
		if model_name in PRODUCT_MAP.keys():
			product = PRODUCT_MAP[model_name]
			if temperature is not None:
				return product.process(message=message, max_tokens=max_tokens, temperature=temperature, **kwargs)
			else:
				return product.process(message=message, max_tokens=max_tokens, **kwargs)
		else:
			raise NotImplementedError(f"{model_name} Not implemented yet")
	@classmethod
	async def async_process(cls, message, model_name, max_tokens=1000, temperature=1.0, **kwargs):
		try:
			if model_name in PRODUCT_MAP.keys():
				product = PRODUCT_MAP[model_name]
				if temperature is not None:
					return await product.async_process(message=message, max_tokens=max_tokens, temperature=temperature, **kwargs)
				else:
					return await product.async_process(message=message, max_tokens=max_tokens, **kwargs)
			else:
				raise NotImplementedError(f"{model_name} Not implemented yet")
		except Exception as e:
			print("using model:", model_name, message)
			print("async_process error:", e)
			raise  
	
	@classmethod
	async def process_n(cls, message, model_name, max_tokens=1000, n=1, temperature=1.0, **kwargs):
		try:
			if model_name in PRODUCT_MAP.keys():
				product = PRODUCT_MAP[model_name]
				# is product has async_process_n method
				if hasattr(product, "async_process_n"):
					if temperature is not None:
						return await product.async_process_n(message=message, max_tokens=max_tokens, n=n, temperature=temperature, **kwargs)
					else:
						return await product.async_process_n(message=message, max_tokens=max_tokens, n=n, **kwargs)
				else:
					if temperature is not None:
						return await asyncio.gather(*[product.async_process(message=message, max_tokens=max_tokens, temperature=temperature, **kwargs) for i in range(n)])
					else:
						return await asyncio.gather(*[product.async_process(message=message, max_tokens=max_tokens, **kwargs) for i in range(n)])
			else:
				raise NotImplementedError(f"{model_name} Not implemented yet")
		except Exception as e:
			print("using model:", model_name, message)
			print("async_process error:", e)
			raise

	@classmethod
	async def gather_multiple_async_messages(cls, messages, model_name, **kwargs):
		"""input a list of messages, return a list of responses"""
		return await asyncio.gather(*[cls.async_process(message, model_name, **kwargs) for message in messages])

	@ classmethod
	async def gather_multiple_async_models(cls, message, model_names, **kwargs):
		"""input a message, return a list of responses from different models"""
		return await asyncio.gather(*[cls.async_process(message, model_name, **kwargs) for model_name in model_names])
	
	@classmethod
	async def gather_multiple_async_models_n(cls, message, model_names, n=1, **kwargs):
		return await asyncio.gather(*[cls.process_n(message, model_name,  n=n, **kwargs) for model_name in model_names])
	
	@classmethod
	def gather_multiple_messages(cls, messages, model_name,  **kwargs):
		return asyncio.run(cls.gather_multiple_async_messages(messages, model_name, **kwargs))
	
	@classmethod
	def gather_multiple_models(cls, message, model_names, **kwargs):
		return asyncio.run(cls.gather_multiple_async_models(message, model_names,  **kwargs))
	
	@classmethod
	def gather_multiple_messages_models(cls, messages, model_names, **kwargs):
		return asyncio.run(cls.gather_multiple_async_messages_models(messages, model_names, **kwargs))
	
	@classmethod
	def gather_multiple_models_n(cls, message, model_names,  n=1, **kwargs):
		return asyncio.run(cls.gather_multiple_async_models_n(message, model_names, n=n, **kwargs))

	@classmethod
	async def gather_multiple_async_messages_n(cls, messages, model_name, n=1, **kwargs):
		"""input a list of messages, return a list of n responses per message"""
		return await asyncio.gather(*[cls.process_n(message, model_name, n=n, **kwargs) for message in messages])

	@classmethod
	def gather_multiple_messages_n(cls, messages, model_name, n=1, **kwargs):
		return asyncio.run(cls.gather_multiple_async_messages_n(messages, model_name, n=n, **kwargs))

	@classmethod
	def print_all_token_count(cls):
		res_text = ""
		for model_name in PRODUCT_MAP.keys():
			product = PRODUCT_MAP[model_name]
			print(model_name, product.token_count)
			tmp = product.token_count / 1e6
			res_text += f"{model_name}: {tmp}\n"
		return res_text

	@classmethod
	def print_token_count(cls, model_name):
		product = PRODUCT_MAP[model_name]
		return product.token_count / 1e6
	@classmethod
	def test_all_models(cls, model_names):
		message = [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Who won the world series in 1924?"},
		]
		for model_name in PRODUCT_MAP.keys():
			if model_name in model_names:
				try:
					print(model_name, cls.process(message, model_name, max_tokens=50))
				except Exception as e:
					print(model_name, "error:", e)

if __name__ == "__main__":
	# Test
	message = [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Who won the world series in 1980?"},
	]
	models = ['GPT-4o']
	print(LLMFactory.gather_multiple_models_n(message, models, max_tokens=100, n=5))
