from typing import Dict, Optional, Literal, Any
import os
import json
from abc import ABC, abstractmethod

try:
    from litellm import completion as _litellm_completion
except ImportError:
    _litellm_completion = None


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        pass


class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        from openai import OpenAI
        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found.")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        if response_format is not None:
            # Use json_object mode for broad compatibility
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        self.model = model

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7) -> str:
        if _litellm_completion is None:
            raise ImportError("litellm is required for OllamaController. pip install litellm")
        response = _litellm_completion(
            model="ollama_chat/{}".format(self.model),
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            response_format=response_format,
        )
        return response.choices[0].message.content


class LLMController:
    def __init__(self, backend: str = "openai", model: str = "gpt-4",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key, base_url)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be 'openai' or 'ollama'")

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
