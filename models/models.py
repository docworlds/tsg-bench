from abc import abstractmethod

from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from openai import OpenAI

from utils.config import load_config, get_config_file_path


class LLM:
    def __init__(self):
        config_path = get_config_file_path()
        self.config = load_config(config_path)

    @abstractmethod
    def invoke(self, text: str):
        raise NotImplementedError


class GPT4o(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.openai = ChatOpenAI(
            api_key=self.config["openai"]["key"],
            model_name="gpt-4o-2024-08-06",
            temperature=temperature,
        )

    def invoke(self, message):
        response = self.openai.invoke(message)
        return response.content.strip()


class GPT4oMini(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.openai = ChatOpenAI(
            api_key=self.config["openai"]["key"],
            model_name="gpt-4o-mini-2024-07-18",
            temperature=temperature,
        )

        self.price = 0

    def invoke(self, message):
        response = self.openai.invoke(message)
        return response.content.strip()


class Claude35Sonnet(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.anthropic = ChatAnthropic(
            api_key=self.config["anthropic"]["key"],
            model_name="claude-3-5-sonnet-20241022",
            temperature=temperature,
        )

    def invoke(self, message):
        response = self.anthropic.invoke(message)
        return response.content.strip()


class Claude35Haiku(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.anthropic = ChatAnthropic(
            api_key=self.config["anthropic"]["key"],
            model_name="claude-3-5-haiku-20241022",
            temperature=temperature,
        )

    def invoke(self, message):
        response = self.anthropic.invoke(message)
        try:
            return response.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            print(f"Error: {message}")
            return ""


class MetaLlama(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "meta-llama/llama-3.3-70b-instruct"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()


class Qwen(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "qwen/qwen-2.5-72b-instruct"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()


class DeepSeek(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "deepseek/deepseek-chat"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()


class MistralMixtral(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "mistralai/mixtral-8x22b-instruct"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()


class MistralLarge(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "mistralai/mistral-large-2411"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()


class Qwen7B(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "qwen/qwen-2.5-7b-instruct"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()


class Mistral7B(LLM):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.config["openrouter"]["key"],
        )
        self.model_name = "mistralai/mistral-7b-instruct"
        self.temperature = temperature

    def invoke(self, message):
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message.content.strip()
