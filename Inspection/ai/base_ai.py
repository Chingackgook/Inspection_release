from importlib import import_module
from datetime import datetime
from typing import List, Dict
from Inspection.utils.path_manager import AI_CHAT_RECORD_PATH
from Inspection.utils.config import CONFIG
from Inspection.ai.ai_adapters.ai_base_adapter import BaseAIAdapter
import os
from collections import defaultdict

# 添加内存缓存
_token_cache = defaultdict(
    lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
)


def record_log(id: str, prompt: str, response: str, token_usage: Dict = None):
    # 为了防止进一步损失，这个代码只要出错，则直接exit

    try:
        if not os.path.exists(AI_CHAT_RECORD_PATH):
            os.makedirs(AI_CHAT_RECORD_PATH)
        log_file_name = id + ".log"
        log_file_path = os.path.join(AI_CHAT_RECORD_PATH, log_file_name)
        # 使用内存缓存避免频繁读取文件
        if token_usage:
            _token_cache[id]["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            _token_cache[id]["completion_tokens"] += token_usage.get("completion_tokens", 0)
            _token_cache[id]["total_tokens"] += token_usage.get("total_tokens", 0)

        # 直接写入,避免读取整个文件
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{datetime.now()} - Conversation Record:\n")
            f.write(f"{datetime.now()} - User: {prompt}\n")
            f.write(f"{datetime.now()} - AI: {response}\n")
            f.write("\n\nToken Usage Information:\n")
            f.write(f"Current conversation token usage:\n")
            f.write(f"prompt_tokens: {token_usage.get('prompt_tokens', 0)}\n")
            f.write(f"completion_tokens: {token_usage.get('completion_tokens', 0)}\n")
            f.write(f"total_tokens: {token_usage.get('total_tokens', 0)}\n")
            f.write("\n\nCumulative token usage:\n")
            f.write(f"prompt_total_tokens: {_token_cache[id]['prompt_tokens']}\n")
            f.write(
                f"completion_total_tokens: {_token_cache[id]['completion_tokens']}\n"
            )
            f.write(f"total_total_tokens: {_token_cache[id]['total_tokens']}\n")
        return True
    except Exception as e:
        print(f"[INS_FATAL] Failed to record log: {e}")
        with open("fatal_error.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} - Failed to record log: {e}\n")
        os._exit(1)


# 创建一个类，传入参数控制为apikey和服务提供厂商和模型，以及可能的代理url
class BaseAI:

    def __init__(self, id: str = "" ):
        self.id: str = (
            id  # 这里可以传入一个id，作为标识，具有相同id的AI会将记录存储在同一个文件中
        )
        self.provider = CONFIG.get("provider", "OpenAI")  # 默认使用OpenAI
        self.history: List[Dict] = []
        self.adapter : BaseAIAdapter  = self._load_adapter()
        self.save = CONFIG.get("ai_logger", True)

    def generate_text(
        self, prompt: str, temperature: int = -1, max_tokens: int = -1 , **kwargs
    ) -> str:
        """用户输入文本，返回生成文本"""
        if temperature == -1:
            temperature = CONFIG.get("temperature", 0.3)
        if max_tokens == -1:
            max_tokens = CONFIG.get("max_tokens", 16000)
        max_retries = CONFIG.get("max_retries", 5)

        additional_params = {}
        if CONFIG.get('additional_ai_params', None) is not None:
            additional_params = CONFIG['additional_ai_params']
        for key, value in kwargs.items():
            additional_params[key] = value

        self.history.append({"role": "user", "content": prompt})
        response, token_dic = self.adapter.generate_text(
            self.history, temperature, max_tokens , max_retries , **additional_params
        )
        self.history.append({"role": "assistant", "content": response})
        if self.save:
            record_log(self.id, prompt, response, token_dic)
        return response
    
    def print_AI_info(self):
        """Print AI information"""
        print(f"AI ID: {self.id}")
        print(f"Provider: {self.provider}")
        print(f"Model: {self.adapter._model}")
        print(f"Base URL: {self.adapter._base_url}")
        print(
            f"API Key: {self.adapter._api_key.replace(self.adapter._api_key[4:-4],'****')}"
        )
        print(f"Current History Length: {len(self.history)}")
        # 返回一个字典，返回这些信息
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.adapter._model,
            "base_url": self.adapter._base_url,
            "api_key": self.adapter._api_key,
            "current_history_length": len(self.history),
        }

    def add_history(self, role: str, content: str):
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        if role == "system":
            self.history.insert(0, {"role": role, "content": content})
        else:
            self.history.append({"role": role, "content": content})

    def print_history(self):
        """Print conversation history"""
        for i in self.history:
            print(f"{i['role']}：{i['content']}")

    def clear_history(self):
        """Clear conversation history"""
        self.history = []

    def copy(self , with_memory: bool = False):
        """Copy current AI without conversation history and return a new instance"""
        new_ai = BaseAI(id=self.id)
        new_ai.provider = self.provider
        new_ai.adapter = new_ai._load_adapter()
        history = []
        if with_memory:
            for i in self.history:
                history.append(i)
        new_ai.history = history
        new_ai.save = self.save
        return new_ai

    def _load_adapter(self):
        """Dynamically load service provider adapter"""
        if self.provider is None:
            raise ValueError("Provider not specified in CONFIG")
        if self.provider == 'OpenAI' or self.provider == 'openAI' or self.provider == 'openai':
            from Inspection.ai.ai_adapters.openai_adapter import OpenAIAdapter
            return OpenAIAdapter()
        elif self.provider == 'Ollama' or self.provider == 'ollama':
            from Inspection.ai.ai_adapters.ollama_adapter import OllamaAdapter
            return OllamaAdapter()
        else:
            try:
                module = import_module(f"Inspection.ai.ai_adapters.{self.provider}_adapter")
                adapter_class = getattr(module, f"{self.provider}Adapter")
                return adapter_class()
            except ModuleNotFoundError as e:
                raise ValueError("Adapter module not found") from e
            except AttributeError as e:
                raise ValueError("Class missing in module") from e


if __name__ == "__main__":
    CONFIG['provider'] = 'ollama'  # 设置使用的AI服务提供商
    CONFIG['model'] = 'gemma:2b'  # 设置使用的模型名称
    CONFIG['base_url'] = 'http://localhost:11434'  # 设置服务的基础URL
    adapter = BaseAI()
    print("AI Information:")
    adapter.print_AI_info()
    while True:
        user_input = input("Please enter: ")
        response = adapter.generate_text(user_input)
        adapter.print_history()
        print(response)
