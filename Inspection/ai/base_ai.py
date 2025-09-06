from importlib import import_module
from datetime import datetime
from typing import List, Dict
from Inspection.utils.path_manager import AI_CHAT_RECORD_PATH
from Inspection.utils.config import CONFIG
import os
from collections import defaultdict

# 添加内存缓存
_token_cache = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

def record_log(id: str, prompt: str, response: str, token_usage: Dict = None):
    """记录日志 - 优化版本"""
    if not os.path.exists(AI_CHAT_RECORD_PATH):
        os.makedirs(AI_CHAT_RECORD_PATH)
    log_file_name = id + ".log"
    log_file_path = os.path.join(AI_CHAT_RECORD_PATH, log_file_name)
    # 使用内存缓存避免频繁读取文件
    if token_usage:
        _token_cache[id]["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
        _token_cache[id]["completion_tokens"] += token_usage.get("completion_tokens", 0)
        _token_cache[id]["total_tokens"] += token_usage.get("total_tokens", 0)
    
    try:
        # 直接写入，避免读取整个文件
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n{datetime.now()} - 对话记录：\n")
            f.write(f"{datetime.now()} - 用户：{prompt}\n")
            f.write(f"{datetime.now()} - AI：{response}\n")
            f.write("\n\nToken使用量信息:\n")
            f.write(f"本次对话使用的token数量:\n")
            f.write(f"prompt_tokens: {token_usage.get('prompt_tokens', 0)}\n")
            f.write(f"completion_tokens: {token_usage.get('completion_tokens', 0)}\n")
            f.write(f"total_tokens: {token_usage.get('total_tokens', 0)}\n")
            f.write("\n\n累计token使用量:\n")
            f.write(f"prompt_total_tokens: {_token_cache[id]['prompt_tokens']}\n")
            f.write(f"completion_total_tokens: {_token_cache[id]['completion_tokens']}\n")
            f.write(f"total_total_tokens: {_token_cache[id]['total_tokens']}\n")
        return True
    except Exception as e:
        print(f"[INS_WARN]Failed to record log: {e}")
        return False


#创建一个类，传入参数控制为apikey和服务提供厂商和模型，以及可能的代理url
class BaseAI():

    def __init__(self,id:str = ""):
        self.id : str = id # 这里可以传入一个id，作为标识，具有相同id的AI会将记录存储在同一个文件中
        self.provider = CONFIG.get("provider", "OpenAI")  # 默认使用OpenAI
        self.history: List[Dict] = []#[{"role": "user", "content": "你是一个人工智能助手"}]
        self.max_history = 20
        self.adapter = self._load_adapter(self.provider)
        self.save=CONFIG.get("ai_Logger",True)

    def generate_text(self, prompt: str,temperature: int=-1,max_tokens: int=16000) -> str:
        """用户输入文本，返回生成文本"""
        if temperature == -1:
            temperature = CONFIG.get("temperature",0.3)
        response , token_dic = self.adapter.generate_text(self.history,prompt,temperature,max_tokens)
        self.check_history()
        if self.save:
            # 记录对话
            record_log(self.id,prompt,response,token_dic)
        return response

    def generate_image(self, prompt: str, filepath: List,temperature: int=0,max_tokens: int=4096) -> str:
        """用户输入包含图片"""
        response = self.adapter.generate_image(self.history,prompt,filepath,temperature,max_tokens)
        self.check_history()
        if self.save:
            # 记录对话
            record_log(self.id,prompt,response)
        return response

    def generate_audio(self, prompt :str,filepath:str,temperature: int=0,max_tokens: int=4096) -> bytes:
        """用户输入包含音频"""
        response = self.adapter.generate_audio(self.history,prompt,filepath,temperature,max_tokens)
        self.check_history()
        return response
        
    def add_history(self, role: str, content: str):
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        # When exceeding limit, prioritize removing earliest non-system conversations
        if len(self.history) >= self.max_history:
            for idx, item in enumerate(self.history):
                if item.get("role") != "system":
                    self.history.pop(idx)
                    break
            else:
                # Only remove earliest system message if all are system messages
                self.history.pop(0)
        if role == 'system':
            self.history.insert(0, {"role": role, "content": content})
        else:
            self.history.append({"role": role, "content": content})

    
    def check_history(self):
        """Check conversation history length, delete earliest record if exceeding maximum length"""
        while len(self.history) >= self.max_history:
            self.history.pop(0)

    
    def print_history(self):
        """Print conversation history"""
        for i in self.history:
            print(f"{i['role']}：{i['content']}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def copy(self):
        """Copy current AI without conversation history"""
        return BaseAI(
            id=self.id,
        )
    
    def copy_with_memory(self):
        """Copy current AI including conversation history"""
        new_ai = BaseAI(
            id=self.id,
        )
        history = []
        for i in self.history:
            history.append(i)
        new_ai.history = history
        return new_ai

    def _load_adapter(self, provider: str):
        """Dynamically load service provider adapter"""
        try:
            module = import_module(f"Inspection.ai.ai_adapters.{provider}_adapter")
            adapter_class = getattr(module, f"{provider}Adapter")
            # 3. Initialize adapter instance
            return adapter_class()
        except ModuleNotFoundError as e:
            raise ValueError("Adapter module not found") from e
        except AttributeError as e:
            raise ValueError("Class missing in module") from e





if __name__ == "__main__":
    adapter = BaseAI()
    while True:
        user_input = input("Please enter: ")
        response = adapter.generate_text(user_input)
        print(response)
