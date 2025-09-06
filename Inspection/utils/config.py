import os
import json

from collections import UserDict
from threading import Lock
from Inspection.utils.path_manager import INSPECTION_DIR

class WriteDict(UserDict):
    def __init__(self, env_key: str, default_dict: dict):
        self.env_key = env_key
        self._lock = Lock()
        super().__init__(default_dict)
        self._write()

    def _write(self):
        with self._lock:
            # 将字典数据序列化为JSON字符串存储到环境变量
            os.environ[self.env_key] = json.dumps(self.data, ensure_ascii=False)

    # 重写修改方法，自动写入环境变量
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._write()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._write()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._write()

    def clear(self):
        super().clear()
        self._write()

    def pop(self, *args, **kwargs):
        result = super().pop(*args, **kwargs)
        self._write()
        return result

    def popitem(self):
        result = super().popitem()
        self._write()
        return result

    def setdefault(self, key, default=None):
        if key not in self.data:
            self[key] = default
        return self[key]

DEFAULT_CONFIG = {
    "path": [],
    "provider": "OpenAI",
    "api_key": "your_openai_api_key_here",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "model_img": "gpt-4o-mini",
    "model_audio": "gpt-4o-mini",

    "temperature": 0.3,

    "simulation_analysis_temprature": 0.5,
    "simulation_generate_code_temprature": 0.3,

    "dumb_analysis_temprature" : 0.3,
    "dumb_generate_code_temprature" : 0.3,

    "adapter_analysis_temprature": 0.3,
    "adapter_generate_code_temprature": 0.3,
    
    "doc_generate_temprature": 0.3,

    "cache_dir": "",
    "dumb_use_simulation": True,
    "dumb_use_v2": True,
    "simulation_use_v2": True,
    "exec_use_subprocess": False,
    "ask": True,
    "ai_Logger": True,
    "auto_suggest": False,
    "force_regenerate": True,
    "record_pkl": True,
    "evaluate_mode": False,

    "/*comment*/": {
        "path":"Represents the path to add custom sys.path for code execution",
        "provider":"AI service provider, supports OpenAI, Google, Mistral, Anthropic, Alibaba, Baidu, Ollama, etc.",
        "api_key":"API key",
        "base_url":"Base URL for the API (required by some providers)",
        "model":"Text model name",
        "model_img":"Image processing model name (not implemented, can input anything)",
        "model_audio":"Audio processing model name (not implemented, can input anything)",
        "temperature":"Temperature for AI-generated content",
        "simulation_analysis_temprature":"Temperature for analysis phase when generating static artifacts",
        "simulation_generate_code_temprature":"Temperature for code generation phase when generating static artifacts",

        "dumb_analysis_temprature":"Temperature for analysis phase in non-intelligent module simulation",
        "dumb_generate_code_temprature":"Temperature for code generation phase in non-intelligent module simulation",   

        "adapter_analysis_temprature":"Temperature for adapter generation analysis phase",
        "adapter_generate_code_temprature":"Temperature for adapter generation code phase",

        "doc_generate_temprature":"Temperature for documentation generation",


        "cache_dir":"Cache directory, uses default cache directory if not set",
        "dumb_use_simulation":"Whether to simulate code execution as baseline",
        "dumb_use_v2":"Whether to use v2 version of non-intelligent module simulation",
        "simulation_use_v2":"Whether to use v2 version of static executable artifact generation",
        "exec_use_subprocess":"Whether to use python -m subprocess when executing code",
        "ask":"Whether to ask user for confirmation when encountering existing code",
        "ai_Logger":"Whether to use AI logging",
        "auto_suggest":"Whether to automatically generate suggestions after simulation execution",
        "force_regenerate":"Whether to force AI to regenerate some content that could use cache",
        "record_pkl":"Whether to record execution results as pkl files",
        "evaluate_mode":"Evaluation mode"
    }
}
CONFIG = None
try:
    with open(os.path.join(INSPECTION_DIR, 'config.json'), 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("[INS_WARN] Configuration file config.json not found, using default configuration and creating config.json in root directory")
    CONFIG = DEFAULT_CONFIG
    with open(os.path.join(INSPECTION_DIR, 'config.json'), 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)

# 使用环境变量存储配置，可在子进程间共享
CONFIG = WriteDict('INSPECTION_CONFIG', CONFIG)


os.environ['OPENAI_API_KEY'] = CONFIG.get('api_key', '')
os.environ['OPENAI_API_BASE'] = CONFIG.get('base_url', '')
os.environ['OPENAI_BASE_URL'] = CONFIG.get('base_url', '')
os.environ['OPENAI_MODEL'] = CONFIG.get('model', 'gpt-4o-mini')
os.environ['OPENAI_MODEL_IMG'] = CONFIG.get('model_img', 'gpt-4o-mini')