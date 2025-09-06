from collections import UserDict
import json

from Inspection.utils.path_manager import INSPECTION_DIR
import os


class ReadDict(UserDict):
    def __init__(self, env_key: str, default_dict: dict = None):
        self.env_key = env_key
        self.default_dict = default_dict or {}
        # super().__init__(self._read())
    
    def _read(self):
        """从环境变量读取数据"""
        try:
            env_data = os.environ.get(self.env_key)
            if env_data:
                return json.loads(env_data)
            else:
                return self.default_dict.copy()
        except json.JSONDecodeError:
            return self.default_dict.copy()
    
    @property
    def data(self):
        """每次访问 data 属性都从环境变量读取"""
        return self._read()
    
    def __getitem__(self, key):
        data = self._read()
        return data[key]
    
    def __iter__(self):
        return iter(self._read())
    
    def __len__(self):
        return len(self._read())
    
    def __contains__(self, key):
        return key in self._read()
    
    def keys(self):
        return self._read().keys()
    
    def values(self):
        return self._read().values()
    
    def items(self):
        return self._read().items()
    
    def get(self, key, default=None):
        data = self._read()
        return data.get(key, default)
    
    def copy(self):
        return self._read().copy()
    
    def __repr__(self):
        return repr(self._read())
    
    def __str__(self):
        return str(self._read())


default_config = {}
with open(os.path.join(INSPECTION_DIR, 'config.json'), 'r', encoding='utf-8') as f:
    try:
        default_config = json.load(f)
    except json.JSONDecodeError:
        print("[INS_ERR] Configuration file format error, using default configuration")
        default_config = {}

# 主要用于执行模块可能是独立的进程，导致主进程修改全局CONFIG无法同步到子进程
OSENV_CONFIG = ReadDict('INSPECTION_CONFIG', default_config)


os.environ['OPENAI_API_KEY'] = OSENV_CONFIG.get('api_key', '')
os.environ['OPENAI_API_BASE'] = OSENV_CONFIG.get('base_url', '')
os.environ['OPENAI_BASE_URL'] = OSENV_CONFIG.get('base_url', '')
os.environ['OPENAI_MODEL'] = OSENV_CONFIG.get('model', 'gpt-4o-mini')
os.environ['OPENAI_MODEL_IMG'] = OSENV_CONFIG.get('model_img', 'gpt-4o-mini')