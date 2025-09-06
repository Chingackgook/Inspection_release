from __future__ import annotations
from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.ChuanhuChatGPT import *
exe = Executor('ChuanhuChatGPT', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
# Import the existing package
import logging
import os
import colorama
import commentjson as cjson
from modules import config
from modules.index_func import *
from modules.presets import *
from modules.utils import *
from modules.models.base_model import BaseLLMModel
from modules.models.base_model import ModelType
from modules.models.OpenAIVision import OpenAIVisionClient
import traceback
from modules.models.OpenAIVision import OpenAIVisionClient
from modules.models.OpenAIInstruct import OpenAI_Instruct_Client
from modules.models.ChatGLM import ChatGLM_Client
from modules.models.Groq import Groq_Client
from modules.models.LLaMA import LLaMA_Client
from modules.models.XMChat import XMChat
from modules.models.StableLM import StableLM_Client
from modules.models.MOSS import MOSS_Client
from modules.models.inspurai import Yuan_Client
from modules.models.minimax import MiniMax_Client
from modules.models.ChuanhuAgent import ChuanhuAgent_Client
from modules.models.GooglePaLM import Google_PaLM_Client
from modules.models.GoogleGemini import GoogleGeminiClient
from modules.models.Azure import Azure_OpenAI_Client
from modules.models.midjourney import Midjourney_Client
from modules.models.spark import Spark_Client
from modules.models.Claude import Claude_Client
from modules.models.Qwen import Qwen_Client
from modules.models.ERNIE import ERNIE_Client
from modules.models.DALLE3 import OpenAI_DALLE3_Client
from modules.models.Ollama import OllamaClient
from modules.models.GoogleGemma import GoogleGemmaClient
# end

import logging
import os
import colorama
import commentjson as cjson

def i18n(text):
    return text

def hide_middle_chars(key):
    return key[:2] + '*' * (len(key) - 4) + key[-2:]

def setPlaceholder(model):
    return 'Type your message here...'

class gr:

    @staticmethod
    def update(label=None, placeholder=None):
        return f'Updated with label: {label}, placeholder: {placeholder}'

    @staticmethod
    def Dropdown(choices=None, visible=True):
        return f'Dropdown with choices: {choices}, visible: {visible}'

def get_model(model_name, lora_model_path=None, access_key=None, temperature=None, top_p=None, system_prompt=None, user_name='', original_model=None) -> BaseLLMModel:
    msg = i18n('模型设置为了：') + f' {model_name}'
    model_type = ModelType.get_type(model_name)
    lora_selector_visibility = False
    lora_choices = ['No LoRA']
    dont_change_lora_selector = False
    model = original_model
    try:
        if model_type == 'OpenAI':
            logging.info(f'正在加载 OpenAI 模型: {model_name}')
            access_key = 'mock_openai_api_key'
            model = OpenAIVisionClient(model_name, api_key=access_key, user_name=user_name)
        else:
            raise ValueError(f'Unimplemented model type: {model_type}')
        logging.info(msg)
    except Exception as e:
        traceback.print_exc()
        msg = f'Error: {e}'
    modelDescription = i18n(model.description)
    presudo_key = hide_middle_chars(access_key)
    if original_model is not None and model is not None:
        model.history = original_model.history
    if dont_change_lora_selector:
        return (model, msg, gr.update(label=model_name, placeholder=setPlaceholder(model=model)), gr.update(), access_key, presudo_key, modelDescription)
    else:
        return (model, msg, gr.update(label=model_name, placeholder=setPlaceholder(model=model)), gr.Dropdown(choices=lora_choices, visible=lora_selector_visibility), access_key, presudo_key, modelDescription)

logging.basicConfig(level=logging.DEBUG)
openai_api_key = os.getenv('OPENAI_API_KEY', '')
client = exe.create_interface_objects(interface_class_name='OpenAIVisionClient', model_name='GPT-4o-mini', api_key=openai_api_key, user_name='')
chatbot = []
stream = False

# Testing billing functionality
logging.info(colorama.Back.GREEN + '测试账单功能' + colorama.Back.RESET)
logging.info(exe.run('billing_info'))

# Testing Q&A functionality
logging.info(colorama.Back.GREEN + '测试问答' + colorama.Back.RESET)
question = '巴黎是中国的首都吗？'
for i in exe.run('predict', inputs=question, chatbot=chatbot, reply_language='中文', should_check_token_count=True):
    logging.info(i)

# Testing memory functionality
logging.info(colorama.Back.GREEN + '测试记忆力' + colorama.Back.RESET)
question = '我刚刚问了你什么问题？'
for i in exe.run('predict', inputs=question, chatbot=chatbot, reply_language='中文', should_check_token_count=True):
    logging.info(i)

# Testing retry functionality
logging.info(colorama.Back.GREEN + '测试重试功能' + colorama.Back.RESET)
for i in exe.run('retry', chatbot=chatbot, reply_language='中文'):
    logging.info(i)
