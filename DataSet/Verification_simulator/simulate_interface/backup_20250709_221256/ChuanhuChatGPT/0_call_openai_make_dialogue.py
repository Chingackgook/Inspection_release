import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.ChuanhuChatGPT import ENV_DIR
from Inspection.adapters.custom_adapters.ChuanhuChatGPT import *
exe = Executor('ChuanhuChatGPT', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import os
from modules.index_func import *
from modules.presets import *
from modules.models.OpenAIVision import OpenAIVisionClient,BaseLLMModel

def get_model(model_name, lora_model_path=None, access_key=None, temperature=None, top_p=None, system_prompt=None, user_name='', original_model=None) -> BaseLLMModel:
    msg = i18n('模型设置为了：') + f' {model_name}'
    model_type = ModelType.get_type(model_name)
    lora_selector_visibility = False
    lora_choices = ['No LoRA']
    dont_change_lora_selector = False
    if model_type != ModelType.OpenAI:
        config.local_embedding = True
    model = original_model
    try:
        if model_type == ModelType.OpenAIVision or model_type == ModelType.OpenAI:
            logging.info(f'正在加载 OpenAI 模型: {model_name}')
            access_key = os.environ.get('OPENAI_API_KEY', access_key)
            model = OpenAIVisionClient(model_name, api_key=access_key, user_name=user_name)
        elif model_type == ModelType.Unknown:
            raise ValueError(f'Unknown model: {model_name}')
        else:
            raise ValueError(f'Unimplemented model type: {model_type}')
        logging.info(msg)
    except Exception as e:
        traceback.print_exc()
        msg = f'{STANDARD_ERROR_MSG}: {e}'
    modelDescription = i18n(model.description)
    presudo_key = hide_middle_chars(access_key)
    if original_model is not None and model is not None:
        model.history = original_model.history
        model.history_file_path = original_model.history_file_path
        model.system_prompt = original_model.system_prompt
    if dont_change_lora_selector:
        return (model, msg, gr.update(label=model_name, placeholder=setPlaceholder(model=model)), gr.update(), access_key, presudo_key, modelDescription, model.stream)
    else:
        return (model, msg, gr.update(label=model_name, placeholder=setPlaceholder(model=model)), gr.Dropdown(choices=lora_choices, visible=lora_selector_visibility), access_key, presudo_key, modelDescription, model.stream)
model_name = 'GPT-4o-mini'
access_key = 'sk-sss'
user_name = 'test_user'
question = '巴黎是中国的首都吗？'
openai_api_key = 'sk-sss'
logging.basicConfig(level=logging.DEBUG)
exe.create_interface_objects(model_name=model_name, api_key=access_key, user_name=user_name)
chatbot = []
logging.info(colorama.Back.GREEN + '测试账单功能' + colorama.Back.RESET)
billing_info = exe.run('billing_info', **{})
logging.info(billing_info)
logging.info(colorama.Back.GREEN + '测试问答' + colorama.Back.RESET)
for i in exe.run('predict', inputs=question, chatbot=chatbot):
    logging.info(i)
logging.info(f'测试问答后history : {exe.adapter.model.history}')
logging.info(colorama.Back.GREEN + '测试记忆力' + colorama.Back.RESET)
question = '我刚刚问了你什么问题？'
for i in exe.run('predict', inputs=question, chatbot=chatbot):
    logging.info(i)
logging.info(f'测试记忆力后history : {exe.adapter.model.history}')
logging.info(colorama.Back.GREEN + '测试重试功能' + colorama.Back.RESET)
for i in exe.run('retry', chatbot=chatbot):
    logging.info(i)
logging.info(f'重试后history : {exe.adapter.model.history}')