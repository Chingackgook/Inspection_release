from __future__ import annotations
from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.gpt_engineer import *
exe = Executor('gpt_engineer', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Union
import backoff
import openai
import pyperclip
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain.schema import messages_from_dict
from langchain.schema import messages_to_dict
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from gpt_engineer.core.token_usage import TokenUsageLog

Message = Union[AIMessage, HumanMessage, SystemMessage]
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AI:

    def __init__(self, model_name='gpt-4-turbo', temperature=0.1, azure_endpoint=None, streaming=True, vision=False):
        self.temperature = temperature
        self.azure_endpoint = azure_endpoint
        self.model_name = model_name
        self.streaming = streaming
        self.vision = 'vision-preview' in model_name or ('gpt-4-turbo' in model_name and 'preview' not in model_name) or 'claude' in model_name
        self.llm = self._create_chat_model()
        self.token_usage_log = TokenUsageLog(model_name)
        logger.debug(f'Using model {self.model_name}')

    def start(self, system: str, user: Any, *, step_name: str) -> List[Message]:
        messages: List[Message] = [SystemMessage(content=system), HumanMessage(content=user)]
        return self.next(messages, step_name=step_name)

    def _extract_content(self, content):
        if isinstance(content, str):
            return content
        elif isinstance(content, list) and content and ('text' in content[0]):
            return content[0]['text']
        else:
            return ''

    def _collapse_text_messages(self, messages: List[Message]):
        collapsed_messages = []
        if not messages:
            return collapsed_messages
        previous_message = messages[0]
        combined_content = self._extract_content(previous_message.content)
        for current_message in messages[1:]:
            if current_message.type == previous_message.type:
                combined_content += '\n\n' + self._extract_content(current_message.content)
            else:
                collapsed_messages.append(previous_message.__class__(content=combined_content))
                previous_message = current_message
                combined_content = self._extract_content(current_message.content)
        collapsed_messages.append(previous_message.__class__(content=combined_content))
        return collapsed_messages

    def next(self, messages: List[Message], prompt: Optional[str]=None, *, step_name: str) -> List[Message]:
        if prompt:
            messages.append(HumanMessage(content=prompt))
        logger.debug('Creating a new chat completion: %s', '\n'.join([m.pretty_repr() for m in messages]))
        if not self.vision:
            messages = self._collapse_text_messages(messages)
        response = self.backoff_inference(messages)
        self.token_usage_log.update_log(messages=messages, answer=response.content, step_name=step_name)
        messages.append(response)
        logger.debug(f'Chat completion finished: {messages}')
        return messages

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=7, max_time=45)
    def backoff_inference(self, messages):
        return self.llm.invoke(messages)

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        data = json.loads(jsondictstr)
        prevalidated_data = [{**item, 'tools': {**item.get('tools', {}), 'is_chunk': False}} for item in data]
        return list(messages_from_dict(prevalidated_data))

    def _create_chat_model(self) -> BaseChatModel:
        if self.azure_endpoint:
            return AzureChatOpenAI(azure_endpoint=self.azure_endpoint, openai_api_version=os.getenv('OPENAI_API_VERSION', '2024-05-01-preview'), deployment_name=self.model_name, openai_api_type='azure', streaming=self.streaming, callbacks=[StreamingStdOutCallbackHandler()])
        elif 'claude' in self.model_name:
            return ChatAnthropic(model=self.model_name, temperature=self.temperature, callbacks=[StreamingStdOutCallbackHandler()], streaming=self.streaming, max_tokens_to_sample=4096)
        elif self.vision:
            return ChatOpenAI(model=self.model_name, temperature=self.temperature, streaming=self.streaming, callbacks=[StreamingStdOutCallbackHandler()], max_tokens=4096)
        else:
            return ChatOpenAI(model=self.model_name, temperature=self.temperature, streaming=self.streaming, callbacks=[StreamingStdOutCallbackHandler()])

def serialize_messages(messages: List[Message]) -> str:
    return AI.serialize_messages(messages)

class ClipboardAI(AI):

    def __init__(self, **_):
        self.vision = False
        self.token_usage_log = TokenUsageLog('clipboard_llm')

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        return '\n\n'.join([f'{m.type}:\n{m.content}' for m in messages])

    @staticmethod
    def multiline_input():
        return 'Sample response from user input.'

    def next(self, messages: List[Message], prompt: Optional[str]=None, *, step_name: str) -> List[Message]:
        if prompt:
            messages.append(HumanMessage(content=prompt))
        logger.debug(f'Creating a new chat completion: {messages}')
        msgs = self.serialize_messages(messages)
        
        # Output file path replaced with FILE_RECORD_PATH
        pyperclip.copy(msgs)
        Path(FILE_RECORD_PATH / 'clipboard.txt').write_text(msgs)  # Changed to use FILE_RECORD_PATH
        print('Messages copied to clipboard and written to clipboard.txt,', len(msgs), 'characters in total')
        
        response = self.multiline_input()
        messages.append(AIMessage(content=response))
        logger.debug(f'Chat completion finished: {messages}')
        return messages

# Parts that may need manual modification:
ai_instance = exe.create_interface_objects(interface_class_name='AI', model_name='gpt-4-turbo', temperature=0.1, azure_endpoint=None, streaming=True, vision=False)  # end
system_message = 'You are a helpful assistant.'  # Parts that may need manual modification: 
user_message = 'What can you do for me?'  # end
step_name = 'initial_query'  # end
messages = exe.run('start', system=system_message, user=user_message, step_name=step_name)
print('Initial messages:', messages)
messages = exe.run('next', messages=messages, prompt='Sample prompt', step_name=step_name)  # Fixed undefined 'prompt'
