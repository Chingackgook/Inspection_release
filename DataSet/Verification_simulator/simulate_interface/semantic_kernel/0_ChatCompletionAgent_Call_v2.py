from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.semantic_kernel import *
exe = Executor('semantic_kernel', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

import asyncio
from typing import Annotated
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents import ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents import FunctionCallContent
from semantic_kernel.contents import FunctionResultContent
from semantic_kernel.filters import AutoFunctionInvocationContext
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel
# add
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
# add end

async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
    await next(context)
    if context.function.plugin_name == 'menu':
        context.terminate = True

class MenuPlugin:

    @kernel_function(description='Provides a list of specials from the menu.')
    def get_specials(self) -> Annotated[str, 'Returns the specials from the menu.']:
        return '\n        Special Soup: Clam Chowder\n        Special Salad: Cobb Salad\n        Special Drink: Chai Tea\n        '

    @kernel_function(description='Provides the price of the requested menu item.')
    def get_item_price(self, menu_item: Annotated[str, 'The name of the menu item.']) -> Annotated[str, 'Returns the price of the menu item.']:
        return '$9.99'

def _create_kernel_with_chat_completionand_filter() -> Kernel:
    # add
    kernel = Kernel()
    custom_client = AsyncOpenAI(
        base_url="https://sg.uiuiapi.com/v1",  
        api_key="sk-sss"  
    )
    chat_completion = OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",  # 模型名称
        async_client=custom_client,   # 传入自定义客户端
        service_id="custom_openai"    # 可选服务标识
    )
    kernel.add_service(chat_completion)
    # add end
    # orign code:
    # kernel.add_service(AzureChatCompletion())
    kernel.add_filter('auto_function_invocation', auto_function_invocation_filter)
    kernel.add_plugin(plugin=MenuPlugin(), plugin_name='menu')
    return kernel

def _write_content(content: ChatMessageContent) -> None:
    last_item_type = type(content.items[-1]).__name__ if content.items else '(empty)'
    message_content = ''
    if isinstance(content.items[-1], FunctionCallContent):
        message_content = f'tool request = {content.items[-1].function_name}'
    elif isinstance(content.items[-1], FunctionResultContent):
        message_content = f'function result = {content.items[-1].result}'
    else:
        message_content = str(content.items[-1])
    print(f"[{last_item_type}] {content.role} : '{message_content}'")

async def run_chat():
    # Parts that may need manual modification:
    agent = exe.create_interface_objects(interface_class_name='ChatCompletionAgent', kernel=_create_kernel_with_chat_completionand_filter(), name='Host', instructions='Answer questions about the menu.')
    # end

    thread: ChatHistoryAgentThread = None
    user_inputs = ['Hello', 'What is the special soup?', 'What is the special drink?', 'Thank you']
    for user_input in user_inputs:
        print(f"# User: '{user_input}'")
        response = await exe.run('get_response', messages=user_input, thread=thread)
        thread = response.thread
        _write_content(response)
    print('================================')
    print('CHAT HISTORY')
    print('================================')
    async for message in thread.get_messages():
        _write_content(message)

# Directly run the main logic
asyncio.run(run_chat())
