
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.composio import ENV_DIR
from Inspection.adapters.custom_adapters.composio import *
exe = Executor('composio','simulation')
FILE_RECORD_PATH = exe.now_record_path


import json
import time
import typing as t
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from composio import ActionType, AppType, TagType
from composio.constants import DEFAULT_ENTITY_ID
from composio.exceptions import InvalidEntityIdError
from composio.tools import ComposioToolSet as BaseComposioToolSet
from composio.tools.toolset import ProcessorsType
from composio_openai import Action
from composio.tools.schema import SchemaType, OpenAISchema

class ComposioToolSet(
    BaseComposioToolSet,
    runtime="openai",
    description_char_limit=1024,
    action_name_char_limit=64,
):
    def __init__(
        self,
    ):
        super().__init__()
        exe.create_interface_objects()
        self.schema = SchemaType.OPENAI

    def handle_tool_calls(
        self,
        response: ChatCompletion,
        entity_id: t.Optional[str] = None,
        check_requested_actions: bool = True,
    ) -> t.List[t.Dict]:
        entity_id = self.validate_entity_id(entity_id or self.entity_id)
        outputs = []
        if response.choices:
            for choice in response.choices:
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        outputs.append(
                            self.execute_tool_call(
                                tool_call=tool_call,
                                entity_id=entity_id or self.entity_id,
                                check_requested_actions=check_requested_actions,
                            )
                        )
        return outputs
    
    def get_actions(
        self, actions: t.Sequence[ActionType]
    ) -> t.List[ChatCompletionToolParam]:
        return self.get_tools(actions=actions)

    def get_tools(
        self,
        actions: t.Optional[t.Sequence[ActionType]] = None,
        apps: t.Optional[t.Sequence[AppType]] = None,
        tags: t.Optional[t.List[TagType]] = None,
        *,
        processors: t.Optional[ProcessorsType] = None,
        check_connected_accounts: bool = True,
    ) -> t.List[ChatCompletionToolParam]:
        """
        Get composio tools wrapped as OpenAI `ChatCompletionToolParam` objects.

        :param actions: List of actions to wrap
        :param apps: List of apps to wrap
        :param tags: Filter the apps by given tags

        :return: Composio tools wrapped as `ChatCompletionToolParam` objects
        """
        self.validate_tools(apps=apps, actions=actions, tags=tags)
        if processors is not None:
            self._processor_helpers.merge_processors(processors)

        return [
            ChatCompletionToolParam(  # type: ignore
                **t.cast(
                    OpenAISchema,
                    self.schema.format(
                        schema.model_dump(
                            exclude_none=True,
                        )
                    ),
                ).model_dump()
            )
            for schema in self.get_action_schemas(
                actions=actions,
                apps=apps,
                tags=tags,
                check_connected_accounts=check_connected_accounts,
                _populate_requested=True,
            )
        ]

    def execute_tool_call(
        self,
        tool_call: ChatCompletionMessageToolCall,
        entity_id: t.Optional[str] = None,
        check_requested_actions: bool = True,
    ) -> t.Dict:
        # Replace the execute_action call with exe.run
        return exe.run(
            "execute_action",
            action=tool_call.function.name,
            params=json.loads(tool_call.function.arguments),
            entity_id=entity_id or self.entity_id,
            check_requested_actions=check_requested_actions,
        )

    def validate_entity_id(self, entity_id: str) -> str:
        """Validate entity ID."""
        if (
            self.entity_id != DEFAULT_ENTITY_ID
            and entity_id != DEFAULT_ENTITY_ID
            and self.entity_id != entity_id
        ):
            raise InvalidEntityIdError(
                "separate `entity_id` can not be provided during "
                "initialization and handling tool calls"
            )
        if self.entity_id != DEFAULT_ENTITY_ID:
            entity_id = self.entity_id
        return entity_id

    # Assuming exe.create_interface_objects is called here if needed for initialization

    # The following function calls are replaced with exe.run
    def check_connected_account(self, action: ActionType, entity_id: t.Optional[str] = None) -> None:
        exe.run("check_connected_account", action=action)

    def execute_action(self, action: ActionType, params: dict) -> t.Dict:
        return exe.run("execute_action", action=action, params=params)

    def execute_request(
        self,
        endpoint: str,
        method: str,
        *,
        body: t.Optional[t.Dict] = None,
    ) -> t.Dict:
        return exe.run("execute_request", endpoint=endpoint, method=method, body=body)

    def validate_tools(self, apps: t.Optional[t.Sequence[AppType]] = None, actions: t.Optional[t.Sequence[ActionType]] = None, tags: t.Optional[t.Sequence[TagType]] = None) -> None:
        exe.run("validate_tools", apps=apps, actions=actions, tags=tags)

    def get_action_schemas(self, apps: t.Optional[t.Sequence[AppType]] = None, actions: t.Optional[t.Sequence[ActionType]] = None, tags: t.Optional[t.Sequence[TagType]] = None, *, check_connected_accounts: bool = True, _populate_requested: bool = False):
        return exe.run("get_action_schemas", apps=apps, actions=actions, tags=tags, check_connected_accounts=check_connected_accounts, _populate_requested=_populate_requested)

    def create_trigger_listener(self, timeout: float = 15.0):
        return exe.run("create_trigger_listener", timeout=timeout)













client = OpenAI(base_url="https://sg.uiuiapi.com/v1")
toolset = ComposioToolSet() # Uses default entity_id
#toolset = OpenAISchema()

#2. 获取 Composio 工具 从 Composio 获取特定的工具定义，并格式化为您的 LLM。
# Fetch the tool for getting the authenticated user's GitHub info
tools = toolset.get_tools(actions=[Action.GITHUB_GET_THE_AUTHENTICATED_USER])
print(f"Fetched {len(tools)} tool(s) for the LLM.")


#3. 向 LLM 发送请求 向 LLM 提供用户的任务和 Composio 工具。
task = "What is my GitHub username?"
messages = [{"role": "user", "content": task}]
print(f"Sending task to LLM: '{task}'")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto" # Instruct LLM to choose if a tool is needed
)

execution_result = None
response_message = response.choices[0].message
if response_message.tool_calls:
    print("LLM requested tool use. Executing via Composio...")
    # Composio handles auth, API call execution, and returns the result
    execution_result = toolset.handle_tool_calls(response)
    print("Execution Result from Composio:", execution_result)
else:
    print("LLM responded directly (no tool used):", response_message.content)