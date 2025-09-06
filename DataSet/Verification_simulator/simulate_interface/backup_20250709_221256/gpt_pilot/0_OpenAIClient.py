import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.gpt_pilot import ENV_DIR
from Inspection.adapters.custom_adapters.gpt_pilot import *
exe = Executor('gpt_pilot', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import datetime
import re
from typing import Optional, Callable  # 修复缺少Callable的导入

import tiktoken
from httpx import Timeout
from openai import AsyncOpenAI, RateLimitError

from core.config import LLMProvider, LLMConfig  # 添加 LLMConfig 的导入
from core.llm.base import BaseLLMClient
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)
tokenizer = tiktoken.get_encoding("cl100k_base")


class OpenAIClient(BaseLLMClient):
    provider = LLMProvider.OPENAI
    stream_options = {"include_usage": True}
    def __init__(self, config):
        super().__init__(config)
        # 这里需要为exe对象也进行init
        self._init_client()
        self.stream_handler = None

    def _init_client(self):
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=Timeout(
                max(self.config.connect_timeout, self.config.read_timeout),
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
            ),
        )

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        completion_kwargs = {
            "model": self.config.model,
            "messages": convo.messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": True,
        }
        if self.stream_options:
            completion_kwargs["stream_options"] = self.stream_options

        if json_mode:
            completion_kwargs["response_format"] = {"type": "json_object"}

        stream = await self.client.chat.completions.create(**completion_kwargs)
        response = []
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in stream:
            if chunk.usage:
                prompt_tokens += chunk.usage.prompt_tokens
                completion_tokens += chunk.usage.completion_tokens

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if not content:
                continue

            response.append(content)
            if self.stream_handler:
                await self.stream_handler(content)

        response_str = "".join(response)

        # Tell the stream handler we're done
        if self.stream_handler:
            await self.stream_handler(None)

        if prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = sum(3 + len(tokenizer.encode(msg["content"])) for msg in convo.messages)
            completion_tokens = len(tokenizer.encode(response_str))
            log.warning(
                "OpenAI response did not include token counts, estimating with tiktoken: "
                f"{prompt_tokens} input tokens, {completion_tokens} output tokens"
            )

        return response_str, prompt_tokens, completion_tokens

    async def api_check(self) -> bool:
        return await exe.run("api_check")

    async def _adapt_messages(self, convo: Convo) -> list[dict[str, str]]:
        return await exe.run("_adapt_messages", convo=convo)

    async def __call__(
        self,
        convo: Convo,
        *,
        temperature: Optional[float] = None,
        parser: Optional[Callable] = None,
        max_retries: int = 3,
        json_mode: bool = False,
    ) -> tuple:
        return await exe.run("__call__", convo=convo, temperature=temperature, parser=parser, max_retries=max_retries, json_mode=json_mode)

    async def rate_limit_sleep(self, err: RateLimitError) -> Optional[datetime.timedelta]:
        return await exe.run("rate_limit_sleep", err=err)


__all__ = ["OpenAIClient"]

# 模拟加载模型
exe.create_interface_objects(api_key="sk-sss", base_url="https://sg.uiuiapi.com/v1", connect_timeout=10, read_timeout=10)

# 示范如何创建一个 OpenAIClient 实例并进行调用
async def example_usage():
    client = OpenAIClient(config=LLMConfig(model="gpt-4o-mini", temperature=0.7))
    convo = Convo()
    convo.user("Hello, how are you?")
    response, request_log = await client(convo, parser=lambda content: {"response": content})
    print(response)

# 运行示例
#await example_usage()
import asyncio

if __name__ == "__main__":
    asyncio.run(example_usage())