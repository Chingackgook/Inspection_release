import httpx
from openai import OpenAI
from typing import List, Dict
from Inspection.utils.config import CONFIG
from Inspection.utils.path_manager import RECORD_PATH
from Inspection.utils.tools import encode_image
from Inspection.ai.ai_adapters.ai_base_adapter import BaseAIAdapter
from datetime import datetime


def log_request(request: httpx.Request):
    import os
    log_dir = RECORD_PATH + "/network_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_name = log_dir + f"/request_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    """Callback function to log raw HTTP request."""
    with open(file_name, "a", encoding="utf-8") as f:
        f.write("--- Raw HTTP Request ---\n")
        f.write(f"{request.method} {request.url}\n")
        for key, value in request.headers.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        # Read and decode the request body
        body = request.read()
        f.write(body.decode("utf-8", errors="ignore"))
        f.write("\n--- End Request ---\n\n")


def log_response(response: httpx.Response):
    """Callback function to log raw HTTP response."""
    import os
    log_dir = RECORD_PATH + "/network_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_name = log_dir + f"/response_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    # The stream must be read to access the content
    response.read()
    with open(file_name, "a", encoding="utf-8") as f:
        f.write("--- Raw HTTP Response ---\n")
        f.write(f"Status Code: {response.status_code}\n")
        for key, value in response.headers.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        # Use .text which is the decoded body
        f.write(response.text)
        f.write("\n--- End Response ---\n\n")


class OpenAIAdapter(BaseAIAdapter):
    def __init__(self):
        # Call parent class initialization
        super().__init__()
    
    def _init_client(self):
        """Initialize OpenAI client"""        
        # Special handling for raw network logging
        if CONFIG.get("log_raw_network", False):
            http_client = httpx.Client(
                event_hooks={"request": [log_request], "response": [log_response]}
            )
        else:
            http_client = httpx.Client()

        self.Client = OpenAI(
            base_url=self._base_url, api_key=self._api_key, http_client=http_client
        )

    def generate_text(
        self, message: List[Dict], temperature: float, max_tokens: int = None , max_retries = 5 , **kwargs
    ) -> tuple:
        # Call model to generate text
        response = ""
        last_error = None
        retry = 0
        while retry < max_retries:
            try:
                response = self.Client.chat.completions.create(
                    model=self._model,
                    messages=message,
                    temperature=temperature,
                    **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                    **kwargs
                )
                if retry != 0:
                    print(f"[INS_INFO]: Retry {retry} times successful")
                break
            except Exception as e:
                print(
                    f"[INS_WARN]: Text generation failed, error: {e}, retrying {retry} times"
                )
                last_error = e
                retry += 1

        if retry == max_retries:
            with open(RECORD_PATH + "/error/generation_error.log", "a") as f:
                f.write(
                    f"{datetime.now()} - Text generation failed after {max_retries} retries. Last error: {last_error}\n"
                )
            raise Exception(
                f"[INS_ERROR]: {max_retries} text generation failures, stopping generation"
            )

        # Get token usage information
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        response_content = response.choices[0].message.content
        
        # Return response content and token usage
        return response_content, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    # Batch upload images
    def generate_image(
        self,
        history: List[Dict],
        prompt: str,
        filepath: List,
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Not enabled in project
        temp = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
        for file in filepath:
            base64_image = encode_image(file)
            temp["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            )
        history.append(temp)
        response = self.Client.chat.completions.create(
            model=self._model,
            messages=history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = response.choices[0].message.content
        history.append({"role": "assistant", "content": response})
        return response

    def generate_audio(
        self,
        history: List[Dict],
        prompt: str,
        filepath: List,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Handle requests containing audio"""
        # OpenAI currently does not support audio processing in this way
        return "Temporarily not supported"




