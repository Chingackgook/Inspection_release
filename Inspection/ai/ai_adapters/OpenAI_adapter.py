from openai import OpenAI
from typing import List, Dict
import base64
from Inspection.utils.config import CONFIG
import json

class OpenAIAdapter():
    def __init__(self):
        config = CONFIG
        self._api_key=config["api_key"]
        self._base_url=config.get("base_url", "https://api.openai.com/v1")
        self._model=config.get("model", "gpt-4o-mini")
        self._model_img=config.get("model_img", "gpt-4o-mini")
        self._model_audio=config.get("model_audio", "gpt-4o-mini")
        try:
            self.Client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        except Exception as e:
            print("[INS_ERROR] Failed to initialize model, please check configuration file or network connection")
            print(f"[INS_ERROR] {e}")
            return

    def generate_text(self, history: List[Dict],promote: str,temperature: int,max_tokens: int) -> tuple:
        #Call model to generate text
        history.append({"role": "user", "content": promote})
        response = ""
        #Retry when timeout is detected
        retry = 1
        max_retries = 50
        while retry < max_retries:
            try:
                response = self.Client.chat.completions.create(
                    model = self._model,
                    messages = history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    )
                if retry != 1:
                    print(f"[INS_INFO]: Retry {retry-1} times successful")
                break
            except Exception as e:
                print(f"[INS_WARN]: Text generation failed, error: {e}, retrying {retry} times")
                retry += 1
        if retry == max_retries:
            #Throw exception
            raise Exception(f"[INS_ERROR]: {max_retries} text generation failures, stopping generation")
        
        # Get token usage information
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        response_content = response.choices[0].message.content
        history.append({"role": "assistant", "content": response_content})

        # Return response content and token usage
        return response_content, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    #Batch upload images
    def generate_image(self, history: List[Dict],prompt :str,filepath: List,temperature: int,max_tokens: int) -> str:
        # Not enabled in project
        temp = {
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
            ]
        }
        for file in filepath:
            base64_image = encode_image(file)
            temp["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            })
        history.append(temp)
        response = self.Client.chat.completions.create(
            model = self._model_img,
            messages = history,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = response.choices[0].message.content
        history.append({"role": "assistant", "content": response})
        return response

    def generate_audio(self, history: List[Dict],prompt: str,filepath: List,temperature: int,max_tokens: int) -> bytes:
        return "Temporarily not supported"
        temp = {
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
            ]
        }
        for file in filepath:
            #Extract file format
            format = filepath[0].split(".")[-1]
            with open(file, "rb") as audio_file:
                base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")
            temp["content"].append({
                "type": "input_audio",
                "input_audio": {
                    "data": base64_audio,
                    "format": format
                },
            })
        print("temp:",temp)
        history.append(temp)
        response = self.Client.chat.completions.create(
            model = self._model_audio,
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages = history,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = response.choices[0].message.content
        history.append({"role": "assistant", "content": response})
        return response
    

#utf-8 encoding
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
