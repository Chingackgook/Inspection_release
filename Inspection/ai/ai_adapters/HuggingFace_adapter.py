from openai import OpenAI
from typing import List, Dict
import base64
from pathlib import Path
from Inspection import BASE_DIR
import json
import os


class HuggingFaceAdapter():
    def __init__(self):
        config_filepath = BASE_DIR + "/Inspection/config.json"
        with open(config_filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # HuggingFace API 配置
        self._api_key = config.get("api_key", "")
        self._base_url = config.get("base_url", "https://api-inference.huggingface.co/v1")
        self._model = config.get("model", "microsoft/DialoGPT-medium")
        self._model_img = config.get("model_img", "microsoft/DialoGPT-medium")
        self._model_audio = config.get("model_audio", "microsoft/DialoGPT-medium")
        
        print("[INS_INFO] HuggingFace adapter initialization completed (functionality not fully implemented yet)")

    def generate_text(self, history: List[Dict],promote: str,temperature: int,max_tokens: int) -> str:
        return "Text not supported yet"

    #Batch upload images
    def generate_image(self, history: List[Dict],prompt :str,filepath: List,temperature: int,max_tokens: int) -> str:
        # Not enabled in project
        return "Images not supported yet"

    def generate_audio(self, history: List[Dict],prompt: str,filepath: List,temperature: int,max_tokens: int) -> bytes:
        # Not enabled in project
        return "Audio not supported yet"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
