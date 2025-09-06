import qianfan
from typing import List, Dict, Optional
import base64
from Inspection import BASE_DIR
import json
import os
from PIL import Image

class BaiduAdapter():
    def __init__(self):
        config_filepath = BASE_DIR + "/Inspection/config.json"
        with open(config_filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Baidu Qianfan API configuration
        # For Baidu, api_key contains access_key and secret_key, separated by ":"
        api_key_parts = config.get("api_key", ":").split(":")
        self._access_key = api_key_parts[0] if len(api_key_parts) > 0 else ""
        self._secret_key = api_key_parts[1] if len(api_key_parts) > 1 else ""
        self._model = config.get("model", "ERNIE-3.5-8K")
        self._model_img = config.get("model_img", "ERNIE-VilG-v2")
        self._model_audio = config.get("model_audio", "ERNIE-3.5-8K")
        
        try:
            # Set Qianfan SDK authentication information
            os.environ["QIANFAN_ACCESS_KEY"] = self._access_key
            os.environ["QIANFAN_SECRET_KEY"] = self._secret_key
            
            # Initialize chat completion client
            self.chat_client = qianfan.ChatCompletion(model=self._model)
            print("[INS_INFO] Baidu Qianfan ERNIE model initialized successfully")
        except Exception as e:
            print("[INS_ERROR] Failed to initialize Baidu model, please check Access Key and Secret Key or network connection")
            print(f"[INS_ERROR] {e}")
            return

    def generate_text(self, history: List[Dict], promote: str, temperature: int, max_tokens: int) -> tuple:
        """Call Baidu Qianfan model to generate text"""
        # Build message list
        messages = self._convert_history_to_qianfan_format(history)
        messages.append({"role": "user", "content": promote})
        
        response = ""
        retry = 1
        max_retries = 5
        
        while retry < max_retries:
            try:
                response = self.chat_client.do(
                    messages=messages,
                    temperature=temperature / 100.0 if temperature > 1 else temperature,  # Qianfan temperature range is 0.0-1.0
                    max_output_tokens=max_tokens
                )
                
                if retry != 1:
                    print(f"[INS_INFO]: Retry {retry-1} times successful")
                break
                
            except Exception as e:
                print(f"[INS_WARN]: Text generation failed, error: {e}, retrying {retry} times")
                retry += 1
        
        if retry == max_retries:
            raise Exception(f"[INS_ERROR]: {max_retries} text generation failures, stopping generation")
        
        # 获取响应内容
        response_content = response.get("result", "")
        history.append({"role": "user", "content": promote})
        history.append({"role": "assistant", "content": response_content})
        
        # Get token usage information
        usage = response.get("usage", {})
        token_usage = {
            "prompt_tokens": usage.get("prompt_tokens", len(promote.split()) * 1.3),
            "completion_tokens": usage.get("completion_tokens", len(response_content.split()) * 1.3),
            "total_tokens": usage.get("total_tokens", (len(promote.split()) + len(response_content.split())) * 1.3)
        }
        
        return response_content, token_usage

    def generate_image(self, history: List[Dict], prompt: str, filepath: List, temperature: int, max_tokens: int) -> str:
        """Handle requests containing images"""
        try:
            # Baidu Qianfan's image processing requires using image understanding model
            print("[INS_INFO] Using Baidu Qianfan image understanding capability to process images")
            
            # Build image description information
            image_info = f"Image analysis request: {prompt}\n"
            for file_path in filepath:
                if os.path.exists(file_path):
                    # Get basic image information
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            format_name = img.format
                            file_size = os.path.getsize(file_path)
                            image_info += f"Image file: {os.path.basename(file_path)} ({format_name}, {width}x{height}, {file_size} bytes)\n"
                    except Exception:
                        file_size = os.path.getsize(file_path)
                        image_info += f"Image file: {os.path.basename(file_path)} ({file_size} bytes)\n"
                else:
                    image_info += f"Image file does not exist: {os.path.basename(file_path)}\n"
            
            # Due to limited image understanding capability of Qianfan SDK, using text description method first
            enhanced_prompt = f"{prompt}\n\n{image_info}\nNote: Currently using text description method to process image information, if image content analysis is needed, please use model that supports vision."
            
            # Use text generation method to handle
            response_content, _ = self.generate_text(history, enhanced_prompt, temperature, max_tokens)
            
            history.append({
                "role": "user", 
                "content": f"[Image Analysis] {prompt} [Contains {len(filepath)} images]"
            })
            history.append({"role": "assistant", "content": response_content})
            
            return response_content
            
        except Exception as e:
            print(f"[INS_ERROR] Image processing failed: {e}")
            return f"Image processing failed: {str(e)}"

    def generate_audio(self, history: List[Dict], prompt: str, filepath: List, temperature: int, max_tokens: int) -> str:
        """Handle requests containing audio"""
        try:
            # Baidu Qianfan currently does not directly support audio processing, using text model to handle audio-related requests
            print("[INS_INFO] Baidu Qianfan currently does not support direct audio processing, will be handled as text request")
            
            # Build audio description information
            audio_info = f"Audio file information: {len(filepath)} audio files - "
            for file_path in filepath:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    file_ext = os.path.splitext(file_path)[1]
                    audio_info += f"[{os.path.basename(file_path)}: {file_ext} format, {file_size} bytes] "
                else:
                    audio_info += f"[File does not exist: {os.path.basename(file_path)}] "
            
            # Convert audio processing request to text description request
            enhanced_prompt = f"{prompt}\n\n{audio_info}\nNote: Due to technical limitations, unable to directly analyze audio content, please respond based on provided audio file information."
            
            # Use text generation method
            return self.generate_text(history, enhanced_prompt, temperature, max_tokens)[0]
            
        except Exception as e:
            print(f"[INS_ERROR] Audio processing failed: {e}")
            return f"Audio processing failed: {str(e)}"

    def _convert_history_to_qianfan_format(self, history: List[Dict]) -> List[Dict]:
        """Convert standard conversation history format to Qianfan format"""
        qianfan_messages = []
        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")
            
            if role in ["user", "assistant"]:
                qianfan_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "system":
                # Qianfan should not directly support system role, convert to user message
                qianfan_messages.append({
                    "role": "user",
                    "content": f"[System Prompt] {content}"
                })
        
        return qianfan_messages

# Tool functions
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")