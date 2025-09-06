from mistralai import Mistral
from typing import List, Dict, Optional
import base64
from Inspection import BASE_DIR
import json
import os
from PIL import Image

class MistralAdapter():
    def __init__(self):
        config_filepath = BASE_DIR + "/Inspection/config.json"
        with open(config_filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Mistral API 配置
        self._api_key = config.get("api_key", "")
        self._model = config.get("model", "mistral-large-latest")
        self._model_img = config.get("model_img", "pixtral-large-latest")
        self._model_audio = config.get("model_audio", "mistral-large-latest")
        
        try:
            self.client = Mistral(api_key=self._api_key)
            print("[INS_INFO] Mistral AI model initialized successfully")
        except Exception as e:
            print("[INS_ERROR] Failed to initialize Mistral model, please check API key or network connection")
            print(f"[INS_ERROR] {e}")
            return

    def generate_text(self, history: List[Dict], promote: str, temperature: int, max_tokens: int) -> tuple:
        """Call Mistral model to generate text"""
        # Build message list
        messages = self._convert_history_to_mistral_format(history)
        messages.append({"role": "user", "content": promote})
        
        response = ""
        retry = 1
        max_retries = 5
        
        while retry < max_retries:
            try:
                response = self.client.chat.complete(
                    model=self._model,
                    messages=messages,
                    temperature=temperature / 100.0 if temperature > 1 else temperature,  # Mistral temperature range is 0.0-1.0
                    max_tokens=max_tokens
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
        response_content = response.choices[0].message.content
        history.append({"role": "user", "content": promote})
        history.append({"role": "assistant", "content": response_content})
        
        # Get token usage information
        usage = response.usage if hasattr(response, 'usage') and response.usage else None
        token_usage = {
            "prompt_tokens": usage.prompt_tokens if usage else len(promote.split()) * 1.3,
            "completion_tokens": usage.completion_tokens if usage else len(response_content.split()) * 1.3,
            "total_tokens": usage.total_tokens if usage else (len(promote.split()) + len(response_content.split())) * 1.3
        }
        
        return response_content, token_usage

    def generate_image(self, history: List[Dict], prompt: str, filepath: List, temperature: int, max_tokens: int) -> str:
        """Handle requests containing images"""
        try:
            # Build message content
            content_parts = [{"type": "text", "text": prompt}]
            
            # Add images
            for file_path in filepath:
                if os.path.exists(file_path):
                    # Encode image to base64
                    base64_image = encode_image(file_path)
                    # Detect image type
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                        image_format = file_path.split('.')[-1].lower()
                        if image_format == 'jpg':
                            image_format = 'jpeg'
                        
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        })
                    else:
                        print(f"[INS_WARN] Unsupported image format: {file_path}")
                else:
                    print(f"[INS_WARN] Image file does not exist: {file_path}")
            
            # Build message list
            messages = self._convert_history_to_mistral_format(history)
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            response = self.client.chat.complete(
                model=self._model_img,
                messages=messages,
                temperature=temperature / 100.0 if temperature > 1 else temperature,
                max_tokens=max_tokens
            )
            
            response_content = response.choices[0].message.content
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
            # Mistral currently has limited audio support, using text model to handle audio-related requests
            print("[INS_INFO] Mistral currently has limited support for direct audio processing, will be handled as text request")
            
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

    def _convert_history_to_mistral_format(self, history: List[Dict]) -> List[Dict]:
        """Convert standard conversation history format to Mistral format"""
        mistral_messages = []
        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")
            
            if role in ["user", "assistant", "system"]:
                mistral_messages.append({
                    "role": role,
                    "content": content
                })
        
        return mistral_messages

# Tool functions
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")