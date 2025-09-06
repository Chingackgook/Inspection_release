import ollama
from typing import List, Dict, Optional
import base64
from Inspection import BASE_DIR
import json
import os
from PIL import Image

class OllamaAdapter():
    def __init__(self):
        config_filepath = BASE_DIR + "/Inspection/config.json"
        with open(config_filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Ollama configuration
        self._host = config.get("base_url", "http://localhost:11434")
        self._model = config.get("model", "llama3.2")
        self._model_img = config.get("model_img", "llama3.2-vision")
        self._model_audio = config.get("model_audio", "llama3.2")
        self._keep_alive = "5m"
        
        try:
            # Initialize Ollama client
            self.client = ollama.Client(host=self._host)
            
            # Test connection
            try:
                self.client.list()
                print("[INS_INFO] Ollama local model service connected successfully")
            except Exception as e:
                print("[INS_WARN] Unable to connect to Ollama service, please ensure Ollama is running")
                print(f"[INS_WARN] Connection error: {e}")
        except Exception as e:
            print("[INS_ERROR] Failed to initialize Ollama client")
            print(f"[INS_ERROR] {e}")
            return

    def generate_text(self, history: List[Dict], promote: str, temperature: int, max_tokens: int) -> tuple:
        """Call Ollama local model to generate text"""
        # Build message list
        messages = self._convert_history_to_ollama_format(history)
        messages.append({"role": "user", "content": promote})
        
        response = ""
        retry = 1
        max_retries = 5
        
        while retry < max_retries:
            try:
                # Call Ollama Chat API
                response = self.client.chat(
                    model=self._model,
                    messages=messages,
                    options={
                        'temperature': temperature / 100.0 if temperature > 1 else temperature,
                        'num_predict': max_tokens,
                    },
                    keep_alive=self._keep_alive
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
        response_content = response.get("message", {}).get("content", "")
        history.append({"role": "user", "content": promote})
        history.append({"role": "assistant", "content": response_content})
        
        # Get token usage information
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        total_duration = response.get("total_duration", 0)
        
        # Estimate token usage
        token_usage = {
            "prompt_tokens": prompt_eval_count if prompt_eval_count > 0 else len(promote.split()) * 1.3,
            "completion_tokens": eval_count if eval_count > 0 else len(response_content.split()) * 1.3,
            "total_tokens": (prompt_eval_count + eval_count) if (prompt_eval_count > 0 and eval_count > 0) else (len(promote.split()) + len(response_content.split())) * 1.3,
            "total_duration_ms": total_duration // 1000000 if total_duration > 0 else 0  # Convert to milliseconds
        }
        
        return response_content, token_usage

    def generate_image(self, history: List[Dict], prompt: str, filepath: List, temperature: int, max_tokens: int) -> str:
        """Handle requests containing images"""
        try:
            print("[INS_INFO] Using Ollama multimodal model to process images")
            
            # Build multimodal message content
            content_parts = [prompt]
            
            # Process image files
            for file_path in filepath:
                if os.path.exists(file_path):
                    try:
                        # Encode image to base64
                        base64_image = encode_image(file_path)
                        content_parts.append({
                            "type": "image",
                            "data": base64_image
                        })
                        print(f"[INS_INFO] Added image: {os.path.basename(file_path)}")
                    except Exception as img_error:
                        print(f"[INS_WARN] Failed to process image {file_path}: {img_error}")
                        # Add image information as text description
                        try:
                            with Image.open(file_path) as img:
                                width, height = img.size
                                format_name = img.format
                                file_size = os.path.getsize(file_path)
                                content_parts.append(f"\n[Image info: {os.path.basename(file_path)} ({format_name}, {width}x{height}, {file_size} bytes)]")
                        except Exception:
                            file_size = os.path.getsize(file_path)
                            content_parts.append(f"\n[Image file: {os.path.basename(file_path)} ({file_size} bytes)]")
                else:
                    content_parts.append(f"\n[Image file does not exist: {os.path.basename(file_path)}]")
            
            # Build message list
            messages = self._convert_history_to_ollama_format(history)
            
            # If there is actual image data, use vision model
            has_images = any(isinstance(part, dict) and part.get("type") == "image" for part in content_parts)
            model_to_use = self._model_img if has_images else self._model
            
            if has_images:
                # Multimodal message format
                messages.append({
                    "role": "user",
                    "content": prompt,
                    "images": [part["data"] for part in content_parts if isinstance(part, dict) and part.get("type") == "image"]
                })
            else:
                # Pure text message format
                full_content = " ".join(str(part) for part in content_parts)
                messages.append({
                    "role": "user",
                    "content": full_content
                })
            
            response = self.client.chat(
                model=model_to_use,
                messages=messages,
                options={
                    'temperature': temperature / 100.0 if temperature > 1 else temperature,
                    'num_predict': max_tokens,
                },
                keep_alive=self._keep_alive
            )
            
            response_content = response.get("message", {}).get("content", "")
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
            # Ollama currently does not directly support audio processing, using text model to handle audio-related requests
            print("[INS_INFO] Ollama currently does not support direct audio processing, will be handled as text request")
            
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

    def _convert_history_to_ollama_format(self, history: List[Dict]) -> List[Dict]:
        """Convert standard conversation history format to Ollama format"""
        ollama_messages = []
        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")
            
            if role in ["user", "assistant"]:
                ollama_messages.append({
                    "role": role,
                    "content": content
                })
            elif role == "system":
                # Ollama supports system role
                ollama_messages.append({
                    "role": "system",
                    "content": content
                })
        
        return ollama_messages

    def list_models(self) -> List[Dict]:
        """Get list of locally available models"""
        try:
            models = self.client.list()
            model_list = []
            for model in models.get('models', []):
                model_info = {
                    'name': model.get('name', ''),
                    'size': model.get('size', 0),
                    'modified_at': model.get('modified_at', ''),
                    'digest': model.get('digest', ''),
                    'details': model.get('details', {})
                }
                model_list.append(model_info)
            return model_list
        except Exception as e:
            print(f"[INS_ERROR] Failed to get model list: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull model to local"""
        try:
            print(f"[INS_INFO] Starting to pull model: {model_name}")
            self.client.pull(model_name)
            print(f"[INS_INFO] Model pull completed: {model_name}")
            return True
        except Exception as e:
            print(f"[INS_ERROR] Failed to pull model: {e}")
            return False

    def check_model_exists(self, model_name: str) -> bool:
        """Check if model exists"""
        try:
            models = self.list_models()
            for model in models:
                if model_name in model.get('name', ''):
                    return True
            return False
        except Exception:
            return False

# Tool functions
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")