import google.generativeai as genai
from typing import List, Dict
import base64
from Inspection import BASE_DIR
import json
import os
from PIL import Image
import io

class GoogleAdapter():
    def __init__(self):
        config_filepath = BASE_DIR + "/Inspection/config.json"
        with open(config_filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Google API 配置
        self._api_key = config.get("api_key", "")
        self._model = config.get("model", "gemini-1.5-flash")
        self._model_img = config.get("model_img", "gemini-1.5-flash")
        self._model_audio = config.get("model_audio", "gemini-1.5-flash")
        
        try:
            genai.configure(api_key=self._api_key)
            self.client = genai.GenerativeModel(self._model)
            print("[INS_INFO] Google AI model initialized successfully")
        except Exception as e:
            print("[INS_ERROR] Failed to initialize Google model, please check API key or network connection")
            print(f"[INS_ERROR] {e}")
            return

    def generate_text(self, history: List[Dict], promote: str, temperature: int, max_tokens: int) -> tuple:
        """Call Google Gemini model to generate text"""
        # Convert history format to Gemini format
        messages = self._convert_history_to_gemini_format(history)
        messages.append(promote)
        
        response = ""
        retry = 1
        max_retries = 5
        
        while retry < max_retries:
            try:
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                response = self.client.generate_content(
                    messages,
                    generation_config=generation_config
                )
                
                if retry != 1:
                    print(f"[INS_INFO]: Retry {retry-1} times successful")
                break
                
            except Exception as e:
                print(f"[INS_WARN]: Text generation failed, error: {e}, retrying {retry} times")
                retry += 1
        
        if retry == max_retries:
            raise Exception(f"[INS_ERROR]: {max_retries} text generation failures, stopping generation")
        
        # Get response content
        response_content = response.text
        history.append({"role": "user", "content": promote})
        history.append({"role": "assistant", "content": response_content})
        
        # Google API does not provide detailed token usage information temporarily, return estimated values
        token_usage = {
            "prompt_tokens": len(promote.split()) * 1.3,  # Estimate
            "completion_tokens": len(response_content.split()) * 1.3,  # Estimate
            "total_tokens": (len(promote.split()) + len(response_content.split())) * 1.3
        }
        
        return response_content, token_usage

    def generate_image(self, history: List[Dict], prompt: str, filepath: List, temperature: int, max_tokens: int) -> str:
        """Handle requests containing images"""
        try:
            # Use model that supports images
            model = genai.GenerativeModel(self._model_img)
            
            # Prepare content list
            content_parts = [prompt]
            
            # Add images
            for file_path in filepath:
                if os.path.exists(file_path):
                    # Open and process image
                    image = Image.open(file_path)
                    content_parts.append(image)
                else:
                    print(f"[INS_WARN] Image file does not exist: {file_path}")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = model.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            response_content = response.text
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
            # Google Gemini 1.5 Flash supports audio processing
            model = genai.GenerativeModel(self._model_audio)
            
            content_parts = [prompt]
            
            # Add audio files
            for file_path in filepath:
                if os.path.exists(file_path):
                    # Upload audio file
                    audio_file = genai.upload_file(path=file_path)
                    content_parts.append(audio_file)
                else:
                    print(f"[INS_WARN] Audio file does not exist: {file_path}")
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = model.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            response_content = response.text
            history.append({
                "role": "user", 
                "content": f"[Audio Analysis] {prompt} [Contains {len(filepath)} audio files]"
            })
            history.append({"role": "assistant", "content": response_content})
            
            return response_content
            
        except Exception as e:
            print(f"[INS_ERROR] Audio processing failed: {e}")
            return f"Audio processing failed: {str(e)}"

    def _convert_history_to_gemini_format(self, history: List[Dict]) -> List[str]:
        """Convert standard conversation history format to Gemini format"""
        messages = []
        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")
            
            if role == "user":
                messages.append(f"User: {content}")
            elif role == "assistant":
                messages.append(f"Assistant: {content}")
            elif role == "system":
                messages.append(f"System: {content}")
        
        return messages

# Tool functions
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")