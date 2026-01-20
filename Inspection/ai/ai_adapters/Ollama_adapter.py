import ollama
from typing import List, Dict
from Inspection.utils.config import CONFIG
from Inspection.utils.path_manager import RECORD_PATH
from Inspection.utils.tools import encode_image
from Inspection.ai.ai_adapters.ai_base_adapter import BaseAIAdapter
from datetime import datetime
import os



class OllamaAdapter(BaseAIAdapter):
    def __init__(self):
        # Ollama specific configuration
        self._keep_alive = "5m"
        
        super().__init__()
    
    def _init_client(self):
        """Initialize Ollama client"""
        # Set default values if not provided
        
        self._host = self._base_url
        # Initialize Ollama client
        self.Client = ollama.Client(host=self._host)

        # Test connection
        try:
            self.Client.list()
            print("[INS_INFO] Ollama local model service connected successfully")
        except Exception as e:
            print(
                "[INS_WARN] Unable to connect to Ollama service, please ensure Ollama is running"
            )
            print(f"[INS_WARN] Connection error: {e}")

    def _ensure_model_available(self, model_name: str) -> bool:
        """Check if model exists, if not automatically pull it"""
        try:
            # Get list of local models
            models = self.Client.list()
            model_list = [model.get("name", "") for model in models.get("models", [])]
            
            # Check if model exists
            model_exists = any(model_name in model for model in model_list)
            
            if not model_exists:
                try:
                    # Pull model
                    self.Client.pull(model_name)
                    return True
                except Exception:
                    return False
            return True
        except Exception as e:
            print(f"[INS_WARN] Failed to check model availability: {e}")
            # If check fails, try to continue anyway
            return True

    def generate_text(
        self, message: List[Dict], temperature: float, max_tokens: int = None, max_retries: int = 5, **kwargs
    ) -> tuple:
        """Call Ollama local model to generate text"""
        # Ensure model is available
        self._ensure_model_available(self._model)
        
        # Call model to generate text
        response = ""
        last_error = None
        retry = 0
        
        # Get additional parameters from CONFIG
        additional_params = {}
        if CONFIG.get('additional_ai_params', None) is not None:
            additional_params = CONFIG['additional_ai_params']

        while retry < max_retries:
            try:
                # Build options
                options = {
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    options["num_predict"] = max_tokens
                
                # Merge additional parameters from CONFIG and kwargs
                options.update(additional_params)
                options.update(kwargs)
                
                # Call Ollama Chat API
                response = self.Client.chat(
                    model=self._model,
                    messages=message,
                    options=options,
                    keep_alive=self._keep_alive,
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
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        
        # Get response content
        response_content = response.get("message", {}).get("content", "")

        # Estimate token usage
        prompt_tokens = prompt_eval_count if prompt_eval_count > 0 else 0
        completion_tokens = eval_count if eval_count > 0 else 0
        total_tokens = prompt_tokens + completion_tokens

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
        """Handle requests containing images"""
        try:
            # Ensure model is available
            self._ensure_model_available(self._model_img)
            
            print("[INS_INFO] Using Ollama multimodal model to process images")

            # Build multimodal message content
            temp = {
                "role": "user",
                "content": prompt,
                "images": []
            }

            # Process image files
            for file_path in filepath:
                if os.path.exists(file_path):
                    try:
                        # Encode image to base64
                        base64_image = encode_image(file_path)
                        temp["images"].append(base64_image)
                        print(f"[INS_INFO] Added image: {os.path.basename(file_path)}")
                    except Exception as img_error:
                        print(
                            f"[INS_WARN] Failed to process image {file_path}: {img_error}"
                        )
                else:
                    print(f"[INS_WARN] Image file does not exist: {file_path}")

            history.append(temp)

            # Call Ollama vision model
            response = self.Client.chat(
                model=self._model_img,
                messages=history,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                keep_alive=self._keep_alive,
            )

            response_content = response.get("message", {}).get("content", "")
            history.append({"role": "assistant", "content": response_content})

            return response_content

        except Exception as e:
            print(f"[INS_ERROR] Image processing failed: {e}")
            return f"Image processing failed: {str(e)}"

    def generate_audio(
        self,
        history: List[Dict],
        prompt: str,
        filepath: List,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Handle requests containing audio"""
        # Ollama currently does not support audio processing
        return "Temporarily not supported"



