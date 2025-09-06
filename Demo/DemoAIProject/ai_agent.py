"""
AI Agent Class - Intelligent agent for calling GPT API
"""
import requests
from typing import Optional, Dict, Any
import os

class AIAgent:
    """Intelligent AI agent class for interacting with GPT API"""
    
    def __init__(self, model = None):
        """
        Initialize AI agent
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            model: Model name to use
        """
        api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if model is None:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if api_key == "your-api-key-here":
            raise ValueError("Please set OPENAI_API_KEY environment variable or provide API key")

        print(f"Using model: {model}")
        print(f"API Base URL: {base_url}")
        print(f"API Key: {api_key}")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    
    def chat(self, message: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Chat with AI
        
        Args:
            message: User message
            system_prompt: System prompt (optional)
            temperature: Temperature parameter to control randomness of response
            
        Returns:
            Dictionary containing AI reply and metadata
        """
        try:
            # Build message list
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": message})
            
            # Build request data
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            # Send request
            url = f"{self.base_url}/chat/completions"
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_reply = result['choices'][0]['message']['content']
                
                return {
                    "success": True,
                    "reply": ai_reply,
                    "tokens_used": result.get('usage', {}),
                    "model": self.model
                }
            else:
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                return {
                    "success": False,
                    "error": error_msg,
                    "reply": None
                }
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout"
            return {"success": False, "error": error_msg, "reply": None}
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Network request error: {str(e)}"
            return {"success": False, "error": error_msg, "reply": None}
        
        except Exception as e:
            error_msg = f"Unknown error: {str(e)}"
            return {"success": False, "error": error_msg, "reply": None}
    
    def generate_text(self, prompt: str) -> Dict[str, Any]:
        """
        Generate text
        
        Args:
            prompt: Prompt text
            
        Returns:
            Generated text and metadata
        """
        return self.chat(prompt, temperature=0.8)
    
    def analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze text
        
        Args:
            text: Text to analyze
            analysis_type: Analysis type (general, sentiment, summary, etc.)
            
        Returns:
            Analysis results
        """
        system_prompts = {
            "general": "You are a text analysis expert, please provide a comprehensive analysis of the following text.",
            "sentiment": "You are a sentiment analysis expert, please analyze the emotional tendency of the following text.",
            "summary": "You are a text summarization expert, please generate a concise summary for the following text."
        }
        
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        return self.chat(text, system_prompt=system_prompt)
    
    def set_model(self, model: str):
        """Set the model to use"""
        self.model = model
    
    def get_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        return {
            "current_model": self.model,
            "base_url": self.base_url
        }
