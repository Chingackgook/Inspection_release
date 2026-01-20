from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from Inspection.utils.config import CONFIG


class BaseAIAdapter(ABC):
    """
    Base adapter class for AI model integrations.
    All AI adapters should inherit from this class to ensure a consistent interface.
    """

    def __init__(self):
        """
        Initialize the adapter with necessary configurations.
        Loads configuration from CONFIG and initializes the client.
        """
        # Load configuration
        self._config = CONFIG
        self._model = self._config.get("model", "")
        self._base_url = self._config.get("base_url", "")
        self._api_key = self._config.get("api_key", "")
        
        # Client will be set by subclass
        self.Client: Any = None
        
        try:
            self._init_client()
        except Exception as e:
            print(f"[INS_ERROR] Failed to initialize model, please check configuration file or network connection {e}")
            raise
    
    @abstractmethod
    def _init_client(self):
        """
        Initialize the client connection.
        This method should be implemented by subclasses to set up their specific client.
        Should set self.Client to the initialized client instance.
        
        Raises:
            Exception: If client initialization fails
        """
        pass

    @abstractmethod
    def generate_text(
        self, 
        message: List[Dict], 
        temperature: float, 
        max_tokens: int = None, 
        max_retries: int = 5, 
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate text response from the AI model.
        
        Args:
            message: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0 or higher)
            max_tokens: Maximum number of tokens to generate (optional)
            max_retries: Maximum number of retry attempts on failure (default: 5)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple containing:
                - str: Generated text response
                - Dict[str, int]: Token usage statistics with keys:
                    - 'prompt_tokens': Number of tokens in the prompt
                    - 'completion_tokens': Number of tokens in the completion
                    - 'total_tokens': Total number of tokens used
                    
        Raises:
            Exception: If generation fails after max_retries attempts
        """
        pass

    @abstractmethod
    def generate_image(
        self,
        history: List[Dict],
        prompt: str,
        filepath: List[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Generate response for requests containing images.
        
        Args:
            history: Conversation history as list of message dictionaries
            prompt: Text prompt to accompany the images
            filepath: List of paths to image files
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text response
        """
        pass

    @abstractmethod
    def generate_audio(
        self,
        history: List[Dict],
        prompt: str,
        filepath: List[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Generate response for requests containing audio.
        
        Args:
            history: Conversation history as list of message dictionaries
            prompt: Text prompt to accompany the audio
            filepath: List of paths to audio files
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: Generated text response
        """
        pass
