from typing import List


# Dead base_ai.py
class BaseAI:
    def __init__(self, id: str = ""):
        self.id: str = id

    def chat(
        self,
        prompt: str,
        image: bool = False,
        audio: bool = False,
        filepath: str = None,
        temperature: int = 0,
        max_tokens: int = 4096,
    ) -> str:
        pass

    def generate_text(
        self, prompt: str, temperature: int = 0, max_tokens: int = 4096
    ) -> str:
        return "this is a base ai response"

    def generate_image(
        self, prompt: str, filepath: List, temperature: int = 0, max_tokens: int = 4096
    ) -> str:
        pass

    def generate_audio(
        self, prompt: str, filepath: str, temperature: int = 0, max_tokens: int = 4096
    ) -> bytes:
        pass

    def check_history(self):
        pass

    def print_history(self):
        pass

    def clear_history(self):
        pass

    def copy(self):
        return BaseAI()

    def print_AI_info(self):
        print(
            "I am a Dead Base AI , use from Inspection.ai.base_ai import BaseAI to get the real BaseAI class"
        )

    def _load_adapter(self, provider: str):
        pass
