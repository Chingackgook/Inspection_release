from typing import List

#Dead base_ai.py
class BaseAI():
    def __init__(self,id:str = ""):
        self.id : str = id
        
    def chat(self, prompt: str,image: bool=False,audio: bool=False, filepath: str = None,temperature: int=0,max_tokens: int=4096) -> str:
        pass

    def generate_text(self, prompt: str,temperature: int=0,max_tokens: int=4096) -> str:
        pass

    def generate_image(self, prompt: str, filepath: List,temperature: int=0,max_tokens: int=4096) -> str:
        pass

    def generate_audio(self, prompt :str,filepath:str,temperature: int=0,max_tokens: int=4096) -> bytes:
        pass
        
    def check_history(self):
        pass
    def print_history(self):
        pass
    def clear_history(self):
        pass
    def copy(self):
        return BaseAI()

    def _load_adapter(self, provider: str):
        pass





if __name__ == "__main__":
    adapter = BaseAI()
    while True:
        user_input = input("Please enter: ")
        response = adapter.chat(user_input)
        print(f": {response}")
        user_imput2= input("Please enter: ")
        response2 = adapter.chat(user_imput2,image=True,filepath=['D:\\桌面\\综合实训\\Inspection\\Inspection\\ai\\image.jpg'])
        print(f": {response2}")
