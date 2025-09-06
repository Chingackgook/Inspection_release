import os
import uuid
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.quivr import ENV_DIR
from Inspection.adapters.custom_adapters.quivr import *
from tempfile import NamedTemporaryFile
import asyncio
from asyncio import to_thread
import openai

# Initialize Executor and setup
exe = Executor('quivr', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)

# Simulate file paths and brain creation
def run_in_event_loop(func, *args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if asyncio.iscoroutinefunction(func):
        result = loop.run_until_complete(func(*args, **kwargs))
    else:
        result = func(*args, **kwargs)
    loop.close()
    return result


def simulate_exe_run():
    # this is a initialization step
    exe.run("from_files", name_quivr="my smart brain", file_paths=[ENV_DIR+"/my_smart_doc.pdf", ENV_DIR+"/my_intelligent_doc.txt"])

    # Replace function calls with exe.run
    print("Starting simulation with exe.run calls...")

    # Running commands using exe.run
    exe.run("save", folder_path=os.path.join(FILE_RECORD_PATH, "path/to/save"))
    exe.run("info")
    
    results = exe.run("asearch", query="What is the meaning of life?", n_results=5)
    print(f"Search results: {results}")
    
    response = exe.run("ask_streaming", question="What is the meaning of life?", run_id=uuid.uuid4())
    print(f"Streaming response: {response}")
    
    response_aask = exe.run("aask", run_id=uuid.uuid4(), question="What is the meaning of life?")
    print(f"Aask response: {response_aask}")
    
    response_ask = exe.run("ask", run_id=uuid.uuid4(), question="What is the meaning of life?")
    print(f"Ask response: {response_ask}")
    
    chat_history = exe.run("get_chat_history", chat_id=uuid.uuid4())
    print(f"Chat history: {chat_history}")
    
    exe.run("add_file")
    brain_from_files = exe.run("afrom_files", name_quivr="user_brain", file_paths=[os.path.join(ENV_DIR, "example.txt")])
    print(f"Brain from files: {brain_from_files}")
    
    brain_from_files = exe.run("from_files", name_quivr="user_brain", file_paths=[os.path.join(ENV_DIR, "example.txt")])
    print(f"Brain from files again: {brain_from_files}")

    print("Simulation complete.")

# Run the simulation directly when executing the script
simulate_exe_run()
