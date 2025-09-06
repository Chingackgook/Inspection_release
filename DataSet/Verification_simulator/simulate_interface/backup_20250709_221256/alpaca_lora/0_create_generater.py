import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.alpaca_lora import ENV_DIR
from Inspection.adapters.custom_adapters.alpaca_lora import *

# 创建 Executor 实例
exe = Executor('alpaca_lora', 'simulation')
FILE_RECORD_PATH = exe.now_record_path


import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    share_gradio: bool = False,
):

    # 替换 evaluate 调用
    def evaluate_wrapper(instruction, input, temperature, top_p, top_k, num_beams, max_new_tokens, stream_output):
        return exe.run("evaluate", instruction=instruction, input=input, temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, max_new_tokens=max_new_tokens, stream_output=stream_output)

    gr.Interface(
        fn=evaluate_wrapper,
        inputs=[
            gr.components.Textbox(
                lines=2,
                label="Instruction",
                placeholder="Tell me about alpacas.",
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Stream output"),
        ],
        outputs=[
            gr.components.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🦙🌲 Alpaca-LoRA",
        description="Alpaca-LoRA is a 7B-parameter LLaMA model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and makes use of the Huggingface LLaMA implementation. For more information, please visit [the project's website](https://github.com/tloen/alpaca-lora).",  # noqa: E501
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)

#fire.Fire(main)


instructions = [
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "List all Canadian provinces in alphabetical order."
]

for instruction in instructions:
    print(f"Instruction: {instruction}")
    result = next(exe.run("evaluate",instruction=instruction))
    print(f"Response: {result}")
    print()