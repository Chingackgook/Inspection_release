import numpy as np
import torch
from PIL import Image
from Inspection.core.executor import Executor
from Inspection import TEST_DATA_PATH
import os

exe = Executor('CLIP', 'simulation')
exe.set_record_function(["tokenize"])
FILE_RECORD_PATH = exe.now_record_path

def test_consistency():
    device = "cpu"
    model_names = exe.run("available_models",record=False)  # 获取所有可用模型
    model_names = [name for name in model_names if "RN50" == name]  # 仅选择包含"RN50"的模型
    for model_name in model_names:
        print(f"Testing model: {model_name}")
        
        # 使用load_model方法替换exe.run("load", ...)调用
        jit_model, transform = exe.create_interface_objects(name=model_name, device=device, jit=True)  # 确保load_model的参数正确
        py_model, _ = exe.create_interface_objects(name=model_name, device=device, jit=False)  # 确保load_model的参数正确

        image = transform(Image.open(os.path.join(TEST_DATA_PATH, "masike.jpg"))).unsqueeze(0).to(device)
        text = exe.run("tokenize" ,texts=["a diagram", "a dog", "a cat","a person"]).to(device)  # 确保tokenize在exe.run中实现

        with torch.no_grad():
            logits_per_image, _ = jit_model(image, text)
            jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            logits_per_image, _ = py_model(image, text)
            py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1):
            print(f"Model {model_name} passed consistency test.")
        else:
            print(f"Model {model_name} failed consistency test.")
            print(f"JIT probabilities: {jit_probs}")
            print(f"Python probabilities: {py_probs}")

test_consistency()
