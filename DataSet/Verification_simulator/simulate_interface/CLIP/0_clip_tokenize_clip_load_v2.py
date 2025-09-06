from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.CLIP import *
import numpy as np
import torch
from PIL import Image
import clip
exe = Executor('CLIP', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = ''

def test_consistency():
    device = 'cpu'
    model_names = exe.run('available_models')
    if not model_names:
        print('No models available.')
        return
    for model_name in model_names:
        print(f'Testing model: {model_name}')
        jit_model, transform = exe.run('load', name=model_name, device=device, jit=True)
        py_model, _ = exe.run('load', name=model_name, device=device, jit=False)
        image = transform(Image.open(RESOURCES_PATH + 'images/test_image.png')).unsqueeze(0).to(device)
        text = exe.run('tokenize', texts=['a diagram', 'a dog', 'a cat']).to(device)
        with torch.no_grad():
            logits_per_image, _ = jit_model(image, text)
            jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            logits_per_image, _ = py_model(image, text)
            py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)
        print(f'Model {model_name} passed consistency test.')
test_consistency()