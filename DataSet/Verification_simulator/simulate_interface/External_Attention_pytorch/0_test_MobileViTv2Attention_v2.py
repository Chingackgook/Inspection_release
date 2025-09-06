from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.External_Attention_pytorch import *
exe = Executor('External_Attention_pytorch', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/External-Attention-pytorch/main.py'
from model.attention.MobileViTv2Attention import *
import torch
from torch import nn
from torch.nn import functional as F

# Generate random input tensor
input_tensor = torch.randn(50, 49, 512)  # Renamed from 'input' to 'input_tensor' to avoid conflict with built-in function

# Create interface objects
sa = exe.create_interface_objects(interface_class_name='MobileViTv2Attention', d_model=512)

# Run the forward pass
output = exe.run('forward', input=input_tensor)

# Get output shape
output_shape = output.shape
print(output_shape)
