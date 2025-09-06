import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.rembg import ENV_DIR
from Inspection.adapters.custom_adapters.rembg import *
exe = Executor('rembg', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包
import json
import io
from typing import IO
import click
from rembg.bg import remove
from rembg.session_factory import new_session 
from rembg.sessions import sessions_names

# 模拟的输入参数
model = "u2net"
extras = json.dumps({
    "alpha_matting": True,
    "alpha_matting_foreground_threshold": 240,
    "alpha_matting_background_threshold": 10,
    "alpha_matting_erode_size": 10,
    "only_mask": False,
    "post_process_mask": False,
    "bgcolor": (0, 0, 0, 0)
})

input_file = ENV_DIR + '/input.jpg'  # 输入文件路径
output_file = FILE_RECORD_PATH + 'output.jpg'  # 输出文件路径

# 读取输入文件数据
with open(input_file, 'rb') as f:
    input_data = io.BytesIO(f.read())  # 使用 BytesIO 包装为流对象

# 创建输出数据流
output_data = io.BytesIO()  # 使用 BytesIO 作为内存中的输出流

# 直接调用的主逻辑
def i_command(model: str, extras: str, input: IO, output: IO, **kwargs) -> None:
    try:
        kwargs.update(json.loads(extras))
    except Exception:
        pass

    # 调用 load_model 进行模型初始化
    exe.create_interface_objects(model=model, **kwargs)

    # 替换 remove 函数调用
    output.write(exe.run("remove", data=input.read(), session=new_session(model, **kwargs), **kwargs))

# 调用主逻辑
i_command(model, extras, input_data, output_data)

# 输出结果（可以根据需要进行处理）
output_data.seek(0)  # 重置指针以读取数据
result = output_data.read()

# 将结果保存到文件
with open(output_file, 'wb') as f:
    f.write(result)
