from CodeAnalysis.tools.dataclass_defs import Call
from pathlib import Path
import re


IGNORE_DIRS: list[str] = [
    '3rdparty',
    'third_party',
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'env',
    '.pytest_cache',
    '.mypy_cache',
    '.tox',
    'node_modules',
    '.eggs',
    'dist',
    'build',
    'checkpoints',
    '.ipynb_checkpoints',
]


IGNORE_PYTHON_FILES: list[str] = [
    "TEST.py" # 项目初期遗留的测试文件
] # 这些文件将不会被分析

PYTHON_MAX_FILE_SIZE_MB: float = 10.0 # 最大文件大小，单位MB，超过该大小的文件将不会被分析

MAX_CALL_STACK_DEPTH: int = 100 # 最大调用栈深度，超过该深度的调用将不会被继续分析


FINAL_CALL_LIST: list[Call] = [
    # 一些深度学习框架的常用推理相关方法
    Call(call_name="forward", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="topk", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="softmax", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="inference_mode", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="no_grad", object_name="ANY", object_type="ANY", call_site=""),

    # 一些启发式的智能模块调用方法
    Call(call_name="_inference", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="_predict", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="_generate", object_name="ANY", object_type="ANY", call_site=""),
    Call(call_name="run", object_name="encoder", object_type="ANY", call_site=""),
    Call(call_name="run", object_name="decoder", object_type="ANY", call_site=""),
    
    # OpenAI官方API调用方法
    Call(call_name="create", object_name="openai.ChatCompletion", object_type="ANY", call_site=""),
    Call(call_name="create", object_name="chat.completions", object_type="ANY", call_site=""),
    
    # 一些启发式的智能代理调用方法
    Call(call_name="invoke", object_name="llm", object_type="ANY", call_site=""),
    Call(call_name="invoke", object_name="model", object_type="ANY", call_site=""),
    Call(call_name="invoke", object_name="agent", object_type="ANY", call_site=""),
    Call(call_name="invoke", object_name="chain", object_type="ANY", call_site=""),
] # 如果在分析过程中调用了该列表中的方法，则认为调用了智能模块


FINAL_OBJECT_CALL_KEY_WORDS: list[str] = [
    'model',
    'detector',
    'classifier',
    'predictor',
] # 如果分析出来的对象调用（使用obj_name()或obj_name.__call__()）的对象名包含该列表中的关键字，则认为调用了智能模块
# 这部分还有一个不可配置的逻辑，即如果能分析出这个对象的类型且类型继承自torch.nn.Module，也认为调用了智能模块


EXCEPT_CALL_PATH_KEY_WORDS: list[str] = [
    'load',
    'train',
] # 如果调用路径包含该列表中的关键字，则不继续深入分析该调用





def get_start_python_files(project_manager):
    """
    遍历所有的python文件,筛选可能的入口文件
    前提条件:文件必须满足以下条件之一:
    1. 包含main函数
    2. 包含if __name__ == '__main__'
    3. 导入了argparse模块
    
    在满足前提条件后,按权重排序
    """
    # 首先筛选出满足前提条件的文件
    candidate_files = []
    
    for py_file in project_manager.python_files.values():
        # 检查是否满足入口脚本的前提条件
        has_main_function = any(func_def.function_name == 'main' for func_def in py_file.top_level_defs)
        has_argparse = 'argparse' in py_file.imports or any(moudle == 'argparse' for moudle, _ in py_file.from_imports_pairs)
        has_main_guard = py_file.has_if_name_main()  # 如果PythonFile类有这个属性
        
        if has_main_function or has_argparse or has_main_guard:
            candidate_files.append(py_file)
    
    # 对满足条件的文件按权重排序
    posible_entry_files = []
    
    # 第一权重:生成脚本
    for py_file in candidate_files:
        must_have_keywords_a = ['gen', 'synthesize']
        must_have_keywords_b = ['audio', 'video', 'image', 'text']
        if any(keyword in Path(py_file.path).name.lower() for keyword in must_have_keywords_a):
            if any(keyword in Path(py_file.path).name.lower() for keyword in must_have_keywords_b):
                posible_entry_files.append((py_file, "文件名包含生成关键词且包含媒体关键词"))

    # 第二权重:文件名包含cli或启动脚本
    for py_file in candidate_files:
        key_words = ['start', 'launch', r'\brun\b', r'\bcli\b']
        file_name_lower = Path(py_file.path).name.lower()
        if any(re.search(keyword, file_name_lower) if keyword.startswith(r'\b') else keyword in file_name_lower for keyword in key_words):
            posible_entry_files.append((py_file, "文件名包含启动/运行/命令行关键词"))
    
    # 第三权重:推理脚本
    for py_file in candidate_files:
        key_words = ['inference', 'predict']
        if any(keyword in Path(py_file.path).name.lower() for keyword in key_words):
            posible_entry_files.append((py_file, "文件名包含推理/预测关键词"))
    
    # 第四权重:媒体处理脚本
    for py_file in candidate_files:
        key_words = ['img', 'image', 'video', 'audio', 'text', 'txt']
        if any(keyword in Path(py_file.path).name.lower() for keyword in key_words):
            posible_entry_files.append((py_file, "文件名包含媒体关键词"))
    
    # 第四权重:示例脚本
    for py_file in candidate_files:
        key_words_example = ['example', 'demo', 'sample']
        if any(keyword in Path(py_file.path).name.lower() for keyword in key_words_example):
            posible_entry_files.append((py_file, "文件名包含示例/演示关键词"))
    
    # 第五权重:测试脚本
    for py_file in candidate_files:
        key_words_test = ['test', 'unittest']
        if any(keyword in Path(py_file.path).name.lower() for keyword in key_words_test):
            posible_entry_files.append((py_file, "文件名包含测试关键词"))
    
    # 第六权重:其他满足前提条件但没有特殊关键词的文件
    for py_file in candidate_files:
        posible_entry_files.append((py_file, "满足入口脚本基本条件(含main函数或argparse)"))

    # 顺序去重
    seen = set()
    unique_entry_files = []
    for item in posible_entry_files:
        if item[0] not in seen:
            unique_entry_files.append(item)
            seen.add(item[0])
    
    return unique_entry_files