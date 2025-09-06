import os
import json


# 为项目根目录 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #根目录
INSPECTION_DIR = BASE_DIR + '/Inspection/' #Inspection目录
CUSTOM_ADAPTER_PATH = BASE_DIR + '/Inspection/adapters/custom_adapters/' #自定义适配器目录
SIMULATION_PATH = BASE_DIR + '/Inspection/simulation/' #模拟执行代码目录
RESOURCES_PATH = BASE_DIR + '/Resources/' #静态资源目录，一些图片，音频，视频等外部资源的路径
EVALUATE_PATH = BASE_DIR + '/Evaluate/' #评估目录
EVALUATE_RQ1_PATH = BASE_DIR + '/Evaluate/RQ1/' #RQ1评估目录
EVALUATE_RQ2_PATH = BASE_DIR + '/Evaluate/RQ2/' #RQ2评估目录
EVALUATE_RQ3_PATH = BASE_DIR + '/Evaluate/RQ3/' #RQ3评估目录
DEMO_PATH = BASE_DIR + '/Demo/' #Demo目录

CACHE_DIR = BASE_DIR #缓存目录


try:
    with open(os.path.join(INSPECTION_DIR, 'config.json'), 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    pass

cache_dir = config.get('cache_dir', '')
if cache_dir!= '':
    CACHE_DIR = cache_dir
    
INTERFACE_DATA_PATH = CACHE_DIR + '/InterfaceData/' #缓存接口数据
INTERFACE_INFO_PATH = CACHE_DIR + '/InterfaceData/InterfaceInfo/' #缓存接口信息
INTERFACE_DOC_PATH = CACHE_DIR + '/InterfaceData/InterfaceDocs/' #缓存接口文档
INTERFACE_TXT_PATH = CACHE_DIR + '/InterfaceData/InterfaceTXT/' #缓存接口文本
RECORD_PATH = CACHE_DIR + '/Records/' #执行记录位置
FILE_RECORD_PATH = RECORD_PATH + 'filerecords/' # 一般不会使用此处，除非LLM出现幻觉，将会退回到此处
SUGGESTION_PATH = CACHE_DIR + '/Suggestions/' #建议位置

AI_CHAT_RECORD_PATH = CACHE_DIR + '/AIChatRecords/' #AI聊天记录位置
ENV_BASE = CACHE_DIR + '/Env/' # 缓存每个项目的一些文件，或是一些超参数
WORKFLOW_PATH = CACHE_DIR + '/Workflow/' #缓存工作流规则
TEST_DATA_PATH = CACHE_DIR + '/TestData/'
# 检查以上缓存路径是否存在，如果不存在则创建
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_and_write_in_txt(path, content):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(content)

def check_TXT_PATH():
    pj_root_dir = os.path.join(INTERFACE_TXT_PATH, 'ProjectRoot.txt')
    api_calls_dir = os.path.join(INTERFACE_TXT_PATH, 'APICalls')
    api_ipl_dir = os.path.join(INTERFACE_TXT_PATH, 'APIImplementations')
    check_path(api_calls_dir)
    check_path(api_ipl_dir)
    check_and_write_in_txt(pj_root_dir, '### 在本文件中编写项目根目录的路径信息')
    ipl_description_path = os.path.join(api_ipl_dir, 'description.txt')
    check_and_write_in_txt(ipl_description_path, '### 在本文件中编写API实现的描述信息')
    ipl_ipl_path = os.path.join(api_ipl_dir, 'implementation.txt')
    check_and_write_in_txt(ipl_ipl_path, '### 在本文件中编写API实现部分的代码')
    ipl_path = os.path.join(api_ipl_dir, 'path.txt')
    check_and_write_in_txt(ipl_path, '### 在本文件中编写API的路径信息')
    ipl_name_path = os.path.join(api_ipl_dir, 'name.txt')
    check_and_write_in_txt(ipl_name_path, '### 在本文件中编写API的名称信息')
    ipl_example_path = os.path.join(api_ipl_dir, 'example.txt')
    check_and_write_in_txt(ipl_example_path, '### 在本文件中编写API调用示例\n### 每个示例信息之间用新起一行的$$$隔开')
    call_description_path = os.path.join(api_calls_dir, 'description.txt')
    check_and_write_in_txt(call_description_path, '### 在本文件中编写API调用的描述信息')
    call_name_path = os.path.join(api_calls_dir, 'name.txt')
    check_and_write_in_txt(call_name_path, '### 在本文件中编写API调用的名称信息，请不要带空格等特殊字符，以英文命名')
    call_code_path = os.path.join(api_calls_dir, 'code.txt')
    check_and_write_in_txt(call_code_path, '### 在本文件中编写API调用的代码')

check_path(RESOURCES_PATH)
# check_path(EVALUATE_RQ1_PATH)
# check_path(EVALUATE_RQ2_PATH)
check_path(INTERFACE_INFO_PATH)
check_path(INTERFACE_DOC_PATH)
check_path(INTERFACE_TXT_PATH)
check_path(RECORD_PATH)
check_path(FILE_RECORD_PATH)
# check_path(TEST_DATA_PATH)
check_path(AI_CHAT_RECORD_PATH)
check_path(ENV_BASE)
check_path(WORKFLOW_PATH)
check_path(SUGGESTION_PATH)
check_TXT_PATH()