import os
import json


# Project root directory - Inspection
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Root directory
INSPECTION_DIR = BASE_DIR + "/Inspection/"  # Inspection directory
CUSTOM_ADAPTER_PATH = (
    BASE_DIR + "/Inspection/adapters/custom_adapters/"
)  # Custom adapter directory
SIMULATOR_PATH = (
    BASE_DIR + "/Inspection/simulation/simulate_interface/"
)  # Simulator directory
DUMB_SIMULATOR_PATH = (
    BASE_DIR + "/Inspection/simulation/dumb_simulator/"
)  # Dumb simulator directory


RESOURCES_PATH = (
    BASE_DIR + "/Resources/"
)  # Static resource directory for images, audio, video and other external resources

DEMO_PATH = BASE_DIR + "/Demo/"  # Demo directory
OTHERTASK_PATH = BASE_DIR + "/Othertask/"  # Other task directory

CACHE_DIR = BASE_DIR  # Cache directory


try:
    with open(os.path.join(INSPECTION_DIR, "config.json"), "r") as f:
        config = json.load(f)
except FileNotFoundError:
    pass

cache_dir = config.get("cache_dir", "")
if cache_dir != "":
    CACHE_DIR = cache_dir

INTERFACE_DATA_PATH = CACHE_DIR + "/InterfaceData/"  # Cache interface data
INTERFACE_INFO_PATH = (
    CACHE_DIR + "/InterfaceData/InterfaceInfo/"
)  # Cache interface information
INTERFACE_DOC_PATH = (
    CACHE_DIR + "/InterfaceData/InterfaceDocs/"
)  # Cache interface documentation
INTERFACE_TXT_PATH = CACHE_DIR + "/InterfaceData/InterfaceTXT/"  # Cache interface text
RECORD_PATH = CACHE_DIR + "/Records/"  # Execution record location
FILE_RECORD_PATH = (
    RECORD_PATH + "file_records/"
)  # Generally not used unless LLM hallucination occurs, will fallback to this location
SUGGESTION_PATH = CACHE_DIR + "/Suggestions/"  # Suggestion location

AI_CHAT_RECORD_PATH = CACHE_DIR + "/AIChatRecords/"  # AI chat record location
ENV_BASE = (
    CACHE_DIR + "/Env/"
)  # Cache some files for each project, or some hyperparameters
WORKFLOW_PATH = CACHE_DIR + "/Workflow/"  # Cache workflow rules


# Check if the above cache paths exist, create them if they don't exist
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_and_write_in_txt(path, content):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)


def check_TXT_PATH():
    pj_root_dir = os.path.join(INTERFACE_TXT_PATH, "ProjectRoot.txt")
    api_calls_dir = os.path.join(INTERFACE_TXT_PATH, "APICalls")
    api_ipl_dir = os.path.join(INTERFACE_TXT_PATH, "APIImplementations")
    check_path(api_calls_dir)
    check_path(api_ipl_dir)
    check_and_write_in_txt(
        pj_root_dir, "### Write project root directory path information in this file"
    )
    ipl_description_path = os.path.join(api_ipl_dir, "description.txt")
    check_and_write_in_txt(
        ipl_description_path,
        "### Write API implementation description information in this file",
    )
    ipl_ipl_path = os.path.join(api_ipl_dir, "implementation.txt")
    check_and_write_in_txt(
        ipl_ipl_path, "### Write API implementation code in this file"
    )
    ipl_path = os.path.join(api_ipl_dir, "path.txt")
    check_and_write_in_txt(ipl_path, "### Write API path information in this file")
    ipl_name_path = os.path.join(api_ipl_dir, "name.txt")
    check_and_write_in_txt(ipl_name_path, "### Write API name information in this file")
    ipl_example_path = os.path.join(api_ipl_dir, "example.txt")
    check_and_write_in_txt(
        ipl_example_path,
        "### Write API call examples in this file\n### Separate each example with $$$ on a new line",
    )
    call_description_path = os.path.join(api_calls_dir, "description.txt")
    check_and_write_in_txt(
        call_description_path, "### Write API call description information in this file"
    )
    call_name_path = os.path.join(api_calls_dir, "name.txt")
    check_and_write_in_txt(
        call_name_path,
        "### Write API call name information in this file, please use English naming without spaces or special characters",
    )
    call_code_path = os.path.join(api_calls_dir, "code.txt")
    check_and_write_in_txt(call_code_path, "### Write API call code in this file")


check_path(CUSTOM_ADAPTER_PATH)
check_path(SIMULATOR_PATH)
check_path(DUMB_SIMULATOR_PATH)


check_path(RESOURCES_PATH)
check_path(INTERFACE_INFO_PATH)
check_path(INTERFACE_DOC_PATH)
check_path(INTERFACE_TXT_PATH)
check_path(RECORD_PATH)
check_path(FILE_RECORD_PATH)
check_path(AI_CHAT_RECORD_PATH)
check_path(ENV_BASE)
check_path(WORKFLOW_PATH)
check_path(SUGGESTION_PATH)
check_TXT_PATH()
