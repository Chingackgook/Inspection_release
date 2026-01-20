from Inspection import BASE_DIR
from Inspection.utils.shared_config import OSENV_CONFIG as CONFIG
from watchdog.events import FileSystemEventHandler
import threading
import os
import re
import keyword
import sys


def select_from_list(options: list, prompt: str = None) -> tuple:
    """
    从选项列表中选择一个值。
    - 支持通过索引或名称选择。
    - 当存在重复选项时，提示用户使用索引选择。
    - 提供退出机制，输入 'q' 或 'exit' 返回 None。
    :param options: 可供选择的选项列表。
    :param prompt: 提示信息。
    :return: (选中的选项, 索引) 或 (None, None)。
    """
    if prompt is None:
        prompt = "Please select an option by entering the corresponding index or name or EXIT to quit: "
    if not options:
        raise ValueError("The options list is empty.")

    while True:
        # 显示选项
        print("\nOptions:")
        for idx, option in enumerate(options):
            print(f"{idx+1}: {str(option)}")

        # 获取用户输入
        choice = input(prompt).strip()

        # 退出机制
        if choice == "EXIT":
            print("[INFO] Exiting selection.")
            return None, None

        if choice == "\n" or choice == "":
            choice = "1"  # 默认选择第一个选项

        # 尝试按索引选择
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx], idx
            else:
                print("[ERROR] Invalid index. Please try again.")
                continue

        # 尝试按名称选择
        matches = [i for i, option in enumerate(options) if str(option) == choice]
        if len(matches) == 1:
            return options[matches[0]], matches[0]
        elif len(matches) > 1:
            print(
                "[ERROR] Multiple options match your input. Please use the index to select."
            )
        else:
            print("[ERROR] Invalid choice. Please try again.")


def stringify_data(data, max_list_length=100, max_str_length=300 , max_total_length=300):
    import json

    def trunc(s: str, limit: int) -> str:
        if len(s) <= limit:
            return s
        tail = f" ... remain {len(s) - limit} ch"
        keep = max(0, limit - len(tail))
        return s[:keep] + tail

    def to_printable(v):
        if isinstance(v, dict):
            return {
                trunc(str(k), max_str_length): to_printable(val)
                for k, val in v.items()
            }

        if isinstance(v, (list, tuple, set)):
            seq = list(v)
            out = [to_printable(x) for x in seq[:max_list_length]]
            if len(seq) > max_list_length:
                out.append(trunc(f"... remain {len(seq)-max_list_length} items", max_str_length))
            return out

        if isinstance(v, (bytes, bytearray)):
            try:
                v = v.decode("utf-8", errors="ignore")
            except Exception:
                v = repr(v)

        if isinstance(v, str):
            return trunc(v, max_str_length)

        return trunc(repr(v), max_str_length)

    printable = to_printable(data)
    text = json.dumps(printable, ensure_ascii=False, separators=(",", ":"))

    if len(text) > max_total_length:
        text = trunc(text, max_total_length)

    return text



def serialize_payload_to_json(payload , profile: str = None):
    """
    Serialize a payload to a JSON-compatible format with size limits.
    """

    import numpy as np
    import json
    if profile is None:
        profile = str(CONFIG.get("json_record_profile", "normal")).lower()
    presets = {
        "compact": {"max_str": 512, "max_array": 64, "max_dict": 64, "max_depth": 8},
        "relaxed": {
            "max_str": 32768,
            "max_array": 2048,
            "max_dict": 2048,
            "max_depth": 12,
        },
        "normal": {"max_str": 4096, "max_array": 512, "max_dict": 512, "max_depth": 10},
        "unlimited": {
            "max_str": sys.maxsize,
            "max_array": sys.maxsize,
            "max_dict": sys.maxsize,
            "max_depth": sys.maxsize,
        },
    }
    if profile not in presets:
        print(f"[INS_WARN] Unknown JSON profile '{profile}', using 'normal' instead.")
    limits = presets.get(profile, presets["normal"])
    max_str = limits["max_str"]
    max_array = limits["max_array"]
    max_dict_items = limits["max_dict"]
    max_depth = limits["max_depth"]

    def truncate(value, depth=0):
        if depth > max_depth:
            return f"__TRUNCATED__(depth>{max_depth})"
        if isinstance(value, str):
            return (
                value if len(value) <= max_str else value[:max_str] + "...<truncated>"
            )
        if isinstance(value, (bytes, bytearray)):
            return truncate(value.decode("utf-8", errors="ignore"), depth)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            sliced = value[:max_array]
            truncated = [truncate(v, depth + 1) for v in sliced]
            if len(value) > max_array:
                truncated.append(f"__TRUNCATED_LIST__(size>{max_array})")
            return truncated
        if isinstance(value, dict):
            items = list(value.items())[:max_dict_items]
            truncated_dict = {str(k): truncate(v, depth + 1) for k, v in items}
            if len(value) > max_dict_items:
                truncated_dict["__TRUNCATED_DICT__"] = f"items>{max_dict_items}"
            return truncated_dict
        try:
            json.dumps(value)
            return value
        except TypeError:
            return truncate(str(value), depth)

    limited = truncate(payload)
    return limited



def save_pkl_with_limit(file_path, *payloads):
    """
    通用工具函数：保存数据到pkl文件，并限制文件大小。
    如果写入过程中超过大小限制，将删除文件并返回 False。
    """
    import dill
    limit_mb = int(CONFIG.get("pkl_record_max_size_mb", 200))
    max_size = limit_mb * 1024 * 1024

    class SizeLimitExceeded(Exception):
        pass

    class LimitedWriter:
        def __init__(self, f, limit_bytes: int):
            self._f = f
            self._limit = limit_bytes
            self.written = 0

        def write(self, b: bytes):
            # 先判断，再写入，避免写入超限数据
            new_total = self.written + len(b)
            if new_total > self._limit:
                raise SizeLimitExceeded(f"Exceeded {self._limit} bytes")
            self.written = new_total
            return self._f.write(b)

        def flush(self):
            return self._f.flush()

    try:
        # 直接写最终文件，但用限流写入器包裹，超限即中断并删除文件
        with open(file_path, "wb") as raw_f:
            lw = LimitedWriter(raw_f, max_size)
            try:
                for payload in payloads:
                    dill.dump(payload, lw)
                raw_f.flush()
            except SizeLimitExceeded:
                try:
                    raw_f.close()
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                print(
                    f"[INS_WARN] Result data exceeds {limit_mb}MB limit, skipping pkl save"
                )
                return False
        return True
    except Exception as e:
        print(f"[INS_WARN] Result cannot be saved as binary format: {e}")
        return False


def run_as_module(
    file_path: str,
    conda_env_name: str = None,
    feed_enter: bool = False,
):
    import subprocess
    from pathlib import Path
    import shlex

    """
    :param file_path: Python file path to execute
    :param conda_env_name: The name of the conda environment to run in. If None, use the current environment.
    :param feed_spaces: 若为 True，则启动一个 feeder 进程持续向 stdin 管道写入【换行符】（通过 bash 管道实现，因此仍然使用 subprocess.run）。
    :param space_interval: feeder 每次写入后的 sleep 秒数（避免占满 CPU）
    :param space_chunk_size: 每次写入的换行数量
    """
    space_interval: float = 1.0
    space_chunk_size: int = 1024
    file_path = Path(file_path).resolve()
    project_root = Path(BASE_DIR).resolve()
    try:
        rel_path = file_path.relative_to(project_root)
    except ValueError:
        raise ValueError(f"File {file_path} is not under project root {project_root}")

    if file_path.suffix != ".py":
        raise ValueError(f"{file_path} is not a .py file")

    rel_path = rel_path.with_suffix("")
    module_name = ".".join(rel_path.parts)

    if conda_env_name:
        command = ["conda", "run", "-n", conda_env_name, "python", "-m", module_name]
    else:
        command = ["python", "-m", module_name]

    if not feed_enter:
        subprocess.run(command, check=True)
        return

    # 使用 bash 管道：feeder stdout -> 目标命令 stdin
    target_cmd = " ".join(shlex.quote(x) for x in command)

    feeder_code = (
        "import sys,time\n"
        f"chunk='\\n' * {int(space_chunk_size)}\n"
        "try:\n"
        "    while True:\n"
        "        sys.stdout.write(chunk)\n"
        "        sys.stdout.flush()\n"
        f"        time.sleep({float(space_interval)})\n"
        "except BrokenPipeError:\n"
        "    pass\n"
    )
    feeder_cmd = f"python -c {shlex.quote(feeder_code)}"
    pipeline = f"{feeder_cmd} | {target_cmd}"
    subprocess.run(["bash", "-lc", pipeline], check=True)


def rename_project(old_name: str, new_name: str):
    import shutil

    print(f"[INS_INFO] Renaming project from {old_name} to {new_name}...")
    if old_name == new_name:
        print(f"[INS_INFO] Old name and new name are the same, no action taken.")
        return
    from Inspection.utils.path_manager import CUSTOM_ADAPTER_PATH

    dirs = os.listdir(CUSTOM_ADAPTER_PATH)
    for dir_name in dirs:
        if not dir_name.endswith(".py"):
            continue
        if old_name == dir_name[:-3]:
            old_path = os.path.join(CUSTOM_ADAPTER_PATH, dir_name)
            new_path = os.path.join(CUSTOM_ADAPTER_PATH, new_name + ".py")
            os.rename(old_path, new_path)
            print(f"[INS_INFO] Renamed adapter file: {old_path} -> {new_path}")
            with open(new_path, "r+") as f:
                # 修改前三行
                lines = f.readlines()
                f.seek(0)
                for i in range(min(4, len(lines))):
                    if lines[i].startswith("ENV_DIR"):
                        lines[i] = f"ENV_DIR = ENV_BASE + '{new_name}/'\n"
                    if lines[i].startswith("#"):
                        lines[i] = f"# {new_name} \n"
                f.writelines(lines)
                f.truncate()
            break
    from Inspection.utils.path_manager import SIMULATOR_PATH, DUMB_SIMULATOR_PATH
    import re

    base = DUMB_SIMULATOR_PATH
    base2 = SIMULATOR_PATH
    dirs = os.listdir(base)
    for dir_name in dirs:
        if old_name == dir_name:
            old_path = os.path.join(base, dir_name)
            new_path = os.path.join(base, new_name)
            # 如果目标路径存在,先删除
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
            shutil.move(old_path, new_path)
            print(f"[INS_INFO] Renamed simulation dir: {old_path} -> {new_path}")
            # 继续进入
            for sub_dir_name in os.listdir(new_path):
                if not sub_dir_name.endswith(".py"):
                    continue
                sub_file_path = os.path.join(new_path, sub_dir_name)
                # 使用全字匹配替换字符串
                with open(sub_file_path, "r") as f:
                    content = f.read()

                # 使用正则表达式进行全字匹配替换
                # \b 表示单词边界,确保只匹配完整的单词
                content = re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, content)
                with open(sub_file_path, "w") as f:
                    f.write(content)

    dirs2 = os.listdir(base2)
    for dir_name in dirs2:
        if old_name == dir_name:
            old_path = os.path.join(base2, dir_name)
            new_path = os.path.join(base2, new_name)
            # 如果目标路径存在,先删除
            if os.path.exists(new_path):
                shutil.rmtree(new_path)
            shutil.move(old_path, new_path)
            print(f"[INS_INFO] Renamed simulation dir: {old_path} -> {new_path}")
            for sub_dir_name in os.listdir(new_path):
                if not sub_dir_name.endswith(".py"):
                    continue
                sub_file_path = os.path.join(new_path, sub_dir_name)
                with open(sub_file_path, "r") as f:
                    content = f.read()
                content = re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, content)
                with open(sub_file_path, "w") as f:
                    f.write(content)

    from Inspection.utils.path_manager import INTERFACE_DOC_PATH

    dirs = os.listdir(INTERFACE_DOC_PATH)
    for dir_name in dirs:
        if dir_name == f"{old_name}.md":
            old_path = os.path.join(INTERFACE_DOC_PATH, dir_name)
            new_path = os.path.join(INTERFACE_DOC_PATH, f"{new_name}.md")
            os.rename(old_path, new_path)
            print(f"[INS_INFO] Renamed interface doc file: {old_path} -> {new_path}")
            break
    from Inspection.utils.path_manager import INTERFACE_INFO_PATH

    dirs = os.listdir(INTERFACE_INFO_PATH)
    for dir_name in dirs:
        if dir_name == f"{old_name}.json":
            old_path = os.path.join(INTERFACE_INFO_PATH, dir_name)
            new_path = os.path.join(INTERFACE_INFO_PATH, f"{new_name}.json")
            os.rename(old_path, new_path)
            print(f"[INS_INFO] Renamed interface info file: {old_path} -> {new_path}")
            break


def get_available_outter_resources_exts(type: str) -> list:
    if type == "images" or type == "image":
        return ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]
    elif type == "audios" or type == "audio":
        return ["wav", "mp3", "flac", "ogg", "aac"]
    elif type == "videos" or type == "video":
        return ["mp4", "avi", "mov", "mkv", "webm"]
    elif type == "texts" or type == "text":
        return ["txt", "json", "xml", "html", "css", "js", "md", "py", "jsonl"]
    else:
        raise ValueError(f"Unknown resource type: {type}")


def to_valid_module_name(filename: str) -> str:
    """
    将一个给定的文件名字符串转换为一个有效的 Python 模块名。
    规则：
    1. 移除 .py 后缀。
    2. 将所有非字母、数字或下划线的字符替换为下划线。
    3. 如果名称以数字开头，在其前面加上一个下划线。
    4. 如果名称是 Python 的关键字，在其末尾加上一个下划线。
    5. 如果结果为空，返回 'module'。

    :param filename: 可能的文件名或字符串。
    :return: 一个有效的 Python 模块名。
    """
    if filename.endswith(".py"):
        name = filename[:-3]
    else:
        name = filename
    # 2. 将所有无效字符替换为下划线
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # 3. 确保不以数字开头
    if name and name[0].isdigit():
        name = "_" + name
    # 4. 检查是否为 Python 关键字
    if keyword.iskeyword(name):
        name += "_"
    # 5. 处理空字符串的情况
    if not name:
        return "module"
    return name


class FileGuardHandler(FileSystemEventHandler):
    """文件守护处理器：限制目录下文件数量不超过200"""

    def __init__(self, path, max_files=200):
        self.path = path
        self.max_files = max_files
        self.lock = threading.Lock()

    def on_created(self, event):
        if not event.is_directory:
            self._check_and_cleanup()

    def _check_and_cleanup(self):
        with self.lock:
            try:
                items = os.listdir(self.path)
                files = [f for f in items if os.path.isfile(os.path.join(self.path, f))]
                if len(files) > self.max_files:
                    # 按修改时间排序，最新的在前
                    files_with_time = [
                        (f, os.path.getmtime(os.path.join(self.path, f))) for f in files
                    ]
                    files_with_time.sort(key=lambda x: x[1], reverse=True)
                    # 删除最新的文件
                    for i in range(len(files) - self.max_files):
                        os.remove(os.path.join(self.path, files_with_time[i][0]))
            except:
                pass


class ReadOnlyGuardHandler(FileSystemEventHandler):
    """只读守护处理器：阻止对目录的任何修改操作"""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        # 设置目录及其所有内容为只读
        self._set_readonly_recursive(path)

    def _set_readonly_recursive(self, path):
        """递归设置目录及其内容为只读"""
        try:
            # 设置目录权限为只读
            os.chmod(path, 0o444)
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o444)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o444)
        except Exception as e:
            print(f"[INS_WARN] Failed to set readonly: {e}")

    def on_modified(self, event):
        """文件被修改时重新设置为只读"""
        self._restore_readonly(event.src_path)

    def on_created(self, event):
        """新文件创建时设置为只读"""
        self._restore_readonly(event.src_path)

    def _restore_readonly(self, path):
        with self.lock:
            try:
                if os.path.isdir(path):
                    self._set_readonly_recursive(path)
                else:
                    os.chmod(path, 0o444)
            except:
                pass


# utf-8 encoding
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
