from CodeAnalysis.core.project_manager import ProjectManager
from CodeAnalysis.core.python_file import PythonFile
from CodeAnalysis.tools.dataclass_defs import Call
from typing import List
import os
import json
from tqdm import tqdm
import shutil

from CodeAnalysis.tools.key_words_manager import EXCEPT_CALL_PATH_KEY_WORDS  , MAX_CALL_STACK_DEPTH, get_start_python_files




class ProjectAnalyzer:
    def __init__(self ,project_path: str, record_debug_info: bool = False , record_base_dir: str = "./FileSummaries/", overwirte: bool = True):
        self.project_path = project_path
        self.record_debug_info = record_debug_info
        self.debug_logs = []  # 用于记录调试信息
        self.record_base_dir = record_base_dir + "pj_root" + self.project_path.replace("/", "_").replace(":", "") + "/"
        if not os.path.exists(self.record_base_dir):
            os.makedirs(self.record_base_dir)
        self.overwirte = overwirte
        if not overwirte:
            path = os.path.join(self.record_base_dir, "analysis_results.json")
            if os.path.exists(path):
                raise FileExistsError(f"Analysis results already exist at {path}. Set overwirte=True to overwrite.")

        self.manager = ProjectManager(project_path)
        self.manager.load_all()
        self.analysis_results : List[dict] = None
        
        # 添加缓存：记录哪些节点无法找到目标方法
        self.dead_end_cache = set()  # 存储格式: "file_path-entry_method"
        # 添加成功缓存：记录哪些节点可以找到目标方法



    def dfs(self, node: PythonFile, entry_method: str, visited: set, current_path: list, all_paths: List[List[dict]]) -> bool:
        """
        node: 当前分析的Python文件对象
        entry_method: 当前分析的入口方法 格式：类名.方法名 或 函数名
        visited: 当前路径下已访问的节点集合，用于避免死循环
        current_path: 当前探索路径（栈结构）
        all_paths: 收集到的所有匹配路径
        返回值: 是否至少找到一个目标方法
        """

        node_id = f"{node.path}-{entry_method}"
        # print(f"Visiting Node: {node_id}")
        
        # 检查缓存：如果已知这个节点是死胡同，直接返回
        if node_id in self.dead_end_cache:
            if self.record_debug_info:
                self.debug_logs.append(f"{'  ' * len(current_path)}[CACHE HIT - DEAD END] Skipping: {node_id}")
            return False
        
        # 关键修改：检查当前路径中是否已经访问过此节点（避免环路）
        if node_id in visited:
            if self.record_debug_info:
                self.debug_logs.append(f"{'  ' * len(current_path)}[CYCLE DETECTED] Skipping: {node_id}")
            return False  # 当前路径出现环

        visited.add(node_id)
        depth = len(current_path)
        indent = "  " * depth
        if self.record_debug_info:
            self.debug_logs.append(f"{indent}[ENTER] File: {node.path}, Method: {entry_method or '<module>'}, Depth: {depth}")


        current_path.append(f"{str(node.path)} <--- {entry_method or '<module>'}")

        matched_call = node.has_target_terminal_call(entry_method)
        found_any = False

        if matched_call:
            if self.record_debug_info:
                self.debug_logs.append(f"{indent}[SUCCESS] Found target method!")
                self.debug_logs.append(f"{indent}  Matched Call: {matched_call.call_name}")
                self.debug_logs.append(f"{indent}  Object Name: {matched_call.object_name}")
                self.debug_logs.append(f"{indent}  Object Type: {matched_call.object_type}")
            matched_node = f"{str(node.path)} <--- Call Matched Method: {matched_call.call_name} , Object Name: {matched_call.object_name}, Possible Types: {matched_call.object_type}"
            all_paths.append(current_path + [matched_node])
            found_any = True

        aim_calls : List[Call] = []
        if entry_method == "":
            aim_calls = node.calls
            for obj_call in node.object_calls:
                new_call = Call(
                    call_name = '__call__',
                    call_site= "<module>",
                    object_name= obj_call.object_name,
                    object_type= obj_call.object_type
                )
                aim_calls.append(new_call)
        else:
            aim_calls = [call for call in node.calls if call.call_site == entry_method]
            obj_calls = [call for call in node.object_calls if call.call_site == entry_method]
            
            for obj_call in obj_calls:
                new_call = Call(
                    call_name = '__call__',
                    call_site= entry_method,
                    object_name= obj_call.object_name,
                    object_type= obj_call.object_type
                )
                aim_calls.append(new_call)

        # 过滤掉调用路径包含 EXCEPT_CALL_PATH_KEY_WORDS 的调用
        filtered_aim_calls = []
        for call in aim_calls:
            found = False
            for keyword in EXCEPT_CALL_PATH_KEY_WORDS:
                if keyword in call.call_name:
                    if self.record_debug_info:
                        self.debug_logs.append(f"{indent}[SKIP] Skipping call due to except keywords: {call.call_name} contains {keyword}")
                    found = True
                    break
            if not found:
                filtered_aim_calls.append(call)
        aim_calls = filtered_aim_calls
        
        
        # 添加深度限制，防止无限递归
        if depth >= MAX_CALL_STACK_DEPTH:
            if self.record_debug_info:
                self.debug_logs.append(f"{indent}[MAX DEPTH] Reached maximum depth {MAX_CALL_STACK_DEPTH}, stopping")
            current_path.pop()
            visited.remove(node_id)
            return found_any
        
    
        for call in aim_calls:
            if self.record_debug_info:
                self.debug_logs.append(f"{indent}[CALL] Analyzing: {call.call_name}")
                self.debug_logs.append(f"{indent}  Object: {call.object_name}, Types: {call.object_type}, Site: {call.call_site}")
            method_str = ""
            if call.object_name != "" and call.object_name is not None:
                
                if len(call.object_type) == 0:
                    continue
                
                for obj_type in call.object_type:
                    method_str = f"{obj_type}.{call.call_name}"
                    
                    method_impl_pyfile = self.manager.get_method_impl_pythonfile(method_str, py_file=node)
                    if method_impl_pyfile is not None:
                        if self.dfs(method_impl_pyfile, method_str, visited, current_path, all_paths):
                            found_any = True
            else:
                method_str = call.call_name
                
                method_impl_pyfile = self.manager.get_method_impl_pythonfile(method_str, py_file=node)
                if method_impl_pyfile is not None:
                    if self.dfs(method_impl_pyfile, method_str, visited, current_path, all_paths):
                        found_any = True
    
        current_path.pop()
        visited.remove(node_id)

        # 如果没有找到任何目标方法，将此节点标记为死胡同
        if not found_any:
            self.dead_end_cache.add(node_id)
            if self.record_debug_info:
                self.debug_logs.append(f"{indent}[EXIT] File: {node.path}, Method: {entry_method or '<module>'} - Not found (cached as dead end)")
        return found_any
    
    def clear_cache(self):
        """清除缓存，在需要重新分析时调用"""
        self.dead_end_cache.clear()
        if self.record_debug_info:
            self.debug_logs.append("[CACHE] Cache cleared")

    def run(self, start_file_path: str = None):
        if start_file_path:
            start_files = [(self.manager.get_file_by_path(start_file_path), "用户指定入口文件")]
        else:
            start_files = get_start_python_files(self.manager)

        print(f"Found {len(start_files)} possible entry files for analysis.")
        results = []
        
        # 使用tqdm显示分析进度
        for idx, (start_file, reason) in enumerate(tqdm(start_files, desc="分析文件", unit="个"), 1):
            if self.record_debug_info:
                self.debug_logs.append(f"\n{'='*80}")
                self.debug_logs.append(f"[START] Analysis {idx}/{len(start_files)}: {start_file.path}")
                self.debug_logs.append(f"[REASON] {reason}")
                self.debug_logs.append(f"{'='*80}")
            visited = set()
            current_path = []
            call_paths : List[List[dict]] = []
            found = self.dfs(start_file, "", visited, current_path, call_paths)
            if self.record_debug_info:
                self.debug_logs.append(f"{'='*80}")
                self.debug_logs.append(f"[END] Analysis {idx}/{len(start_files)}: {'Found' if found else 'Not Found'}")
                self.debug_logs.append(f"{'='*80}\n")
            if found:
                results.append({
                    "entry_file_path": str(start_file.path),
                    "entry_file_index": idx,
                    "reason": reason,
                    "call_paths": call_paths
                })

        # 将结果存储到 self.analysis_results
        self.analysis_results = {
            "possible_entry_files": [
                {
                    "entry_file_path": str(file.path),
                    "entry_file_index": en_idx,
                    "reason": reason
                } for en_idx, (file, reason) in enumerate(start_files, 1)
            ],
            "analysis_results": results
        }

        print(f"Analysis complete.")


    def save_analysis_result(self):
        """保存分析结果到JSON文件"""
        if self.analysis_results is None:
            print("No analysis results to save. Please run analysis first.")
            return
        
        if self.overwirte:
            # 清空目录
            if os.path.exists(self.record_base_dir):
                shutil.rmtree(self.record_base_dir)
            os.makedirs(self.record_base_dir)

        record_base_dir = self.record_base_dir
        json_format_record_path = record_base_dir + "analysis_results.json"
        possible_entry_files_record_path = record_base_dir + "possible_entry_files.json"
        root_json_path = record_base_dir + "project_root.txt"
        default_name_path = record_base_dir + "default_name.txt"

        # 创建目录
        if not os.path.exists(record_base_dir):
            os.makedirs(record_base_dir)

        with open(root_json_path, 'w', encoding='utf-8') as f:
            f.write(self.project_path)
        
        with open(default_name_path, 'w', encoding='utf-8') as f:
            name = self.project_path.split("/")[-1].replace("-","_").replace(".","_")
            f.write(name)


        # 写入分析结果JSON文件

        with open(json_format_record_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results["analysis_results"], f, ensure_ascii=False, indent=4)

        # 写入可能的入口文件JSON文件
        with open(possible_entry_files_record_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results["possible_entry_files"], f, ensure_ascii=False, indent=4)

        # 记录调试信息到文件
        if self.record_debug_info and self.debug_logs:
            debug_log_path = record_base_dir + "debug_logs.txt"
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.debug_logs))
        
        

        self.manager.write_all_file_summaries(record_base_dir + 'project_pyfile_results/')
        
        print(f"Results saved to {json_format_record_path} and {possible_entry_files_record_path}.")

        self._save_call_implementation_pairs()
        

    
    def _save_call_implementation_pairs(self):
        if self.analysis_results is None:
            print("No analysis results to save. Please run analysis first.")
            return
        
        
        save_dir = self.record_base_dir + "call_implementation_pairs/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for result in self.analysis_results["analysis_results"]:
            entry_file_index = result["entry_file_index"]
            call_python_file = self.manager.get_file_by_path(result["entry_file_path"])
            call_paths = result.get("call_paths", [])
            all_pairs = []
            call_path_idx = 1
            for call_path in call_paths:
                base_class_code = ""
                if len(call_path) < 2:
                    continue  # 没有实现部分，跳过
                impl_python_file_path = call_path[1].split(" <--- ")[0].strip()
                impl_pytho_file = self.manager.get_file_by_path(impl_python_file_path)
                method = call_path[1].split(" <--- ")[1].strip()
                method_data = None
                if method.find(".")!=-1:
                    cls_data = None
                    class_name = method.split(".")[0]
                    method_name = method.split(".")[1]
                    for cls in impl_pytho_file.class_defs:
                        if cls.class_name == class_name:
                            cls_data = cls
                            for func in cls.methods:
                                if func.function_name == method_name:
                                    method_data = func
                                    break
                            break
                    base_classes = cls_data.base_classes if cls_data else []
                    for base_class in base_classes:
                        base_class_impl_py_file = self.manager.get_class_impl_pythonfile(base_class,impl_pytho_file)
                        code = base_class_impl_py_file.code if base_class_impl_py_file else ""
                        base_class_code += f"\n\n\n# {class_name} base class: {base_class} in file {base_class_impl_py_file.path if base_class_impl_py_file else 'Not Found'}\n"
                        base_class_code += code
                        base_class_code += f"\n\n\n"

                else:
                    class_name = None
                    method_name = method
                    for func in impl_pytho_file.top_level_defs:
                        if func.function_name == method_name:
                            method_data = func
                            break
                if method_data is None:
                    raise ValueError(f"Cannot find method data for {method} in file {impl_pytho_file.path}")

                method_str = method.replace(".", "_").replace("(", "_").replace(")", "_")

                pair_dict = {
                    "call_path_index": call_path_idx,
                    "entered_from": str(call_python_file.path),
                    "project_root": str(self.project_path),
                    "call_intelligent_path": call_path,
                    "call_data":{
                        "name": f"call_{method_str}",
                        "code": call_python_file.code,
                        "description": "",
                        "path": str(call_python_file.path),
                    },
                    "implementation_data":{
                        "name": f"impl_{method_str}",
                        "class": class_name,
                        "method": method_name,
                        "arguments": method_data.position_parameters + method_data.keyword_parameters,
                        "implementation": impl_pytho_file.code + (base_class_code if base_class_code else ""),
                        "path": str(impl_pytho_file.path),
                        "description": "",
                        "example": []
                    },
                }
                call_path_idx += 1
                all_pairs.append(pair_dict)
                

            name = "entry_idx_" + str(entry_file_index) +"_"+  str(call_python_file.name).replace(".py","") + ".json"
            py_data_path = os.path.join(save_dir, "py_data/", f"entry_idx_" + str(entry_file_index) +"_"+  str(call_python_file.name).replace(".py","") + "/")
            for pair in all_pairs:
                py_path = os.path.join(py_data_path, f"path_idx_{pair['call_path_index']}")
                if not os.path.exists(py_path):
                    os.makedirs(py_path)
                call_py_path = os.path.join(py_path, "call_code.py")
                impl_py_path = os.path.join(py_path, "implementation_code.py")
                with open(call_py_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Call File Path: {pair['call_data']['path']}\n")
                    f.write(f"# {'='*80}\n")
                    f.write(f"# Call Name: {pair['call_data']['name'][5:]}\n")
                    f.write(f"# {'='*80}\n")
                    f.write(f"# Call Code\n")
                    f.write(f"# {'='*80}\n\n")
                    f.write(pair['call_data']['code'])
                with open(impl_py_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Implementation File Path: {pair['implementation_data']['path']}\n")
                    f.write(f"# {'='*80}\n")
                    f.write(f"# Implementation Name: {pair['implementation_data']['name'][5:]}\n")
                    f.write(f"# {'='*80}\n")
                    f.write(f"# Implementation Code\n")
                    f.write(f"# {'='*80}\n\n")
                    f.write(pair['implementation_data']['implementation'])

        

            json_file_path = os.path.join(save_dir, name)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_pairs, f, ensure_ascii=False, indent=4)

    def save_analyzer(self):
        import pickle
        save_path = self.record_base_dir + "project_analyzer.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"ProjectAnalyzer object saved to {save_path}")

    def load_analyzer(self, load_path: str = None):
        if load_path is None:
            load_path = self.record_base_dir + "project_analyzer.pkl"
        import pickle
        with open(load_path, 'rb') as f:
            loaded_analyzer = pickle.load(f)
        self.project_path = loaded_analyzer.project_path
        self.record_debug_info = loaded_analyzer.record_debug_info
        self.manager = loaded_analyzer.manager
        self.debug_logs = loaded_analyzer.debug_logs
        self.record_base_dir = loaded_analyzer.record_base_dir
        self.analysis_results = loaded_analyzer.analysis_results
        self.dead_end_cache = loaded_analyzer.dead_end_cache
        print(f"ProjectAnalyzer object loaded from {load_path}")