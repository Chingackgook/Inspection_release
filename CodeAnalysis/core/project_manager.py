from CodeAnalysis.core.python_file import PythonFile


from collections import deque
from pathlib import Path
from typing import Dict, List , Optional
from tqdm import tqdm
import re

from CodeAnalysis.tools.key_words_manager import IGNORE_DIRS , IGNORE_PYTHON_FILES , PYTHON_MAX_FILE_SIZE_MB

class ProjectManager:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.ignore_dirs = IGNORE_DIRS
        self.ignore_python_files: List[str] = IGNORE_PYTHON_FILES
        self.python_files: Dict[str, PythonFile] = {}
        self.max_file_size_bytes = PYTHON_MAX_FILE_SIZE_MB * 1024 * 1024  # 转换为字节
        self.skipped_files = []  # 记录跳过的文件

        self.all_class_def_full_info = []

        self.all_class_def_names = set()
        self.all_top_level_function_names = set()
    


    def load_all(self):
        """加载并解析所有Python文件，使用进度条显示"""
        files = []
        print(f"开始收集Python文件: {self.project_path}")
        print(f"文件大小限制: {self.max_file_size_bytes / (1024 * 1024):.1f}MB")
        
        self._collect_python_files(self.project_path, files)
        
        print(f"\n共收集到 {len(files)} 个Python文件")
        if self.skipped_files:
            print(f"跳过 {len(self.skipped_files)} 个超大文件")
        
        print("\n开始解析文件...")
        
        # 使用tqdm显示解析进度
        for file_path in tqdm(sorted(files), desc="解析文件", unit="个"):
            try:
                rel_path = str(file_path.relative_to(self.project_path))
                py_file = PythonFile(str(file_path), project_manager=self)
                self.python_files[rel_path] = py_file

                # 收集所有类名
                for cls_def in py_file.class_defs:
                    self.all_class_def_names.add(cls_def.class_name)
                    self.all_class_def_full_info.append(cls_def)
                # 收集所有顶层函数名
                for func_def in py_file.top_level_defs:
                    self.all_top_level_function_names.add(func_def.function_name)
            except Exception as e:
                tqdm.write(f"❌ 解析失败: {file_path} - {e}")
        
        self._process_from_module_import_star()
    
    def get_file_by_path(self, relative_path: str) -> PythonFile:
        """获取指定的Python文件对象，支持绝对路径和相对路径"""
        # 如果是绝对路径，转换为相对路径
        path_obj = Path(relative_path)
        if path_obj.is_absolute():
            try:
                rel_path = str(path_obj.relative_to(self.project_path))
            except ValueError:
                raise ValueError(f"绝对路径 {relative_path} 不在项目目录 {self.project_path} 下")
        else:
            rel_path = relative_path

        file = self.python_files.get(rel_path)
        if file is None:
            # 打印所有可用的文件路径以供调试
            available_files = "\n".join(self.python_files.keys())
            print(f"可用的文件路径:\n{available_files}")
            raise ValueError(f"文件未找到: {relative_path}")
        return file
    
    def get_method_impl_pythonfile(self, method_name: str, py_file: PythonFile) -> Optional[PythonFile]:
        """
        method_name格式：
        classname.methodname
        或
        functionname
        """
        # print(f"查找方法实现: {method_name} 从文件 {py_file.path} 开始")
        if method_name.find('.') != -1:
            class_name = method_name.split('.')[0]
            method_name_part = method_name.split('.')[1]
            # 先在当前文件中查找
            for cls_def in py_file.class_defs:
                if cls_def.class_name == class_name:
                    for method in cls_def.methods:
                        if method.function_name == method_name_part:
                            return py_file
        
            module_name = ""
            for moudle, import_name in py_file.from_imports_pairs:
                if import_name == class_name:
                    module_name = moudle
                    break

            if module_name != "":
                # 使用链式导入追踪查找类
                going_process_modules = [(module_name, py_file)]
                processed_modules = set()  # 只存储 (module_name, file_path) 避免重复访问
                
                while len(going_process_modules) > 0:
                    now_module, current_file = going_process_modules.pop(0)
                    module_key = (now_module, str(current_file.path))
                    
                    # 关键修复：检查是否已处理过此模块+文件组合
                    if module_key in processed_modules:
                        continue
                    processed_modules.add(module_key)
                    
                    module_pyfile = self.get_python_file_with_module_import_statement(now_module, current_file=current_file)
                    if module_pyfile is not None:
                        for cls_def in module_pyfile.class_defs:
                            if cls_def.class_name == class_name:
                                for method in cls_def.methods:
                                    if method.function_name == method_name_part:
                                        return module_pyfile
                        # 继续处理该模块的from_imports_pairs
                        for moudle, import_name in module_pyfile.from_imports_pairs:
                            new_key = (moudle, str(module_pyfile.path))
                            if import_name == class_name and new_key not in processed_modules:
                                going_process_modules.append((moudle, module_pyfile))
            else:
                # 粗略匹配
                py_file_result = self.get_class_impl_pythonfile(class_name)
                if py_file_result is not None:
                    for cls_def in py_file_result.class_defs:
                        if cls_def.class_name == class_name:
                            for method in cls_def.methods:
                                if method.function_name == method_name_part:
                                    return py_file_result

        else:
            # 先在当前文件中查找
            for func_def in py_file.top_level_defs:
                if func_def.function_name == method_name:
                    return py_file
            
            module_name = ""
            for moudle, import_name in py_file.from_imports_pairs:
                if import_name == method_name:
                    module_name = moudle
                    break
            
            if module_name != "":
                going_process_modules = [(module_name, py_file)]
                processed_modules = set()
                
                while len(going_process_modules) > 0:
                    now_module, current_file = going_process_modules.pop(0)
                    module_key = (now_module, str(current_file.path))
                    
                    # 关键修复：检查是否已处理过此模块+文件组合
                    if module_key in processed_modules:
                        continue
                    processed_modules.add(module_key)
                    
                    module_pyfile = self.get_python_file_with_module_import_statement(now_module, current_file=current_file)
                    if module_pyfile is not None:
                        for func_def in module_pyfile.top_level_defs:
                            if func_def.function_name == method_name:
                                return module_pyfile
                        # 继续处理该模块的from_imports_pairs
                        for moudle, import_name in module_pyfile.from_imports_pairs:
                            new_key = (moudle, str(module_pyfile.path))
                            if import_name == method_name and new_key not in processed_modules:
                                going_process_modules.append((moudle, module_pyfile))
            else:
                # 粗略匹配
                py_file_result = self.get_function_impl_pythonfile(method_name)
                if py_file_result is not None:
                    for func_def in py_file_result.top_level_defs:
                        if func_def.function_name == method_name:
                            return py_file_result

        # 最后的全局搜索
        for py_file_obj in self.python_files.values():
            # 检查类方法
            for cls_def in py_file_obj.class_defs:
                for method in cls_def.methods:
                    full_method_name = f"{cls_def.class_name}.{method.function_name}"
                    if full_method_name == method_name:
                        return py_file_obj
            # 检查顶层函数
            for func_def in py_file_obj.top_level_defs:
                if func_def.function_name == method_name:
                    return py_file_obj
        return None
    
    def get_class_impl_pythonfile(self, class_name: str, py_file: PythonFile = None) -> PythonFile | None:
        """
        获取实现指定类的Python文件对象
        如果提供了py_file，则从该文件的导入关系开始精确查找
        """
        # 如果提供了py_file，先在当前文件中查找
        if py_file is not None:
            for cls_def in py_file.class_defs:
                if cls_def.class_name == class_name:
                    return py_file
            
            # 查找导入的模块
            module_name = ""
            for moudle, import_name in py_file.from_imports_pairs:
                if import_name == class_name:
                    module_name = moudle
                    break
            
            if module_name != "":
                # 使用链式导入追踪查找类
                going_process_modules = [(module_name, py_file)]
                processed_modules = set()
                
                while len(going_process_modules) > 0:
                    now_module, current_file = going_process_modules.pop(0)
                    module_key = (now_module, str(current_file.path))
                    
                    if module_key in processed_modules:
                        continue
                    processed_modules.add(module_key)
                    
                    module_pyfile = self.get_python_file_with_module_import_statement(now_module, current_file=current_file)
                    if module_pyfile is not None:
                        for cls_def in module_pyfile.class_defs:
                            if cls_def.class_name == class_name:
                                return module_pyfile
                        # 继续处理该模块的from_imports_pairs
                        for moudle, import_name in module_pyfile.from_imports_pairs:
                            new_key = (moudle, str(module_pyfile.path))
                            if import_name == class_name and new_key not in processed_modules:
                                going_process_modules.append((moudle, module_pyfile))
        
        # 最后的全局搜索（兜底方案）
        for py_file_obj in self.python_files.values():
            for cls_def in py_file_obj.class_defs:
                if cls_def.class_name == class_name:
                    return py_file_obj
        
        return None
    
    def get_function_impl_pythonfile(self, function_name: str, py_file: PythonFile = None) -> PythonFile | None:
        """
        获取实现指定函数的Python文件对象
        如果提供了py_file,则从该文件的导入关系开始精确查找
        """
        # 如果提供了py_file,先在当前文件中查找
        if py_file is not None:
            for func_def in py_file.top_level_defs:
                if func_def.function_name == function_name:
                    return py_file
            
            # 查找导入的模块
            module_name = ""
            for moudle, import_name in py_file.from_imports_pairs:
                if import_name == function_name:
                    module_name = moudle
                    break
            
            if module_name != "":
                # 使用链式导入追踪查找函数
                going_process_modules = [(module_name, py_file)]
                processed_modules = set()
                
                while len(going_process_modules) > 0:
                    now_module, current_file = going_process_modules.pop(0)
                    module_key = (now_module, str(current_file.path))
                    
                    if module_key in processed_modules:
                        continue
                    processed_modules.add(module_key)
                    
                    module_pyfile = self.get_python_file_with_module_import_statement(now_module, current_file=current_file)
                    if module_pyfile is not None:
                        for func_def in module_pyfile.top_level_defs:
                            if func_def.function_name == function_name:
                                return module_pyfile
                        # 继续处理该模块的from_imports_pairs
                        for moudle, import_name in module_pyfile.from_imports_pairs:
                            new_key = (moudle, str(module_pyfile.path))
                            if import_name == function_name and new_key not in processed_modules:
                                going_process_modules.append((moudle, module_pyfile))
        
        # 最后的全局搜索(兜底方案)
        for py_file_obj in self.python_files.values():
            for func_def in py_file_obj.top_level_defs:
                if func_def.function_name == function_name:
                    return py_file_obj
        
        return None
    
    
    def write_all_file_summaries(self, output_dir):
        output_path = Path(output_dir)
        
        for rel_path, py_file in self.python_files.items():
            # 保持原始目录结构
            summary_path = output_path / rel_path
            # 将 .py 后缀替换为 _summary.txt
            summary_path = summary_path.with_suffix('.py_summary.txt')
            # 确保父目录存在
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            py_file.wirte_summary_to_file(str(summary_path))

                    
    
    def get_python_file_with_module_import_statement(self, module_path: str, current_file: PythonFile = None) -> Optional[PythonFile]:
        """
        根据模块导入语句获取对应的Python文件对象
        Args:
            module_path: 模块路径，如 "..module.b", "Inspection.a.b", "a.b.c"
            current_file: 当前PythonFile对象，用于解析相对导入
        Returns:
            对应的PythonFile对象，如果找不到则返回None
        """
        # 处理相对导入（以.开头）
        if module_path.startswith('.'):
            if current_file is None:
                return None
            
            # 计算相对导入的层级
            level = len(module_path) - len(module_path.lstrip('.'))
            module_name = module_path.lstrip('.')
            
            # 获取当前文件的相对路径
            current_rel_path = str(current_file.path.relative_to(self.project_path))
            current_dir = Path(current_rel_path).parent
            
            # 向上level-1层
            for _ in range(level - 1):
                current_dir = current_dir.parent
            
            # 构建完整的模块路径
            if module_name:
                full_path = current_dir / module_name.replace('.', '/')
            else:
                full_path = current_dir
            
            # 尝试可能的文件路径
            possible_paths = [
                str(full_path) + '.py',
                str(full_path / '__init__.py'),
            ]
            
            for rel_path in possible_paths:
                if rel_path in self.python_files:
                    return self.python_files[rel_path]
            
            return None
        
        # 处理绝对导入
        parts = module_path.split('.')
        
        # 尝试不同长度的路径组合
        for i in range(len(parts), 0, -1):
            # 构建可能的文件路径
            possible_paths = [
                # 作为文件：a/b/c.py
                '/'.join(parts[:i]) + '.py',
                # 作为包：a/b/c/__init__.py
                '/'.join(parts[:i]) + '/__init__.py',
            ]
            
            for rel_path in possible_paths:
                if rel_path in self.python_files:
                    return self.python_files[rel_path]
        
        return None
    





    def _process_from_module_import_star(self):
        for py_file in self.python_files.values():
            new_pairs = []
            for from_moudle , import_name in py_file.from_imports_pairs:
                if import_name == "*":
                    # 处理星号导入
                    module_pyfile = self.get_python_file_with_module_import_statement(from_moudle, current_file=py_file)
                    if module_pyfile is not None:
                        # 获取该模块的所有顶层定义
                        for func_def in module_pyfile.top_level_defs:
                            new_pairs.append((from_moudle, func_def.function_name))
                        for cls_def in module_pyfile.class_defs:
                            new_pairs.append((from_moudle, cls_def.class_name))
            py_file.from_imports_pairs = [pair for pair in py_file.from_imports_pairs if pair[1] != "*"]
            if len(new_pairs) > 0:
                py_file.from_imports_pairs.extend(new_pairs)


    def _collect_python_files(self, path: Path, files: List[Path]):
        """使用迭代方式收集Python文件，避免递归栈溢出，并过滤超大文件"""
        dirs_to_process = deque([path])
        while dirs_to_process:
            current_dir = dirs_to_process.popleft()
            try:
                for item in current_dir.iterdir():
                    if item.name in self.ignore_dirs or item.name.startswith('.'):
                        continue
                    if item.is_file() and item.suffix == '.py':
                        file_name = item.name
                        if file_name in self.ignore_python_files:
                            continue
                        # 检查文件大小
                        file_size = item.stat().st_size
                        if file_size > self.max_file_size_bytes:
                            size_mb = file_size / (1024 * 1024)
                            self.skipped_files.append((str(item), size_mb))
                            print(f"⚠️  跳过超大文件: {item} ({size_mb:.2f}MB)")
                        else:
                            files.append(item)
                    elif item.is_dir():
                        dirs_to_process.append(item)
            except (PermissionError, FileNotFoundError) as e:
                print(f"⚠️  无法访问: {current_dir} ({e})")
            
