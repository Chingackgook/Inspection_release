from pathlib import Path
from typing import List , Optional , Tuple
from CodeAnalysis.tools.codeprocesser import extract_from_import
from CodeAnalysis.tools.codeprocesser import extract_import_objects
from CodeAnalysis.tools.codeprocesser import get_functions_and_class_methods
from CodeAnalysis.tools.codeprocesser import get_function_param_names
from CodeAnalysis.tools.codeprocesser import get_all_function_calls
from CodeAnalysis.tools.codeprocesser import get_object_class_names
from CodeAnalysis.tools.codeprocesser import get_object_call_invocations

from CodeAnalysis.tools.dataclass_defs import FunctionDef , ClassDef , Call , ObjectCall
from CodeAnalysis.tools.key_words_manager import FINAL_CALL_LIST , FINAL_OBJECT_CALL_KEY_WORDS



class PythonFile:
    def __init__(self, path: str , project_manager=None):
        self.path = Path(path)
        self.code = self.path.read_text()
        self.name = self.path.name
        self.imports: List[str] = []
        self.from_imports_pairs: List[Tuple[str, str]] = []
        self.top_level_defs: List[FunctionDef] = []
        self.class_defs: List[ClassDef] = []
        self.calls : List[Call] = None
        self.object_calls : List[ObjectCall] = None
        self.project_manager = project_manager
        self._solve_imports()
        self._solve_defs()

    def _solve_imports(self):
        imports = extract_import_objects(self.code)
        from_imports_objects = extract_from_import(self.code)
        self.imports = imports
        self.from_imports_pairs = from_imports_objects

    def _solve_defs(self):
        top_level_functions, class_methods = get_functions_and_class_methods(self.code)
        for func in top_level_functions:
            position_params , keyword_params = get_function_param_names(self.code, func , "")
            func_def = FunctionDef(
                function_name=func,
                position_parameters=position_params,
                keyword_parameters=keyword_params
            )
            self.top_level_defs.append(func_def)
        
        for cls in class_methods:
            class_name = cls['class_name']
            base_classes = cls.get('base_classes', [])  # 获取父类列表
            methods = []
            init_method = None
            for method in cls['methods']:
                position_params , keyword_params = get_function_param_names(self.code, method , class_name)
                func_def = FunctionDef(
                    function_name=method,
                    position_parameters=position_params,
                    keyword_parameters=keyword_params
                )
                if method == '__init__':
                    init_method = func_def
                else:
                    methods.append(func_def)
            class_def = ClassDef(
                class_name=class_name,
                base_classes=base_classes,  # 设置父类列表
                init_method=init_method,
                methods=methods
            )
            self.class_defs.append(class_def)

    
    def _solve_calls(self):
        self.calls = []
        calls_info = get_all_function_calls(self.code)
        for call_item in calls_info.get('calls', []):
            call_name = call_item.get('name', '')
            obj_name = call_item.get('obj', None)
            parent_name = call_item.get('parent_name', '')
            # 获取对象类型（如果有对象名）
            possible_obj_type = []
            
            # 首先遍历已经分析出的对象类型
            already_found = False
            for call in self.calls:
                if call.object_name == obj_name and call.call_site == parent_name:
                    # 检查是否已经存在相同的调用,避免重复添加
                    is_duplicate = any(
                        c.call_name == call_name and 
                        c.call_site == parent_name and 
                        c.object_name == obj_name 
                        for c in self.calls
                    )
                    if not is_duplicate:
                        self.calls.append(
                            Call(
                                call_name=call_name,
                                call_site=parent_name,
                                object_name=obj_name,
                                object_type=call.object_type  # 复用已知的类型
                            )
                        )
                    already_found = True
                    break
            if already_found:
                continue
                

            if obj_name:
                if obj_name == 'self':
                    class_name = parent_name.split('.')[0]
                    possible_obj_type = [class_name]
                else:
                    # 尝试获取该对象的类型，并包括继承的类型
                    possible_obj_type = get_object_class_names(self.code, obj_name)
                    all_class_defs = self.project_manager.all_class_def_full_info
                    going_extend = []
                    for possible_type in possible_obj_type:
                        for cls_def in all_class_defs:
                            baseclasses = cls_def.base_classes
                            for basecls in baseclasses:
                                if possible_type == basecls:
                                    going_extend.append(cls_def.class_name)
                    possible_obj_type.extend(going_extend)

                
                if len(possible_obj_type) == 0:
                    # 也可能是import进来的包内的方法
                    for imp in self.imports:
                        if imp == obj_name:
                            possible_obj_type = []
                            obj_name = None
                            self.from_imports_pairs.append( (imp , call_name) )
                            break

            call = Call(
                call_name=call_name,
                call_site=parent_name,
                object_name=obj_name,
                object_type=possible_obj_type
            )
            self.calls.append(call)
    
    def _solve_object_calls(self):
        # 该方法不会在solve中自动调用，会在has_aim_call中按需调用
        all_class_defs = self.project_manager.all_class_def_names
        
        
        imported_class_names = set()
        for  _ , imp in self.from_imports_pairs:
            if imp in all_class_defs:
                imported_class_names.add(imp)

        self.object_calls = []
        obj_call_datas = get_object_call_invocations(self.code , list(imported_class_names))

        for obj_call in obj_call_datas:
            obj_name = obj_call.get('obj_name')
            parent_name = obj_call.get('parent_name')
            class_type = obj_call.get('class_type')
            object_call = ObjectCall(
                call_site=parent_name,
                object_name=obj_name,
                object_type=class_type
            )
            self.object_calls.append(object_call)
    
    def has_if_name_main(self) -> bool:
        """
        检查文件中是否包含 if __name__ == '__main__' 语句
        """
        lines = self.code.splitlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("if __name__") and "__main__" in stripped_line:
                return True
        return False


    def has_target_terminal_call(self , entry_method: str) -> Optional[Call]:
        obj_call = self.has_object_call(entry_method)
        method_function_call = self.has_call(entry_method)
        if obj_call is not None:
            return obj_call
        if method_function_call is not None:
            return method_function_call
        return None

    def has_object_call(self , call_site: str) -> Optional[Call]:
        if self.object_calls is None:
            self._solve_object_calls()
        
        avaliavle_obj_names_keywords = FINAL_OBJECT_CALL_KEY_WORDS

        for obj_call in self.object_calls:

            if obj_call.call_site != call_site:
                continue
            if any(keyword in str(obj_call.object_name) for keyword in avaliavle_obj_names_keywords):
                return Call(
                    call_name="__call__",
                    call_site=obj_call.call_site,
                    object_name=obj_call.object_name,
                    object_type=obj_call.object_type
                )
            elif len(obj_call.object_type) > 0:
                for obj_type in obj_call.object_type:
                    # 检查所有的obj_type是不是继承自nn.Module
                    class_impl_pyfile = self.project_manager.get_class_impl_pythonfile(obj_type)
                    if class_impl_pyfile is not None:
                        for cls_def in class_impl_pyfile.class_defs:
                            if cls_def.class_name == obj_type:
                                if 'nn.Module' in cls_def.base_classes:
                                    return Call(
                                        call_name="__call__",
                                        call_site=obj_call.call_site,
                                        object_name=obj_call.object_name,
                                        object_type=obj_call.object_type
                                    )


        return None

    def has_call(self , call_site: str) -> Optional[Call]:
        aim_calls = FINAL_CALL_LIST

        if self.calls is None:
            self._solve_calls()
        
        """检查当前文件是否在调用点call_site
        包含目标调用"""
        for aim_call in aim_calls:
            for call in self.calls:
                if call.call_site != call_site:
                    continue
                if call.call_name == aim_call.call_name:
                    if aim_call.object_name != "ANY":
                        if call.object_name is None or aim_call.object_name is None:
                            continue
                        if call.object_name in aim_call.object_name or aim_call.object_name in call.object_name:
                            if aim_call.object_type is None or call.object_type is None:
                                continue
                            if aim_call.object_type == "ANY" or any(otype in call.object_type for otype in aim_call.object_type):
                                return call
                    else:
                        return call
        return None
                
        


    def wirte_summary_to_file(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write(f"File: {self.path}\n")
            f.write("Imports:\n")
            for imp in self.imports:
                f.write(f"  {imp}\n")
            f.write("From Imports:\n")
            for from_imp in self.from_imports_pairs:
                f.write(f"  {from_imp[0]} import {from_imp[1]}\n")
            f.write("Top-level Functions:\n")
            for func in self.top_level_defs:
                f.write(f"  {func.function_name}({', '.join(func.position_parameters + func.keyword_parameters)})\n")
            f.write("Classes:\n")
            for cls in self.class_defs:
                f.write(f"  Class: {cls.class_name}\n")
                f.write(f"    Base Classes: {', '.join(cls.base_classes)}\n")
                if cls.init_method:
                    f.write(f"    __init__({', '.join(cls.init_method.position_parameters + cls.init_method.keyword_parameters)})\n")
                for method in cls.methods:
                    f.write(f"    Method: {method.function_name}({', '.join(method.position_parameters + method.keyword_parameters)})\n")
            if self.calls is not None:
                f.write("Function Calls:\n")
                for call in self.calls:
                    f.write(f"  Call Name: {call.call_name}, Call Site: {call.call_site}, Object Name: {call.object_name}, Object Types: {', '.join(call.object_type)}\n")
            if self.object_calls is not None:
                f.write("Object Calls:\n")
                for obj_call in self.object_calls:
                    f.write(f"  Call Site: {obj_call.call_site}, Object Name: {obj_call.object_name}, Object Types: {', '.join(obj_call.object_type)}\n")

            f.write("\n\n\n\nCode:\n")
            f.write(self.code)