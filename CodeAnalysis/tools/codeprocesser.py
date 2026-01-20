import re
import sys
import ast
from typing import List, Dict, Set



if sys.version_info < (3, 9):
    import astor  # Used to convert AST to source code, Python 3.8 and below versions need to install astor library


def has_syntax_error(code_str: str) -> bool:
    """
    判断输入的 Python 代码字符串是否存在语法错误。

    参数:
        code_str (str): 待检测的 Python 代码字符串。

    返回:
        bool: 有语法错误返回 True，无语法错误返回 False。
    """
    try:
        ast.parse(code_str)
        return False
    except SyntaxError:
        return True


def extract_code(text , language=None , first_only=False):
    """
    Extract code blocks from text
    """
    if language is None:
        pattern = re.compile(r'```(.*?)```', re.DOTALL)
    else:
        pattern = re.compile(rf'```{language}(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    matchstr = ""
    for match in matches:
        if first_only:
            return match.strip()
        matchstr += match.strip() + "\n"
    return matchstr



def extract_from_import(code):
    """
    提取代码中的 from ... import ... 的导入信息
    返回: List[Tuple[str, str]] - [(模块路径, 导入对象名), ...]
    - 模块路径: from 后面的部分（如 'os.path'）
    - 导入对象名: import 后面的对象名，包括别名（如果使用 as）
    - 不忽略 * 导入
    """
    tree = ast.parse(code)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # 获取模块路径，如果为相对导入，module可能为None
            module = node.module if node.module else ''
            # 处理相对导入的点号
            if node.level > 0:
                module = '.' * node.level + module
            
            for alias in node.names:
                # 如果有别名，使用别名；否则使用原名
                imported_name = alias.asname if alias.asname else alias.name
                imports.append((module, imported_name))
    
    return imports
    
def extract_import_objects(code):
    """
    提取代码中的普通 import 语句导入的包名列表
    - 对于 import xxx，提取 xxx
    - 对于 import xxx as yyy，提取 yyy（别名）
    - 对于 import xxx.yyy，提取 xxx.yyy（完整包路径）
    - 对于 import xxx.yyy as zzz，提取 zzz（别名）
    """
    try:
        tree = ast.parse(code)
        objects = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # 如果有别名，使用别名；否则使用完整的包名
                    if alias.asname:
                        objects.append(alias.asname)
                    else:
                        # 保留完整的包路径，如 xxx.yyy.zzz
                        objects.append(alias.name)
        return objects
    except Exception as e:
        print(f"[INS_WARN] {e}")
        print("[INS_WARN] AST parsing failed, using regex to match import statements")
        # 使用正则表达式回退方案
        code = re.sub(r'#.*', '', code)  # 去除注释
        pattern = re.compile(r'^\s*import\s+([^\n]+)', re.MULTILINE)
        objects = []
        for match in pattern.findall(code):
            # 处理多个导入：import a, b, c
            for item in match.split(','):
                item = item.strip()
                if not item:
                    continue
                # 处理 as 别名
                if ' as ' in item:
                    parts = item.split(' as ')
                    alias = parts[1].strip()
                    objects.append(alias)
                else:
                    # 保留完整的包路径
                    objects.append(item.strip())
        return objects



def get_functions_and_class_methods(code: str):
    """
    提取代码中的顶层函数和类的方法信息。
    
    返回:
        tuple: (top_level_functions, class_methods)
        - top_level_functions: List[str] 顶层函数名列表
        - class_methods: List[dict] 每个元素是字典 {'class_name': str, 'methods': List[str]}
    """
    try:
        top_level_functions = []
        class_methods = []

        class FunctionAndMethodVisitor(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack = []
                self.current_class_methods = []
                self.current_class_name = None

            def visit_FunctionDef(self, node):
                if self._is_top_level_function():
                    # 顶层函数,不过滤私有函数
                    top_level_functions.append(node.name)
                elif self._is_method():
                    # 类中的方法,不过滤私有方法
                    self.current_class_methods.append(node.name)
                
                # 进入函数作用域，但不递归访问嵌套函数
                self.scope_stack.append('function')
                # 不调用 self.generic_visit(node) 来避免访问嵌套函数
                self.scope_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                if self._is_top_level():
                    # 只处理顶层类
                    # 检查是否有 @dataclass 装饰器，如果是dataclass则跳过
                    if not self._is_dataclass(node):
                        self.current_class_name = node.name
                        self.current_class_methods = []
                        
                        self.scope_stack.append('class')
                        self.generic_visit(node)  # 访问类中的方法
                        self.scope_stack.pop()
                        
                        # 将类和其方法添加到结果中
                        # 提取父类名称
                        base_classes = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                base_classes.append(base.id)
                            elif isinstance(base, ast.Attribute):
                                # 保留完整的模块路径，如 module.ClassName
                                def get_full_attr(node):
                                    parts = []
                                    while isinstance(node, ast.Attribute):
                                        parts.append(node.attr)
                                        node = node.value
                                    if isinstance(node, ast.Name):
                                        parts.append(node.id)
                                    return ".".join(reversed(parts))
                                base_classes.append(get_full_attr(base))
                        
                        class_methods.append({
                            'class_name': self.current_class_name,
                            'methods': self.current_class_methods.copy(),
                            'base_classes': base_classes  # 添加父类列表
                        })
                        
                        self.current_class_name = None
                        self.current_class_methods = []

            def _is_top_level(self):
                return not self.scope_stack

            def _is_top_level_function(self):
                return self.scope_stack == []

            def _is_method(self):
                return self.scope_stack and self.scope_stack[-1] == 'class'

            def _is_dataclass(self, node):
                for decorator in node.decorator_list:
                    # 支持 from dataclasses import dataclass 或直接 @dataclass
                    if (
                        isinstance(decorator, ast.Name) and decorator.id == 'dataclass'
                    ) or (
                        isinstance(decorator, ast.Attribute) and decorator.attr == 'dataclass'
                    ):
                        return True
                return False

        tree = ast.parse(code)
        visitor = FunctionAndMethodVisitor()
        visitor.visit(tree)
        
        return top_level_functions, class_methods

    except SyntaxError:
        print("[INS_WARN] AST parsing failed, using regex to match function and class definitions")
        # AST解析失败时的正则表达式回退方案
        top_level_functions = []
        class_methods = []
        
        # 匹配顶层函数和类
        lines = code.split('\n')
        current_class = None
        current_indent = -1
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            indent = len(line) - len(line.lstrip())
            
            # 顶层定义（缩进为0）
            if indent == 0:
                # 匹配类定义
                class_match = re.match(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
                if class_match:
                    if current_class:
                        # 保存之前的类
                        class_methods.append({
                            'class_name': current_class['name'],
                            'methods': current_class['methods']
                        })
                    current_class = {'name': class_match.group(1), 'methods': []}
                    current_indent = indent
                    continue
                
                # 匹配函数定义,不过滤私有函数
                func_match = re.match(r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
                if func_match:
                    func_name = func_match.group(1)
                    top_level_functions.append(func_name)
                    current_class = None  # 重置当前类
                    continue
            
            # 类内方法定义,不过滤私有方法
            elif current_class and indent > current_indent:
                func_match = re.match(r'(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
                if func_match:
                    method_name = func_match.group(1)
                    current_class['methods'].append(method_name)
        
        # 处理最后一个类
        if current_class:
            class_methods.append({
                'class_name': current_class['name'],
                'methods': current_class['methods']
            })
        
        return top_level_functions, class_methods



def get_class_instance_names(code: str, class_name: str) -> List[str]:
    """
    获取代码中由指定类构造的对象名称列表。
    - 只匹配赋值语句右值为 class_name(...) 的情形(支持 module.ClassName)。
    - 支持链式调用,如 class_name(...).to("cuda") 或 class_name(...).to("cpu")。
    - 支持函数返回值追踪:如果函数返回 class_name 实例,则接收返回值的变量也会被识别。
    - 支持变量传递追踪:如果函数返回变量,追踪该变量的类型。
    - 赋值左值可为变量、属性(self.attr)、多重赋值或解构赋值。
    返回按出现顺序去重的名称列表。
    """
    if not class_name:
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_class_instance_names parse failed: {exc}")
        return []

    def _attr_to_str(node: ast.Attribute) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    def _matches_constructor(call_node: ast.Call) -> bool:
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id == class_name
        if isinstance(func, ast.Attribute):
            return func.attr == class_name
        return False

    def _is_constructor_chain(node) -> bool:
        """
        检查节点是否是构造函数的链式调用,如 Model().to("cuda")
        """
        if isinstance(node, ast.Call):
            # 直接的构造函数调用
            if _matches_constructor(node):
                return True
            # 链式调用:检查是否为 constructor().method() 形式
            if isinstance(node.func, ast.Attribute):
                # 递归检查链的起始是否为构造函数
                return _is_constructor_chain(node.func.value)
        return False

    def _collect_targets(target) -> List[str]:
        names: List[str] = []
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            dotted = _attr_to_str(target)
            if dotted:
                names.append(dotted)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names.extend(_collect_targets(elt))
        return names

    def _get_value_name(node) -> str | None:
        """获取赋值右值的变量名（用于变量传递）"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return _attr_to_str(node)
        return None

    # 第一步:收集函数定义,分析哪些函数返回目标类实例
    functions_returning_class = set()
    func_to_return_vars: Dict[str, List[str]] = {}
    var_to_class: Dict[str, bool] = {}  # 记录变量是否为目标类实例
    var_to_var: Dict[str, str] = {}  # 记录变量间的引用关系: target -> source
    
    class FunctionAnalyzer(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None
        
        def visit_FunctionDef(self, node):
            old_function = self.current_function
            self.current_function = node.name
            
            # 收集函数内的赋值，查找目标类实例
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, ast.Call) and _is_constructor_chain(stmt.value):
                        for target in stmt.targets:
                            for var_name in _collect_targets(target):
                                var_to_class[var_name] = True
                elif isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.value, ast.Call) and _is_constructor_chain(stmt.value):
                        target_name = _attr_to_str(stmt.target) if isinstance(stmt.target, ast.Attribute) else stmt.target.id if isinstance(stmt.target, ast.Name) else None
                        if target_name:
                            var_to_class[target_name] = True
            
            # 收集返回语句
            return_vars = []
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value:
                    # 检查返回值是否为构造函数或其链式调用
                    if isinstance(stmt.value, ast.Call) and _is_constructor_chain(stmt.value):
                        functions_returning_class.add(self.current_function)
                    # 检查返回的是否为变量
                    else:
                        return_var = _get_value_name(stmt.value)
                        if return_var:
                            return_vars.append(return_var)
            
            if return_vars:
                func_to_return_vars[node.name] = return_vars
            
            self.generic_visit(node)
            self.current_function = old_function
        
        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)
    
    analyzer = FunctionAnalyzer()
    analyzer.visit(tree)
    
    # 第二步:检查函数返回的变量是否为目标类实例
    for func_name, return_vars in func_to_return_vars.items():
        for var_name in return_vars:
            if var_to_class.get(var_name, False):
                functions_returning_class.add(func_name)
                break

    # 第三步:收集实例名称和变量传递关系
    result: List[str] = []
    seen = set()
    instance_vars = set()  # 记录所有目标类实例的变量名

    def _record(name: str):
        if name and name not in seen:
            seen.add(name)
            result.append(name)
            instance_vars.add(name)

    for node in ast.walk(tree):
        call_node = None
        targets = None

        if isinstance(node, ast.Assign):
            value = node.value
            # 情况1:直接的构造函数调用或链式调用
            if isinstance(value, ast.Call) and _is_constructor_chain(value):
                call_node = value
                targets = node.targets
            # 情况2:调用返回该类实例的函数
            elif isinstance(value, ast.Call):
                func_name = None
                if isinstance(value.func, ast.Name):
                    func_name = value.func.id
                elif isinstance(value.func, ast.Attribute):
                    func_name = value.func.attr
                
                if func_name and func_name in functions_returning_class:
                    call_node = value
                    targets = node.targets
            # 情况3:变量传递 (d = v)
            else:
                source_var = _get_value_name(value)
                if source_var:
                    for target in node.targets:
                        for target_name in _collect_targets(target):
                            var_to_var[target_name] = source_var
                    
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            # 情况1:类型注解赋值,直接构造
            if isinstance(value, ast.Call) and _is_constructor_chain(value):
                call_node = value
                targets = [node.target]
            # 情况2:类型注解赋值,函数返回值
            elif isinstance(value, ast.Call):
                func_name = None
                if isinstance(value.func, ast.Name):
                    func_name = value.func.id
                elif isinstance(value.func, ast.Attribute):
                    func_name = value.func.attr
                
                if func_name and func_name in functions_returning_class:
                    call_node = value
                    targets = [node.target]
            # 情况3:变量传递
            else:
                source_var = _get_value_name(value)
                if source_var:
                    target_name = _attr_to_str(node.target) if isinstance(node.target, ast.Attribute) else node.target.id if isinstance(node.target, ast.Name) else None
                    if target_name:
                        var_to_var[target_name] = source_var

        if call_node and targets:
            for tgt in targets:
                for name in _collect_targets(tgt):
                    _record(name)

    # 第四步:追踪变量传递链，找到所有间接引用
    def _resolve_var(var_name: str, visited: Set[str] = None) -> bool:
        """递归检查变量是否最终指向目标类实例"""
        if visited is None:
            visited = set()
        
        if var_name in visited:
            return False  # 避免循环引用
        
        if var_name in instance_vars:
            return True
        
        visited.add(var_name)
        
        if var_name in var_to_var:
            source = var_to_var[var_name]
            return _resolve_var(source, visited)
        
        return False
    
    # 收集所有通过变量传递指向实例的变量
    for target_var, source_var in var_to_var.items():
        if _resolve_var(target_var):
            _record(target_var)

    return result



def get_function_param_names(code: str, func_name: str, class_name: str = ""):
    """
    获取指定函数或方法定义的参数名列表。
    - class_name 为空:匹配顶层函数定义(不含嵌套函数)。
    - class_name 非空:仅匹配该类的直接方法定义(不含嵌套类/函数);若首参为 self 会被移除。
    - 返回两个列表: (位置参数列表, 关键字参数列表)
    - 位置参数包含: 位置仅参数 + 普通位置参数
    - 关键字参数包含: 关键字仅参数
    - 忽略 *args 和 **kwargs
    """
    if not func_name:
        return [], []
    if class_name is None:
        class_name = ""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_function_param_names parse failed: {exc}")
        return [], []

    def _collect(args: ast.arguments, drop_self: bool) -> tuple[List[str], List[str]]:
        positional: List[str] = []
        keyword_only: List[str] = []

        # 位置仅参数 (Python 3.8+)
        for arg in getattr(args, "posonlyargs", []):
            positional.append(arg.arg)

        # 普通位置参数
        for arg in args.args:
            positional.append(arg.arg)

        # 如果是方法且第一个参数是 self，则移除
        if drop_self and positional and positional[0] == "self":
            positional.pop(0)

        # 关键字仅参数
        for arg in args.kwonlyargs:
            keyword_only.append(arg.arg)

        return positional, keyword_only

    class ParamCollector(ast.NodeVisitor):
        def __init__(self):
            self.positional_params: List[str] = []
            self.keyword_params: List[str] = []
            self.found = False
            self.class_stack: List[str] = []

        def visit_ClassDef(self, node):
            if self.found:
                return
            if self.class_stack:
                # 忽略嵌套类
                return
            self.class_stack.append(node.name)
            if not class_name or node.name == class_name:
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.visit(stmt)
            self.class_stack.pop()

        def visit_FunctionDef(self, node):
            if self.found:
                return
            inside_class = bool(self.class_stack)

            if class_name:
                if inside_class and self.class_stack[-1] == class_name and node.name == func_name:
                    self.positional_params, self.keyword_params = _collect(node.args, drop_self=True)
                    self.found = True
                return

            if not inside_class and node.name == func_name:
                self.positional_params, self.keyword_params = _collect(node.args, drop_self=False)
                self.found = True

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

    collector = ParamCollector()
    collector.visit(tree)
    return collector.positional_params, collector.keyword_params


def get_all_function_calls(code: str) -> dict:
    """
    提取代码中的所有函数调用信息，并标注每个调用的上层函数或方法（包括类名+方法名）。
    如果是嵌套调用，返回最顶层的函数/方法名称。
    返回结构:
    {
        'calls': [
            {
                'call_type': 'function' | 'method',
                'name': str,           # 被调用的函数/方法名（如 func 或 method）
                'obj': str | None,     # 方法调用时的对象名（如 obj），函数调用为 None
                'parent_name': str,    # 调用发生的最顶层函数/方法名（如 func 或 Class.method）
            }
        ]
    }
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_all_function_calls parse failed: {exc}")
        return {'calls': []}

    calls = []

    # 记录当前所在的函数/方法/类
    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope_stack = []

        def visit_FunctionDef(self, node):
            self.scope_stack.append({'type': 'function', 'name': node.name})
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            self.scope_stack.append({'type': 'class', 'name': node.name})
            self.generic_visit(node)
            self.scope_stack.pop()

        def visit_Call(self, node):
            func = node.func
            if isinstance(func, ast.Name):
                call_type = 'function'
                name = func.id
                obj = None
            elif isinstance(func, ast.Attribute):
                call_type = 'method'
                name = func.attr
                obj = self._get_full_name(func.value)
            else:
                # 忽略复杂的函数调用表达式，不记录到结果中
                self.generic_visit(node)
                return

            # 查找最顶层的函数/方法/类方法
            parent_name = None
            for scope in reversed(self.scope_stack):
                if scope['type'] == 'function':
                    # 判断是否在类作用域内
                    if len(self.scope_stack) >= 2 and self.scope_stack[-2]['type'] == 'class':
                        parent_name = f"{self.scope_stack[-2]['name']}.{scope['name']}"
                    else:
                        parent_name = scope['name']
                    break
                elif scope['type'] == 'class':
                    # 查找类方法
                    if len(self.scope_stack) >= 2 and self.scope_stack[-2]['type'] == 'function':
                        parent_name = f"{scope['name']}.{self.scope_stack[-2]['name']}"
                        break
            if parent_name is None:
                parent_name = '<module>'

            calls.append({
                'call_type': call_type,
                'name': name,
                'obj': obj,
                'parent_name': parent_name
            })

            self.generic_visit(node)

        def _get_full_name(self, node):
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                parts = []
                current = node
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    return ".".join(reversed(parts))
            return None

    visitor = CallVisitor()
    visitor.visit(tree)

    return {'calls': calls}


def get_object_class_names(code: str, obj_name: str) -> List[str]:
    """
    根据对象名推断其所属的类名列表。
    
    分析策略:
    1. 查找赋值语句: obj_name = ClassName(...) 或 obj_name = ClassName(...).method()
    2. 查找属性赋值: self.obj_name = ClassName(...)
    3. 查找类型注解: obj_name: ClassName = ...
    4. 查找函数参数的类型注解: def func(obj_name: ClassName)**
    5. 支持链式调用: ClassName().to('cpu')
    6. 支持多重赋值: a = b = ClassName()
    7. 支持变量传递: a = A(); b = a; c = b  # c 可追溯到 A
    8. 支持函数返回值追踪: obj = func(); 追踪 func 的返回值类型
    9. 只识别首字母大写的名称作为构造函数（遵循Python命名约定）
    
    参数:
        code (str): Python 代码字符串
        obj_name (str): 对象名称，支持属性形式如 "self.model" 或 "obj.attr"
    
    返回:
        List[str]: 可能的类名列表，按出现顺序返回
    """
    if not obj_name:
        return []
    
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_object_class_names parse failed: {exc}")
        return []
    
    class_names = []
    seen = set()
    
    # 构建变量到类名的映射表: {var_name: [class_names]}
    var_to_classes: Dict[str, List[str]] = {}
    # 构建变量间的引用关系: {var_name: var_name}
    var_to_var: Dict[str, str] = {}
    # 构建函数到返回变量的映射: {func_name: [return_var_names]}
    func_to_returns: Dict[str, List[str]] = {}
    # 构建变量到函数调用的映射: {var_name: func_name}
    var_to_func: Dict[str, str] = {}
    
    def _is_class_name(name: str) -> bool:
        """判断名称是否像类名（首字母大写）"""
        return name and name[0].isupper()
    
    def _get_target_name(target) -> str | None:
        """获取赋值目标的名称"""
        if isinstance(target, ast.Name):
            return target.id
        elif isinstance(target, ast.Attribute):
            parts = []
            node = target
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
                return ".".join(reversed(parts))
        return None
    
    def _get_class_from_call(node) -> str | None:
        """从调用节点中提取类名，只有首字母大写才认为是构造函数"""
        if not isinstance(node, ast.Call):
            return None
        
        func = node.func
        
        # 直接调用: ClassName(...)
        if isinstance(func, ast.Name):
            if _is_class_name(func.id):
                return func.id
        
        # 模块调用: module.ClassName(...)
        elif isinstance(func, ast.Attribute):
            if _is_class_name(func.attr):
                return func.attr
        
        return None
    
    def _get_func_name_from_call(node) -> str | None:
        """从调用节点中提取函数名（小写开头的函数）"""
        if not isinstance(node, ast.Call):
            return None
        
        func = node.func
        
        # 直接调用: func_name(...)
        if isinstance(func, ast.Name):
            if not _is_class_name(func.id):
                return func.id
        
        return None
    
    def _extract_class_from_chain(node) -> str | None:
        """从链式调用中提取最初的构造函数类名"""
        current = node
        
        # 先检查当前节点是否直接是构造函数调用
        if isinstance(current, ast.Call):
            class_name = _get_class_from_call(current)
            if class_name:
                return class_name
            
            # 如果不是构造函数，检查是否是链式调用 Constructor().method()
            if isinstance(current.func, ast.Attribute):
                # 递归检查链的起点
                return _extract_class_from_chain(current.func.value)
        
        return None
    
    def _get_value_name(node) -> str | None:
        """获取赋值右值的变量名（用于变量传递）"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        return None
    
    def _collect_targets(target) -> List[str]:
        """收集赋值目标名称（处理多重赋值）"""
        names = []
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            name = _get_target_name(target)
            if name:
                names.append(name)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names.extend(_collect_targets(elt))
        return names
    
    def _record_class(var_name: str, class_name: str):
        """记录变量到类名的映射"""
        if class_name:
            if var_name not in var_to_classes:
                var_to_classes[var_name] = []
            if class_name not in var_to_classes[var_name]:
                var_to_classes[var_name].append(class_name) 
    
    def _extract_class_from_annotation(annotation) -> str | None:
        """从类型注解中提取类名，支持泛型类型"""
        if isinstance(annotation, ast.Name):
            if _is_class_name(annotation.id):
                return annotation.id
        elif isinstance(annotation, ast.Attribute):
            if _is_class_name(annotation.attr):
                return annotation.attr
        # 处理泛型类型，如 Optional[BaseSession], Union[str, int], List[Model] 等
        elif isinstance(annotation, ast.Subscript):
            # 从泛型类型中提取第一个类型参数
            if isinstance(annotation.slice, ast.Index):  # Python 3.8
                inner = annotation.slice.value
            else:  # Python 3.9+
                inner = annotation.slice
            
            # 处理 Union[Type1, Type2, ...] 或 Optional[Type]
            if isinstance(inner, ast.Tuple):
                # 取第一个非 None 的类型
                for elt in inner.elts:
                    if isinstance(elt, ast.Constant) and elt.value is None:
                        continue
                    # 递归提取
                    result = _extract_class_from_annotation(elt)
                    if result:
                        return result
            else:
                # 单个类型参数，如 Optional[BaseSession] 或 List[Model]
                return _extract_class_from_annotation(inner)
        
        return None
    
    # 第一遍：收集所有赋值信息和函数定义（包括参数类型注解）
    class AssignmentCollector(ast.NodeVisitor):
        def __init__(self):
            self.current_func = None
        
        def visit_FunctionDef(self, node):
            """收集函数的返回值信息和参数类型注解"""
            prev_func = self.current_func
            self.current_func = node.name
            
            # 收集参数的类型注解
            for arg in node.args.args:
                if arg.annotation:
                    class_name = _extract_class_from_annotation(arg.annotation)
                    if class_name:
                        _record_class(arg.arg, class_name)
            
            # 收集 keyword-only 参数的类型注解
            for arg in node.args.kwonlyargs:
                if arg.annotation:
                    class_name = _extract_class_from_annotation(arg.annotation)
                    if class_name:
                        _record_class(arg.arg, class_name)
            
            # 收集 positional-only 参数的类型注解 (Python 3.8+)
            for arg in getattr(node.args, 'posonlyargs', []):
                if arg.annotation:
                    class_name = _extract_class_from_annotation(arg.annotation)
                    if class_name:
                        _record_class(arg.arg, class_name)
            
            # 查找所有 return 语句
            return_vars = []
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value:
                    # 提取返回的变量名
                    return_var = _get_value_name(child.value)
                    if return_var:
                        return_vars.append(return_var)
            
            if return_vars:
                func_to_returns[node.name] = return_vars
            
            self.generic_visit(node)
            self.current_func = prev_func
        
        def visit_AsyncFunctionDef(self, node):
            """处理异步函数定义"""
            self.visit_FunctionDef(node)
        
        def visit_Assign(self, node):
            # 处理多重赋值的所有目标
            all_targets = []
            for target in node.targets:
                all_targets.extend(_collect_targets(target))
            
            # 检查右值是否为构造调用
            if isinstance(node.value, ast.Call):
                class_name = _extract_class_from_chain(node.value)
                if class_name:
                    for var_name in all_targets:
                        _record_class(var_name, class_name)
                else:
                    # 检查是否为函数调用
                    func_name = _get_func_name_from_call(node.value)
                    if func_name:
                        for var_name in all_targets:
                            var_to_func[var_name] = func_name
            else:
                # 检查右值是否为变量引用（变量传递）
                value_name = _get_value_name(node.value)
                if value_name:
                    for var_name in all_targets:
                        var_to_var[var_name] = value_name
            
            self.generic_visit(node)
        
        def visit_AnnAssign(self, node):
            target_name = _get_target_name(node.target)
            if not target_name:
                self.generic_visit(node)
                return
            
            # 从类型注解中提取类名
            class_name = _extract_class_from_annotation(node.annotation)
            if class_name:
                _record_class(target_name, class_name)
            
            # 从赋值值中提取
            if node.value:
                if isinstance(node.value, ast.Call):
                    class_name = _extract_class_from_chain(node.value)
                    if class_name:
                        _record_class(target_name, class_name)
                    else:
                        # 检查是否为函数调用
                        func_name = _get_func_name_from_call(node.value)
                        if func_name:
                            var_to_func[target_name] = func_name
                else:
                    # 变量传递
                    value_name = _get_value_name(node.value)
                    if value_name:
                        var_to_var[target_name] = value_name
            
            self.generic_visit(node)
    
    collector = AssignmentCollector()
    collector.visit(tree)
    
    # 第二遍：追踪变量传递链和函数返回值，找到最终的类名
    def _resolve_class_names(var_name: str, visited: Set[str] = None) -> List[str]:
        """递归追踪变量传递链、函数返回值，找到所有可能的类名"""
        if visited is None:
            visited = set()
        
        if var_name in visited:
            return []  # 避免循环引用
        
        visited.add(var_name)
        result = []
        
        # 直接映射到类名
        if var_name in var_to_classes:
            result.extend(var_to_classes[var_name])
        
        # 追踪变量传递
        if var_name in var_to_var:
            source_var = var_to_var[var_name]
            result.extend(_resolve_class_names(source_var, visited))
        
        # 追踪函数调用的返回值
        if var_name in var_to_func:
            func_name = var_to_func[var_name]
            if func_name in func_to_returns:
                for return_var in func_to_returns[func_name]:
                    result.extend(_resolve_class_names(return_var, visited))
        
        return result
    
    # 查找目标变量的所有类名
    resolved_classes = _resolve_class_names(obj_name)
    
    # 去重并保持顺序
    for cls in resolved_classes:
        if cls not in seen:
            seen.add(cls)
            class_names.append(cls)
    
    return class_names

def get_object_call_invocations(code: str , class_names: List[str] = []) -> List[dict]:
    """
    提取代码中所有对象的 __call__ 调用(obj() 语法糖形式)。
    
    参数:
        code (str): Python 代码字符串
        class_names (List[str]): 类名列表，用于查找这些类的实例
    
    返回:
        List[dict]: 每个元素包含:
            - 'obj_name': str, 被调用的对象名(如 'func_obj' 或 'self.model')
            - 'class_type': List[str], 推断出的类名列表,无法推断则为空列表
            - 'parent_name': str, 调用所在的函数/方法名
            - 'line': int, 调用所在行号
    """

    object_to_classes = {}
    for class_name in class_names:
        instances = get_class_instance_names(code, class_name)
        for instance in instances:
            if instance not in object_to_classes:
                object_to_classes[instance] = []
            if class_name not in object_to_classes[instance]:
                object_to_classes[instance].append(class_name)
    
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_object_call_invocations parse failed: {exc}")
        return []
    
    invocations = []
    
    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.scope_stack = []
        
        def visit_FunctionDef(self, node):
            self.scope_stack.append({'type': 'function', 'name': node.name})
            self.generic_visit(node)
            self.scope_stack.pop()
        
        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)
        
        def visit_ClassDef(self, node):
            self.scope_stack.append({'type': 'class', 'name': node.name})
            self.generic_visit(node)
            self.scope_stack.pop()
        
        def visit_Call(self, node):
            obj_name = None
            
            # 提取被调用的对象名
            if isinstance(node.func, ast.Name):
                # 形式: obj()
                obj_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # 形式: self.obj() 或 a.b.obj()
                obj_name = self._get_full_name(node.func)
            
            # 检查是否在已知对象列表中
            if obj_name and obj_name in object_to_classes:
                parent_name = self._get_parent_name()
                
                invocations.append({
                    'obj_name': obj_name,
                    'class_type': object_to_classes[obj_name],
                    'parent_name': parent_name,
                    'line': node.lineno
                })
            
            self.generic_visit(node)
        
        def _get_full_name(self, node):
            """获取完整的属性访问路径"""
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                parts = []
                current = node
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    return ".".join(reversed(parts))
            return None
        
        def _get_parent_name(self):
            """获取调用所在的函数/方法名"""
            for scope in reversed(self.scope_stack):
                if scope['type'] == 'function':
                    # 检查是否在类中
                    if len(self.scope_stack) >= 2 and self.scope_stack[-2]['type'] == 'class':
                        return f"{self.scope_stack[-2]['name']}.{scope['name']}"
                    return scope['name']
            return '<module>'
    
    visitor = CallVisitor()
    visitor.visit(tree)
    
    return invocations