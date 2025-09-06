import re
import sys
import ast
from typing import List

if sys.version_info < (3, 9):
    import astor  # Used to convert AST to source code, Python 3.8 and below versions need to install astor library

def extract_python_code(text:str):
    """
    Extract Python code blocks from text
    """
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    matchstr = ""
    for match in matches:
        matchstr += match.strip() + "\n"
    if matchstr == '':
        print("[INS_WARN] Python code block not found, code block format is incorrect")
        if text.find('```python') != -1:
            return text.replace('```python', '').replace('```', '').strip()
        else:
            return text.strip()
    return matchstr

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


def remove_assignments(name, code, use_regex=False):
    """
    Use AST to remove assignment statements with specified variable names from code.
    If AST parsing fails, fall back to using regex processing.
    """
    if isinstance(name, str):
        names_to_remove = {name}
    else:
        names_to_remove = set(name)

    if use_regex:
        # 使用正则表达式方式处理
        for var in names_to_remove:
            code = re.sub(rf'^\s*{re.escape(var)}\s*=\s*[^#\n]*(\s*#.*)?\n?', '', code, flags=re.MULTILINE)
        return code
    try:
        # 定义 AST 处理器
        class InlineRemover(ast.NodeTransformer):
            def visit_Assign(self, node):
                new_targets = [
                    t for t in node.targets
                    if not (isinstance(t, ast.Name) and t.id in names_to_remove)
                ]
                if not new_targets:
                    return None
                node.targets = new_targets
                return node

        # 尝试使用 AST 方式
        tree = ast.parse(code)
        tree = InlineRemover().visit(tree)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
        else:
            import astor
            return astor.to_source(tree)

    except Exception as e:
        # 回退到正则处理方式（简化方案：按行删除赋值语句）
        for var in names_to_remove:
            code = re.sub(rf'^\s*{re.escape(var)}\s*=\s*[^#\n]*(\s*#.*)?\n?', '', code, flags=re.MULTILINE)

        return code


def extract_import_statements(code: str):
    """
    提取代码中的import语句，返回两个字符串：
    1. from __future__ 的导入语句
    2. 其他导入语句
    如果 AST 解析失败，则使用正则表达式匹配。
    """
    try:
        tree = ast.parse(code)
        future_imports = []
        other_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == '__future__':
                    for alias in node.names:
                        future_imports.append(f"from __future__ import {alias.name}")
                else:
                    for alias in node.names:
                        if node.module is None:
                            continue
                        if alias.asname:
                            other_imports.append(f"from {node.module} import {alias.name} as {alias.asname}")
                        else:
                            other_imports.append(f"from {node.module} import {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        other_imports.append(f"import {alias.name} as {alias.asname}")
                    else:
                        other_imports.append(f"import {alias.name}")
        future_imports_str = "\n".join(future_imports)
        other_imports_str = "\n".join(other_imports)

        return future_imports_str , other_imports_str
    except Exception as e:
        # ast解析失败时，使用正则匹配import语句，然后将 from __future__ 的语句排在前面
        print(f"[INS_WARN] {e}")
        print("[INS_WARN] AST parsing failed, using regex to match import statements")
        import re
        pattern = re.compile(r'^\s*(import\s+[^\n]+|from\s+[^\n]+import\s+[^\n]+)', re.MULTILINE)
        matches = pattern.findall(code)
        future_lines, other_lines = [], []
        for line in matches:
            if line.startswith('from __future__'):
                future_lines.append(line)
            else:
                other_lines.append(line)
        return "\n".join(future_lines), "\n".join(other_lines)

def extract_from_import_object(code):
    """
    提取代码中的 from ... import ... 的对象名列表，支持多行import、as重命名和去除注释
    """
    try:
        tree = ast.parse(code)
        objects = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != '*':
                        as_name = alias.asname
                        objects.append(alias.name)
                        if as_name:
                            objects.append(as_name)
        return objects
    except Exception as e:
        print(f"[INS_WARN] {e}")
        print("[INS_WARN] AST parsing failed, using regex to match object names from 'from ... import ...' statements")
        code = re.sub(r'#.*', '', code)              # 去除注释
        code = re.sub(r'\\\n', '', code)             # 去除反斜线换行
        code = re.sub(r'\(\s*\n\s*', '(', code)      # 合并括号换行
        code = re.sub(r'\n\s*\)', ')', code)
        pattern = re.compile(r'from\s+\S+\s+import\s+([^\n]+)')
        objects = []
        for match in pattern.findall(code):
            match = match.replace('(', '').replace(')', '')
            for item in match.split(','):
                item = item.strip()
                if not item or item == '*':
                    continue
                parts = item.split(' as ')
                name = parts[0].strip()
                alias = parts[1].strip() if len(parts) == 2 else None
                objects.append(name)
                if alias:
                    objects.append(alias)
        return objects



def remove_definitions_by_names(code: str, name_list: list = None):
    """
    移除代码中名称在 name_list 中的所有函数或类定义（支持嵌套）
    name_list: [str]，表示 from ... import ... 的对象名列表
    """
    class RemoveDefinitionsByName(ast.NodeTransformer):
        def __init__(self, name_set):
            self.name_set = name_set

        def visit_FunctionDef(self, node):
            if node.name in self.name_set:
                print(f"[INS_WARN] Removing incorrect function definition: {node.name}")
                return None  # 删除该函数
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            if node.name in self.name_set:
                print(f"[INS_WARN] Removing incorrect class definition: {node.name}")
                return None  # 删除该类
            self.generic_visit(node)
            return node
    try:
        if name_list is None:
            name_list = extract_from_import_object(code)
        target_names = set(name_list)
        tree = ast.parse(code)
        transformer = RemoveDefinitionsByName(target_names)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        if sys.version_info >= (3, 9):
            return ast.unparse(new_tree)
        else:
            import astor
            # 注意：需要安装 astor 库
            # pip install astor
            return astor.to_source(new_tree)
    except Exception as e:
        print(f"[INS_WARN] AST parsing failed {e}, using regex to match function/class definitions")
    if not name_list:
        return code
    for name in name_list:
        # 注意：支持 def/class，无参数也可匹配，处理缩进体
        pattern = re.compile(
            rf"""
            ^[ \t]*               # 行首缩进
            (?:@.*\n)*            # 可选的装饰器行
            (def|class)[ \t]+     # def 或 class
            {re.escape(name)}     # 函数或类名
            [ \t]*(\(.*?\))?[ \t]*:  # 可选的括号（参数列表）
            (?:\n                 # 函数或类体开始
                (?:[ \t]+.*\n?)+  # 缩进内容，至少一行
            )?
            """,
            re.MULTILINE | re.VERBOSE
        )
        code = pattern.sub('', code)
    return code


def get_class_code(file_path, class_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    class_code = []
    inside_class = False
    class_indent = None

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f'class {class_name}'):
            inside_class = True
            class_indent = len(line) - len(stripped)
            continue

        if inside_class:
            current_indent = len(line) - len(line.lstrip())
            if line.strip() == '':  # 空行也要保留
                class_code.append(line)
            elif current_indent > class_indent:
                class_code.append(line)
            else:
                break  # 缩进不再大于类定义的缩进，结束
    return ''.join(class_code)


def get_class_or_function_def_name(code: str):
    """
    提取模块顶层函数、类定义，类体中的方法，但不包含任何嵌套函数/类。
    """
    try:
        names = []

        class DefVisitor(ast.NodeVisitor):
            def __init__(self):
                self.scope_stack = []

            def visit_FunctionDef(self, node):
                if self._is_top_level_function() or self._is_method():
                    if not node.name.startswith('_') or node.name == '__init__' or node.name == '__call__':
                        names.append(node.name)
                self.scope_stack.append('function')
                self.generic_visit(node)
                self.scope_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                if self._is_top_level():
                    # 检查是否有 @dataclass 装饰器
                    if not self._is_dataclass(node):
                        names.append(node.name)
                self.scope_stack.append('class')
                self.generic_visit(node)
                self.scope_stack.pop()

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
        visitor = DefVisitor()
        visitor.visit(tree)
        return names

    except SyntaxError:
        print("[INS_WARN] AST parsing failed, using regex to match class or function definition names")
        # 正则无法判断装饰器，只能简单过滤
        pattern = re.compile(r'^\s*(?:async\s+)?(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.MULTILINE)
        result = [match[1] for match in pattern.findall(code)]
        return [name for name in result if name == '__init__' or not name.startswith('_')]

    
def get_inherit_info(code):
    """
    提取代码中的类继承信息，返回为一个元组(class1_name, class2_name)表示class1是class2的子类。
    只返回有继承父类的类信息，支持 module.B 形式，只返回 B。
    优先使用 AST，如果解析失败则回退使用正则匹配。
    """
    try:
        inherit_info = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.bases:
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        inherit_info.append((node.name, base.id))
                    elif isinstance(base, ast.Attribute):
                        # 只取最后一级名字
                        inherit_info.append((node.name, base.attr))
        return inherit_info
    except SyntaxError:
        print("[INS_WARN] AST parsing failed, using regex to match class inheritance information")
        pattern = re.compile(r'^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)', re.MULTILINE)
        matches = pattern.findall(code)
        # 只返回括号内最后一个点后的名字
        def get_last_name(s):
            return s.strip().split('.')[-1]
        return [(match[0], get_last_name(match[1])) for match in matches if match[1].strip()]


def transform_exe_attributes_and_call_name(code: str, exceptlist=[]) -> str:
    """
    替换 exe.xxx 为 exe.adapter.xxx，但跳过 exe.xxx() 的方法调用形式。
    另外，如果遇到 exe.run(...)，把第一个参数是字符串时，字符串里的.替换为_。
    """
    try:
        class ExeAttrTransformer(ast.NodeTransformer):
            def visit_Attribute(self, node):
                self.generic_visit(node)
                if isinstance(node.value, ast.Name) and node.value.id == 'exe':
                    if node.attr in exceptlist:
                        return node
                    parent = getattr(node, 'parent', None)
                    if not (isinstance(parent, ast.Call) and parent.func is node):
                        return ast.copy_location(
                            ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id='exe', ctx=ast.Load()),
                                    attr='adapter',
                                    ctx=node.ctx
                                ),
                                attr=node.attr,
                                ctx=node.ctx
                            ),
                            node
                        )
                return node

            def visit_Call(self, node):
                self.generic_visit(node)

                # 判断是否是 exe.run(...) 形式
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and
                            node.func.value.id == 'exe' and
                            node.func.attr == 'run'):

                        if node.args:
                            first_arg = node.args[0]
                            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                                # 替换字符串中的点
                                new_str = first_arg.value.replace('.', '_')
                                node.args[0] = ast.Constant(value=new_str, kind=None)

                return node

        def set_parents(tree):
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

        tree = ast.parse(code)
        set_parents(tree)
        tree = ExeAttrTransformer().visit(tree)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
        else:
            import astor
            return astor.to_source(tree)
    except Exception as e:
        print(f"[INS_WARN] transform_exe_attributes_and_call_name failed: {e}")
        return code

    
def remove_function_calls(code: str, function_names: list) -> str:
    """
    移除代码中指定函数名的所有调用（不处理方法调用）。
    如果赋值语句右侧包含这些函数调用，则整个赋值语句被删除。
    参数：
        code (str): 源代码字符串。
        function_names (list): 要移除的函数名列表。
    返回：
        str: 修改后的代码字符串。
    """
    if not function_names:
        return code

    class FunctionCallRemover(ast.NodeTransformer):
        def visit_Expr(self, node):
            # 删除独立的函数调用语句
            if (isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Name) and
                node.value.func.id in function_names):
                print(f"[INS_WARN] Removing function call statement: {node.value.func.id}")
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            # 如果赋值右侧包含目标函数调用，则删除整个赋值语句
            if self.contains_target_func(node.value):
                print(f"[INS_WARN] Removing assignment statement: {ast.dump(node)}")
                return None
            return self.generic_visit(node)

        def contains_target_func(self, node):
            # 检查表达式树中是否有目标函数调用
            for subnode in ast.walk(node):
                if (isinstance(subnode, ast.Call) and
                    isinstance(subnode.func, ast.Name) and
                    subnode.func.id in function_names):
                    # 返回函数名
                    return True
            return False

        def visit_Call(self, node):
            # 表达式内部的函数调用，用 None 替换
            if isinstance(node.func, ast.Name) and node.func.id in function_names:
                print(f"[INS_WARN] Removing function call: {node.func.id}")
                return ast.Constant(value=None)
            return self.generic_visit(node)
    try:
        tree = ast.parse(code)
        tree = FunctionCallRemover().visit(tree)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
        else:
            import astor
            return astor.to_source(tree)
    except Exception as e:
        print(f"[INS_WARN] AST parsing failed")
        return code

def remove_imports_from_code(source_code: str, imports_to_remove: str) -> str:
    """
    移除源代码中指定的导包语句，source_code 为源代码，imports_to_remove 为要移除的导包字符串列表。
    """
    imports_to_remove = set(imports_to_remove.splitlines())
    try:
        tree = ast.parse(source_code)
        class ImportRemover(ast.NodeTransformer):
            def visit_Import(self, node):
                new_names = []
                for alias in node.names:
                    line = f"import {alias.name}" if not alias.asname else f"import {alias.name} as {alias.asname}"
                    if line not in imports_to_remove:
                        new_names.append(alias)
                if not new_names:
                    print(f"[INS_WARN] Removing import statement: import {node.names[0].name}")
                    return None
                node.names = new_names
                return node

            def visit_ImportFrom(self, node):
                from_lines = []
                for alias in node.names:
                    line = (f"from {node.module} import {alias.name}"
                            if not alias.asname else f"from {node.module} import {alias.name} as {alias.asname}")
                    from_lines.append(line)
                new_names = [alias for alias, line in zip(node.names, from_lines) if line not in imports_to_remove]
                if not new_names:
                    print(f"[INS_WARN] Removing import statement: from {node.module}")
                    return None
                node.names = new_names
                return node

        tree = ImportRemover().visit(tree)
        ast.fix_missing_locations(tree)
        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
        else:
            import astor
            return astor.to_source(tree)

    except Exception:
        print("[INS_WARN] AST parsing failed, returning original code")
        return source_code  # 如果 AST 解析失败，返回原始代码
    
    


def extract_function_names_from_adapter(code: str):
    """
    从Python代码中提取所有 if name == '...' 或 if name == "..." 的右值。
    为适应新版本，提取 if dispatch_key == '...' 或 if dispatch_key == "..." 的右值.
    :param code: Python 代码字符串
    :return: 右值列表
    """
    pattern = r"if\s+(?:name|dispatch_key)\s*==\s*(['\"])(.*?)\1"
    matches = re.findall(pattern, code)
    return [match[1] for match in matches if match[1]]



def extract_class_names_from_adapter(code: str):
    """
    从Python代码中提取所有 interface_class_name == '...' 或 interface_class_name == "..." 的右值。

    :param code: Python 代码字符串
    :return: 右值列表
    """
    pattern = r"interface_class_name\s*==\s*(['\"])(.*?)\1"
    matches = re.findall(pattern, code)
    return [match[1] for match in matches if match[1]]



def replace_assignment(code: str, var_name: str, new_rhs_code: str) -> str:
    """
    使用 AST 替换指定变量或属性的赋值右值中包含 path/to 的字符串。
    替换优先级：
    1. 全局变量赋值
    2. 函数调用关键字参数
    3. 函数内变量赋值
    4. 函数默认值
    """

    def _matches_target(target, var_name):
        if isinstance(target, ast.Name):
            if '.' in var_name:
                var_name = var_name.split('.')[-1]  # 只匹配最后一部分
            return target.id == var_name
        elif isinstance(target, ast.Attribute):
            if '.' in var_name:
                parts = var_name.split('.')
                if len(parts) == 2:
                    return target.attr == parts[1] and getattr(target.value, 'id', None) == parts[0]
            else:
                return target.attr == var_name
        return False

    class PathStringReplacer(ast.NodeTransformer):
        def __init__(self, new_expr_ast):
            self.new_expr_ast = new_expr_ast  # 已经是 AST 表达式节点
            self.replaced = False

        def visit_Constant(self, node):
            if isinstance(node.value, str) and ("path/to" in node.value.lower() or "path to" in node.value.lower()):
                self.replaced = True
                return ast.copy_location(self.new_expr_ast, node)
            return node

        def visit_Str(self, node):  # 兼容 Python < 3.8
            if "path/to" in node.s.lower() or "path to" in node.s.lower():
                self.replaced = True
                return ast.copy_location(self.new_expr_ast, node)
            return node

    def replace_rhs_node(rhs_node, new_rhs_code):
        expr_ast = ast.parse(new_rhs_code, mode='eval').body  # 保留表达式结构
        replacer = PathStringReplacer(expr_ast)
        new_node = replacer.visit(rhs_node)
        return new_node, replacer.replaced

    class PriorityAssignReplacer(ast.NodeTransformer):
        def __init__(self):
            self.global_assignments = []
            self.keyword_arg_calls = []
            self.function_assignments = []
            self.default_value_assignments = []
            self.scope_stack = []

        def visit_FunctionDef(self, node):
            if node.args.defaults:
                default_start = len(node.args.args) - len(node.args.defaults)
                for i, default in enumerate(node.args.defaults):
                    arg_name = node.args.args[default_start + i].arg
                    if arg_name == var_name:
                        self.default_value_assignments.append((node, 'args', default_start + i, default))
            if node.args.kw_defaults:
                for i, default in enumerate(node.args.kw_defaults):
                    if default is not None and node.args.kwonlyargs[i].arg == var_name:
                        self.default_value_assignments.append((node, 'kw_defaults', i, default))
            self.scope_stack.append('function')
            self.generic_visit(node)
            self.scope_stack.pop()
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            self.scope_stack.append('class')
            self.generic_visit(node)
            self.scope_stack.pop()
            return node

        def visit_Assign(self, node):
            if len(node.targets) == 1:
                target = node.targets[0]
                if _matches_target(target, var_name):
                    if not self.scope_stack:
                        self.global_assignments.append((node, node.value))
                    else:
                        self.function_assignments.append((node, node.value))
            self.generic_visit(node)
            return node

        def visit_Call(self, node):
            for kw in node.keywords:
                if kw.arg == var_name:
                    self.keyword_arg_calls.append((node, kw, kw.value))
            self.generic_visit(node)
            return node

        def replace_first_assignment(self, new_rhs_code: str):

            # 1. 全局变量赋值
            for assign_node, rhs_node in self.global_assignments:
                new_node, replaced = replace_rhs_node(rhs_node, new_rhs_code)
                if replaced:
                    assign_node.value = new_node
                    print(f"[INS_INFO] Replacing global assignment: {var_name}")
                    return

            # 2. 函数调用关键字参数
            for call_node, kw_node, rhs_node in self.keyword_arg_calls:
                new_node, replaced = replace_rhs_node(rhs_node, new_rhs_code)
                if replaced:
                    kw_node.value = new_node
                    print(f"[INS_INFO] Replacing function call keyword argument: {var_name}")
                    return

            # 3. 函数内变量赋值
            for assign_node, rhs_node in self.function_assignments:
                new_node, replaced = replace_rhs_node(rhs_node, new_rhs_code)
                if replaced:
                    assign_node.value = new_node
                    print(f"[INS_INFO] Replacing function assignment: {var_name}")
                    return

            # 4. 函数默认值
            for func_node, param_type, index, rhs_node in self.default_value_assignments:
                new_node, replaced = replace_rhs_node(rhs_node, new_rhs_code)
                if replaced:
                    if param_type == 'args':
                        func_node.args.defaults[index] = new_node
                    elif param_type == 'kw_defaults':
                        func_node.args.kw_defaults[index] = new_node
                    print(f"[INS_INFO] Replacing function default value: {var_name}")
                    return

    try:
        tree = ast.parse(code)
        replacer = PriorityAssignReplacer()
        replacer.visit(tree)
        replacer.replace_first_assignment(new_rhs_code)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            new_code = ast.unparse(tree)
        else:
            import astor
            new_code = astor.to_source(tree)

        return new_code

    except Exception as e:
        print(f"[INS_WARN] replace_assignment failed: {e}")
        return code



def replace_dict_value(code: str, target_key: str, new_value_code: str) -> str:
    """
    使用 AST 修改 dict 字面量中指定 key 的 value，只影响 dict 语法，避免误伤。
    new_value_code 需符合 Python 表达式字符串格式，例如 '"new_string"'。
    """
    class DictValueReplacer(ast.NodeTransformer):
        def _value_contains_pathto(self, value_node):
            """检查值节点是否包含 path/to 或 path to"""
            try:
                if sys.version_info >= (3, 9):
                    value_code = ast.unparse(value_node)
                else:
                    import astor
                    value_code = astor.to_source(value_node)
                value_lower = value_code.lower()
                return ("path/to" in value_lower) or ("path to" in value_lower)
            except Exception:
                return False
        
        def visit_Dict(self, node):
            self.generic_visit(node)
            for idx, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and target_key.find(key.value) != -1:
                    # 检查原值是否包含 path/to 或 path to
                    if self._value_contains_pathto(node.values[idx]):
                        print(f"[INS_INFO] Original dictionary value contains path/to or path to, performing replacement")
                        node.values[idx] = ast.parse(new_value_code, mode='eval').body
            return node
        
    try:
        tree = ast.parse(code)
        tree = DictValueReplacer().visit(tree)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            new_code = ast.unparse(tree)
        else:
            import astor
            new_code = astor.to_source(tree)
        return new_code
    except Exception as e:
        print(f"[INS_WARN] Failed to replace dictionary value: {e}")
        return code



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

def replace_file_variable_in_code(code: str, fixed_path: str) -> str:
    """
    替换 Python 代码中非字符串、非注释部分的 __file__ 变量为固定路径。
    """
    # 正则只匹配： __file__ 前后不为引号的情况
    pattern = re.compile(r'(?<!["\'])\b__file__\b(?!["\'])')

    # 替换
    return pattern.sub(f"'{fixed_path}'", code)


def remove_future_imports(code: str) -> str:
    """
    去除 Python 代码中所有 from __future__ import 语句（支持单行、多行括号、多行续行）
    """
    # 处理三种情况：
    # 1. 单行：from __future__ import something
    # 2. 括号包裹：from __future__ import (a, b)
    # 3. 反斜线续行：from __future__ import a, \n    b
    pattern = re.compile(
        r'^[ \t]*from[ \t]+__future__[ \t]+import[ \t]+'
        r'(?:\\\n|.*\([^)]*\)|.*(?:\n[ \t]+[^\n]*)*)',
        re.MULTILINE
    )
    original_code = code
    code = pattern.sub('', code)
    if has_syntax_error(code):
        # 如果替换后仍有语法错误，说明可能有其他问题，返回原代码
        return original_code
    return code

def remove_exe_imports(code: str) -> str:
    """
    去除代码中from ... import exe 的语句。
    """
    # 处理多种情况：
    # 1. from xxx import exe
    # 2. from xxx import exe, other
    # 3. from xxx import other, exe
    # 4. from xxx import other, exe, another
    # 5. from xxx import (exe)
    # 6. from xxx import (exe, other)
    # 7. 多行import with括号或反斜线续行
    # 先处理单独的 from ... import exe 语句（整行删除）
    pattern1 = re.compile(r'^\s*from\s+[^\s]+\s+import\s+exe\s*(?:#.*)?$', re.MULTILINE)
    code = pattern1.sub('', code)
    # 处理 from ... import exe, other 的情况（只删除exe部分）
    pattern2 = re.compile(r'(\bfrom\s+[^\s]+\s+import\s+)exe\s*,\s*', re.MULTILINE)
    code = pattern2.sub(r'\1', code)
    # 处理 from ... import other, exe 的情况（只删除exe部分）
    pattern3 = re.compile(r'(\bfrom\s+[^\s]+\s+import\s+[^,\n]+),\s*exe\b', re.MULTILINE)
    code = pattern3.sub(r'\1', code)
    # 处理括号内的情况 from ... import (exe, other) 或 from ... import (other, exe)
    def clean_parentheses_imports(match):
        full_import = match.group(0)
        # 在括号内容中移除exe
        inside_parens = re.sub(r'\bexe\s*,\s*', '', full_import)  # exe在前
        inside_parens = re.sub(r',\s*exe\b', '', inside_parens)   # exe在后
        inside_parens = re.sub(r'\(\s*exe\s*\)', '()', inside_parens)  # 只有exe
        return inside_parens
    # 处理括号形式的import
    pattern4 = re.compile(r'from\s+[^\s]+\s+import\s+\([^)]*\bexe\b[^)]*\)', re.MULTILINE)
    code = pattern4.sub(clean_parentheses_imports, code)
    # 清理可能产生的空import语句
    empty_import_pattern = re.compile(r'^\s*from\s+[^\s]+\s+import\s+\(\s*\)\s*$', re.MULTILINE)
    code = empty_import_pattern.sub('', code)
    # 清理多余的空行
    code = re.sub(r'\n\s*\n', '\n\n', code)
    
    return code
    

def reset_exe_run_create_with_origin(code: str, available_funcs :List[str] , available_classes :List[str]) -> str:
    """
    重置 exe.run(...) 和 exe.create_interface_objects(...) 的调用，
    如果函数名或类名不在 available_funcs_classes 中，则试图使用原始名称替换。
    """
    try:
        available_funcs_classes = available_funcs + available_classes
        class ExeRunAndCreateTransformer(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                # 处理 exe.run(...)
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "exe"
                    ):
                        # 处理 exe.run
                        if node.func.attr == "run":
                            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                func_name_str = node.args[0].value
                                if func_name_str not in available_funcs:
                                    if len(available_classes) == 1:
                                        # 如果只有一个可用类，本来的形式肯恩是CLSNAME.method ,注册上去可能是 methon ,所以
                                        for methond in available_funcs:
                                            if func_name_str.endswith(methond):
                                                node.args[0] = ast.Constant(value=methond, kind=None)
                                                return node
                                    if 'call' in available_funcs and func_name_str == '__call__':
                                        # 特例处理 __call__ 方法
                                        node.args[0] = ast.Constant(value='call', kind=None)
                                        return node
                                    if '__call__' in available_funcs and func_name_str == 'call':
                                        # 特例处理 call 方法
                                        node.args[0] = ast.Constant(value='__call__', kind=None)
                                        return node
                                    print(f"[INS_WARN] Function name '{func_name_str}' in exe.run() is not in available function list, reset to original name")
                                    new_call = ast.Call(
                                        func=ast.Name(id=func_name_str, ctx=ast.Load()),
                                        args=node.args[1:],
                                        keywords=node.keywords
                                    )
                                    return new_call
                        # 处理 exe.create_interface_objects
                        if node.func.attr == "create_interface_objects":
                            # 检查关键字参数中的 interface_class_name
                            class_name_str = None
                            
                            # 首先检查关键字参数
                            for kw in node.keywords:
                                if kw.arg == "interface_class_name":
                                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                        class_name_str = kw.value.value
                                        break
                            
                            # 如果没有找到关键字参数，再检查位置参数
                            if class_name_str is None and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                                class_name_str = node.args[0].value
                            
                            if class_name_str and class_name_str not in available_funcs_classes:
                                print(f"[INS_WARN] Class name '{class_name_str}' in exe.create_interface_objects() is not in available list, reset to original name")
                                new_instance = ast.Call(
                                    func=ast.Name(id=class_name_str, ctx=ast.Load()),
                                    args=node.args[1:] if node.args else [],
                                    keywords=[kw for kw in node.keywords if kw.arg != "interface_class_name"]
                                )
                                return new_instance
                return node

        tree = ast.parse(code)
        transformer = ExeRunAndCreateTransformer()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            return ast.unparse(tree)
        else:
            import astor
            return astor.to_source(tree)
    except Exception as e:
        print(f"[INS_WARN] reset_exe_run_create_with_origin failed: {e}")
        return code

def clean_path_to_in_code(code: str) -> str:
    """
    处理给定的 Python 代码字符串，将所有包含 'path to' 或 'path/to' 的字符串常量（不含 save、output 等）直接替换为空字符串。
    对于函数/方法调用的关键字参数，如果值包含 path/to，则删除该参数。
    """
    class PathToCleaner(ast.NodeTransformer):
        def visit_Constant(self, node):
            # 只在非函数调用参数时替换
            if isinstance(node.value, str):
                # 检查父节点是否为 ast.keyword（即函数参数），如果是则不处理
                parent = getattr(node, 'parent', None)
                if not isinstance(parent, ast.keyword):
                    s_lower = node.value.lower()
                    if (
                        ('path/to' in s_lower or 'path to' in s_lower)
                        and not any(x in s_lower for x in ('save', 'record', 'output', 'result', 'log'))
                    ):
                        return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_Call(self, node):
            # 删除包含 path/to 的关键字参数
            new_keywords = []
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    s_lower = kw.value.value.lower()
                    if (
                        ('path/to' in s_lower or 'path to' in s_lower)
                        and not any(x in s_lower for x in ('save', 'record', 'output', 'result', 'log'))
                    ):
                        continue  # 删除该参数
                new_keywords.append(kw)
            node.keywords = new_keywords
            self.generic_visit(node)
            return node

        def generic_visit(self, node):
            # 给所有子节点加 parent 属性
            for child in ast.iter_child_nodes(node):
                child.parent = node
            return super().generic_visit(node)

    try:
        tree = ast.parse(code)
        cleaner = PathToCleaner()
        cleaned_tree = cleaner.visit(tree)
        ast.fix_missing_locations(cleaned_tree)
        if sys.version_info >= (3, 9):
            return ast.unparse(cleaned_tree)
        else:
            import astor
            return astor.to_source(cleaned_tree)
    except Exception as e:
        print(f"[INS_WARN] Failed to process placeholder path/to: {e}")
        return code