import re
import sys
import ast
from typing import List
from Inspection.utils.tools import get_available_outter_resources_exts
import os

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


def extract_python_code(text: str):
    """
    Extract Python code blocks from text, handling nested blocks by matching the outermost layer.
    """
    start_token = "```python"
    end_token = "```"
    
    # 找到所有的开始标记的位置
    starts = [m.start() for m in re.finditer(re.escape(start_token), text)]
    # 找到所有的结束标记的位置
    # 注意：这里需要小心，因为 ```python 内部也包含 ```。
    # 我们先找到所有 ``` 的位置，然后排除掉那些属于 ```python 的部分
    all_backticks = [m.start() for m in re.finditer(re.escape(end_token), text)]
    
    # 过滤掉属于 start_token 的 end_token
    # 如果一个 ``` 的位置等于某个 ```python 的位置，那它其实是开始标记的一部分
    real_ends = []
    for pos in all_backticks:
        is_part_of_start = False
        for s_pos in starts:
            if pos == s_pos:
                is_part_of_start = True
                break
        if not is_part_of_start:
            real_ends.append(pos)
            
    # 将所有标记按位置排序：(position, type) type=1 for start, -1 for end
    tokens = []
    for pos in starts:
        tokens.append((pos, 1))
    for pos in real_ends:
        tokens.append((pos, -1))
        
    tokens.sort(key=lambda x: x[0])
    
    results = []
    nesting_level = 0
    current_start_idx = -1
    
    for pos, token_type in tokens:
        if token_type == 1: # Start
            if nesting_level == 0:
                current_start_idx = pos + len(start_token)
            nesting_level += 1
        elif token_type == -1: # End
            if nesting_level > 0:
                nesting_level -= 1
                if nesting_level == 0 and current_start_idx != -1:
                    # 闭合了最外层
                    results.append(text[current_start_idx:pos].strip())
                    current_start_idx = -1
    
    matchstr = "\n".join(results)

    if matchstr == "":
        print("[INS_WARN] Python code block not found, code block format is incorrect")
        if text.find("```python") != -1:
            return text.replace("```python", "").replace("```", "").strip()
        else:
            return text.strip()
    return matchstr


def extract_code(text, language=None, first_only=False):
    """
    Extract code blocks from text
    """
    if language is None:
        pattern = re.compile(r"```(.*?)```", re.DOTALL)
    else:
        pattern = re.compile(rf"```{language}(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    matchstr = ""
    for match in matches:
        if first_only:
            return match.strip()
        matchstr += match.strip() + "\n"
    return matchstr


def remove_assignments(name, code, use_regex=False):
    original_code = code
    processed_code = code

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
            code = re.sub(
                rf"^\s*{re.escape(var)}\s*=\s*[^#\n]*(\s*#.*)?\n?",
                "",
                code,
                flags=re.MULTILINE,
            )
        processed_code = code
    try:
        # 定义 AST 处理器
        class InlineRemover(ast.NodeTransformer):
            def visit_Assign(self, node):
                new_targets = [
                    t
                    for t in node.targets
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
            processed_code = ast.unparse(tree)
        else:
            processed_code = astor.to_source(tree)

    except Exception as e:
        # 回退到正则处理方式（简化方案：按行删除赋值语句）
        for var in names_to_remove:
            code = re.sub(
                rf"^\s*{re.escape(var)}\s*=\s*[^#\n]*(\s*#.*)?\n?",
                "",
                code,
                flags=re.MULTILINE,
            )
        processed_code = code
    
    if has_syntax_error(processed_code):
        print("[INS_WARN] Removal resulted in syntax error, returning original code")
        return original_code
    return processed_code


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
                if node.module == "__future__":
                    for alias in node.names:
                        future_imports.append(f"from __future__ import {alias.name}")
                else:
                    for alias in node.names:
                        if node.module is None:
                            continue
                        if alias.asname:
                            other_imports.append(
                                f"from {node.module} import {alias.name} as {alias.asname}"
                            )
                        else:
                            other_imports.append(
                                f"from {node.module} import {alias.name}"
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        other_imports.append(f"import {alias.name} as {alias.asname}")
                    else:
                        other_imports.append(f"import {alias.name}")

        # 去重
        future_imports_str = "\n".join(sorted(set(future_imports)))
        other_imports_str = "\n".join(sorted(set(other_imports)))

        return future_imports_str, other_imports_str
    except Exception as e:
        # ast解析失败时，使用正则匹配import语句，然后将 from __future__ 的语句排在前面
        print(f"[INS_WARN] {e}")
        print("[INS_WARN] AST parsing failed, using regex to match import statements")
        import re

        pattern = re.compile(
            r"^\s*(import\s+[^\n]+|from\s+[^\n]+import\s+[^\n]+)", re.MULTILINE
        )
        matches = pattern.findall(code)
        future_lines, other_lines = [], []
        for line in matches:
            if line.startswith("from __future__"):
                future_lines.append(line)
            else:
                other_lines.append(line)

        # 去重
        return "\n".join(sorted(set(future_lines))), "\n".join(sorted(set(other_lines)))


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
                    if alias.name != "*":
                        as_name = alias.asname
                        objects.append(alias.name)
                        if as_name:
                            objects.append(as_name)
        return objects
    except Exception as e:
        print(f"[INS_WARN] {e}")
        print(
            "[INS_WARN] AST parsing failed, using regex to match object names from 'from ... import ...' statements"
        )
        code = re.sub(r"#.*", "", code)  # 去除注释
        code = re.sub(r"\\\n", "", code)  # 去除反斜线换行
        code = re.sub(r"\(\s*\n\s*", "(", code)  # 合并括号换行
        code = re.sub(r"\n\s*\)", ")", code)
        pattern = re.compile(r"from\s+\S+\s+import\s+([^\n]+)")
        objects = []
        for match in pattern.findall(code):
            match = match.replace("(", "").replace(")", "")
            for item in match.split(","):
                item = item.strip()
                if not item or item == "*":
                    continue
                parts = item.split(" as ")
                name = parts[0].strip()
                alias = parts[1].strip() if len(parts) == 2 else None
                objects.append(name)
                if alias:
                    objects.append(alias)
        return objects


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
        code = re.sub(r"#.*", "", code)  # 去除注释
        pattern = re.compile(r"^\s*import\s+([^\n]+)", re.MULTILINE)
        objects = []
        for match in pattern.findall(code):
            # 处理多个导入：import a, b, c
            for item in match.split(","):
                item = item.strip()
                if not item:
                    continue
                # 处理 as 别名
                if " as " in item:
                    parts = item.split(" as ")
                    alias = parts[1].strip()
                    objects.append(alias)
                else:
                    # 保留完整的包路径
                    objects.append(item.strip())
        return objects


def remove_definitions_by_names(code: str, name_list=None):
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

    new_code = ""
    try:
        if name_list is None:
            name_list = extract_from_import_object(code)
        if isinstance(name_list, str):
            name_list = [name_list]
        target_names = set(name_list)
        tree = ast.parse(code)
        transformer = RemoveDefinitionsByName(target_names)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        if sys.version_info >= (3, 9):
            new_code = ast.unparse(new_tree)
        else:
            new_code = astor.to_source(new_tree)
    except Exception as e:
        print(
            f"[INS_WARN] AST parsing failed {e}, using regex to match function/class definitions"
        )
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
                re.MULTILINE | re.VERBOSE,
            )
            new_code = pattern.sub("", code)
    if new_code == "" or has_syntax_error(new_code):
        print(
            "[INS_WARN] Removal resulted in empty or invalid code, returning original code"
        )
        return code
    else:
        return new_code


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
                    # 顶层函数，过滤下划线开头的私有函数
                    top_level_functions.append(node.name)
                elif self._is_method():
                    # 类中的方法，包含__init__和__call__，但过滤其他私有方法
                    self.current_class_methods.append(node.name)

                # 进入函数作用域，但不递归访问嵌套函数
                self.scope_stack.append("function")
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

                        self.scope_stack.append("class")
                        self.generic_visit(node)  # 访问类中的方法
                        self.scope_stack.pop()

                        # 将类和其方法添加到结果中
                        class_methods.append(
                            {
                                "class_name": self.current_class_name,
                                "methods": self.current_class_methods.copy(),
                            }
                        )

                        self.current_class_name = None
                        self.current_class_methods = []

            def _is_top_level(self):
                return not self.scope_stack

            def _is_top_level_function(self):
                return self.scope_stack == []

            def _is_method(self):
                return self.scope_stack and self.scope_stack[-1] == "class"

            def _is_dataclass(self, node):
                for decorator in node.decorator_list:
                    # 支持 from dataclasses import dataclass 或直接 @dataclass
                    if (
                        isinstance(decorator, ast.Name) and decorator.id == "dataclass"
                    ) or (
                        isinstance(decorator, ast.Attribute)
                        and decorator.attr == "dataclass"
                    ):
                        return True
                return False

        tree = ast.parse(code)
        visitor = FunctionAndMethodVisitor()
        visitor.visit(tree)

        return top_level_functions, class_methods

    except SyntaxError:
        print(
            "[INS_WARN] AST parsing failed, using regex to match function and class definitions"
        )
        # AST解析失败时的正则表达式回退方案
        top_level_functions = []
        class_methods = []

        # 匹配顶层函数和类
        lines = code.split("\n")
        current_class = None
        current_indent = -1

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())

            # 顶层定义（缩进为0）
            if indent == 0:
                # 匹配类定义
                class_match = re.match(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", stripped)
                if class_match:
                    if current_class:
                        # 保存之前的类
                        class_methods.append(
                            {
                                "class_name": current_class["name"],
                                "methods": current_class["methods"],
                            }
                        )
                    current_class = {"name": class_match.group(1), "methods": []}
                    current_indent = indent
                    continue

                # 匹配函数定义
                func_match = re.match(
                    r"(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)", stripped
                )
                if func_match:
                    func_name = func_match.group(1)
                    if not func_name.startswith("_"):
                        top_level_functions.append(func_name)
                    current_class = None  # 重置当前类
                    continue

            # 类内方法定义
            elif current_class and indent > current_indent:
                func_match = re.match(
                    r"(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)", stripped
                )
                if func_match:
                    method_name = func_match.group(1)
                    if not method_name.startswith("_") or method_name in [
                        "__init__",
                        "__call__",
                    ]:
                        current_class["methods"].append(method_name)

        # 处理最后一个类
        if current_class:
            class_methods.append(
                {
                    "class_name": current_class["name"],
                    "methods": current_class["methods"],
                }
            )

        return top_level_functions, class_methods


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
        print(
            "[INS_WARN] AST parsing failed, using regex to match class inheritance information"
        )
        pattern = re.compile(
            r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", re.MULTILINE
        )
        matches = pattern.findall(code)

        # 只返回括号内最后一个点后的名字
        def get_last_name(s):
            return s.strip().split(".")[-1]

        return [
            (match[0], get_last_name(match[1])) for match in matches if match[1].strip()
        ]


def replace_call_with_new_method(
    code: str,
    positionlist: List[str],  # 位置参数对应的新形参列表
    preobj,  # 可以是 str 或 List[str]
    premethod: str,
    newobj: str,
    newmethod: str,
    first_arg: str,
) -> str:
    """
    将指定函数/方法调用替换为新对象的方法：
    - 新方法的第一个参数固定为 first_arg 表达式；
    - 原先的位置参数会根据 positionlist 映射为关键字参数；
    - 原关键字参数保持不变，*args/**kwargs 会被保留。

    参数:
        preobj: 可以是单个对象名(str)或对象名列表(List[str])，支持点号分隔的属性访问
    """

    if not premethod or not newobj or not newmethod or not first_arg:
        return code

    positionlist = positionlist or []

    # 统一处理 preobj 为列表格式
    if isinstance(preobj, str):
        preobj_list = [preobj.strip()] if preobj.strip() else []
    elif isinstance(preobj, list):
        preobj_list = [obj.strip() for obj in preobj if obj and obj.strip()]
    else:
        preobj_list = []

    try:
        tree = ast.parse(code)
        ast.parse(first_arg, mode="eval")
        ast.parse(newobj, mode="eval")
    except SyntaxError as exc:
        print(f"[INS_WARN] replace_call_with_new_method parse failed: {exc}")
        return code

    def _attr_parts(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return list(reversed(parts))
        return None

    # 预处理所有目标对象的部分列表
    target_parts_list = []
    for obj in preobj_list:
        if obj:
            target_parts_list.append(obj.split("."))

    def _is_target(call_node: ast.Call) -> bool:
        if preobj_list:
            if (
                isinstance(call_node.func, ast.Attribute)
                and call_node.func.attr == premethod
            ):
                attr_parts = _attr_parts(call_node.func.value)
                # 检查是否匹配任一目标对象
                return any(
                    attr_parts == target_parts for target_parts in target_parts_list
                )
            return False
        return isinstance(call_node.func, ast.Name) and call_node.func.id == premethod

    def _build_first_arg():
        return ast.parse(first_arg, mode="eval").body

    def _build_new_func():
        value = ast.parse(newobj, mode="eval").body
        return ast.Attribute(value=value, attr=newmethod, ctx=ast.Load())

    class CallRewriter(ast.NodeTransformer):
        def __init__(self):
            self.changed = False

        def visit_Call(self, node):
            # 先递归处理子节点
            self.generic_visit(node)

            if not _is_target(node):
                return node

            fallback_args = []
            converted_keywords = []
            existing_kw_names = {kw.arg for kw in node.keywords if kw.arg}
            used_param_names = set(existing_kw_names)
            param_idx = 0

            # 将位置参数转换为关键字参数
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    fallback_args.append(arg)
                    continue

                param_name = None
                while param_idx < len(positionlist):
                    candidate = positionlist[param_idx]
                    param_idx += 1
                    if not candidate or candidate in used_param_names:
                        continue
                    param_name = candidate
                    used_param_names.add(candidate)
                    break

                if param_name is None:
                    fallback_args.append(arg)
                else:
                    converted_keywords.append(ast.keyword(arg=param_name, value=arg))

            # 构建新的函数调用
            new_call = ast.Call(
                func=_build_new_func(),
                args=[_build_first_arg()] + fallback_args,
                keywords=converted_keywords + node.keywords,
            )
            self.changed = True
            return ast.copy_location(new_call, node)

        def visit_Attribute(self, node):
            # 处理链式调用：obj.method().other() 或 func().other()
            # 先递归处理子节点
            self.generic_visit(node)

            # 检查是否是基于目标调用的属性访问
            if isinstance(node.value, ast.Call) and _is_target(node.value):
                # 这是链式调用，先替换内部调用
                replaced_call = self.visit_Call(node.value)
                if replaced_call != node.value:
                    # 构建新的属性访问
                    new_attr = ast.Attribute(
                        value=replaced_call, attr=node.attr, ctx=node.ctx
                    )
                    return ast.copy_location(new_attr, node)

            return node

    transformer = CallRewriter()
    new_tree = transformer.visit(tree)
    if not transformer.changed:
        return code

    ast.fix_missing_locations(new_tree)
    if sys.version_info >= (3, 9):
        return ast.unparse(new_tree)
    return astor.to_source(new_tree)


def append_call_with_new_method(
    code: str,
    positionlist: List[str],  # 位置参数对应的新形参列表
    preobj,  # 可以是 str 或 List[str]
    premethod: str,
    newobj: str,
    newmethod: str,
    first_arg: str,
) -> str:
    """
    在匹配的函数/方法调用所在语句的下一行，追加一个新对象的方法调用：
    - 新方法的第一个参数固定为 first_arg 表达式；
    - 原先的位置参数根据 positionlist 映射为关键字参数；
    - 原关键字参数保持不变，*args/**kwargs 会被保留；
    - 原语句不变，仅在其后插入新调用（若同一语句内存在多个匹配调用，将插入等量的新调用）。
    参数:
        preobj: 可以是单个对象名(str)或对象名列表(List[str])，支持点号分隔的属性访问
    """
    if not premethod or not newobj or not newmethod or not first_arg:
        return code

    positionlist = positionlist or []

    # 统一处理 preobj 为列表格式
    if isinstance(preobj, str):
        preobj_list = [preobj.strip()] if preobj.strip() else []
    elif isinstance(preobj, list):
        preobj_list = [obj.strip() for obj in preobj if obj and obj.strip()]
    else:
        preobj_list = []

    try:
        tree = ast.parse(code)
        ast.parse(first_arg, mode="eval")
        ast.parse(newobj, mode="eval")
    except SyntaxError as exc:
        print(f"[INS_WARN] append_call_with_new_method parse failed: {exc}")
        return code

    def _attr_parts(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return list(reversed(parts))
        return None

    # 预处理所有目标对象的部分列表
    target_parts_list = []
    for obj in preobj_list:
        if obj:
            target_parts_list.append(obj.split("."))

    def _is_target(call_node: ast.Call) -> bool:
        if preobj_list:
            if (
                isinstance(call_node.func, ast.Attribute)
                and call_node.func.attr == premethod
            ):
                attr_parts = _attr_parts(call_node.func.value)
                return any(
                    attr_parts == target_parts for target_parts in target_parts_list
                )
            return False
        return isinstance(call_node.func, ast.Name) and call_node.func.id == premethod

    def _build_first_arg():
        return ast.parse(first_arg, mode="eval").body

    def _build_new_func():
        value = ast.parse(newobj, mode="eval").body
        return ast.Attribute(value=value, attr=newmethod, ctx=ast.Load())

    def _build_new_call_from(original_call: ast.Call) -> ast.Call:
        fallback_args = []
        converted_keywords = []
        existing_kw_names = {kw.arg for kw in original_call.keywords if kw.arg}
        used_param_names = set(existing_kw_names)
        param_idx = 0

        # 将位置参数转换为关键字参数
        for arg in original_call.args:
            if isinstance(arg, ast.Starred):
                fallback_args.append(arg)
                continue

            param_name = None
            while param_idx < len(positionlist):
                candidate = positionlist[param_idx]
                param_idx += 1
                if not candidate or candidate in used_param_names:
                    continue
                param_name = candidate
                used_param_names.add(candidate)
                break

            if param_name is None:
                fallback_args.append(arg)
            else:
                converted_keywords.append(ast.keyword(arg=param_name, value=arg))

        new_call = ast.Call(
            func=_build_new_func(),
            args=[_build_first_arg()] + fallback_args,
            keywords=converted_keywords + original_call.keywords,
        )
        return ast.copy_location(new_call, original_call)

    class BlockAppender(ast.NodeTransformer):
        def __init__(self):
            self.changed = False

        def _process_stmt_list(self, stmts):
            new_stmts = []
            for stmt in stmts:
                new_stmts.append(stmt)

                # 收集该语句内所有匹配的调用，但不递归进入子代码块
                matched_calls = []
                
                # 使用自定义遍历代替 ast.walk，避免进入 body/orelse 等语句块
                nodes_to_check = [stmt]
                while nodes_to_check:
                    node = nodes_to_check.pop(0)
                    
                    if isinstance(node, ast.Call) and _is_target(node):
                        matched_calls.append(node)
                    
                    # 遍历子节点，但跳过语句块字段
                    for field, value in ast.iter_fields(node):
                        if field in ('body', 'orelse', 'finalbody'):
                            # Lambda 的 body 是表达式，属于当前语句的一部分，不应跳过
                            if not isinstance(node, ast.Lambda):
                                continue
                        
                        if isinstance(value, list):
                            for item in value:
                                if isinstance(item, ast.AST):
                                    nodes_to_check.append(item)
                        elif isinstance(value, ast.AST):
                            nodes_to_check.append(value)

                # 在本语句后插入新调用表达式（每个匹配调用插入一次）
                for call in matched_calls:
                    new_call = _build_new_call_from(call)
                    expr = ast.Expr(value=new_call)
                    expr = ast.copy_location(expr, stmt)
                    new_stmts.append(expr)
                    self.changed = True

            return new_stmts

        def visit_Module(self, node):
            node.body = self._process_stmt_list(node.body)
            self.generic_visit(node)
            return node

        def visit_FunctionDef(self, node):
            node.body = self._process_stmt_list(node.body)
            self.generic_visit(node)
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            node.body = self._process_stmt_list(node.body)
            self.generic_visit(node)
            return node

        def visit_If(self, node):
            node.body = self._process_stmt_list(node.body)
            node.orelse = self._process_stmt_list(node.orelse)
            self.generic_visit(node)
            return node

        def visit_For(self, node):
            node.body = self._process_stmt_list(node.body)
            node.orelse = self._process_stmt_list(node.orelse)
            self.generic_visit(node)
            return node

        def visit_AsyncFor(self, node):
            return self.visit_For(node)

        def visit_While(self, node):
            node.body = self._process_stmt_list(node.body)
            node.orelse = self._process_stmt_list(node.orelse)
            self.generic_visit(node)
            return node

        def visit_With(self, node):
            node.body = self._process_stmt_list(node.body)
            self.generic_visit(node)
            return node

        def visit_AsyncWith(self, node):
            return self.visit_With(node)

        def visit_Try(self, node):
            node.body = self._process_stmt_list(node.body)
            node.orelse = self._process_stmt_list(node.orelse)
            node.finalbody = self._process_stmt_list(node.finalbody)
            for h in node.handlers:
                h.body = self._process_stmt_list(h.body)
            self.generic_visit(node)
            return node

        def visit_ExceptHandler(self, node):
            node.body = self._process_stmt_list(node.body)
            self.generic_visit(node)
            return node

    transformer = BlockAppender()
    new_tree = transformer.visit(tree)
    if not transformer.changed:
        return code

    ast.fix_missing_locations(new_tree)
    if sys.version_info >= (3, 9):
        return ast.unparse(new_tree)
    return astor.to_source(new_tree)


def get_class_instance_names(code: str, class_name: str) -> List[str]:
    """
    获取代码中由指定类构造的对象名称列表。

    支持：
    - 构造调用：Class(...) / module.Class(...)
    - 链式调用：Class(...).to(...)
    - 函数返回追踪（含返回注解）
    - 变量传递追踪
    - 赋值左值：变量 / 属性 / 多重 / 解构
    - 类型注解：
        * 赋值注解：x: Class = ...
        * 仅注解：x: Class
        * 参数注解：def f(x: Class)
        * 返回注解：def f() -> Class
        * Optional / Union / List / Tuple / | 等
    返回按出现顺序去重的名称列表。
    """
    if not class_name:
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_class_instance_names parse failed: {exc}")
        return []

    # ------------------ 基础工具 ------------------

    def _attr_to_str(node: ast.Attribute):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        return None

    def _get_value_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return _attr_to_str(node)
        return None

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

    # ------------------ 构造 & 链式调用 ------------------

    def _matches_constructor(call_node: ast.Call) -> bool:
        func = call_node.func
        if isinstance(func, ast.Name):
            return func.id == class_name
        if isinstance(func, ast.Attribute):
            return func.attr == class_name
        return False

    def _is_constructor_chain(node) -> bool:
        if isinstance(node, ast.Call):
            if _matches_constructor(node):
                return True
            if isinstance(node.func, ast.Attribute):
                return _is_constructor_chain(node.func.value)
        return False

    # ------------------ 注解分析 ------------------

    def _annotation_matches_class(node) -> bool:
        """
        判断类型注解是否指向 class_name
        支持：
        - Class
        - module.Class
        - Optional[Class]
        - List[Class] / Tuple[Class, ...]
        - Union[Class, X]
        - Class | X (py3.10+)
        """
        if node is None:
            return False

        if isinstance(node, ast.Name):
            return node.id == class_name

        if isinstance(node, ast.Attribute):
            return node.attr == class_name

        if isinstance(node, ast.Subscript):
            return _annotation_matches_class(node.value) or _annotation_matches_class(
                node.slice
            )

        # py3.8 compatibility
        if isinstance(node, ast.Index):
            return _annotation_matches_class(node.value)

        # PEP 604: Class | Other
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return _annotation_matches_class(node.left) or _annotation_matches_class(
                node.right
            )

        if isinstance(node, (ast.Tuple, ast.List)):
            return any(_annotation_matches_class(elt) for elt in node.elts)

        return False

    # ------------------ 第一阶段：函数级分析 ------------------

    functions_returning_class = set()
    func_to_return_vars = {}
    var_to_class = {}
    var_to_var = {}
    result: List[str] = []
    seen = set()
    instance_vars = set()

    def _record(name: str):
        if name and name not in seen:
            seen.add(name)
            result.append(name)
            instance_vars.add(name)

    class FunctionAnalyzer(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None

        def visit_FunctionDef(self, node):
            old_function = self.current_function
            self.current_function = node.name

            # 返回类型注解
            if node.returns and _annotation_matches_class(node.returns):
                functions_returning_class.add(node.name)

            # 参数注解
            for arg in node.args.args:
                if arg.annotation and _annotation_matches_class(arg.annotation):
                    var_to_class[arg.arg] = True
                    _record(arg.arg)

            # 函数体内赋值
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, ast.Call) and _is_constructor_chain(
                        stmt.value
                    ):
                        for target in stmt.targets:
                            for var_name in _collect_targets(target):
                                var_to_class[var_name] = True

                elif isinstance(stmt, ast.AnnAssign):
                    if stmt.annotation and _annotation_matches_class(stmt.annotation):
                        target_name = (
                            _attr_to_str(stmt.target)
                            if isinstance(stmt.target, ast.Attribute)
                            else (
                                stmt.target.id
                                if isinstance(stmt.target, ast.Name)
                                else None
                            )
                        )
                        if target_name:
                            var_to_class[target_name] = True

                    if isinstance(stmt.value, ast.Call) and _is_constructor_chain(
                        stmt.value
                    ):
                        target_name = (
                            _attr_to_str(stmt.target)
                            if isinstance(stmt.target, ast.Attribute)
                            else (
                                stmt.target.id
                                if isinstance(stmt.target, ast.Name)
                                else None
                            )
                        )
                        if target_name:
                            var_to_class[target_name] = True

            # return 分析
            return_vars = []
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and stmt.value:
                    if isinstance(stmt.value, ast.Call) and _is_constructor_chain(
                        stmt.value
                    ):
                        functions_returning_class.add(self.current_function)
                    else:
                        rv = _get_value_name(stmt.value)
                        if rv:
                            return_vars.append(rv)

            if return_vars:
                func_to_return_vars[node.name] = return_vars

            self.generic_visit(node)
            self.current_function = old_function

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

    analyzer = FunctionAnalyzer()
    analyzer.visit(tree)

    # 二次传播：函数返回变量 -> 函数返回 Class
    for func_name, return_vars in func_to_return_vars.items():
        for var_name in return_vars:
            if var_to_class.get(var_name):
                functions_returning_class.add(func_name)
                break

    # ------------------ 第二阶段：全局赋值与调用 ------------------

    # 类定义中 self
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            _record("self")
            break

    for node in ast.walk(tree):
        call_node = None
        targets = None

        if isinstance(node, ast.Assign):
            value = node.value

            if isinstance(value, ast.Call) and _is_constructor_chain(value):
                call_node = value
                targets = node.targets

            elif isinstance(value, ast.Call):
                func_name = (
                    value.func.id
                    if isinstance(value.func, ast.Name)
                    else (
                        value.func.attr
                        if isinstance(value.func, ast.Attribute)
                        else None
                    )
                )
                if func_name and func_name in functions_returning_class:
                    call_node = value
                    targets = node.targets

            else:
                source_var = _get_value_name(value)
                if source_var:
                    for target in node.targets:
                        for target_name in _collect_targets(target):
                            var_to_var[target_name] = source_var

        elif isinstance(node, ast.AnnAssign):
            # 注解直接判定
            if node.annotation and _annotation_matches_class(node.annotation):
                target_name = (
                    _attr_to_str(node.target)
                    if isinstance(node.target, ast.Attribute)
                    else node.target.id if isinstance(node.target, ast.Name) else None
                )
                if target_name:
                    _record(target_name)

            value = node.value
            if isinstance(value, ast.Call) and _is_constructor_chain(value):
                call_node = value
                targets = [node.target]

            elif isinstance(value, ast.Call):
                func_name = (
                    value.func.id
                    if isinstance(value.func, ast.Name)
                    else (
                        value.func.attr
                        if isinstance(value.func, ast.Attribute)
                        else None
                    )
                )
                if func_name and func_name in functions_returning_class:
                    call_node = value
                    targets = [node.target]

            elif value is not None:
                source_var = _get_value_name(value)
                if source_var:
                    target_name = (
                        _attr_to_str(node.target)
                        if isinstance(node.target, ast.Attribute)
                        else (
                            node.target.id
                            if isinstance(node.target, ast.Name)
                            else None
                        )
                    )
                    if target_name:
                        var_to_var[target_name] = source_var

        if call_node and targets:
            for tgt in targets:
                for name in _collect_targets(tgt):
                    _record(name)

    # ------------------ 第三阶段：变量传递闭包 ------------------
    def _resolve_var(var_name: str, visited=None) -> bool:
        if visited is None:
            visited = set()
        if var_name in visited:
            return False
        if var_to_class.get(var_name):
            return True
        if var_name in instance_vars:
            return True
        visited.add(var_name)
        if var_name in var_to_var:
            return _resolve_var(var_to_var[var_name], visited)
        return False

    for target_var in list(var_to_var.keys()):
        if _resolve_var(target_var):
            _record(target_var)

    return result


def get_function_param_names(
    code: str, func_name: str, class_name: str = "", baseclasses: List[str] = None
):
    """
    获取指定函数或方法定义的参数名列表。
    - class_name 为空:匹配顶层函数定义(不含嵌套函数)。
    - class_name 非空:仅匹配该类的直接方法定义(不含嵌套类/函数);若首参为 self 会被移除。
    - baseclasses: 基类列表，如果指定类中未找到方法，则尝试在基类中查找(按列表顺序)。
    - 返回两个列表: (位置参数列表, 关键字参数列表)
    - 位置参数包含: 位置仅参数 + 普通位置参数
    - 关键字参数包含: 关键字仅参数
    - 忽略 *args 和 **kwargs
    """
    if not func_name:
        return [], []
    if class_name is None:
        class_name = ""
    if baseclasses is None:
        baseclasses = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"[INS_WARN] get_function_param_names parse failed: {exc}")
        return [], []

    def _collect(args: ast.arguments, drop_self: bool):
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

    # 确定搜索顺序
    search_order = []
    if class_name:
        search_order.append(class_name)
    if baseclasses:
        search_order.extend(baseclasses)
    
    # 用于快速判断是否需要进入某类的集合
    target_classes = set(search_order)

    class ParamCollector(ast.NodeVisitor):
        def __init__(self):
            self.found_methods = {}  # class_name -> (pos, kw)
            self.top_level_result = None
            self.class_stack: List[str] = []

        def visit_ClassDef(self, node):
            if self.class_stack:
                # 忽略嵌套类
                return
            
            # 只有当类在目标列表中时才进入查找
            if node.name in target_classes:
                self.class_stack.append(node.name)
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.visit(stmt)
                self.class_stack.pop()

        def visit_FunctionDef(self, node):
            inside_class = bool(self.class_stack)

            # 匹配方法名
            if node.name != func_name:
                return

            if inside_class:
                current_class = self.class_stack[-1]
                # 记录该类下的方法参数
                self.found_methods[current_class] = _collect(
                    node.args, drop_self=True
                )
                return

            # 匹配顶层函数
            if not class_name and not inside_class:
                self.top_level_result = _collect(
                    node.args, drop_self=False
                )

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

    collector = ParamCollector()
    collector.visit(tree)

    # 如果没有指定类名，返回顶层函数结果
    if not class_name:
        return collector.top_level_result if collector.top_level_result else ([], [])

    # 按优先级查找结果
    for cls in search_order:
        if cls in collector.found_methods:
            return collector.found_methods[cls]

    return [], []


def remove_function_calls(code: str, function_names) -> str:
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
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in function_names
            ):
                print(
                    f"[INS_WARN] Removing function call statement: {node.value.func.id}"
                )
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
                if (
                    isinstance(subnode, ast.Call)
                    and isinstance(subnode.func, ast.Name)
                    and subnode.func.id in function_names
                ):
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
                    line = (
                        f"import {alias.name}"
                        if not alias.asname
                        else f"import {alias.name} as {alias.asname}"
                    )
                    if line not in imports_to_remove:
                        new_names.append(alias)
                if not new_names:
                    print(
                        f"[INS_WARN] Removing import statement: import {node.names[0].name}"
                    )
                    return None
                node.names = new_names
                return node

            def visit_ImportFrom(self, node):
                from_lines = []
                for alias in node.names:
                    line = (
                        f"from {node.module} import {alias.name}"
                        if not alias.asname
                        else f"from {node.module} import {alias.name} as {alias.asname}"
                    )
                    from_lines.append(line)
                new_names = [
                    alias
                    for alias, line in zip(node.names, from_lines)
                    if line not in imports_to_remove
                ]
                if not new_names:
                    print(f"[INS_WARN] Removing import statement: from {node.module}")
                    return None
                node.names = new_names
                return node

        tree = ImportRemover().visit(tree)
        ast.fix_missing_locations(tree)
        if sys.version_info >= (3, 9):
            new_code = ast.unparse(tree)
            if has_syntax_error(new_code):
                return source_code
            else:
                return new_code
        else:
            new_code = astor.to_source(tree)
            if has_syntax_error(new_code):
                return source_code
            else:
                return new_code

    except Exception:
        print("[INS_WARN] AST parsing failed, returning original code")
        return source_code  # 如果 AST 解析失败，返回原始代码


def replace_assignment_which_with_path_to(
    code: str, var_name: str, new_rhs_code: str
) -> str:
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
            if "." in var_name:
                var_name = var_name.split(".")[-1]  # 只匹配最后一部分
            return target.id == var_name
        elif isinstance(target, ast.Attribute):
            if "." in var_name:
                parts = var_name.split(".")
                if len(parts) == 2:
                    return (
                        target.attr == parts[1]
                        and getattr(target.value, "id", None) == parts[0]
                    )
            else:
                return target.attr == var_name
        return False

    class PathStringReplacer(ast.NodeTransformer):
        def __init__(self, new_expr_ast):
            self.new_expr_ast = new_expr_ast  # 已经是 AST 表达式节点
            self.replaced = False

        def visit_Constant(self, node):
            if isinstance(node.value, str) and (
                "path/to" in node.value.lower() or "path to" in node.value.lower()
            ):
                self.replaced = True
                return ast.copy_location(self.new_expr_ast, node)
            return node

        def visit_Str(self, node):  # 兼容 Python < 3.8
            if "path/to" in node.s.lower() or "path to" in node.s.lower():
                self.replaced = True
                return ast.copy_location(self.new_expr_ast, node)
            return node

    def replace_rhs_node(rhs_node, new_rhs_code):
        expr_ast = ast.parse(new_rhs_code, mode="eval").body  # 保留表达式结构
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
                        self.default_value_assignments.append(
                            (node, "args", default_start + i, default)
                        )
            if node.args.kw_defaults:
                for i, default in enumerate(node.args.kw_defaults):
                    if default is not None and node.args.kwonlyargs[i].arg == var_name:
                        self.default_value_assignments.append(
                            (node, "kw_defaults", i, default)
                        )
            self.scope_stack.append("function")
            self.generic_visit(node)
            self.scope_stack.pop()
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            self.scope_stack.append("class")
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
                    print(
                        f"[INS_INFO] Replacing function call keyword argument: {var_name}"
                    )
                    return

            # 3. 函数内变量赋值
            for assign_node, rhs_node in self.function_assignments:
                new_node, replaced = replace_rhs_node(rhs_node, new_rhs_code)
                if replaced:
                    assign_node.value = new_node
                    print(f"[INS_INFO] Replacing function assignment: {var_name}")
                    return

            # 4. 函数默认值
            for (
                func_node,
                param_type,
                index,
                rhs_node,
            ) in self.default_value_assignments:
                new_node, replaced = replace_rhs_node(rhs_node, new_rhs_code)
                if replaced:
                    if param_type == "args":
                        func_node.args.defaults[index] = new_node
                    elif param_type == "kw_defaults":
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
                        print(
                            f"[INS_INFO] Original dictionary value contains path/to or path to, performing replacement"
                        )
                        node.values[idx] = ast.parse(new_value_code, mode="eval").body
            return node

    try:
        tree = ast.parse(code)
        tree = DictValueReplacer().visit(tree)
        ast.fix_missing_locations(tree)

        if sys.version_info >= (3, 9):
            new_code = ast.unparse(tree)
        else:
            new_code = astor.to_source(tree)
        return new_code
    except Exception as e:
        print(f"[INS_WARN] Failed to replace dictionary value: {e}")
        return code


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
        r"^[ \t]*from[ \t]+__future__[ \t]+import[ \t]+"
        r"(?:\\\n|.*\([^)]*\)|.*(?:\n[ \t]+[^\n]*)*)",
        re.MULTILINE,
    )
    original_code = code
    code = pattern.sub("", code)
    if has_syntax_error(code):
        # 如果替换后仍有语法错误，说明可能有其他问题，返回原代码
        return original_code
    return code



def clean_path_to_in_code(code: str) -> str:
    """
    path/to 处理规则：
    1. suffix 可识别 → 替换为 RESOURCES_PATH + "{type}/default.{suffix}"
    2. suffix 不可识别 → 回退到原始逻辑（删除 / 置空）
    """
    _SUFFIX_TO_TYPE = {}
    for _type in ("images", "audios", "videos", "texts"):
        for _ext in get_available_outter_resources_exts(_type):
            _SUFFIX_TO_TYPE[_ext] = _type

    def contains_path_to(s: str) -> bool:
        s = s.lower()
        return "path/to" in s or "path to" in s

    def infer_type_and_suffix(s: str):
        _, ext = os.path.splitext(s.lower())
        if not ext:
            return None, None
        suffix = ext.lstrip(".")
        return _SUFFIX_TO_TYPE.get(suffix), suffix

    def infer_folder_type(s: str):
        s_lower = s.lower().replace("\\", "/").rstrip("/")
        if s_lower.endswith("/images") or s_lower.endswith(" images"):
            return "images"
        if s_lower.endswith("/videos") or s_lower.endswith(" videos"):
            return "videos"
        if s_lower.endswith("/audios") or s_lower.endswith(" audios"):
            return "audios"
        if s_lower.endswith("/texts") or s_lower.endswith(" texts"):
            return "texts"
        if s_lower.endswith("/inputs") or s_lower.endswith("/input"):
            return "images"
        return None

    def make_resource_expr(resource_type: str, suffix: str) -> ast.AST:
        return ast.BinOp(
            left=ast.Name(id="RESOURCES_PATH", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(
                value=f"{resource_type}/default.{suffix}"
            ),
        )

    def make_folder_resource_expr(resource_type: str) -> ast.AST:
        return ast.BinOp(
            left=ast.Name(id="RESOURCES_PATH", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(
                value=f"{resource_type}/default_{resource_type}_folder"
            ),
        )

    class PathToCleaner(ast.NodeTransformer):

        def visit_Constant(self, node):
            if isinstance(node.value, str) and contains_path_to(node.value):
                resource_type, suffix = infer_type_and_suffix(node.value)

                if resource_type:
                    # ✅ 可识别资源 → 语义替换
                    return ast.copy_location(
                        make_resource_expr(resource_type, suffix),
                        node,
                    )
                
                folder_type = infer_folder_type(node.value)
                if folder_type:
                    # ✅ 可识别资源文件夹 → 语义替换
                    return ast.copy_location(
                        make_folder_resource_expr(folder_type),
                        node,
                    )
                else:
                    # ❌ 不可识别 → 原始逻辑：置空（非 keyword）
                    parent = getattr(node, "parent", None)
                    if not isinstance(parent, ast.keyword):
                        return ast.copy_location(
                            ast.Constant(value=""),
                            node,
                        )
            return node

        def visit_Call(self, node):
            new_keywords = []
            for kw in node.keywords:
                if (
                    isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                    and contains_path_to(kw.value.value)
                ):
                    resource_type, suffix = infer_type_and_suffix(
                        kw.value.value
                    )
                    if resource_type:
                        # ✅ 替换为资源表达式
                        kw.value = make_resource_expr(resource_type, suffix)
                        new_keywords.append(kw)
                    else:
                        folder_type = infer_folder_type(kw.value.value)
                        if folder_type:
                            # ✅ 替换为资源文件夹表达式
                            kw.value = make_folder_resource_expr(folder_type)
                            new_keywords.append(kw)
                        else:
                            # ❌ 不可识别 → 原始逻辑：删除参数
                            continue
                else:
                    new_keywords.append(kw)

            node.keywords = new_keywords
            self.generic_visit(node)
            return node

        def generic_visit(self, node):
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
            if astor is None:
                raise RuntimeError("astor is required for Python < 3.9")
            return astor.to_source(cleaned_tree)

    except Exception as e:
        print(f"[INS_WARN] Failed to process placeholder path/to: {e}")
        return code