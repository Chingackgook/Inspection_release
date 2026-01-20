from Inspection.utils.path_manager import SIMULATOR_PATH, BASE_DIR
from Inspection.utils.config import CONFIG

import os
import ast
import sys
import re


class Injector:
    """
    将dumbsimulator生成的模拟参数函数，注入到simulation的代码中，生成注入后文件存入dumbsimulator目录下
    """

    def __init__(self, name):
        """
        name:项目名
        """
        self.pj_name = name

    def get_call_code(self, call_idx, api_name):
        dir_name = SIMULATOR_PATH + f"{self.pj_name}/"
        pyfiles = os.listdir(dir_name)
        pyfiles = [f for f in pyfiles if f.endswith(".py")]
        for file in pyfiles:
            match = re.match(r"call(\d+)_(.+)_simulator\.py", file)
            if match:
                idx = int(match.group(1))
                apiname_in_file = match.group(2)
                if idx == call_idx and apiname_in_file == api_name:
                    with open(dir_name + file, "r") as f:
                        code = f.read()
                    return code
        return None

    def inject(self, call_idx, api_name, module_path):
        simulate_code = self.get_call_code(call_idx, api_name)
        if simulate_code is None:
            print(
                f"[INS_ERR] Could not find simulator code for call {call_idx} of {api_name}"
            )
            return None
        injected_code = self.use_ast_inject_exe(simulate_code, api_name, module_path)
        return injected_code

    def use_ast_inject_exe(self, simulate_code, api_name, module_path):

        import_path = self._module_path_to_import_statement(module_path)
        tree = ast.parse(simulate_code)
        transformer = ExeRunInjector(api_name=api_name, import_path=import_path)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        injected_code = ""
        if sys.version_info >= (3, 9):
            injected_code = ast.unparse(new_tree)
        else:
            import astor

            injected_code = astor.to_source(new_tree)

        return injected_code

    def _module_path_to_import_statement(self, module_path: str):
        try:
            module_path = os.path.abspath(module_path)
            base_dir = os.path.abspath(BASE_DIR)
            if not module_path.startswith(base_dir):
                print(
                    f"[INS_ERR] Module path {module_path} is not under project root {base_dir}"
                )
                return None
            relative_path = os.path.relpath(module_path, base_dir)
            if relative_path.endswith(".py"):
                relative_path = relative_path[:-3]
            elif relative_path.endswith(".pyx"):
                relative_path = relative_path[:-4]
            import_path = relative_path.replace(os.sep, ".")
            import_path = import_path.lstrip(".")
            if not import_path or ".." in import_path:
                print(f"[INS_ERR] Generated import path is invalid: {import_path}")
                return None
            return import_path
        except Exception as e:
            print(f"[INS_ERR] Error occurred while converting module path: {e}")
            return None


class ExeRunInjector(ast.NodeTransformer):
    def __init__(self, api_name, import_path):
        self.api_name = api_name
        self.import_path = import_path
        self.kwarg_id = f"{api_name}_inject_kwarg"

    def is_exe_run_call(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "exe"
            and node.func.attr == "run"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == self.api_name
        )

    def make_import_and_kwarg(self):
        import_node = ast.ImportFrom(
            module=self.import_path,
            names=[
                ast.alias(name="dumb_simulator", asname=None),
                ast.alias(name="set_exe", asname=None),
            ],
            level=0,
        )
        # exe.set_record_function([""])
        call_unset_record_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="exe", ctx=ast.Load()),
                    attr="set_record_function",
                    ctx=ast.Load(),
                ),
                args=[ast.List(elts=[ast.Constant(value="")], ctx=ast.Load())],
                keywords=[],
            )
        )

        # dumb_simulator() -> bark_inject_kwarg
        assign_dumb_node = ast.Assign(
            targets=[ast.Name(id=self.kwarg_id, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id="dumb_simulator", ctx=ast.Load()), args=[], keywords=[]
            ),
        )

        # set_exe(exe)
        call_exeinit_node = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="set_exe", ctx=ast.Load()),
                args=[ast.Name(id="exe", ctx=ast.Load())],
                keywords=[],
            )
        )

        # exe.set_record_function(["api_name"])
        call_exe_record_node = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="exe", ctx=ast.Load()),
                    attr="set_record_function",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.List(elts=[ast.Constant(value=self.api_name)], ctx=ast.Load())
                ],
                keywords=[],
            )
        )
        return [
            import_node,
            call_unset_record_node,
            call_exeinit_node,
            assign_dumb_node,
            call_exe_record_node,
        ]

    def visit_Module(self, node):
        self.generic_visit(node)
        insert_nodes = self.make_import_and_kwarg()
        # 寻找 exe 的初始化语句
        insert_index = 0
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "exe":
                        insert_index = i + 1  # 插入在 exe 初始化语句之后
                        break
        node.body = node.body[:insert_index] + insert_nodes + node.body[insert_index:]
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if self.is_exe_run_call(node):
            return ast.Call(
                func=node.func,
                args=[node.args[0]],
                keywords=[
                    ast.keyword(
                        arg=None, value=ast.Name(id=self.kwarg_id, ctx=ast.Load())
                    )
                ],
            )
        return node

    def visit_Assign(self, node):
        # 检查是否是 exe 的赋值语句
        self.generic_visit(node)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "exe":
                # 检查赋值的值是否是类初始化
                if isinstance(node.value, ast.Call):
                    # 确保类初始化有至少两个参数
                    if len(node.value.args) > 1:
                        # 将第二个参数改为字符串 "dumb"
                        node.value.args[1] = ast.Constant(s="dumb")
                break
        return node
