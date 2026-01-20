from dataclasses import dataclass
from typing import List

@dataclass
class FunctionDef:
    function_name: str
    position_parameters: List[str]
    keyword_parameters: List[str]

@dataclass
class ClassDef:
    class_name: str
    base_classes: List[str]
    init_method: FunctionDef
    methods: List[FunctionDef]

@dataclass
class Call:
    call_name: str # 为方法名或函数名
    call_site: str # 调用该方法的顶层函数或顶层类方法的名称 格式：类名.方法名 或  函数名
    object_name: str
    object_type: List[str] # 可能的类型列表

@dataclass
class Object:
    call_site: str # 该对象的位置在顶层函数或顶层类方法的名称 格式：类名.方法名 或  函数名
    object_name: str
    object_type: List[str] # 可能的类型列表


@dataclass
class ObjectCall:
    call_site: str # 调用该对象的顶层函数或顶层类方法的名称 格式：类名.方法名 或  函数名
    object_name: str
    object_type: List[str] # 可能的类型列表

