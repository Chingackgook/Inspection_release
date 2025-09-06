from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.marvin import *
import sys
from typing import Annotated
from pydantic import Field, TypeAdapter
import marvin
exe = Executor('marvin', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/marvin/examples/hello_generate.py'
Fruit = Annotated[str, Field(description='A fruit')]
fruits = exe.run('generate', target=Fruit, n=3, instructions='high vitamin C content')
assert len(fruits) == 3
print('Results are a valid list of Fruit:')
print(f'{TypeAdapter(list[Fruit]).validate_python(fruits)}')
bizarro_names = exe.run('generate', target=str, n=len(fruits), instructions=f'bizarro sitcom character names based on these fruit: {fruits}')
print(bizarro_names)