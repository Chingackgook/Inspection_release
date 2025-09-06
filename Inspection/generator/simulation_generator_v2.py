# 模拟执行代码生成器
from Inspection.ai.base_ai import BaseAI
from Inspection.core.code_processor import extract_python_code, extract_import_statements
from Inspection.utils.path_manager import INTERFACE_DOC_PATH, CUSTOM_ADAPTER_PATH, INTERFACE_INFO_PATH , BASE_DIR ,EVALUATE_RQ1_PATH
from Inspection.core.code_processor import *
from Inspection.utils.interface_reader import InterfaceInfoReader, InterfaceDocReader
import os
import json
import time
from Inspection.utils.path_manager import SIMULATION_PATH
from Inspection.utils.config import CONFIG


class SimulationGenerator:
    def __init__(self, name: str):
        self.name = name
        self.base_ai_instance :BaseAI = None
        self.prompt_user_for_overwrite = CONFIG.get('ask', True)  # 询问是否覆盖
        self.simulation_path = SIMULATION_PATH + 'simulate_interface/' + self.name + '/'
        self.analysis_data = ""
        self.evaluate_mode = CONFIG.get('evaluate_mode', False)  # 是否开启评估模式生成
        self.analysis_temprature = CONFIG.get('simulation_analysis_temprature', 0.5)  # 分析温度
        self.generate_code_temprature = CONFIG.get('simulation_generate_code_temprature', 0.3)  # 生成代码温度
    
    def set_base_ai(self, base_ai_instance):
        self.base_ai_instance = base_ai_instance

    def _init_evaluate_mode(self):
        write_dir = EVALUATE_RQ1_PATH + 'simulation/' + self.name + '/'
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        file_name_base = str(self.call_idx) + '_' + self.call_name + '_v2'
        # 自动递增文件名
        idx = 1
        while os.path.exists(write_dir + file_name_base + f'_genidx_{idx}.py'):
            idx += 1
        # Evaluate!!!!
        if idx >  CONFIG.get('evaluate_simulation_gen_times', 0):
            raise FileExistsError(f"Maximum file count reached for {self.name} in evaluation mode.")
        # 正式启用时删除包裹的代码块
        self.write_path = write_dir + file_name_base + f'_genidx_{idx}.py'
        self.analysis_data_path = write_dir + file_name_base + f'_genidx_{idx}_analysis.md'

    def _write_analysis_data(self):
        with open(self.analysis_data_path, 'w') as f:
            f.write(self.analysis_data)

    def generate_test_code(self):
        # 未实现V2版本的测试代码生成，直接调用V1版本的生成器
        from Inspection.generator.discarded.simulation_generator import SimulationGenerator
        simulation_generator = SimulationGenerator(self.name)
        simulation_generator.generate_test_code()

    def generate_simulation(self, call_idx :int = 0):
        # 生成根据调用信息，以及原调用步骤生成的模拟执行代码
        simulation_path = self.simulation_path
        start_time = time.time()
        try:
            info_reader = InterfaceInfoReader(self.name)
        except FileNotFoundError:
            print(f"[INS_ERR] Static executable artifact generation failed, please check if interface information exists in {INTERFACE_INFO_PATH} and JSON format is correct")
            return
        try:
            doc = InterfaceDocReader(self.name).get_doc()
        except FileNotFoundError:
            print(f"[INS_ERR] Static executable artifact generation failed, please check if interface documentation exists in {INTERFACE_DOC_PATH}")
            return
        call_data= info_reader.get_call_str_by_idx(call_idx)
        call_name = info_reader.get_call_dict_by_idx(call_idx).get('Name', 'unknown')
        self.call_name = call_name
        self.call_idx = call_idx
        call_path = info_reader.get_call_dict_by_idx(call_idx).get('Path', '')
        future_import_data , import_data = extract_import_statements(call_data)
        self.analysis_data_path = simulation_path + str(call_idx) + '_' + call_name + '_v2_analysis.md'
        self.write_path = simulation_path + str(call_idx) + '_' + call_name + '_v2.py'
        # 检查文件是否已经存在模拟执行代码
        if os.path.exists(self.write_path) and self.prompt_user_for_overwrite and not self.evaluate_mode:
            print(f"[INS_WARN] Static executable artifact {self.name}/ {call_idx}_{call_name}_v2 already exists")
            ch = input("Overwrite? (y/n)")
            if ch != 'y':
                return
        # 创建目录
        if not os.path.exists(simulation_path):
            os.makedirs(simulation_path)
        exe_data = get_class_code(CUSTOM_ADAPTER_PATH+self.name+'.py', 'CustomAdapter')
        avaliable_fuc = extract_function_names_from_adapter(exe_data)
        avaliable_class = extract_class_names_from_adapter(exe_data)
        avaliable_fuc_str = '\n'.join([f" - {fuc}" for fuc in avaliable_fuc])
        avaliable_class_str = '\n'.join([f" - {cls}" for cls in avaliable_class])
        if avaliable_class == []:
            avaliable_class_str = "Note: No available classes, all calls are independent function calls"
        self.doc = doc
        self.call_data = call_data # 调用部分源代码
        self.exe_data = exe_data # exe对象的实现代码（实际上是适配器的实现代码）
        self.avaliable_fuc = avaliable_fuc # 可用的独立函数，方法列表
        self.avaliable_class = avaliable_class # 可用的类列表
        self.avaliable_fuc_str = avaliable_fuc_str # 已注册的独立函数，方法列表
        self.avaliable_class_str = avaliable_class_str # 已注册的类列表
        self.analysis_data = "" # 用于存储分析结果，最后会写入记录md文件中
        if self.evaluate_mode:
            # 如果是评估模式生成，则使用评估模式的路径
            self._init_evaluate_mode()
            print(f"[INS_INFO] Evaluation generation mode")
        self._write_analysis_data()
        print('[INS_INFO] Analysis process located in ' + self.analysis_data_path + ' for viewing anytime')
        # 以上是该方法的一些准备工作，下面开始进行ai生成模拟执行代码
        executable_code = self.change_code_to_executable(call_data)
        if not executable_code:
            print(f"[INS_ERR] Static executable artifact generation failed, unable to convert original call to static executable artifact")
            return
        executable_code = remove_definitions_by_names(executable_code,extract_from_import_object(import_data))

        exe_replaced_code = self.change_interface_to_exe_methods_withLLM(executable_code)
        if not exe_replaced_code:
            print(f"[INS_ERR] Static executable artifact generation failed, unable to replace interface calls with exe.run form")
            return
        result = exe_replaced_code
        result = remove_assignments('exe', result , use_regex=True)
        result = remove_assignments('os.environ[\'OPENAI_API_KEY\']', result, use_regex=True)
        result = remove_future_imports(result)
        result = remove_exe_imports(result)
        if call_path:
            result = replace_file_variable_in_code(result, call_path)
        result = f"""
{future_import_data}
from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.{self.name} import *
exe = Executor('{self.name}','simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '{call_path}'
{import_data}
# end\n\n
"""  +  result
        optimized_result = self.optimize_code(result)
        optimized_result = self.change_external_resources_path(optimized_result)
        optimized_result = remove_assignments('exe.now_record_path', optimized_result, use_regex=True)

        total_time = time.time() - start_time
        self.analysis_data += f"\n\n\n$$$$$静态可执行工件生成耗时$$$$$\n"
        self.analysis_data += f"Total time: {total_time:.2f} seconds\n"
        self._write_analysis_data()

        self.write_code_to_file(optimized_result)


    def write_code_to_file(self, code: str):
        with open(self.write_path, 'w') as f:
            f.write(code)
        if not self.evaluate_mode:
            with open(BASE_DIR + '/rc.py', 'w') as f:
                f.write(code) # 用于快速测试
        print(f"[INS_INFO] Static executable artifact {self.name} generated successfully, saved to {self.write_path}")

    def change_code_to_executable(self, code: str):
        ai = self.base_ai_instance.copy()
        print('[INS_INFO] Analyzing code logic and converting to executable code ...')
        analysis_prompt = f"""
Here is a piece of code that calls an intelligent module:
```python
{code}
```
Here are some key top-level functions and methods that may appear in this code snippet:
```function/method list:
{self.avaliable_fuc_str}
```
Here is the API documentation for the key functions/methods:
```api documentation:
{self.doc}
```
Please explain in detail what the main execution logic of this code is. Provide a thorough analysis.
"""
        self.analysis_data += "$$$$$代码逻辑分析$$$$$\n"
        self.analysis_data += ai.generate_text(analysis_prompt, temperature=self.analysis_temprature)
        self._write_analysis_data()
        suppose_prompt = f"""
Based on the above analysis, if we directly run this piece of code using Python’s `exec` function, what potential problems might occur?
How should this code be modified, with minimal changes to its logic, so that it can be executed directly via the `exec` function.
Generate a plan for modifying this code (do not generate the modified code yet).
Tips:
1. Remove or replace any interactive input mechanisms (such as input(Pay attention to the dead loop), argparse, typer_app, https request, or web UI interactions) with hardcoded values
 (especially for the input file , if it has a default value, please use the default value given in the call data, or use a placeholder path like 'path/to/..').
2. If the code is a Python module without an `if __name__ == "__main__"` block or any other execution entry point, you need to add an entry point and provide appropriate input data.
 (You need to ensure that the code is able to execute key functions or methods such as {self.avaliable_fuc_str}, rather than simply invoking non-essential content.)
Please analyze first — do not generate the modified code yet.
"""
        self.analysis_data += "\n\n\n$$$$$代码执行补全分析$$$$$\n"
        self.analysis_data += ai.generate_text(suppose_prompt, temperature=self.analysis_temprature)
        self._write_analysis_data()
        generate_prompt = f"""
{code}

Based on the analysis above, modify this piece of code so that it can be directly executed using the `exec` function.

Requirements:
1. The generated code must be complete Python code that can be run directly via `exec`.
2. Keep the original logic structure as intact as possible; avoid leaving out any parts of the original code.
3. Output **only** the Python code, wrapped in ```python ... ```, with no extra explanations or text.

"""
        self.analysis_data += "\n\n\n$$$$$代码执行补全结果$$$$$\n"
        result = ai.generate_text(generate_prompt, temperature=self.generate_code_temprature)
        self.analysis_data += result
        self._write_analysis_data()
        result = extract_python_code(result)
        return result
    
    def change_interface_to_exe_methods_withLLM(self, code: str):
        """
        将代码中的接口调用替换为exe.run("function_name" , **kwargs)的形式
        :param code: Python代码字符串
        :return: 替换后的代码字符串
        """
        ai = self.base_ai_instance.copy()
        print('[INS_INFO] Replacing interface calls with exe.run form ...')
        analysis_prompt = f"""
Here is a code snippet that calls an intelligent module:
{code}
Here is the list of key top-level functions or methods that may appear in this code snippet:
```Methods/Functions List:
{self.avaliable_fuc_str}
```End
```Available Classes List:
{self.avaliable_class_str}
```End
Q1: Identify which key functions/methods from the above list are actually called in this code snippet. You must only select functions/methods from the provided list, and methods must belong to the available classes
(Be careful with the same method names in different classes! Only select the methods that belong to the available classes {self.avaliable_class_str}).
Q2: For each function/method you found in Q1, categorize it: indicate whether it is a method of a class (specify which class and the object that calls it , the class name must be in the available classes list), or a top-level function (not belonging to any class).
Q3: For each object identified in Q2, locate the part of the code where the object is initialized and tell me the class name and the initialization parameters for that object.(the class name must be in the available classes list: {self.avaliable_class_str})

"""
        self.analysis_data += "\n\n\n$$$$$代码关键函数/方法定位与分类分析$$$$$\n"
        self.analysis_data += ai.generate_text(analysis_prompt, temperature=self.analysis_temprature)
        self._write_analysis_data()
        replace_prompt = f"""
Here is the API documentation for the key functions/methods:
{self.doc}
For the functions/methods identified as being called in the previous step, please do the following:
1. If it is a top-level function call (e.g., `var = func1(arg1, arg2)`), rewrite it according to the parameter signature in the API documentation as `var = exe.run("func1", arg1=arg1, arg2=arg2)` (you must use keyword arguments for all parameters except the first one).
2. If it is a class method call (e.g., `var = obj.func2(arg1, arg2)`), rewrite it according to the parameter signature in the API documentation as `var = exe.run("func2", arg1=arg1, arg2=arg2)`.
   (Note: the methods of this `obj` must be called through `exe.run` — do not call them directly using `obj.method()`)
3. For the objects that the class methods were called on in step 2, replace their original initialization with:
   `obj = exe.create_interface_objects(interface_class_name='ClassNameOfTheObject', **kwargs)`,  
   where `kwargs` are the original initialization parameters of that class , you must use keyword arguments for all parameters from the API documentation.
   (If all the key functions in the code are top-level functions, you should not replace any object initialization.)
4. Do not modify any parameter values in the replaced function or method calls. Make sure to preserve the context of the original method, such as indexing or other related operations.
Please generate an analysis based on the above four points and provide a complete replacement plan (no need to generate executable code for now). Assume that the exe object has already been implemented.
"""
        self.analysis_data += "\n\n\n$$$$$代码接口调用替换分析$$$$$\n"
        analysis_result = ai.generate_text(replace_prompt, temperature=self.analysis_temprature)
        self.analysis_data += analysis_result
        self._write_analysis_data()
        ai.clear_history()
        if self.avaliable_class == []:
            replace_class_init_prompt = ""
        else:
            replace_class_init_prompt = f"""
2. For the initialization of objects of class `{self.avaliable_class_str}`, replace them with `exe.create_interface_objects(interface_class_name='ClassName', **kwargs)`.  
   Do not replace the initialization of any other objects.
"""
        generate_prompt = f"""
{code}
Analysis:
{analysis_result}
Based on the analysis above, please rewrite the code by replacing the interface calls as follows:
1. Only replace the identified interface calls using the format `exe.run("function_name", **kwargs)`, you can only replace {self.avaliable_fuc_str} functions and methods , do not replace any other functions or methods.
{replace_class_init_prompt}
3. Just perform the code replacement , Please return all other irrelevant code exactly as it is, without any changes.
4. The replacement method behaves the same as the original. Do not omit any surrounding code or context during replacement.
   Return the full updated code, and wrap it in ```python ... ``` blocks.
"""
        self.analysis_data += "\n\n\n$$$$$代码接口调用替换结果$$$$$\n"
        result = ai.generate_text(generate_prompt, temperature=self.generate_code_temprature)
        result = extract_python_code(result)  # 提取python代码
        self.analysis_data += result
        self._write_analysis_data()
        result = transform_exe_attributes_and_call_name(result,exceptlist=['now_record_path'])
        result = reset_exe_run_create_with_origin(result, self.avaliable_fuc , self.avaliable_class)
        return result
    

    def change_external_resources_path(self, code: str):
        """
        将一些图片，音频，视频等外部资源的路径替换为本项目默认静态资源
        """
        ai_instance = self.base_ai_instance.copy()
        print('[INS_INFO] Attempting to replace external resource paths with default static resources ...')
        ask_prompt = f"""
Here is a piece of Python code:
{code}
Please analyze whether there are placeholder paths in this code that contain "path/to" or similar placeholder patterns.
Focus only on variables or dictionary values that contain placeholder paths like:
- "path/to/image.jpg"
- "path/to/audio.mp3" 
- "path/to/video.mp4"
- "path/to/some_file"
- similar placeholder patterns

For each placeholder path found, determine:
1. Whether it should correspond to a single file or a folder
2. Whether it's an image, audio, or video file based on the context or file extension
3. The corresponding variable names or python dictionary keys in the code ()
4. The placeholder value (the right side of the assignment statement)

Only analyze paths that are clearly placeholders, ignore real file paths or paths that don't contain placeholder patterns.
Classify the placeholder resources into three categories: images, audios, and videos.(pdf file will be treated as images)
"""
        self.analysis_data += "\n\n\n$$$$$External Resource Path Analysis$$$$$\n"
        self.analysis_data += ai_instance.generate_text(ask_prompt, temperature=self.analysis_temprature)
        self._write_analysis_data()

        format_prompt = r"""
For the placeholder paths identified above (only those containing "path/to" or similar placeholder patterns), please return in the following JSON format:
```json
{
    "images": [
        {
            "name": "some_img",
            "is_folder": false,
            "value": "path/to/image.jpg",
            "suffix": "jpg"
        },
        {
            "name": "some_pdf_path",
            "is_folder": false,
            "value": "path/to/some_file.pdf",
            "suffix": "pdf"
        }
    ],
    "audios": [
        {
            "name": "some_audio", 
            "is_folder": true,
            "value": "path/to/audios/",
            "suffix": ""
        }
    ],
    "videos": [
        {
            "name": "some_video_path",
            "is_folder": false,
            "value": "path/to/video.mp4", 
            "suffix": "mp4"
        }
    ]
}
```
Please note:
1. ONLY include variables/paths that contain placeholder patterns like "path/to"
2. Do not include real file paths or existing file references
3. The returned JSON format must strictly follow the above format
4. Variable names and paths must be string types
5. If there are no placeholder resources of a certain type, return an empty list for that field

Where `name` is the corresponding variable name or dictionary key in the code, `value` is the placeholder path (the right side of the assignment statement as a string).
`is_folder` indicates whether the path should be a file or a folder.
`suffix` indicates the file extension (empty string if it's a folder).
Please wrap the returned JSON content with ```json ...```
"""
        self.analysis_data += "\n\n\n$$$$$External Resource Path Format Analysis$$$$$\n"
        analysis_result = ai_instance.generate_text(format_prompt, temperature=0)
        self.analysis_data += analysis_result
        self._write_analysis_data()
        try:
            resource_info = extract_code(analysis_result, 'json')
            resource_info = json.loads(resource_info)
        except json.JSONDecodeError:
            print(f"[INS_ERR] External resource path analysis result parsing failed, please check if the returned json format is correct")
            return code
        # 接下来使用正则表达式将代码中的路径替换为默认静态资源路径
        from Inspection.core.code_processor import replace_assignment
        images_list = resource_info.get('images', [])
        audios_list = resource_info.get('audios', [])
        videos_list = resource_info.get('videos', [])
        def need_replace(value: str):
            if value == None:
                return False
            try:
                if value.find('output') != -1 or value.find('result') != -1 or value.find('save') != -1:
                    # 如果路径中包含output或result，表示是输出结果路径
                    return False
                if value.find('path') != -1 and value.find('to') != -1:
                    # 这种情况往往表示这是一个占位符路径
                    return True
                if value.find('FILE_RECORD_PATH') != -1 or value.find('ENV_DIR') != -1:
                    # 如果路径中包含FILE_RECORD_PATH，表示已经出现幻觉了，需要替换
                    return True
                return False
            except Exception as e:
                print(f"[INS_WARN] Error occurred while checking if path needs replacement: {e}")
                return False

        for img_data in images_list:
            variable_name = img_data.get('name', '')
            is_folder = img_data.get('is_folder', False)
            if not need_replace(img_data.get('value', '')):
                # 如果不需要替换，则跳过
                continue
            if is_folder:
                # 如果是文件夹，则替换为默认静态资源路径下的images文件夹
                new_value = f"RESOURCES_PATH + 'images/test_images_floder'"
            else:
                suffix = img_data.get('suffix', '')
                if suffix == '' or suffix == 'mp4':
                    suffix = 'jpg'  # 默认图片格式为jpg
                new_value = f"RESOURCES_PATH + 'images/test_image.{suffix}'"
            code = replace_assignment(code, variable_name, new_value)
            code = replace_dict_value(code, variable_name, new_value)
        for audio_data in audios_list:
            variable_name = audio_data.get('name', '')
            is_folder = audio_data.get('is_folder', False)
            if not need_replace(audio_data.get('value', '')):
                continue
            if is_folder:
                # 如果是文件夹，则替换为默认静态资源路径下的audios文件夹
                new_value = f"RESOURCES_PATH + 'audios/test_audios_floder/'"
            else:
                suffix = audio_data.get('suffix', '')
                if suffix == '':
                    suffix = 'mp3'  # 默认音频格式为mp3
                new_value = f"RESOURCES_PATH + 'audios/test_audio.{suffix}'"
            code = replace_assignment(code, variable_name, new_value)
            code = replace_dict_value(code, variable_name, new_value)
        for video_data in videos_list:
            variable_name = video_data.get('name', '')
            is_folder = video_data.get('is_folder', False)
            if not need_replace(video_data.get('value', '')):
                continue
            if is_folder:
                # 如果是文件夹，则替换为默认静态资源路径下的videos文件夹
                new_value = f"RESOURCES_PATH + 'videos/test_videos_floder/'"
            else:
                suffix = video_data.get('suffix', '')
                if suffix == '':
                    suffix = 'mp4'
                new_value = f"RESOURCES_PATH + 'videos/test_video.{suffix}'"
            code = replace_assignment(code, variable_name, new_value)
            code = replace_dict_value(code, variable_name, new_value)
        code = clean_path_to_in_code(code)  # 清理代码中的path/to占位符
        return code
    

    def optimize_code(self , source_code: str):
        ai_instance = self.base_ai_instance.copy()
        print('[INS_INFO] Optimizing ...')
        optimization_int_prompt = f"""
Here is a piece of Python code:
{source_code}
Q1: Please find the places in this code where files are **Final** output, please tell me the variable names of the output files 
please answer the variable names in a list format wrapped in ```list ... ```
e.g. 
```list
['output_path_1', 'output_path_2']
```
if there are no output files, please return an empty list.
Q2: Please find potential syntax errors in this code. Does it use `if __name__ == '__main__'` or unitest to run the main logic?
Please answer each question one by one
"""
        analysis_data = ai_instance.generate_text(optimization_int_prompt, temperature=self.analysis_temprature)
        output_files_list = extract_code(analysis_data, 'list',first_only=True)
        if has_syntax_error(output_files_list):
            output_files_list = []
        try:
            output_files_list = eval(output_files_list) if output_files_list else []
        except:
            output_files_list = [str(output_files_list).strip('[]')]
        output_files_list_str = ', '.join(output_files_list)
        self.analysis_data += "\n\n\n$$$$$Code Optimization Analysis$$$$$\n"
        self.analysis_data += analysis_data
        self._write_analysis_data()
        optimization_prompt = "Please optimize the source code:\n"
        q1 = f"""
Corresponding to Q1, please replace the **final output** file `{output_files_list_str}` root paths with an existing global variable FILE_RECORD_PATH 
(already exists, no need to define, type is string)
"""
        q2 = f"""
Corresponding to Q2, please check for simple syntax errors in this code and fix them. If it uses `if __name__ == '__main__'` to run the main logic, please remove the `if __name__ == '__main__'` and change it to run the main logic directly.
If it uses unittest to run the main logic, please remove the unittest code and change it to run the main logic directly.
Please preserve the original code structure and logic as much as possible, and you can add necessary comments. Only return the modified code.
Wrap the generated code with ```python ... ```
"""
        if len(output_files_list) > 0:
            optimization_prompt += q1 + '\n'
        optimization_prompt += q2 + '\n'
        result = ai_instance.generate_text(optimization_prompt, temperature=self.generate_code_temprature)
        self.analysis_data += "\n\n\n$$$$$Code Optimization Result$$$$$\n"
        self.analysis_data += result
        self._write_analysis_data()
        optimized_code = extract_python_code(result)
        return optimized_code
    

    def genrate_all_simulation(self):
        # Generate all simulation execution code
        try:
            info_reader = InterfaceInfoReader(self.name)
        except FileNotFoundError:
            print(f"[INS_ERR] Simulation execution code generation failed, please check if {INTERFACE_INFO_PATH} contains interface information and if json format is correct")
            return
        for i in range(len(info_reader.info_json['API_Calls'])):
            self.generate_simulation(i)
            self.base_ai_instance.clear_history()

        

if __name__ == "__main__":
    simulation=SimulationGenerator('bark')
    simulation.set_base_ai(BaseAI())
    simulation.generate_simulation(0)
    # print(get_class_code(BASE_DIR+'/Inspection/adapters/custom_adapters/clip.py', 'CustomAdapter'))
