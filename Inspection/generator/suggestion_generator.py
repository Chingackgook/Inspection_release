from Inspection.utils.interface_reader import InterfaceInfoReader , InterfaceDocReader
from Inspection import SUGGESTION_PATH  , RECORD_PATH
from Inspection.utils.result_reader import RecordJsonReader
import os , time
from Inspection.ai.base_ai import BaseAI

def stringify_data(data, max_list_length=100):
    """
    递归地将数据中的所有值转换为字符串（长度最多300字符），
    支持字典和列表结构嵌套处理，限制列表最大长度。
    """
    if isinstance(data, dict):
        return {k: stringify_data(v, max_list_length) for k, v in data.items()}
    
    elif isinstance(data, list):
        # 限制列表最大长度，并在末尾添加提示
        result = [stringify_data(item, max_list_length) for item in data[:max_list_length]]
        if len(data) > max_list_length:
            result.append(f"... remain {len(data) - max_list_length} items")
        return result
    
    else:
        the_str = str(data)
        if len(the_str) > 300:
            return the_str[:300] + ' ... ' + f"remain {len(the_str) - 300} ch"
        return the_str


class SuggestionGenerator:
    """
    A class to generate suggestions for code inspection.
    """

    def __init__(self, name):
        self.name = name
        self.doc = InterfaceDocReader(self.name).get_doc()

        
    def set_base_ai(self, base_ai):
        self.BaseAI:BaseAI = base_ai

    def get_simulate_result(self , simulate_type :str , final_max_length=5000):
        dumb_path = RECORD_PATH + f"/{simulate_type}/"
        # 找到所有以self.name开头的文件
        files = os.listdir(dumb_path)
        files = [f for f in files if f.startswith(self.name)]
        if len(files) == 0:
            print(f"[INS_WARN] No simulation execution results exist for {self.name}")
            return None
        files.sort()
        pjresult_dir = os.path.join(dumb_path, files[-1]) #最新的文件
        pjresults = os.listdir(pjresult_dir)
        pjresults = [f for f in pjresults if f.endswith(self.api_name)]
        if len(pjresults) == 0:
            print(f"[INS_WARN] No simulation execution results exist for {self.api_name}")
            return None
        final_result = ""
        for i in range(len(pjresults)):
            pjresult_path = os.path.join(pjresult_dir, pjresults[i], 'result_data.json')
            simulate_result_reader = RecordJsonReader(pjresult_path)
            args = simulate_result_reader.get_args()
            args_str = stringify_data(args)
            result = simulate_result_reader.get_interface_return()
            return_str = stringify_data(result)
            issuccess = simulate_result_reader.get_is_success()
            fail_reason = simulate_result_reader.get_fail_reason()
            final_result += f"""
Execution {i+1}:
Parameters:
{args_str}
Return Value:
{return_str}
Execution Success:
{issuccess}
            """
            if not issuccess:
                final_result += f"Failure Reason:\n{fail_reason}"
            if i>= 5:
                break
        return final_result[:final_max_length] # 限制长度，避免过长的文本影响生成建议的质量
    

    def generate_suggestions(self, api_name: str, idx: int = 0, simulate_type : str =""):
        """
        idx为接口调用的索引
        """
        api_call_data = InterfaceInfoReader(self.name).get_call_str_by_idx(idx)
        self.api_name = api_name
        simulate_result = self.get_simulate_result(simulate_type=simulate_type)
        if simulate_result is None:
            print(f"[INS_WARN] Simulation execution results do not exist, unable to generate suggestions")
            return
        
        print(f"[INS_INFO] Generating inspection suggestions for interface {api_name}")
        promote = f"""
{self.doc}
The above is the documentation information for the API
{api_call_data}
The above is the code for the API {api_name} call, please understand this code first
Here I need to inspect this API, I will send you the simulation results of this API
"""
        _ = self.BaseAI.generate_text(promote, max_tokens=4096)
        promote = f"""
{simulate_result}
The above is the simulation result for API {api_name}, please provide modifications or inspection suggestions for my source code based on these results
"""
        suggestion = self.BaseAI.generate_text(promote, max_tokens=12000)
        now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        file_pre_info =f"""
# Generation Time: {now}
# Project Name: {self.name}
# API Name: {api_name}
# API Call Index: {idx}
# Simulation Type: {simulate_type}
# Simulation Results:
{simulate_result}\n\n\n
# Review Suggestions:
"""
        suggestion_gen_path = SUGGESTION_PATH + f"/{self.name}_{api_name}_call{idx}_{now}.md"
        with open(suggestion_gen_path, "a") as f:
            f.write(file_pre_info)
            f.write(suggestion)
        print(f"[INS_INFO] Review suggestions saved to {suggestion_gen_path}")

        

if __name__ == "__main__":
    # 测试代码
    su = SuggestionGenerator("fish_speech_fixed")
    su.set_base_ai(BaseAI())
    su.generate_suggestions("encode", 0, "simulation")