from Inspection import INTERFACE_TXT_PATH
from Inspection import INTERFACE_INFO_PATH

import os
import json


def trim_txt(str:str):
    # 去除以 ### 开头的行
    lines = str.split('\n')
    lines = [line for line in lines if not line.startswith('###')]
    return '\n'.join(lines)

def spit_txt(str:str):
    # 将txt文件中的内容按$$$分割
    lines = str.split('\n')
    result = []
    tempresult = ''
    for line in lines:
        if not line.startswith('$$$'):
            tempresult += line + '\n'
        else:
            result.append(tempresult)
            tempresult = ''
    result.append(tempresult)
    return result
    

class InterfaceWriter:
    def __init__(self,project_name,cover=False):
        self.project_name = project_name
        self.cover = cover
        self.pj_root_data = ""
        self.api_call_data = {}
        self.api_ipl_data = {}
        
    def check_info_exist(self):
        #检查命名为project_name的json接口信息文件是否存在
        #如果存在则返回True，否则返回False
        info_path = os.path.join(INTERFACE_INFO_PATH, self.project_name + '.json')
        if os.path.exists(info_path):
            return True
        else:
            return False
        
    def read_txt(self):
        project_root = ""
        api_call_data = {}
        api_ipl_data = {}
        pj_root_dir = os.path.join(INTERFACE_TXT_PATH, 'ProjectRoot.txt')
        with open(pj_root_dir, 'r') as f:
            project_root = f.read()
            project_root = trim_txt(project_root)
            # 去除所有空格和换行符
            project_root = project_root.replace(' ', '').replace('\n', '')
            self.pj_root_data = project_root
        callsdir = os.path.join(INTERFACE_TXT_PATH, 'APICalls')
        with open(os.path.join(callsdir, 'code.txt'), 'r') as f:
            c_code = f.read()
            c_code = trim_txt(c_code)
        with open(os.path.join(callsdir, 'name.txt'), 'r') as f:
            c_name = f.read()
            c_name = trim_txt(c_name)
            c_name = c_name.replace(' ', '_')
            c_name = c_name.replace('-', '_')
            c_name = c_name.replace('.', '_')
        with open(os.path.join(callsdir, 'description.txt'), 'r') as f:
            c_description = f.read()
            c_description = trim_txt(c_description)
        with open(os.path.join(callsdir, 'path.txt'), 'r') as f:
            c_path = f.read()
            c_path = trim_txt(c_path)
        api_call_data['Name'] = c_name.strip()
        api_call_data['Description'] = c_description
        api_call_data['Code'] = c_code  
        api_call_data['Path'] = c_path
        self.api_call_data = api_call_data
        ipl_dir = os.path.join(INTERFACE_TXT_PATH, 'APIImplementations')
        with open(os.path.join(ipl_dir, 'implementation.txt'), 'r') as f:
            i_implementation = f.read()
            i_implementation = trim_txt(i_implementation)
        with open(os.path.join(ipl_dir, 'path.txt'), 'r') as f:
            i_path = f.read()
            i_path = trim_txt(i_path)
        with open(os.path.join(ipl_dir, 'name.txt'), 'r') as f:
            i_name = f.read()
            i_name = trim_txt(i_name)
            i_name = i_name.replace(' ', '_')
            i_name = i_name.replace('-', '_')
            i_name = i_name.replace('.', '_')
        with open(os.path.join(ipl_dir, 'description.txt'), 'r') as f:
            i_description = f.read()
            i_description = trim_txt(i_description)
        with open(os.path.join(ipl_dir, 'example.txt'), 'r') as f:
            i_example = f.read()
            i_example = trim_txt(i_example)
            i_example = spit_txt(i_example)
        api_ipl_data['Name'] = i_name.strip()
        api_ipl_data['Description'] = i_description
        api_ipl_data['Path'] = i_path
        api_ipl_data['Implementation'] = i_implementation
        api_ipl_data['Examples'] = i_example
        self.api_ipl_data = api_ipl_data
        
    def write(self, calls = True ,ipls = True):
        """
        calls: bool 追加模式下是否添加API_Calls
        ipls: bool 追加模式下是否添加API_Implementations
        """
        self.read_txt()
        if(self.check_info_exist()):
            if self.cover:
                info_path = os.path.join(INTERFACE_INFO_PATH, self.project_name + '.json')
                with open(info_path, 'w') as f:
                    data = {
                        'Project_Root': f'{self.pj_root_data}',
                        'API_Calls': [self.api_call_data],
                        'API_Implementations': [self.api_ipl_data]
                    }
                    json.dump(data, f, indent=4 , ensure_ascii=False)
            else:
                from Inspection.utils.interface_reader import InterfaceInfoReader
                info_reader = InterfaceInfoReader(self.project_name)
                json_data = info_reader.info_json
                if calls:
                    json_data['API_Calls'].append(self.api_call_data)
                if ipls:
                    json_data['API_Implementations'].append(self.api_ipl_data)
                info_path = os.path.join(INTERFACE_INFO_PATH, self.project_name + '.json')
                with open(info_path, 'w') as f:
                    json.dump(json_data, f, indent=4 , ensure_ascii=False)
        else:
            data = {
                'Project_Root': f'{self.pj_root_data}',
                'API_Calls': [self.api_call_data],
                'API_Implementations': [self.api_ipl_data]
            }
            #将数据写入json文件
            info_path = os.path.join(INTERFACE_INFO_PATH, self.project_name + '.json')
            with open(info_path, 'w') as f:
                json.dump(data, f, indent=4 , ensure_ascii=False)


class InterfaceRewriter:
    def __init__(self, project_name):
        self.project_name = project_name
        self.info_path = os.path.join(INTERFACE_INFO_PATH, self.project_name + '.json')

    def rewrite(self):
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"{self.info_path} 不存在")
        with open(self.info_path, 'r') as f:
            data = json.load(f)

        # 写 ProjectRoot.txt
        pj_root_dir = os.path.join(INTERFACE_TXT_PATH, 'ProjectRoot.txt')
        with open(pj_root_dir, 'w') as f:
            f.write(data.get('Project_Root', ''))

        # 写 APICalls
        callsdir = os.path.join(INTERFACE_TXT_PATH, 'APICalls')
        os.makedirs(callsdir, exist_ok=True)
        api_calls = data.get('API_Calls', [{}])
        api_call = api_calls[0] if api_calls else {}
        with open(os.path.join(callsdir, 'code.txt'), 'w') as f:
            f.write(api_call.get('Code', ''))
        with open(os.path.join(callsdir, 'name.txt'), 'w') as f:
            f.write(api_call.get('Name', ''))
        with open(os.path.join(callsdir, 'description.txt'), 'w') as f:
            f.write(api_call.get('Description', ''))
        with open(os.path.join(callsdir, 'path.txt'), 'w') as f:
            f.write(api_call.get('Path', ''))

        # 写 APIImplementations
        ipl_dir = os.path.join(INTERFACE_TXT_PATH, 'APIImplementations')
        os.makedirs(ipl_dir, exist_ok=True)
        api_ipls = data.get('API_Implementations', [{}])
        api_ipl = api_ipls[0] if api_ipls else {}
        with open(os.path.join(ipl_dir, 'implementation.txt'), 'w') as f:
            f.write(api_ipl.get('Implementation', ''))
        with open(os.path.join(ipl_dir, 'path.txt'), 'w') as f:
            f.write(api_ipl.get('Path', ''))
        with open(os.path.join(ipl_dir, 'name.txt'), 'w') as f:
            f.write(api_ipl.get('Name', ''))
        with open(os.path.join(ipl_dir, 'description.txt'), 'w') as f:
            f.write(api_ipl.get('Description', ''))
        # Examples 需要合并为一个字符串，分隔符为 $$$
        examples = api_ipl.get('Examples', [])
        examples_txt = ""
        for i, ex in enumerate(examples):
            examples_txt += ex.strip()
            if i != len(examples) - 1:
                examples_txt += '\n$$$\n'
        with open(os.path.join(ipl_dir, 'example.txt'), 'w') as f:
            f.write(examples_txt)

        if len(api_calls)>1 or len(api_ipls)>1:
            print(f"[INS_WARN] {self.project_name} interface information contains multiple API_Calls or API_Implementations, currently only processing the first one, data may be lost when writing back")


if __name__ == "__main__":
    # 测试代码
    rewriter = InterfaceRewriter('FlexLLMGen')
    rewriter.rewrite()