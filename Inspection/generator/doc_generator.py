# 根据接口的调用信息，生成一个接口文档
from Inspection.utils.interface_reader import InterfaceInfoReader
from Inspection.ai.base_ai import BaseAI
from Inspection.core.code_processor import get_class_or_function_def_name,get_inherit_info
from Inspection import INTERFACE_INFO_PATH, INTERFACE_DOC_PATH
from Inspection.utils.config import CONFIG
import os

class DocGenerator:
    def __init__(self, name : str):
        self.name = name
        self.BaseAI :BaseAI = None
        self.inpath = INTERFACE_INFO_PATH
        self.outpath = INTERFACE_DOC_PATH
        self.ask = CONFIG.get('ask', True)
        self.temprature = CONFIG.get('doc_generate_temprature', 0.3)
        self.check_dir()

    def set_base_ai(self, base_ai):
        self.BaseAI = base_ai

    def check_dir(self):
        if not os.path.exists(self.inpath):
            os.makedirs(self.inpath)
            print('[INS_WARN] Interface information folder not found, automatically created')
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
            print('[INS_WARN] Interface documentation folder not found, automatically created')

    def generate_doc(self):
        name = self.name
        try:
            interface_info_reader = InterfaceInfoReader(name)
            ipl_datas = interface_info_reader.get_implementations()
        except Exception as e:
            print(f"[INS_ERR] Failed to read interface information: {e}")
            return
        
        if os.path.exists(self.outpath + name + '.md') and self.ask:
            print(f"[INS_WARN] Document {name} already exists")
            ch = input("Overwrite? (y/n)")
            if ch != 'y':
                return
        final_doc = ""
        for ipl_data in ipl_datas:
            namelist = get_class_or_function_def_name(ipl_data)
            inherit_info = get_inherit_info(ipl_data)
            inherit_info = [info for info in inherit_info if info[1] in namelist]
            namestr = ''
            for n in namelist:
                namestr += f' - {n}\n'
            promote = f"""
    {ipl_data}
    Based on the above interface implementation information,
    please generate API documentation for the following functions and classes:
    {namestr}
    Requirements:
    1. If it is a class, you need to generate API documentation for the class’s initializer, its attributes, and each public method in the class.
    2. The API documentation should include: function/method name, parameter description, return value description, parameter value range, and a brief explanation of its purpose.
    """
            result_doc = self.BaseAI.generate_text(promote)
            if len(inherit_info) > 0:
                inherit_info_prompt = ""
                for info in inherit_info:
                    if info[1] not in namelist and namestr.find(info[1]) == -1:
                        print(f"[INS_WARN] {info[1]} is not in the implementation list, skipping inheritance info.")
                        continue
                    inherit_info_prompt += f"""
    {info[0]} inherits from {info[1]}. Please merge the API documentation of {info[1]} into {info[0]} by listing all public methods and attributes from {info[1]} directly under {info[0]}’s documentation, as if they were defined in {info[0]}. Do not keep a separate documentation section for {info[1]}; treat all inherited content as part of {info[0]}.
    """
                if inherit_info_prompt != "":
                    print(f"[INS_INFO] Found inheritance relationship, processing inheritance information...")
                    self.BaseAI.clear_history()
                    result_doc = self.BaseAI.generate_text(result_doc + inherit_info_prompt)

            final_doc += result_doc + "\n\n"

        with open(self.outpath + name + '.md', 'w') as f:
            f.write(final_doc)
        print('[INS_INFO] Documentation generated successfully, saved to: ' + self.outpath + name + '.md')

if __name__ == "__main__":
    doc = DocGenerator('bark')
    doc.generate_doc()