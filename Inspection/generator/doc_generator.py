from Inspection.utils.readers import InterfaceInfoReader, ProjectRegistrationInfoReader
from Inspection.ai.base_ai import BaseAI
from Inspection.core.code_processor import (
    get_functions_and_class_methods,
    get_inherit_info,
)
from Inspection import INTERFACE_INFO_PATH, INTERFACE_DOC_PATH
from Inspection.utils.config import CONFIG
import os


class DocGenerator:
    def __init__(self, name: str):
        self.name = name
        self.ai: BaseAI = None
        self.inpath = INTERFACE_INFO_PATH
        self.outpath = INTERFACE_DOC_PATH
        self.ask = CONFIG.get("ask", True)
        self.temprature = CONFIG.get("doc_generate_temprature", 0.3)
        self.check_dir()

    def generate_doc(self):
        if not ProjectRegistrationInfoReader().can_generate_doc(self.name):
            print(
                f"[INS_ERROR] Project {self.name} can not generate doc, not found {self.name} in {INTERFACE_INFO_PATH}"
            )
            return

        if os.path.exists(self.outpath + self.name + ".md") and self.ask:
            print(f"[INS_WARN] Document {self.name} already exists")
            ch = input("Overwrite? (y/n)")
            if ch != "y":
                return

        try:
            interface_info_reader = InterfaceInfoReader(self.name)
            ipl_datas = interface_info_reader.get_implementations()
        except Exception as e:
            print(f"[INS_ERR] Failed to read interface information: {e}")
            return

        final_doc = ""
        for ipl_data in ipl_datas:
            functions_names, class_datas = get_functions_and_class_methods(ipl_data)
            class_names = [data["class_name"] for data in class_datas]

            inherit_info = get_inherit_info(ipl_data)

            function_name_str = ""
            for fn in functions_names:
                function_name_str += f" - {fn}\n"

            class_str = ""
            for cd in class_datas:
                methods = cd["methods"]
                method_str = ""
                for m in methods:
                    method_str += f"    - {m}\n"
                class_str += f' - {cd["class_name"]}:\n{method_str}'

            promote = f"""
{ipl_data}
Based on the above interface implementation information,
please generate API documentation for the following functions and classes:
Functions:
{function_name_str}
Classes:
{class_str}

Requirements:
1. If it is a class, you need to generate API documentation for the class’s initializer, its attributes, and each public method in the class.
2. The API documentation should include: function/method name, parameter description, return value description, parameter value range, and a brief explanation of its purpose.
"""
            result_doc = self.ai.generate_text(promote)

            inherit_info = [info for info in inherit_info if info[1] in class_names]
            if len(inherit_info) > 0:
                inherit_info_prompt = ""
                for info in inherit_info:
                    inherit_info_prompt += f"""
{info[0]} inherits from {info[1]}. Please merge the API documentation of {info[1]} into {info[0]} by listing all public methods and attributes from {info[1]} directly under {info[0]}’s documentation, as if they were defined in {info[0]}. Do not keep a separate documentation section for {info[1]}; treat all inherited content as part of {info[0]}.
"""
                if inherit_info_prompt != "":
                    print(
                        f"[INS_INFO] Found inheritance relationship, processing inheritance information..."
                    )
                    self.ai.clear_history()
                    result_doc = self.ai.generate_text(result_doc + inherit_info_prompt)

            final_doc += result_doc + "\n\n"

        with open(self.outpath + self.name + ".md", "w") as f:
            f.write(final_doc)
        print(
            "[INS_INFO] Documentation generated successfully, saved to: "
            + self.outpath
            + self.name
            + ".md"
        )

    def set_base_ai(self, base_ai):
        self.ai = base_ai

    def check_dir(self):
        if not os.path.exists(self.inpath):
            os.makedirs(self.inpath)
            print(
                "[INS_WARN] Interface information folder not found, automatically created"
            )
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
            print(
                "[INS_WARN] Interface documentation folder not found, automatically created"
            )


if __name__ == "__main__":
    doc = DocGenerator("bark")
    doc.generate_doc()
