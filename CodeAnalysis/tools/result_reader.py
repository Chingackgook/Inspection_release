import json
import os


class AnalysisResultReader:
    def __init__(self, result_directory):
        self.result_directory = result_directory
        self.overall_results = self._load_overall_results()
        self.possible_entries = self._load_possible_entries()


    def _load_overall_results(self):
        file_dir = f"{self.result_directory}/analysis_results.json"
        with open(file_dir, "r") as f:
            data = json.load(f)
        return data
    
    def _load_possible_entries(self):
        file_dir = f"{self.result_directory}/possible_entry_files.json"
        with open(file_dir, "r") as f:
            data = json.load(f)
        return data

    def get_and_write_inspection_format_data(
            self,
            entry_idx = None,
            call_path_index = None,
            project_root = None, #如果用户指定，则使用用户指定的路径作为根路径
            out_put_dir = None, # 输出的json文件存放路径
            json_name = None, # 如果用户指定，则使用用户指定的名称作为输出名称
            repo_name = None # 如果用户指定，则在最终的json文件中增加该信息
        ):
        if project_root is None:
            with open(f"{self.result_directory}/project_root.txt", "r") as f:
                project_root = f.read().strip()

        if out_put_dir is None:
            out_put_dir = self.result_directory
        
        if not os.path.exists(out_put_dir):
            os.makedirs(out_put_dir)
        
        if json_name is None:
            with open(f"{self.result_directory}/default_name.txt", "r") as f:
                json_name = f.read().strip()

        if entry_idx is None:
            entry_idx = self.overall_results[0]['entry_file_index']
            print(f"No entry_idx provided, using default entry_idx: {entry_idx}")
        
        if call_path_index is None:
            call_path_index = 1

        detailed_result_dir = f"{self.result_directory}/call_implementation_pairs"
        possible_files = os.listdir(detailed_result_dir)
        detailed_file_path = None
        for file in possible_files:
            if file.startswith(f"entry_idx_{entry_idx}_"):
                detailed_file_path = f"{detailed_result_dir}/{file}"
                break
        
        if detailed_file_path is None:
            raise FileNotFoundError(f"No detailed result file found for entry index {entry_idx} , {self.result_directory}")
        with open(detailed_file_path, "r") as f:
            print(f"Loading detailed results from {detailed_file_path}")
            detailed_data = json.load(f)

        selected_data = None
        for path_data in detailed_data:
            call_idx = path_data['call_path_index']
            if call_idx == call_path_index:
                selected_data = path_data
                break


        data = {}
        data['Project_Root'] = project_root
        data['API_Calls'] = []
        API_Call = {}
        API_Call['Name'] = selected_data['call_data']['name']
        API_Call['Description'] = ''
        API_Call['Path'] = selected_data['call_data']['path']
        with open(selected_data['call_data']['path'], "r") as f:
            code = f.read()
        API_Call['Code'] = code
        data['API_Calls'].append(API_Call)
        data['API_Implementations'] = []
        API_Implementation = {}
        API_Implementation['Name'] = selected_data['implementation_data']['name']
        API_Implementation['Description'] = ''
        API_Implementation['Path'] = selected_data['implementation_data']['path']
        with open(selected_data['implementation_data']['path'], "r") as f:
            code = f.read()
        API_Implementation['Implementation'] = code
        API_Implementation['Examples'] = []
        data['API_Implementations'].append(API_Implementation)
        data['API_name'] = selected_data['call_data']['name'][5:]
        data['PJ_name'] = json_name
        if repo_name is not None:
            data['Repo_Name'] = repo_name
        data['Call_Final_Intelligent_Interface_Path'] = selected_data['call_intelligent_path']

        output_file_path = f"{out_put_dir}/{json_name}.json"
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        return data , output_file_path
    