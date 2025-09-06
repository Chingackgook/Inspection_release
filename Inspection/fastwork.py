from Inspection.core.workflow import Workflow
from Inspection.utils.file_lister import FileLister

from Inspection.utils.path_manager import INTERFACE_INFO_PATH
from Inspection.utils.config import CONFIG

from Inspection.ai.base_ai import BaseAI
import os



def fastwork():
    CONFIG['evaluate_mode'] = False
    CONFIG['ask'] = False
    print(os.environ.get('INSPECTION_CONFIG'))
    file_lister = FileLister(INTERFACE_INFO_PATH, 'json')
    file_lister.print_file_list("Available interface information files:")
    selected_file_name = file_lister.choose_file("Please enter the project for quick execution: ")
    workflow = Workflow(selected_file_name, [] , BaseAI())
    workflow.add_step("generate_doc")
    workflow.add_step("generate_adapter")
    workflow.add_step("generate_simulation" , simulate_idx= 0)
    workflow.add_step("simulation", simulate_idx= 0)
    workflow.run()

if __name__ == "__main__":
    fastwork()