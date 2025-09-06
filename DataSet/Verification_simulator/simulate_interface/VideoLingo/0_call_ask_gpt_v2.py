from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.VideoLingo import *
exe = Executor('VideoLingo', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/VideoLingo/core/translate_lines.py'
from core.prompts import generate_shared_prompt
from core.prompts import get_prompt_faithfulness
from core.prompts import get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
# end

from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
console = Console()

def valid_translate_result(result: dict, required_keys: list, required_sub_keys: list):
    if not all((key in result for key in required_keys)):
        return {'status': 'error', 'message': f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    for key in result:
        if not all((sub_key in result[key] for sub_key in required_sub_keys)):
            return {'status': 'error', 'message': f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}
    return {'status': 'success', 'message': 'Translation completed'}

def translate_lines(lines, previous_content_prompt, after_content_prompt, things_to_note_prompt, summary_prompt, index=0):
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt)

    def retry_translation(prompt, length, step_name):

        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length + 1)], ['direct'])

        def valid_express(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length + 1)], ['free'])
        
        for retry in range(3):
            if step_name == 'faithfulness':
                result = exe.run('ask_gpt', prompt=prompt + retry * ' ', resp_type='json', valid_def=valid_faith, log_title=f'translate_{step_name}')
            elif step_name == 'expressiveness':
                result = exe.run('ask_gpt', prompt=prompt + retry * ' ', resp_type='json', valid_def=valid_express, log_title=f'translate_{step_name}')
            if len(lines.split('\n')) == len(result):
                return result
            if retry != 2:
                console.print(f'[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed, Retry...[/yellow]')
        raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed after 3 retries. Please check {FILE_RECORD_PATH}/gpt_log/error.json for more details.[/red]')
    
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split('\n')), 'faithfulness')
    for i in faith_result:
        faith_result[i]['direct'] = faith_result[i]['direct'].replace('\n', ' ')
    
    reflect_translate = False
    if not reflect_translate:
        translate_result = '\n'.join([faith_result[i]['direct'].strip() for i in faith_result])
        table = Table(title='Translation Results', show_header=False, box=box.ROUNDED)
        table.add_column('Translations', style='bold')
        for (i, key) in enumerate(faith_result):
            table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
            if i < len(faith_result) - 1:
                table.add_row('[yellow]' + '-' * 50 + '[/yellow]')
        console.print(table)
        return (translate_result, lines)
    
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split('\n')), 'expressiveness')
    table = Table(title='Translation Results', show_header=False, box=box.ROUNDED)
    table.add_column('Translations', style='bold')
    for (i, key) in enumerate(express_result):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row('[yellow]' + '-' * 50 + '[/yellow]')
    console.print(table)
    
    translate_result = '\n'.join([express_result[i]['free'].replace('\n', ' ').strip() for i in express_result])
    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check {FILE_RECORD_PATH}/gpt_log/translate_expressiveness.json[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')
    
    return (translate_result, lines)

# Main logic execution
lines = 'All of you know Andrew Ng as a famous computer science professor at Stanford.\nHe was really early on in the development of neural networks with GPUs.\nOf course, a creator of Coursera and popular courses like deeplearning.ai.\nAlso the founder and creator and early lead of Google Brain.'
previous_content_prompt = None
after_content_prompt = None
things_to_note_prompt = None
summary_prompt = None
translate_lines(lines, previous_content_prompt, after_content_prompt, things_to_note_prompt, summary_prompt)
