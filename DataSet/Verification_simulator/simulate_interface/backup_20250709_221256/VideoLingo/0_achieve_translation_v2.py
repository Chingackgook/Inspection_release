from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.VideoLingo import ENV_DIR
from Inspection.adapters.custom_adapters.VideoLingo import *
exe = Executor('VideoLingo','simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
chunk_size = 600  # 每个块的字符数
max_i = 10        # 最大句子数
# end

# 导入原有的包
import pandas as pd
import json
import concurrent.futures
from core.translate_lines import translate_lines
from core._4_1_summarize import search_things_to_note_in_prompt
from core._8_1_audio_task import check_len_then_trim
from core._6_gen_sub import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.models import *

console = Console()



with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
    f.write('This is the first sentence.\nThis is the second sentence.\nThis is the third sentence.')
with open(_4_1_TERMINOLOGY, 'w', encoding='utf-8') as f:
    json.dump({'theme': 'Test Theme'}, f)

def split_chunks_by_chars(chunk_size, max_i):
    """Split text into chunks based on character count, return a list of multi-line text chunks"""
    with open(_3_2_SPLIT_BY_MEANING, 'r', encoding='utf-8') as file:
        sentences = file.read().strip().split('\n')
    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-3:]

def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:2]

def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i) 
    (translation, english_result) = exe.run('translate_lines', lines=chunk, previous_content_prompt=previous_content_prompt, after_content_prompt=after_content_prompt, things_to_note_prompt=things_to_note_prompt, summary_prompt=theme_prompt, index=i)
    return (i, english_result, translation)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 直接运行主逻辑
console.print('[bold green]Start Translating All...[/bold green]')
chunks = split_chunks_by_chars(chunk_size=chunk_size, max_i=max_i)
with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
    theme_prompt = json.load(file).get('theme')
with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'), transient=True) as progress:
    task = progress.add_task('[cyan]Translating chunks...', total=len(chunks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for (i, chunk) in enumerate(chunks):
            future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
            futures.append(future)
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            progress.update(task, advance=1)
results.sort(key=lambda x: x[0])
(src_text, trans_text) = ([], [])
for (i, chunk) in enumerate(chunks):
    chunk_lines = chunk.split('\n')
    src_text.extend(chunk_lines)
    chunk_text = ''.join(chunk_lines).lower()
    matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text)) for r in results]
    best_match = max(matching_results, key=lambda x: x[1])
    if best_match[1] < 0.9:
        console.print(f'[yellow]Warning: No matching translation found for chunk {i}[/yellow]')
        raise ValueError(f'Translation matching failed (chunk {i})')
    elif best_match[1] < 1.0:
        console.print(f'[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]')
    trans_text.extend(best_match[0][2].split('\n'))
df_text = pd.DataFrame({'text': ['This is a cleaned chunk.']})
df_text['text'] = df_text['text'].str.strip('"').str.strip()
df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
console.print(df_time)
df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], 0) if 0 > load_key('min_trim_duration') else x['Translation'], axis=1)
console.print(df_time)
df_time.to_excel(_4_2_TRANSLATION, index=False)
console.print('[bold green]✅ Translation completed and results saved.[/bold green]')
