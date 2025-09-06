from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.memvid import *
exe = Executor('memvid', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/memvid/examples/file_chat.py'
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json
from memvid import MemvidEncoder
from memvid import MemvidChat
from memvid.config import get_default_config
from memvid.config import get_codec_parameters
from tqdm import tqdm
from bs4 import BeautifulSoup
'\nfile_chat.py - Enhanced script for testing MemvidChat with external files\n\nThis script allows you to:\n1. Create a memory video from your own files with configurable parameters\n2. Chat with the created memory using different LLM providers\n3. Store results in output/ directory to avoid contaminating the main repo\n4. Handle FAISS training issues gracefully\n5. Configure chunking and compression parameters\n'
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json
sys.path.insert(0, str(Path('/mnt/autor_name/haoTingDeWenJianJia/memvid/examples/file_chat.py').parent.parent))
from memvid import MemvidEncoder, MemvidChat
from memvid.config import get_default_config, get_codec_parameters

def setup_output_dir():
    """Create output directory if it doesn't exist"""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_memory_name(input_source):
    """Generate a meaningful name for the memory files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if isinstance(input_source, list):
        base_name = f'files_{len(input_source)}items'
    else:
        dir_name = Path(input_source).name
        base_name = f'dir_{dir_name}'
    return f'{base_name}_{timestamp}'

def collect_files_from_directory(directory_path, extensions=None):
    """Collect supported files from a directory"""
    if extensions is None:
        extensions = {'.txt', '.md', '.pdf', '.doc', '.docx', '.rtf', '.epub', '.html', '.htm'}
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f'Directory does not exist: {directory_path}')
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f'*{ext}'))
    return [str(f) for f in files if f.is_file()]

def create_memory_with_fallback(encoder, video_path, index_path):
    """Create memory with graceful FAISS fallback for training issues"""
    try:
        build_stats = encoder.build_video(str(video_path), str(index_path))
        return build_stats
    except Exception as e:
        error_str = str(e)
        if 'is_trained' in error_str or 'IndexIVFFlat' in error_str or 'training' in error_str.lower():
            print(f'⚠️  FAISS IVF training failed: {e}')
            print(f'🔄 Auto-switching to Flat index for compatibility...')
            original_index_type = encoder.config['index']['type']
            encoder.config['index']['type'] = 'Flat'
            try:
                encoder._setup_index()
                build_stats = encoder.build_video(str(video_path), str(index_path))
                print(f'✅ Successfully created memory using Flat index')
                return build_stats
            except Exception as fallback_error:
                print(f'❌ Fallback also failed: {fallback_error}')
                raise
        else:
            raise

def create_memory_from_files(files, output_dir, memory_name, **config_overrides):
    """Create a memory video from a list of files with configurable parameters"""
    print(f'Creating memory from {len(files)} files...')
    start_time = time.time()
    config = get_default_config()
    for key, value in config_overrides.items():
        if key in ['chunk_size', 'overlap']:
            config['chunking'][key] = value
        elif key == 'index_type':
            config['index']['type'] = value
        elif key == 'codec':
            config[key] = value
    encoder = MemvidEncoder(config)
    actual_codec = encoder.config.get('codec')
    video_ext = get_codec_parameters(actual_codec).get('video_file_type', 'mp4')
    processed_count = 0
    skipped_count = 0
    for file_path in files:
        file_path = Path(file_path)
        print(f'Processing: {file_path.name}')
        try:
            chunk_size = config['chunking']['chunk_size']
            overlap = config['chunking']['overlap']
            if file_path.suffix.lower() == '.pdf':
                encoder.add_pdf(str(file_path), chunk_size, overlap)
            elif file_path.suffix.lower() == '.epub':
                encoder.add_epub(str(file_path), chunk_size, overlap)
            elif file_path.suffix.lower() in ['.html', '.htm']:
                try:
                    from bs4 import BeautifulSoup
                except ImportError:
                    print(f'Warning: BeautifulSoup not available for HTML processing. Skipping {file_path.name}')
                    skipped_count += 1
                    continue
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    for script in soup(['script', 'style']):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
                    clean_text = ' '.join((chunk for chunk in chunks if chunk))
                    if clean_text.strip():
                        encoder.add_text(clean_text, chunk_size, overlap)
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        encoder.add_text(content, chunk_size, overlap)
            processed_count += 1
        except Exception as e:
            print(f'Warning: Could not process {file_path.name}: {e}')
            skipped_count += 1
            continue
    if processed_count == 0:
        raise ValueError('No files were successfully processed')
    video_path = Path(FILE_RECORD_PATH) / f'{memory_name}.{video_ext}'
    index_path = Path(FILE_RECORD_PATH) / f'{memory_name}_index.json'
    print(f'\n🎬 Building memory video: {video_path}')
    print(f'📊 Total chunks to encode: {len(encoder.chunks)}')
    build_stats = create_memory_with_fallback(encoder, video_path, index_path)
    print(f'\n🎉 Memory created successfully!')
    print(f'  📁 Video: {video_path}')
    print(f'  📋 Index: {index_path}')
    print(f'  📊 Chunks: {build_stats.get('total_chunks', 'unknown')}')
    print(f'  🎞️  Frames: {build_stats.get('total_frames', 'unknown')}')
    print(f'  📏 Video size: {video_path.stat().st_size / (1024 * 1024):.1f} MB')
    metadata = {'created': datetime.now().isoformat(), 'source_files': files, 'video_path': str(video_path), 'index_path': str(index_path), 'config_used': config, 'processing_stats': {'files_processed': processed_count, 'files_skipped': skipped_count}, 'build_stats': build_stats}
    metadata_path = Path(FILE_RECORD_PATH) / f'{memory_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  📄 Metadata: {metadata_path}')
    return (str(video_path), str(index_path))

def load_existing_memory(memory_path):
    """Load and validate existing memory from the output directory"""
    memory_path = Path(memory_path)
    if memory_path.is_dir():
        video_files = []
        for ext in ['mp4', 'avi', 'mkv']:
            video_files.extend(memory_path.glob(f'*.{ext}'))
        if not video_files:
            raise ValueError(f'No video files found in {memory_path}')
        video_path = video_files[0]
        possible_index_paths = [video_path.with_name(video_path.stem + '_index.json'), video_path.with_suffix('.json'), video_path.with_suffix('_index.json')]
        index_path = None
        for possible_path in possible_index_paths:
            if possible_path.exists():
                index_path = possible_path
                break
        if not index_path:
            raise ValueError(f'No index file found for {video_path}')
    elif memory_path.suffix in ['.mp4', '.avi', '.mkv']:
        video_path = memory_path
        index_path = memory_path.with_name(memory_path.stem + '_index.json')
    else:
        base_path = memory_path
        video_path = None
        for ext in ['mp4', 'avi', 'mkv']:
            candidate = base_path.with_suffix(f'.{ext}')
            if candidate.exists():
                video_path = candidate
                break
        if not video_path:
            raise ValueError(f'No video file found with base name: {memory_path}')
        index_path = base_path.with_suffix('_index.json')
    if not video_path.exists():
        raise ValueError(f'Video file not found: {video_path}')
    if not index_path.exists():
        raise ValueError(f'Index file not found: {index_path}')
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        chunk_count = len(index_data.get('metadata', []))
        print(f'✅ Index contains {chunk_count} chunks')
    except Exception as e:
        raise ValueError(f'Index file corrupted: {e}')
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f'✅ Video file: {video_size_mb:.1f} MB')
    print(f'Loading existing memory:')
    print(f'  📁 Video: {video_path}')
    print(f'  📋 Index: {index_path}')
    return (str(video_path), str(index_path))

def start_chat_session(video_path, index_path, provider='openai', model=None):
    """Start an interactive chat session"""
    print(f'\nInitializing chat with {provider}...')
    try:
        chat = exe.create_interface_objects(interface_class_name='MemvidChat', video_file=video_path, index_file=index_path, llm_provider=provider, llm_model=model)
        print('✓ Chat initialized successfully!')
        print('\nStarting interactive session...')
        print('Commands:')
        print('  - Type your questions normally')
        print("  - Type 'quit' or 'exit' to end")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'stats' to see session statistics")
        print('=' * 50)
        user_input = 'What is the memory about?'
        response = exe.run('chat', message=user_input, stream=True)
    except Exception as e:
        print(f'Error initializing chat: {e}')
        return False
    return True
output_dir = setup_output_dir()
input_dir = RESOURCES_PATH + 'images/test_images_floder'
provider = 'openai'
model = None
memory_name = None
try:
    files = collect_files_from_directory(input_dir)
    if not files:
        print(f'No supported files found in {input_dir}')
    else:
        print(f'Found {len(files)} files to process')
        memory_name = generate_memory_name(input_dir)
        video_path, index_path = create_memory_from_files(files, output_dir, memory_name)
        success = start_chat_session(video_path, index_path, provider, model)
except Exception as e:
    print(f'Error: {e}')