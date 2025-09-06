# quivr 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'quivr/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/quivr/core')
# 以上是自动生成的代码，请勿修改



import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

# 假设以下类已经定义
# LLMEndpoint, VectorStore, Embeddings, StorageBase, ChatHistory, BrainInfo, SearchResult, RetrievalConfig, QuivrQARAG, QuivrQARAGLangGraph, QuivrKnowledge, ParsedRAGResponse, ParsedRAGChunkResponse

from quivr_core import Brain

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.brain = None  # 用于存储Brain对象

    def create_interface_objects(self, name_quivr: str, **kwargs):
        """
        加载模型，创建Brain对象
        """
        # 从kwargs中获取参数
        llm = kwargs.get('llm', None)
        vector_db = kwargs.get('vector_db', None)
        embedder = kwargs.get('embedder', None)
        storage = kwargs.get('storage', None)
        user_id = kwargs.get('user_id', None)
        chat_id = kwargs.get('chat_id', None)

        # 创建Brain对象
        name = name_quivr
        self.brain = Brain(name=name, llm=llm, vector_db=vector_db, embedder=embedder, storage=storage, user_id=user_id, chat_id=chat_id)
        self.result.set_result(fuc_name='create_interface_objects', is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=self.brain)

    def run(self, name: str, **kwargs):
        """
        执行具体智能模块的入口方法
        """
        if name == 'print_info':
            self.brain.print_info()
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=None)
        
        elif name == 'save':
            folder_path = kwargs.get('folder_path', '')
            result = self.brain.save(folder_path)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path=result, except_data=None, interface_return=result)

        elif name == 'load':
            folder_path = kwargs.get('folder_path', '')
            self.brain = Brain.load(folder_path)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=self.brain)

        elif name == 'info':
            info = self.brain.info()
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=info)

        elif name == 'asearch':
            query = kwargs.get('query', '')
            n_results = kwargs.get('n_results', 5)
            filter = kwargs.get('filter', None)
            fetch_n_neighbors = kwargs.get('fetch_n_neighbors', 20)
            results = self.brain.asearch(query, n_results=n_results, filter=filter, fetch_n_neighbors=fetch_n_neighbors)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=results, interface_return=results)

        elif name == 'ask_streaming':
            question = kwargs.get('question', '')
            system_prompt = kwargs.get('system_prompt', None)
            retrieval_config = kwargs.get('retrieval_config', None)
            rag_pipeline = kwargs.get('rag_pipeline', None)
            list_files = kwargs.get('list_files', None)
            chat_history = kwargs.get('chat_history', None)
            for chunk in self.brain.ask_streaming(question, system_prompt=system_prompt, retrieval_config=retrieval_config, rag_pipeline=rag_pipeline, list_files=list_files, chat_history=chat_history):
                self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=chunk, interface_return=chunk)

        elif name == 'aask':
            run_id = kwargs.get('run_id', uuid.uuid4())
            question = kwargs.get('question', '')
            system_prompt = kwargs.get('system_prompt', None)
            retrieval_config = kwargs.get('retrieval_config', None)
            rag_pipeline = kwargs.get('rag_pipeline', None)
            list_files = kwargs.get('list_files', None)
            chat_history = kwargs.get('chat_history', None)
            response = self.brain.aask(run_id=run_id, question=question, system_prompt=system_prompt, retrieval_config=retrieval_config, rag_pipeline=rag_pipeline, list_files=list_files, chat_history=chat_history)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=response, interface_return=response)

        elif name == 'ask':
            run_id = kwargs.get('run_id', uuid.uuid4())
            question = kwargs.get('question', '')
            system_prompt = kwargs.get('system_prompt', None)
            retrieval_config = kwargs.get('retrieval_config', None)
            rag_pipeline = kwargs.get('rag_pipeline', None)
            list_files = kwargs.get('list_files', None)
            chat_history = kwargs.get('chat_history', None)
            response = self.brain.ask(run_id=run_id, question=question, system_prompt=system_prompt, retrieval_config=retrieval_config, rag_pipeline=rag_pipeline, list_files=list_files, chat_history=chat_history)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=response, interface_return=response)

        elif name == 'get_chat_history':
            chat_id = kwargs.get('chat_id', uuid.uuid4())
            chat_history = self.brain.get_chat_history(chat_id)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=chat_history, interface_return=chat_history)

        elif name == 'add_file':
            file_path = kwargs.get('file_path', '')
            self.brain.add_file(file_path)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=None)

        elif name == 'afrom_files':
            name_quivr = kwargs.get('name_quivr', '')
            file_paths = kwargs.get('file_paths', [])
            vector_db = kwargs.get('vector_db', None)
            storage = kwargs.get('storage', None)
            llm = kwargs.get('llm', None)
            embedder = kwargs.get('embedder', None)
            skip_file_error = kwargs.get('skip_file_error', False)
            processor_kwargs = kwargs.get('processor_kwargs', None)
            self.brain = Brain.afrom_files(name=name_quivr, file_paths=file_paths, vector_db=vector_db, storage=storage, llm=llm, embedder=embedder, skip_file_error=skip_file_error, processor_kwargs=processor_kwargs)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=self.brain)

        elif name == 'from_files':
            name_quivr = kwargs.get('name_quivr', '')
            file_paths = kwargs.get('file_paths', [])
            vector_db = kwargs.get('vector_db', None)
            storage = kwargs.get('storage', None)
            llm = kwargs.get('llm', None)
            embedder = kwargs.get('embedder', None)
            skip_file_error = kwargs.get('skip_file_error', False)
            processor_kwargs = kwargs.get('processor_kwargs', None)
            self.brain = Brain.from_files(name=name_quivr, file_paths=file_paths)
            self.result.set_result(fuc_name=name, is_success=True, fail_reason='', is_file=False, file_path='', except_data=None, interface_return=self.brain)

        else:
            self.result.set_result(fuc_name=name, is_success=False, fail_reason='Function not found', is_file=False, file_path='', except_data=None, interface_return=None)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('print_info')
adapter_addtional_data['functions'].append('save')
adapter_addtional_data['functions'].append('load')
adapter_addtional_data['functions'].append('info')
adapter_addtional_data['functions'].append('asearch')
adapter_addtional_data['functions'].append('ask_streaming')
adapter_addtional_data['functions'].append('aask')
adapter_addtional_data['functions'].append('ask')
adapter_addtional_data['functions'].append('get_chat_history')
adapter_addtional_data['functions'].append('add_file')
adapter_addtional_data['functions'].append('afrom_files')
adapter_addtional_data['functions'].append('from_files')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
