from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.deep_searcher import *
import argparse
import logging
import sys
import warnings
from deepsearcher.configuration import Configuration
from deepsearcher.configuration import init_config
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.offline_loading import load_from_website
from deepsearcher.online_query import query
from deepsearcher.utils import log
exe = Executor('deep_searcher', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    """
    Main entry point for the DeepSearcher CLI.

    This function executes the appropriate action based on hardcoded values for the query.
    Returns:
        None
    """
    command = 'query'
    query_value = 'What is the capital of France?'
    max_iter_value = 3
    config = Configuration()
    init_config(config=config)
    if command == 'query':
        final_answer, refs, consumed_tokens = exe.run('query', original_query=query_value, max_iter=max_iter_value)
        log.color_print('\n==== FINAL ANSWER====\n')
        log.color_print(final_answer)
        log.color_print('\n### References\n')
        for i, ref in enumerate(refs):
            log.color_print(f'{i + 1}. {ref.text[:60]}… {ref.reference}')
    elif command == 'load':
        load_path = ['']
        urls = [url for url in load_path if url.startswith('http')]
        local_files = [file for file in load_path if not file.startswith('http')]
        kwargs = {'collection_name': 'example_collection', 'collection_desc': 'An example collection description', 'force_new_collection': False, 'batch_size': 256}
        if len(urls) > 0:
            load_from_website(urls, **kwargs)
        if len(local_files) > 0:
            load_from_local_files(local_files, **kwargs)
    else:
        print('Please provide a query or a load argument.')
main()