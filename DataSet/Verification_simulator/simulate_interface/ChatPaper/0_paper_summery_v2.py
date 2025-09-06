from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.ChatPaper import *
import argparse
import base64
import configparser
import datetime
import json
import os
import re
from collections import namedtuple
import arxiv
import numpy as np
import openai
import requests
import tenacity
import tiktoken
import fitz
import io
from PIL import Image
import time
exe = Executor('ChatPaper', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

class Args:
    pdf_path = RESOURCES_PATH + 'images/test_image.pdf'
    query = 'all: ChatGPT robot'
    key_word = 'reinforcement learning'
    filter_keys = 'ChatGPT robot'
    max_results = 1
    sort = 'Relevance'
    save_image = False
    file_format = 'md'
    language = 'zh'
args = Args()

def chat_paper_main(args):
    if args.sort == 'Relevance':
        sort = arxiv.SortCriterion.Relevance
    elif args.sort == 'LastUpdatedDate':
        sort = arxiv.SortCriterion.LastUpdatedDate
    else:
        sort = arxiv.SortCriterion.Relevance
    if args.pdf_path:
        reader1 = exe.create_interface_objects(interface_class_name='Reader', key_word=args.key_word, query=args.query, filter_keys=args.filter_keys, sort=sort, args=args)
        exe.run('show_info')
        paper_list = []
        if args.pdf_path.endswith('.pdf'):
            paper_list.append(Paper(path=args.pdf_path))
        else:
            for root, dirs, files in os.walk(args.pdf_path):
                print('root:', root, 'dirs:', dirs, 'files:', files)
                for filename in files:
                    if filename.endswith('.pdf'):
                        paper_list.append(Paper(path=os.path.join(root, filename)))
        print('------------------paper_num: {}------------------'.format(len(paper_list)))
        [print(paper_index, paper_name.path.split('\\')[-1]) for paper_index, paper_name in enumerate(paper_list)]
        exe.run('summary_with_chat', paper_list=paper_list)
    else:
        reader1 = exe.create_interface_objects(interface_class_name='Reader', key_word=args.key_word, query=args.query, filter_keys=args.filter_keys, sort=sort, args=args)
        exe.run('show_info')
        filter_results = exe.run('filter_arxiv', max_results=args.max_results)
        paper_list = exe.run('download_pdf', filter_results=filter_results)
        exe.run('summary_with_chat', paper_list=paper_list)
start_time = time.time()
chat_paper_main(args)
print('summary time:', time.time() - start_time)