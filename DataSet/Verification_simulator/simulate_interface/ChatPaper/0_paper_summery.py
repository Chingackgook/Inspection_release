
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.ChatPaper import ENV_DIR
exe = Executor('ChatPaper', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

import argparse
import base64
import configparser
import datetime
import json
import re
from collections import namedtuple

import arxiv
import openai
import requests
import tenacity
import tiktoken

import fitz, io
from PIL import Image

from chat_paper import Paper
from chat_paper import Reader

def chat_paper_main():
    # 创建一个Reader对象
    exe.create_interface_objects(key_word='Machine Learning',
                    query='deep learning',
                    filter_keys='neural network',
                    root_path='./',
                    gitee_key='',
                    sort=arxiv.SortCriterion.SubmittedDate,
                    user_name='default',
                    args=argparse.Namespace(language='en', file_format='md', save_image=False))
    
    reader = exe.adapter.reader
    # 获取arXiv搜索结果
    filter_results = reader.filter_arxiv(max_results=10)

    # 下载PDF
    papers = reader.download_pdf(filter_results)

    # 使用聊天模型进行总结，替换为exe.run的调用
    htmls = []
    for paper_index, paper in enumerate(papers):
        # 第一步先用title，abs，和introduction进行总结。
        text = ''
        text += 'Title:' + paper.title
        text += 'Url:' + paper.url
        text += 'Abstract:' + paper.abs
        text += 'Paper_info:' + paper.section_text_dict['paper_info']
        text += list(paper.section_text_dict.values())[0]  # introduction
        
        chat_summary_text = exe.run("chat_summary", text=text)
        # 检查chat_summary是否已经实现，应该存在
        # 如果发生错误，例如未实现的方法，将会触发异常处理

        htmls.append('## Paper:' + str(paper_index + 1))
        htmls.append('\n\n\n')
        htmls.append(chat_summary_text)

        # 第二步总结方法：
        method_key = ''
        for parse_key in paper.section_text_dict.keys():
            if 'method' in parse_key.lower() or 'approach' in parse_key.lower():
                method_key = parse_key
                break

        if method_key != '':
            text = ''
            method_text = paper.section_text_dict[method_key]
            summary_text = "<summary>" + chat_summary_text
            text = summary_text + "\n\n<Methods>:\n\n" + method_text
            chat_method_text = exe.run("chat_method", text=text)
            # 检查chat_method是否已经实现，应该存在
            # 如果发生错误，例如未实现的方法，将会触发异常处理
            htmls.append(chat_method_text)

        htmls.append("\n" * 4)

        # 第三步总结全文，并打分：
        conclusion_key = ''
        for parse_key in paper.section_text_dict.keys():
            if 'conclu' in parse_key.lower():
                conclusion_key = parse_key
                break

        text = ''
        summary_text = "<summary>" + chat_summary_text
        if conclusion_key != '':
            conclusion_text = paper.section_text_dict[conclusion_key]
            text = summary_text + "\n\n<Conclusion>:\n\n" + conclusion_text
        else:
            text = summary_text
            
        chat_conclusion_text = exe.run("chat_conclusion", text=text)
        # 检查chat_conclusion是否已经实现，应该存在
        # 如果发生错误，例如未实现的方法，将会触发异常处理
        htmls.append(chat_conclusion_text)
        htmls.append("\n" * 4)

        # 整合成一个文件，打包保存下来
        date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
        export_path = FILE_RECORD_PATH
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        file_name = os.path.join(export_path,
                                date_str + '-' + reader.validateTitle(paper.title[:80]) + ".md")
        reader.export_to_markdown("\n".join(htmls), file_name=file_name)
        htmls = []

# 直接运行主逻辑
chat_paper_main()
