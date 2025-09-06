import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.flair import ENV_DIR
from Inspection.adapters.custom_adapters.flair import *
exe = Executor('flair', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
from flair.data import Sentence
from flair.nn import Classifier
from flair.embeddings import WordEmbeddings
from flair.data import Corpus
from pathlib import Path

# 创建一个句子
sentence = Sentence('I love Berlin.')

# 加载NER标注器
exe.create_interface_objects(model_path=os.path.join(ENV_DIR, 'load_model_path', 'pytorch_model.bin'))

# 对句子进行NER预测
mock_sentences = [sentence]  # 模拟输入
exe.run("predict", sentences=mock_sentences, mini_batch_size=32, 
         return_probabilities_for_all_classes=False, verbose=False, 
         label_name=None, return_loss=False, embedding_storage_mode="none")

# 打印带有所有注释的句子
print(sentence)

# 模拟评估
mock_data_points = [sentence]  # 模拟输入
exe.run("evaluate", data_points=mock_data_points, gold_label_type="NER", 
         out_path=os.path.join(FILE_RECORD_PATH, "evaluation_results.txt"), 
         embedding_storage_mode="none", mini_batch_size=32, 
         main_evaluation_metric=("micro avg", "f1-score"), 
         exclude_labels=["O"], gold_label_dictionary=None, return_loss=True)

# 模拟打印预测
mock_batch = [sentence]  # 模拟输入
exe.run("_print_predictions", batch=mock_batch, gold_label_type="NER")


