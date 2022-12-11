import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm
import math
from torch.utils import data
import warnings
import logging
import torch
warnings.filterwarnings('ignore')
from sys import path

import sys

sys.path.append("..")

typo_df = pd.read_csv('../../data/typo_dict.tsv', sep='\t', header=None)
typo_dict_lower = {row[0][0].lower()+row[0][1:]: row[1][0].lower()+row[1][1:] for index, row in typo_df.iterrows()}
typo_dict_capital = {row[0][0].upper()+row[0][1:]: row[1][0].upper()+row[1][1:] for index, row in typo_df.iterrows()}
typo_dict = dict(typo_dict_capital, **typo_dict_lower)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Dict2Obj(dict):
    def __getattr__(self, key):
        value = self.get(key)
        return Dict(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


def readjson(path):
    '''读取json,json_list就返回list,json就返回dict'''
    with open(path, 'r', encoding='utf-8') as load_f:
        data_ = json.load(load_f)
    return data_


def savejson(file_name, json_info, indent=4):
    '''json list保存为json'''
    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(json_info, fp, indent=indent, sort_keys=False)

def get_2_part_decomposition(qdt):
    if '[INQL]' in qdt and '[INQR]' in qdt:
        result=' '.join([x for x in qdt.split() if x!='[DES]']).strip()
    else:
        qdt_token_list=qdt.split()
        des_num=qdt.count('[DES]')
        if des_num==1 or des_num==2:
            des_flag = 1
        elif des_num == 3 or des_num==4:
            des_flag = 2
        else:
            des_flag=3
        current_des=0
        qdt_2_part=[]
        for token in qdt_token_list:
            if token=='[DES]':
                current_des+=1
                if current_des==des_flag:
                    qdt_2_part.append(token)
            else:
                qdt_2_part.append(token)
        result=' '.join(qdt_2_part)
    return result

class DataGen(data.Dataset):
    def __init__(self, encoder_question):
        self.encoding = encoder_question

    def __len__(self):
        return len(self.encoding)

    def __getitem__(self, index):
        return self.encoding[index]

def clean_question(question,two_end=True,typo=False):
    if question == '':
        return ''
    question = question.strip()
    if two_end:
        if question[0] in ['/', ']']:
            question = question[1:]
        while question[-1] in ['/', '?', '`', '.', ':','>',';']:
            question = question[:-1]
    if typo:
        question=question.replace(" 's","'s")
        # typo
        for typo in list(typo_dict.keys()):
            if question.find(typo) == 0:
                question = question.replace(typo + " ", typo_dict[typo] + " ")
            elif typo == question.split()[-1]:
                question = question.replace(" " + typo, " " + typo_dict[typo])
            else:
                question = question.replace(" " + typo + " ", " " + typo_dict[typo] + " ")

    question = ' '.join(question.split())
    question = question.strip()
    return question
def clean_cwq_sparql(sparql):
    keyword_list = ['FILTER (?x != ?c)', 'DISTINCT', 'EXISTS', 'FILTER']
    sparql_list = sparql.split('\n')
    for x in keyword_list:
        sparql_list = [line for line in sparql_list if x not in line]
    sparql = '\n'.join(sparql_list)
    return sparql


def is_clean_sparql_correct():
    json_list = readjson('../../data/cwq/ComplexWebQuestions_dev.json')
    excutor = FBExcutor()
    for js in json_list:
        sparql = js['sparql']
        ans = excutor.query_db(sparql)
        cleaned_sparql = clean_cwq_sparql(sparql)
        cleaned_ans = excutor.query_db(cleaned_sparql)


def cal_f1(golden, pred):
    if len(pred) == 0 and len(golden) == 0:
        return 1
    elif len(pred) == 0 and len(golden) != 0:
        return 0
    elif len(pred) != 0 and len(golden) == 0:
        return 0
    else:
        p = len([x for x in pred if x in golden]) / len(pred)
        r = len([x for x in golden if x in pred]) / len(golden)
        if p == 0 or r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)


def get_gpu_memory(gpu_id=1):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        return 0
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler).free / 1024 / 1024
    return meminfo


def find_free_gpu(gpu_id=None, threshold=22000):
    # This mapping is based on author's device, use your own mapping, do not use ours (e.g. {0:0,1:1,2:2,3:3...})
    nvidia2torh = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}  # convert nvidia id to torch id

    torch2nvidia = {v: k for k, v in nvidia2torh.items()}
    assert gpu_id == None or gpu_id in [0, 2, 4, 1, 3], 'This gpu num not exist!'
    if gpu_id != None:
        if get_gpu_memory(torch2nvidia[gpu_id]) > threshold:
            print('Current gpu: ', gpu_id, ' is avaliable!')
            return gpu_id
        else:
            print('Current gpu: ', gpu_id, ' out of memory, searching avaliable gpu...')
    print('Searching avaliable gpu...')
    while True:
        for gpu_id in [0, 2, 4, 1, 3]:
            if get_gpu_memory(gpu_id) > threshold:
                print('Find gpu :', nvidia2torh[gpu_id], ' is avaliable! ')
                return nvidia2torh[gpu_id]
def get_confusion_matrix(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives




def outer_inner_list2qdt(outer_question, inner_question):

    all_node = [
        {'nodeType': 'Entity', 'entityID': 0, 'nodeID': 0}
    ]
    all_edge = []
    if len(inner_question) != 0:
        all_node.append({'nodeType': 'Entity', 'entityID': 1, 'nodeID': 1})

    node_id_now = len(all_node)
    for outer_des in outer_question:
        all_node.append({'nodeType': 'Description', 'value': outer_des, 'entityID': 0, 'nodeID': node_id_now,
                         'hasRefer': '[ENT]' in outer_des})
        all_edge.append({'from': 0, 'to': node_id_now})
        if '[ENT]' in outer_des:
            all_edge.append({'from': node_id_now, 'to': 1})
        node_id_now += 1
    for inner_des in inner_question:
        all_node.append({'nodeType': 'Description', 'value': inner_des, 'entityID': 1, 'nodeID': node_id_now,
                         'hasRefer': '[ENT]' in inner_des})
        all_edge.append({'from': 1, 'to': node_id_now})
        node_id_now += 1

    tree_structure = {"nodes": all_node, "edge": all_edge}

    return tree_structure


def linear_qdt_to_tree(linear_qdt,seq_eval=False):

    if '[INQL]' not in linear_qdt and '[INQR]' not in linear_qdt:
        outer_des = [x.strip() for x in linear_qdt.split('[DES]')]
        inner_des = []
    else:
        inq_split = [x.strip() for x in re.split("\[INQL\]|\[INQR\]",linear_qdt)]
        if len(inq_split) == 2:
            linear_qdt = linear_qdt.replace('[INQL]', '[DES]').replace('[INQR]', '[DES]')
            outer_des = [x.strip() for x in linear_qdt.split('[DES]')]
            inner_des = []
        elif len(inq_split) == 3:
            inner_des = [x.strip() for x in inq_split[1].split('[DES]')]  # 内层描述
            outer_des = linear_qdt.replace('[INQL] ' + inq_split[1] + ' [INQR]', '[ENT]')
            outer_des = [x.strip() for x in outer_des.split('[DES]')]
        else:
            inner_des = ' [DES] '.join(inq_split[1:-1])
            inner_des = [x.strip() for x in inner_des.split('[DES]')]
            outer_des = ' [ENT] '.join([inq_split[0], inq_split[-1]])
            outer_des = [x.strip() for x in outer_des.split('[DES]')]
    outer_des = [x for x in outer_des if x!='']
    inner_des = [x for x in inner_des if x != '']
    if seq_eval:
        if len(inner_des)!=0:
            outer_des=[' '.join(outer_des)]
            inner_des=[' '.join(inner_des)]
        else:
            outer_des = [' '.join(outer_des[:int(len(outer_des)/2)]),' '.join(outer_des[int(len(outer_des)/2):])]

    qdt = outer_inner_list2qdt(outer_des, inner_des)
    return outer_des, inner_des, qdt

def tree2linear_qdt(qdt,two_inq=True):
    #qdt->linear qdt
    this_example = []
    for out_des in qdt['root_question']:
        if 'inner_questions' in out_des:
            inner_part = ' [DES] '.join([x['description'] for x in out_des['inner_questions']['IQ1']])
            if two_inq:
                this_out = out_des['description'].replace('[IQ1]', '[INQL] ' + inner_part + ' [INQR]')
            else:
                this_out = out_des['description'].replace('[IQ1]', '[INQ] ' + inner_part + ' [INQ]')
        else:
            this_out = out_des['description']
        this_example.append(this_out)
    this_example = ' [DES] '.join(this_example).strip()
    return this_example

