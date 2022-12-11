from email import header
from email.policy import default
from textwrap import indent


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_realtions.py
@Time    :   2022/03/08 10:51:08
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
'''

# here put the import lib

import json
from typing import DefaultDict
from pkg_resources import run_script

from tqdm import tqdm
import csv
import sys
import os
# print(sys.path)
sys.path.append(os.getcwd())
# sys.path.append("..") # CWQ dir

from components.utils import dump_json, load_json
import torch
from tqdm import tqdm
import pandas as pd


def gen_rel_match_data_set(split, filter_base=True):
    """make sentence-relation pair for cross-encoder training"""
    res_lines = []

    if filter_base:
        relation_data_file_name = f'data/rel_match/CWQ_{split}_relation_data_old.json'
    else:
        relation_data_file_name = f'data/rel_match/CWQ_{split}_relation_data.json'

    train_relation_map = load_json(f'data/label_maps/CWQ_{split}_label_maps.json')

    
    print(f'Loading {split}...')
    with open(relation_data_file_name,'r') as f:
        databank = json.load(f)
        print(len(databank))
        idx =0
        for qid,data in tqdm(databank.items(),desc=f'Processing {split}'):

            if split=='train': # for training
                for rel in data['positive_rels']:
                    if filter_base: # filter base relation
                        if rel.startswith('base.'):
                            continue

                    if '\t' in rel: # process illegal relation
                        rel = rel.split('\t')[0]
                    line = str(idx)+"\t"+data['question'] + '\t' + rel+'\t'+'1'
                    res_lines.append(line+'\n')
                    idx+=1
                for rel in data['negative_rels']:
                    if '\t' in rel:
                        rel = rel.split('\t')[0]
                    line = str(idx)+"\t"+data['question'] + '\t' + rel+'\t'+'0'
                    res_lines.append(line+'\n')
                    idx+=1
            else:
                added_rels = set()
                for rel in data['cand_rels']: # take cand_rels
                    if filter_base: # filter base relation
                        if rel.startswith('base.'):
                            continue

                    if '\t' in rel: # process illegal relation
                        rel = rel.split('\t')[0]
                    added_rels.add(rel)
                    if rel in data['positive_rels']:
                        line = str(idx)+"\t"+data['question'] + '\t' + rel+'\t'+'1'
                    else:
                        line = str(idx)+"\t"+data['question'] + '\t' + rel+'\t'+'0'
                    res_lines.append(line+'\n')
                    idx+=1
                
                for rel in train_relation_map.keys(): # add rels in train set
                    if rel not in added_rels:
                        added_rels.add(rel)
                        if '\t' in rel: # process illegal relation
                            rel = rel.split('\t')[0]
                        if rel in data['positive_rels']:
                            line = str(idx)+"\t"+data['question'] + '\t' + rel+'\t'+'1'
                        else:
                            line = str(idx)+"\t"+data['question'] + '\t' + rel+'\t'+'0'
                        res_lines.append(line+'\n')
                        idx+=1

    with open(f'data/rel_match/CWQ_{split}_rel_match.tsv','w') as f:
        f.writelines(res_lines)


def make_pseudo_sorted_dataset(split):
    pseudo_sorted_data = {}
    rel_data_bank = load_json(f'data/rel_match/CWQ_{split}_relation_data.json')
    for qid,data in tqdm(rel_data_bank.items(),desc='Building pseudo cand rel set'):
        pos_rels = data['positive_rels']
        neg_rels = data['negative_rels']
        pseudo_sorted_data[qid] = pos_rels+neg_rels
    
    dump_json(pseudo_sorted_data,f'data/rel_match/CWQ_{split}_cand_rels_sorted.json',indent=4)


def make_sorted_dataset(split, from_cand=False):
    # with open('logits.pt','r') as f:
    
    logits = torch.load(f'data/rel_match/predict/{split}/logits.pt',map_location=torch.device('cpu'))

    # print(logits)
    # print(len(logits))
    logits_list = list(logits.squeeze().numpy())
    print('Logits len:',len(logits_list))

    split_tsv = f'CWQ_{split}_rel_match_old.tsv'
    
    # with open(split_tsv,'r') as f:
    #     tsv_lines = f.readlines()
    # print('TSV lines:', len(tsv_lines))

    split_df = pd.read_csv(split_tsv, header=None, delimiter='\t'
                            ,names=['id','question','rel','label']
                            ,error_bad_lines=True
                            ,quoting=csv.QUOTE_NONE)
                            #,dtype={"id":int, "question":str, "rel":str, 'label':int}
                            
    print('Dataframe len:',len(split_df))

    
    # length check
    assert len(logits_list) == len(split_df)


    enumerate_cand_rel_bank  = load_json(f'data/rel_match/CWQ_{split}_relation_data_old.json')


    # with open('logits.txt','w') as f:
    #     for logit in tqdm(logits_list):
    #         f.write(str(logit)+'\n')

    split_dataset = f'data/origin/ComplexWebQuestions_{split}.json'
    if split=='test_new':
        split_dataset= 'data/origin/ComplexWebQuestions_test.json'
    split_dataset = load_json(split_dataset)
    question2id = {x['question']:x['ID'] for x in split_dataset}

    # cand_rel_bank = {} # Dict[Question, List[Relation:logit]]
    cand_rel_bank = DefaultDict(dict)
    cand_rel_bank_from_enumerate = {}
    for idx,logit in tqdm(enumerate(logits_list),total=len(logits_list),desc=f'Reading logits {split}'):
        logit = float(logit)
        question = split_df.loc[idx]['question']
        rel = split_df.loc[idx]['rel']
        cwq_id = question2id.get(question,None)

        if not cwq_id:
            continue
        else:
            if not from_cand:
                cand_rel_bank[cwq_id][rel]=logit
            else:
                # only retain the relations in cand rels
                if cwq_id in cand_rel_bank_from_enumerate:
                    cand_rel_enumreate = cand_rel_bank_from_enumerate[cwq_id]
                else:
                    cand_rel_enumreate = set(enumerate_cand_rel_bank[cwq_id]['cand_rels'])
                    cand_rel_bank_from_enumerate[cwq_id] = cand_rel_enumreate
                
                if rel in cand_rel_enumreate: # only retain the relations in cand rels
                    cand_rel_bank[cwq_id][rel]=logit

    cand_rel_file_name = f'data/rel_match/CWQ_{split}_cand_rels_from_cand.json' if from_cand else f'data/rel_match/CWQ_{split}_cand_rels.json'
    dump_json(cand_rel_bank,cand_rel_file_name,indent=4)
    

    final_candRel_map = {} # Dict[Question,List[Rel]]   sorted by logits

    for ori_data in tqdm(split_dataset,total=len(split_dataset),desc=f'Sort by logits {split}'):
        qid = ori_data['ID']
        cand_rel_map = cand_rel_bank.get(qid,None)
        if not cand_rel_map:
            final_candRel_map[qid]=[]
        else:
            cand_rel_list = list(cand_rel_map.keys())
            cand_rel_list.sort(key=lambda x:float(cand_rel_map[x]),reverse=True)
            final_candRel_map[qid]=cand_rel_list

    sorted_cand_rel_name = f'data/rel_match/CWQ_{split}_cand_rels_sorted.json' if not from_cand else f'data/rel_match/CWQ_{split}_cand_rels_sorted_from_cand.json'
    dump_json(final_candRel_map,sorted_cand_rel_name,indent=4)
    

def check_sorted_dataset_coverage(split,sorted_rel_file):
    
    sorted_rel_bank = load_json(sorted_rel_file)
    label_map_file = f'data/label_maps/CWQ_{split}_label_maps.json'
    if split=='test_new':
        label_map_file = f'data/label_maps/CWQ_test_label_maps.json'
    label_maps_bank = load_json(label_map_file)

    gt_0_pre = 0
    gt_0_recall = 0
    top1_pre = 0
    top1_recall = 0
    top3_pre = 0
    top3_recall = 0
    top4_pre = 0
    top4_recall = 0
    top5_pre = 0
    top5_recall = 0
    top10_pre = 0
    top10_recall = 0
    total = len(sorted_rel_bank)
    for qid in sorted_rel_bank:
        cand_rel_list = sorted_rel_bank[qid]
        gold_rel_set = label_maps_bank[qid]['rel_label_map'].keys()

        
        top1_cand = set([x[0] for x in cand_rel_list[:1]])
        top3_cand = set([x[0] for x in cand_rel_list[:3]])
        top4_cand = set([x[0] for x in cand_rel_list[:4]])
        top5_cand = set([x[0] for x in cand_rel_list[:5]])
        top10_cand = set([x[0] for x in cand_rel_list[:10]])

        gt_0_cand = set([x[0] for x in cand_rel_list if x[1]>=0])
        if not gt_0_cand:
            gt_0_cand = set([x[0] for x in cand_rel_list[:1]])

        gt_0_pre += len(gt_0_cand & gold_rel_set) / len(gt_0_cand) if gt_0_cand else 0
        gt_0_recall += len(gt_0_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        gt_0_f1 = 2*gt_0_pre*gt_0_recall/(gt_0_pre+gt_0_recall) if (gt_0_pre+gt_0_recall)>0 else 0

        top1_pre += len(top1_cand & gold_rel_set) / len(top1_cand) if top1_cand else 0
        top1_recall += len(top1_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        top1_f1 = 2*top1_pre*top1_recall/(top1_pre+top1_recall) if (top1_pre+top1_recall)>0 else 0
        
        top3_pre += len(top3_cand & gold_rel_set) / len(top3_cand) if top3_cand else 0
        top3_recall += len(top3_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        top3_f1 = 2*top3_pre*top3_recall/(top3_pre+top3_recall) if (top3_pre+top3_recall)>0 else 0

        top4_pre += len(top4_cand & gold_rel_set) / len(top4_cand) if top4_cand else 0
        top4_recall += len(top4_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        top4_f1 = 2*top4_pre*top4_recall/(top4_pre+top4_recall) if (top4_pre+top4_recall)>0 else 0

        top5_pre += len(top5_cand & gold_rel_set) / len(top5_cand) if top5_cand else 0
        top5_recall += len(top5_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        top5_f1 = 2*top5_pre*top5_recall/(top5_pre+top5_recall) if (top5_pre+top5_recall)>0 else 0

        top10_pre += len(top10_cand & gold_rel_set) / len(top10_cand) if top10_cand else 0
        top10_recall += len(top10_cand & gold_rel_set) / len(gold_rel_set) if gold_rel_set else 0
        top10_f1 = 2*top10_pre*top10_recall/(top10_pre+top10_recall) if (top10_pre+top10_recall)>0 else 0
    
    print(split,'TOTAL: ',total)
    print(f'GT 0 pre: {gt_0_pre/total}, GT 0 recall: {gt_0_recall/total}, GT 0 f1: {gt_0_f1/total}') 
    print(f'TOP 1 pre: {top1_pre/total}, TOP 1 recall: {top1_recall/total}, TOP 1 f1: {top1_f1/total}') 
    print(f'TOP 3 pre: {top3_pre/total}, TOP 3 recall: {top3_recall/total}, TOP 3 f1: {top3_f1/total}') 
    print(f'TOP 4 pre: {top4_pre/total}, TOP 4 recall: {top4_recall/total}, TOP 4 f1: {top4_f1/total}') 
    print(f'TOP 5 pre: {top5_pre/total}, TOP 5 recall: {top5_recall/total}, TOP 5 f1: {top5_f1/total}') 
    print(f'TOP 10 pre: {top10_pre/total}, TOP 10 recall: {top10_recall/total}, TOP 10 f1: {top10_f1/total}') 


def check_cand_dataset_coverage(split):
    """Check the recall of candidate relations"""
    data_bank = load_json(f'data/rel_match/CWQ_{split}_relation_data_old.json')
    cand_recall = 0
    for qid,data in data_bank.items():
        positive_rels = data['positive_rels']
        negative_rels = data['negative_rels']
        cand_rels = data['cand_rels']

        cand_recall += len(set(cand_rels)& set(positive_rels))/len(set(positive_rels))

    print(f'{split} cand recall: {cand_recall/len(data_bank)}')


def make_sorted_dataset_from_logits(split, logits_file, tsv_file, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logits = torch.load(logits_file,map_location=torch.device('cpu'))

    # print(logits)
    # print(len(logits))
    logits_list = list(logits.squeeze().numpy())
    print('Logits len:',len(logits_list))

    tsv_df = pd.read_csv(tsv_file, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
                            #quoting=csv.QUOTE_NONE,
                            

    print('Tsv len:', len(tsv_df))
    # print(tsv_df.head())
    print('Question Num:',len(tsv_df['question'].unique()))

    assert(len(logits_list)==len(tsv_df))

    split_dataset = load_json(f'data/origin/ComplexWebQuestions_{split}.json')
    question2id = {x['question']:x['ID'] for x in split_dataset}

    # cand_rel_bank = {} # Dict[Question, List[Relation:logit]]
    cand_rel_bank = DefaultDict(dict)
    for idx,logit in tqdm(enumerate(logits_list),total=len(logits_list),desc=f'Reading logits {split}'):
        logit = float(logit)
        question = tsv_df.loc[idx]['question']
        rel = tsv_df.loc[idx]['relation'].split("|")[0]
        cwq_id = question2id.get(question,None)

        if not cwq_id:
            print(question)
            cand_rel_bank[cwq_id]= {}
        else:
            cand_rel_bank[cwq_id][rel]=logit

    cand_rel_logit_map = {}
    for qid in tqdm(cand_rel_bank,total=len(cand_rel_bank),desc='Sorting rels...'):
        cand_rel_maps = cand_rel_bank[qid]
        cand_rel_list = [(rel,logit) for rel,logit in cand_rel_maps.items()]
        cand_rel_list.sort(key=lambda x:x[1],reverse=True)
        
        cand_rel_logit_map[qid]=cand_rel_list

    dump_json(cand_rel_logit_map,os.path.join(output_dir,f'CWQ_{split}_cand_rel_logits.json'),indent=4)

    final_candRel_map = DefaultDict(list) # Dict[Question,List[Rel]]   sorted by logits

    for ori_data in tqdm(split_dataset,total=len(split_dataset),desc=f'{split} Dumping... '):
        qid = ori_data['ID']
        # cand_rel_map = cand_rel_bank.get(qid,None)
        cand_rel_list = cand_rel_logit_map.get(qid,None)
        if not cand_rel_list:
            final_candRel_map[qid]=[]
        else:
            # cand_rel_list = list(cand_rel_map.keys())
            # cand_rel_list.sort(key=lambda x:float(cand_rel_map[x]),reverse=True)
            final_candRel_map[qid]=[x[0] for x in cand_rel_list]

    sorted_cand_rel_name = os.path.join(output_dir,f'CWQ_{split}_cand_rels_sorted.json')
    dump_json(final_candRel_map,sorted_cand_rel_name,indent=4)     


if __name__=='__main__':
    
    output_dir = 'data/rel_match/relations/sorted_results_new'
    # split = 'dev'
    for split in ['dev']:
        logits_file = f'data/rel_match/relations/predictions_new/{split}/logits.pt'
        tsv_file = f'data/rel_match/relations/biEncoder_top200_new/CWQ.{split}.biEncoder.train_all.richRelation.top200.tsv'
        make_sorted_dataset_from_logits(split, logits_file, tsv_file, output_dir)

    # split = 'test'
    for split in ['dev']:
        sorted_logit_file_name = os.path.join(output_dir,f'CWQ_{split}_cand_rel_logits.json')
        check_sorted_dataset_coverage(split,sorted_logit_file_name)



    # for split in ['train','test','dev']:
    #     # gen_rel_match_data_set(split)
    #     check_cand_dataset_coverage(split)
    

    # gen_rel_match_data_set('test')

    # make_pseudo_sorted_dataset('dev')
    # make_pseudo_sorted_dataset('train')
    # for split in ['test','dev','train']:
    #     make_sorted_dataset(split)
    
    # 根据logits重建排序后的候选关系    
    # make_sorted_dataset('test_new')    
    # make_sorted_dataset('test',from_cand=True)
    # make_sorted_dataset('dev',from_cand=True)
    # make_sorted_dataset('train',from_cand=True)
    
    # 检查排序后的候选关系的覆盖率
    # check_sorted_dataset_coverage('test')
    # check_sorted_dataset_coverage('dev')
    # check_sorted_dataset_coverage('train')
    # check_sorted_dataset_coverage('test_new')

    
    # check_sorted_dataset_coverage('dev')
    # check_sorted_dataset_coverage('train')

