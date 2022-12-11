#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   process_data.py
@Time    :   2022/03/16 15:44:34
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
'''

# here put the import lib

from concurrent.futures import process
import json
from typing import OrderedDict
from inputDataset.gen_sparql_dataset import _vanilla_linearization_sparql_method
from inputDataset.gen_dataset import _vanilla_linearization_method
from components.utils import dump_json, load_json
from tqdm import tqdm

def merge_all_data_file(split):
    origin_dataset = load_json(f'data/origin/ComplexWebQuestions_{split}.json')
    
    sexpr_dataset = load_json(f'data/sexpr/CWQ.{split}.expr.json')
    sexpr_dataset = {x['ID']:x for x in sexpr_dataset}
    
    qdt_predict_data = load_json(f'data/qdt/CWQ_{split}_new_qdt.json')
    # qdt_predict_data = {x['ID']:x for x in qdt_predict_data}
    qdt_annotate_data = load_json(f'data/qdt/CWQ_{split}_qdt_linear.json')
    qdt_annotate_data = {x['ID']:x for x in qdt_annotate_data}
    
    cand_entity_data = load_json(f'data/linking_results/merged_CWQ_{split}_linking_results.json')
    cand_relation_data = load_json(f'data/rel_match/relations/sorted_results_new/CWQ_{split}_cand_rel_logits.json')
    
    gold_label_maps_data = load_json(f'data/label_maps/CWQ_{split}_label_maps.json')

    merged_data_list = []
    for example in tqdm(origin_dataset,total=len(origin_dataset),desc=f'{split}'):
        qid = example['ID']
        comp_type = example['compositionality_type']
        question = example['question']
        
        # sparql
        sparql = example['sparql']

        if split=='test':
            gold_answer = example['answer']
        else:
            gold_answer = [x['answer_id'] for x in example['answers']]

        # sexpr
        sexpr = sexpr_dataset[qid]['SExpr']

        # qdt, annotate and predict
        qdt_annotate = qdt_annotate_data.get(qid,None)
        if qdt_annotate:
            qdt_annotate = qdt_annotate['linear_qdt']
        
        qdt_predict = qdt_predict_data[qid]

        # entity
        gold_entity_map = gold_label_maps_data[qid]['entity_label_map']
        cand_entity_map = cand_entity_data[qid]

        # relation
        gold_relation_map = gold_label_maps_data[qid]['rel_label_map']

        cand_relation_list = cand_relation_data.get(qid,[])[:10] # retain top 10 relations

        # type
        gold_type_map = gold_label_maps_data[qid]['type_label_map']

        new_data = {}
        new_data['ID']=qid
        new_data['comp_type']=comp_type
        new_data['question']=question
        new_data['answer']=gold_answer
        new_data['sparql']=sparql
        new_data['sexpr']=sexpr
        new_data['normed_sexpr']= _vanilla_linearization_method(sexpr)
        new_data['qdt_annotate']=qdt_annotate
        new_data['qdt_predict']=qdt_predict
        new_data['gold_entity_map']=gold_entity_map
        new_data['gold_relation_map']=gold_relation_map
        new_data['gold_type_map']=gold_type_map
        new_data['cand_entity_map']=cand_entity_map
        new_data['cand_relation_map']=cand_relation_list

        merged_data_list.append(new_data)
    
    dump_json(merged_data_list,f'data/merged_all_data/CWQ_{split}_all_data.json',ensure_ascii=False,indent=4)

def sample_all_data_file(split):
    data_list = load_json(f'data/merged_all_data/CWQ_{split}_all_data.json')
    if split == 'train':
        sample_data = data_list[:5000]
    else:
        sample_data = data_list[:500]
    
    dump_json(sample_data,f'data/merged_all_data/CWQ_{split}_sample_all_data.json',ensure_ascii=False,indent=4)

def check_mtlt5_top_k_exact_match(top_k_prediction_file):
    print('new topk file name:',top_k_prediction_file)
    data_list = load_json(top_k_prediction_file)
    ex_cnt = 0
    contain_ex_cnt = 0
    real_total = 0
    for data in data_list:
        predictions = data['predictions']
        gen_label = data['gen_label']

        if gen_label.lower()!='null':
            real_total+=1
            
        if predictions[0].lower() == gen_label.lower():
            ex_cnt +=1
        
        for pred in predictions:
            if pred.lower() == gen_label.lower():
                contain_ex_cnt +=1
                break
    total = len(data_list)
    
    print(f"""Total:{total}, 
                ex_cnt:{ex_cnt}, 
                ex_rate:{ex_cnt/total}, 
                real_ex_rate:{ex_cnt/real_total}, 
                contain_ex_cnt:{contain_ex_cnt}, 
                contain_ex_rate:{contain_ex_cnt/total}, 
                real_contain_ex_rate:{contain_ex_cnt/real_total}
        """)    

def check_legacy_top_k_exact_match(top_k_prediction_file):
    print('legacy topk file name:',top_k_prediction_file)
    data_map = load_json(top_k_prediction_file)
    # print(len(data_map))

    dataset_file = 'data/merged_all_data/CWQ_test_all_data.json'
    dataset_list = load_json(dataset_file)

    ex_cnt = 0
    contain_ex_cnt = 0
    real_total = 0
    for ex in dataset_list:
        ID = ex['ID']
        normed_sexpr = ex['normed_sexpr'].replace(" , ",", ")

        if normed_sexpr.lower()!='null':
            real_total+=1
        predictions = data_map[ID]
        if predictions[0].lower()==normed_sexpr.lower():
            ex_cnt+=1
        
        if any([x.lower()==normed_sexpr.lower() for x in predictions]):
            contain_ex_cnt+=1
    

    total = len(dataset_list)
    
    print(f"""Total:{total}, 
                ex_cnt:{ex_cnt}, 
                ex_rate:{ex_cnt/total}, 
                real_ex_rate:{ex_cnt/real_total}
                contain_ex_cnt:{contain_ex_cnt}, 
                contain_ex_rate:{contain_ex_cnt/total},
                real_contain_ex_rate:{contain_ex_cnt/real_total}
        """)    

def check_sparql_top_k_exact_match(top_k_sparql_prediction_file):
    print('legacy topk file name:',top_k_sparql_prediction_file)
    pred_sparql_data_map = load_json(top_k_sparql_prediction_file)

    dataset_file = 'data/merged_all_data/CWQ_test_all_data.json'
    dataset_list = load_json(dataset_file)

    ex_cnt = 0
    contain_ex_cnt = 0
    real_total = 0
    for ex in dataset_list:
        ID = ex['ID']
        sparql = ex['sparql']
        normed_sparql = _vanilla_linearization_sparql_method(sparql,{},{},{})

        pred_sparqls = pred_sparql_data_map[ID]
        if pred_sparqls[0].lower() == normed_sparql.lower().replace('{','').replace('}',''):
            ex_cnt+=1

        for pred in pred_sparqls:
            if pred.lower() == normed_sparql.lower().replace('{','').replace('}',''):
                contain_ex_cnt+=1
                break

        if normed_sparql!="null":
            real_total+=1
    
    total = len(dataset_list)

    print(f"""Total:{total}, 
                ex_cnt:{ex_cnt}, 
                ex_rate:{ex_cnt/total}, 
                real_ex_rate:{ex_cnt/real_total}
                contain_ex_cnt:{contain_ex_cnt}, 
                contain_ex_rate:{contain_ex_cnt/total},
                real_contain_ex_rate:{contain_ex_cnt/real_total}
        """)      
            

def process_qdt_data(split):    
    origin_qdt_file = f'data/qdt/Clue_Decipher/cwq_qa_decomp_seq2seq_all_all_2inq_prefix_decipher_{split}.json'
    new_qdt_file = f'data/qdt/CWQ_{split}_new_qdt_decipher.json'
    origin_data_bank = load_json(origin_qdt_file)
    new_data_bank = OrderedDict()
    for data in origin_data_bank:
        qid = data['ID']
        qdt = data['pred']
        new_data_bank[qid] = qdt

    dump_json(new_data_bank,new_qdt_file,indent=4)
    print(f'{split} Done')
    


if __name__=='__main__':
    for split in ['test','dev','train']:
        process_qdt_data(split)        
        # merge_all_data_file(split)
        # sample_all_data_file(split)
    
    # 测试直接生成SPARQL的准确率
    # top_k_sparql_file = 'results/sparql_gen/CWQ_test_sparql_nlq_qdt/top_k_predictions.json'
    # check_sparql_top_k_exact_match(top_k_sparql_file)
    
    """
    # 测试生成SExpr的准确率
    # new_top_k_prediction_file = 'exps/gen_multitask/CWQ_not_lower/top_k_predictions.json'
    # new_top_k_prediction_file = 'exps/gen_multitask/CWQ_not_lower_1epoch/top_k_predictions.json'
    # new_top_k_prediction_file = 'exps/gen_multitask/CWQ_not_lower_10epoch/beam_10_top_k_predictions.json'
    new_top_k_prediction_file= 'exps/gen_multitask/CWQ_not_lower_5epoch/beam_10_top_k_predictions.json'
    check_mtlt5_top_k_exact_match(new_top_k_prediction_file)

    new_top_k_prediction_file= 'exps/gen_multitask/CWQ_not_lower_5epoch_retry/beam_10_top_k_predictions.json'
    check_mtlt5_top_k_exact_match(new_top_k_prediction_file)
    
    new_top_k_prediction_file= 'exps/gen_multitask/CWQ_not_lower_10epoch/beam_50_top_k_predictions.json'
    check_mtlt5_top_k_exact_match(new_top_k_prediction_file)

    #new_top_k_prediction_file= 'exps/gen_multitask/CWQ_not_lower_10epoch/beam_10_top_k_predictions.json'
    new_top_k_prediction_file= 'exps/gen_multitask/CWQ_not_lower_10epoch_retry/beam_50_top_k_predictions.json'
    check_mtlt5_top_k_exact_match(new_top_k_prediction_file)


    legacy_top_k_prediction_file = 'results/gen/CWQ_test_nlq_newqdt_candEnt/top_k_predictions.json'
    check_legacy_top_k_exact_match(legacy_top_k_prediction_file)
    """


