#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   analyse.py
@Time    :   2022/02/27 12:05:40
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
'''

# here put the import lib
from functools import reduce
from components.utils import dump_json, load_json
from components.grail_utils import extract_mentioned_entities, extract_mentioned_relations_from_sexpr, extract_mentioned_relations_from_sparql
from executor.logic_form_util import lisp_to_sparql,execute_query
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, precision_score,recall_score,f1_score
import os

def analyse_candEnt_failed():
    candEnt_failed_res = load_json("results/gen/CWQ_test_nlq_qdt_candEnt/gen_failed_results.json")
    goldEnt_failed_res = load_json("results/gen/CWQ_test_nlq_qdt_goldEnt/gen_failed_results.json")


    print(len(candEnt_failed_res))
    print(len(goldEnt_failed_res))

    candEnt_failed_qids = {x["qid"] for x in candEnt_failed_res}
    goldEnt_failed_qids = {x["qid"] for x in goldEnt_failed_res}

    print('Common failed: ', len(candEnt_failed_qids&goldEnt_failed_qids))
    print('Cand Failed: ', len(candEnt_failed_qids-goldEnt_failed_qids))
    print('Gold Failed: ', len(goldEnt_failed_qids-candEnt_failed_qids))

    print(list(candEnt_failed_qids-goldEnt_failed_qids)[:10])
    

def analyse_denormalize_failed(exp_name):

    dirname = "results/gen/CWQ_test_"+exp_name
    denormlize_failed_data = load_json(os.path.join(dirname,"gen_failed_results.json"))
    ent_failed_examples =[]
    rel_failed_examples = []
    sketch_failed_examples = []

    for failed_example in tqdm(denormlize_failed_data,total=len(denormlize_failed_data)):
        qid = failed_example['qid']
        gt_expr = failed_example['gt_sexpr']
        # denorm_expr = failed_example['denormed_sexpr']
        denorm_expr = failed_example['denormed_pred'][0] # first expr

        gt_entities = extract_mentioned_entities(gt_expr)
        pred_entities = extract_mentioned_entities(denorm_expr)

        gt_relations = extract_mentioned_relations_from_sexpr(gt_expr)
        pred_relations = extract_mentioned_relations_from_sexpr(denorm_expr)

        try:
            gt_answers = execute_query(lisp_to_sparql(gt_expr))
        except:
            pred_answers = []
        try:
            pred_answers=execute_query(lisp_to_sparql(denorm_expr))
        except:
            pred_answers = []


        if pred_answers != gt_answers:
            # 首先探测 结构错误
            gt_expr_sketch = gt_expr
            for ent in gt_entities:
                gt_expr_sketch = gt_expr_sketch.replace(ent,'[ENT]')
            for rel in gt_relations:
                gt_expr_sketch = gt_expr_sketch.replace(rel,'[REL]')    
            
            pred_expr_sketch = denorm_expr
            for ent in pred_entities:
                pred_expr_sketch = pred_expr_sketch.replace(ent,'[ENT]')
            for rel in pred_relations:
                pred_expr_sketch = pred_expr_sketch.replace(rel, '[REL]')

            if gt_expr_sketch.lower() != pred_expr_sketch.lower():
                sketch_failed_examples.append({
                    "ID":qid,
                    "gt_sexpr":gt_expr,
                    "normed_sexpr":failed_example['gt_normed_sexpr'],
                    'denormed_sexpr':denorm_expr,
                    "gt_relations":gt_relations,
                    "pred_relations":pred_relations
                }     
                )
            else:
                # 实体错误
                if set(pred_entities) != set(gt_entities):
                        ent_failed_examples.append({
                            "ID":qid,
                            "gt_sexpr":gt_expr,
                            "normed_sexpr":failed_example['gt_normed_sexpr'],
                            'denormed_sexpr':denorm_expr,
                            "gt_entities":gt_entities,
                            "pred_enties":pred_entities
                        })
                
                # 关系错误
                if set(pred_relations) != set(gt_relations):
                    rel_failed_examples.append({
                        "ID":qid,
                        "gt_sexpr":gt_expr,
                        "normed_sexpr":failed_example['gt_normed_sexpr'],
                        'denormed_sexpr':denorm_expr,
                        "gt_relations":gt_relations,
                        "pred_relations":pred_relations
                    })

    print(f'Total: {len(denormlize_failed_data)}, Sketch Error: {len(sketch_failed_examples)}, Entity Error: {len(ent_failed_examples)}, Relation Error: {len(rel_failed_examples)}') 

    dump_json(ent_failed_examples,os.path.join(dirname,"entity_failed_examples.json"),indent=4,ensure_ascii=False)
    dump_json(rel_failed_examples,os.path.join(dirname,"rel_failed_examples.json"),indent=4,ensure_ascii=False)
    dump_json(sketch_failed_examples,os.path.join(dirname,"sketch_failed_examples.json"),indent=4,ensure_ascii=False)


def compare_gen_failed(exp_name1, exp_name2):
    dirname1 = "results/gen/CWQ_test_"+exp_name1
    denormlize_failed_data1 = load_json(os.path.join(dirname1,"gen_failed_results.json"))
    print(len(denormlize_failed_data1))

    dirname2 = "results/gen/CWQ_test_"+exp_name2
    denormlize_failed_data2 = load_json(os.path.join(dirname2,"gen_failed_results.json"))
    print(len(denormlize_failed_data2))

    qids_1 = set(x['qid'] for x in denormlize_failed_data1)
    qids_2 = set(x['qid'] for x in denormlize_failed_data2)
    print('both:',len(qids_1 & qids_2))
    print('exp2 failed:',len(qids_2 - qids_1))
    print('exp1 failed:',len(qids_1 - qids_2))
    print('exp2 failed sample:', list(qids_2 - qids_1)[:10])


def evaluate_clf_res(res_file):
    """Analyse the relation classification task in MTL gen model"""
    total_pre = 0
    total_recall = 0
    total_f1 = 0

    data_bank = load_json(res_file)
    for data in data_bank:
        pred_clf_labels = [0 if x <0.5 else 1 for x in data['pred_clf_labels']]
        gold_clf_labels = data['gold_clf_labels']
        # print(pred_clf_labels)
        # print(gold_clf_labels)
        p,r,f,_ = precision_recall_fscore_support(gold_clf_labels,pred_clf_labels,average='binary',zero_division=0)
        # print(p,r,f)
        total_pre += p
        total_recall += r
        total_f1 += f
    
    total = len(data_bank)
    print(f'Total:{total}, P:{total_pre/total}, R:{total_recall/total}, F:{total_f1/total}')


def error_analysis(gen_failed_file):
    data_bank = load_json(gen_failed_file)
    total = len(data_bank)
    print(total)

    normed_ex_cnt = 0
    null_gt_cnt = 0
    sketch_ex_cnt = 0
    mask_ent_ex_cnt = 0
    mask_rel_ex_cnt = 0
    for qid,data in data_bank.items():
        gt_expr = data['gt_sexpr']
        
        if gt_expr =="null":
            null_gt_cnt+=1
            continue

        gt_normed_sexpr = data['gt_normed_sexpr']
        preds = data['pred']
        pred_normed = data['pred_normed']
        if any([x==gt_normed_sexpr for x in pred_normed]):
            normed_ex_cnt+=1
            continue
        
        gt_entities = list(set(extract_mentioned_entities(gt_expr)))
        gt_relations = list(set(extract_mentioned_relations_from_sexpr(gt_expr)))

        gt_sketch = reduce(lambda x,y:x.replace(y,'[ENT]'),gt_entities,gt_expr)
        gt_sketch = reduce(lambda x,y:x.replace(y,'[REL]'),gt_relations,gt_sketch)
        # print(gt_sketch)


        for denorm_sexpr in preds:
            pred_entities = list(set(extract_mentioned_entities(denorm_sexpr)))
            pred_relations = list(set(extract_mentioned_relations_from_sexpr(denorm_sexpr)))
            
            denorm_sketch = reduce(lambda x,y:x.replace(y,'[ENT]'),pred_entities,denorm_sexpr)
            denorm_sketch = reduce(lambda x,y:x.replace(y,'[REL]'),pred_relations,denorm_sketch)

            if denorm_sketch==gt_sketch:
                sketch_ex_cnt+=1

                denorm_mask_ent_sketch = reduce(lambda x,y:x.replace(y,'[ENT]'),pred_entities,denorm_sexpr)
                gt_mask_ent_sketch = reduce(lambda x,y:x.replace(y,'[ENT]'),gt_entities,gt_expr)
                if denorm_mask_ent_sketch==gt_mask_ent_sketch:
                    mask_ent_ex_cnt += 1
                
                denorm_mask_rel_sketch = reduce(lambda x,y:x.replace(y,'[REL]'),pred_relations,denorm_sexpr)
                gt_mask_rel_sketch = reduce(lambda x,y:x.replace(y,'[REL]'),gt_relations,gt_expr)

                if denorm_mask_rel_sketch==gt_mask_rel_sketch:
                    mask_rel_ex_cnt += 1

                break

        # pred_entities = extract_mentioned_entities(denorm_expr)            
        # pred_relations = extract_mentioned_relations_from_sexpr(denorm_expr)

    
    print('Normed SExpr match:',normed_ex_cnt)
    print('Null gt cnt:',null_gt_cnt)
    print('Sketch wrong:',total-normed_ex_cnt-sketch_ex_cnt)
    print('Sketch ex cnt:',sketch_ex_cnt)
    print('Only Entity wrong:',mask_ent_ex_cnt)
    print('Only Relation wrong:',mask_rel_ex_cnt)
    print('Both Entity and Relation wrong:',sketch_ex_cnt-mask_ent_ex_cnt-mask_rel_ex_cnt)
            

def compare_failed_QIDs(exp1_file,exp2_file):
    # pass
    exp_result1 = load_json(exp1_file)
    exp_result2 = load_json(exp2_file)

    #exp1_failed_QIDs = set([x['qid'] for x in exp_result1 if x['answer_acc']==False])
    #exp2_failed_QIDs = set([x['qid'] for x in exp_result2 if x['answer_acc']==False])

    exp1_failed_QIDs = set([x['qid'] for x in exp_result1])
    exp2_failed_QIDs = set([x['qid'] for x in exp_result2])

    print(len(exp1_failed_QIDs))
    print(len(exp2_failed_QIDs))

    print('exp1 failed only:', len(exp1_failed_QIDs-exp2_failed_QIDs))
    print('exp2 failed only:', len(exp2_failed_QIDs-exp1_failed_QIDs))
    print('exp1 and exp2 both failed:', len(exp1_failed_QIDs & exp2_failed_QIDs))
    print('exp1 failed qids:',list(exp1_failed_QIDs-exp2_failed_QIDs)[:10])
    print('exp2 failed qids:',list(exp2_failed_QIDs-exp1_failed_QIDs)[:10])


if __name__=='__main__':
    # analyse_candEnt_failed()
    # exp_name = "nlq_candEnt"
    # exp_name = "newqdt_candEnt"
    # exp_name = "qdt_candEnt"

    # exp_name = "nlq_only"
    # exp_name = "nlq_newqdt_candEnt"
    # analyse_denormalize_failed(exp_name)

    # 对比两者差异
    # exp_name1 = "nlq_newqdt_candEnt"
    # exp_name2 = "nlq_newqdt_candEnt_candRel_top1"
    # compare_gen_failed(exp_name1,exp_name2)


    # predict_file_name = 'exps/gen_multitask/CWQ_not_lower_10epoch/beam_50_top_k_predictions.json'
    # evaluate_clf_res(predict_file_name)
    
    # gen_failed_file_name = 'results/error_analysis/CWQ_not_lower_10epoch_gen_failed/res_map.json'
    # error_analysis(gen_failed_file_name)

    #exp1_result_file = 'exps/final/CWQ_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json'
    exp1_result_file = 'exps/final/CWQ_generation_structure_concat_add_prefix_warmup_epochs_5_15epochs/beam_50_top_k_predictions.json_gen_failed_results.json'
    #exp2_result_file = 'exps/final/CWQ_generation_structure_concat_add_prefix_warmup_epochs_5_15epochs/beam_50_top_k_predictions.json_gen_sexpr_results.json_new.json'
    exp2_result_file = 'exps/final/CWQ_relation_entity_concat_add_prefix_warmup_epochs_5_15epochs/beam_50_top_k_predictions.json_gen_failed_results.json'

    compare_failed_QIDs(exp1_result_file, exp2_result_file)
