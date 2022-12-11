#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eval_topk_prediction.py
@Time    :   2022/01/08 21:26:57
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
'''

# here put the import lib
import argparse
from ast import dump
from cmath import exp
from importlib.resources import path
from math import fabs
from pickle import load
from typing import OrderedDict
from cwq_evaluate import cwq_evaluate_valid_results
from components.utils import dump_json, load_json
from tqdm import tqdm
from executor.sparql_executor import execute_query_with_odbc, execute_query_with_odbc_filter_answer, get_label, execute_query
from executor.logic_form_util import lisp_to_sparql
import re
import json
import os
from entity_linker import surface_index_memory

def is_value_tok(t):
    if t[0].isalpha():
        return False
    return (process_literal(t) != 'null')

# copied from grail value extractor
def process_literal(value: str):  # process datetime mention; append data type
    pattern_date = r"(?:(?:jan.|feb.|mar.|apr.|may|jun.|jul.|aug.|sep.|oct.|nov.|dec.) the \d+(?:st|nd|rd|th), \d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
    pattern_datetime = r"\d{4}-\d{2}-\d{2}t[\d:z-]+"
    pattern_float = r"(?:[-]*\d+[.]*\d*e[+-]\d+|(?<= )[-]*\d+[.]\d*|^[-]*\d+[.]\d*)"
    pattern_yearmonth = r"\d{4}-\d{2}"
    pattern_year = r"(?:(?<= )\d{4}|^\d{4})"
    pattern_int = r"(?:(?<= )[-]*\d+|^[-]*\d+)"
    if len(re.findall(pattern_datetime, value)) == 1:
        value = value.replace('t', "T").replace('z', 'Z')
        return f'{value}^^http://www.w3.org/2001/XMLSchema#dateTime'
    elif len(re.findall(pattern_date, value)) == 1:
        if value.__contains__('-'):
            return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
        elif value.__contains__('/'):
            fields = value.split('/')
            value = f"{fields[2]}-{fields[0]}-{fields[1]}"
            return f'{value}^^http://www.w3.org/2001/XMLSchema#date'
    elif len(re.findall(pattern_yearmonth, value)) == 1:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#gYearMonth'
    elif len(re.findall(pattern_float, value)) == 1:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#float'
    elif len(re.findall(pattern_year, value)) == 1 and int(value) <= 2015:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#gYear'
    elif len(re.findall(pattern_int, value)) == 1:
        return f'{value}^^http://www.w3.org/2001/XMLSchema#integer'
    else:
        return 'null'


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on, can be `test`, `dev` and `train`')
    parser.add_argument('--pred_file', default=None, help='topk prediction file')
    parser.add_argument('--revise_only', action='store_true', dest='revise_only', default=False, help='only do revising')
    parser.add_argument('--server_ip', default=None, help='server ip for debugging')
    parser.add_argument('--server_port', default=None, help='server port for debugging')
    parser.add_argument('--qid',default=None,type=str, help='single qid for debug, None by default' )

    args = parser.parse_args()
    if args.pred_file is None:
        args.pred_file = f'results/gen/CWQ_{args.split}/top_k_predictions.json'

    print(f'split:{args.split}, topk_file:{args.pred_file}')
    return args


dev_el_results = load_json(f'data/CWQ_dev_entities.json')
train_el_results = load_json(f'data/CWQ_train_entities.json')
test_el_results = load_json(f'data/CWQ_test_entities.json')


def type_checker(token:str)->str:
    """Check the type of a token, e.g. Integer, Float or date.
       Return original token if no type is detected."""
    
    pattern_year = r"\d{4}"
    pattern_year_month = r"\d{4}-\d{2}"
    pattern_year_month_date = r"\d{4}-\d{2}-\d{2}"
    if re.match(pattern_year, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month_date, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    else:
        return token

    return token

def denormalize_sparql_new(normed_sparql, entity_label_map,
                                        type_label_map,
                                        rel_label_map,
                                        train_entity_map,
                                        surface_index):
    
    #label_entity_map = {l.lower():e for e,l in entity_label_map.items()}
    

    expr = normed_sparql

    # replace entity names
    # for v,e in entity_label_map.items():
    #     expr = expr.replace(v , e ) # original case
    #     expr = expr.replace(v.lower(), e) # lower case

    # expr = expr.replace('( greater equal', '( ge')
    # expr = expr.replace('( greater than', '( gt')
    # expr = expr.replace('( less equal', '( le')
    # expr = expr.replace('( less than', '( lt')

    # TODO need something to replace '{'
    expr = expr.replace('select distinct ?x where','select distinct ?x where {')
    expr = expr+" }"

    expr = expr.replace('!=',' != ')
    expr = expr.replace('[lt]','<')
    expr = expr.replace('[le]','<=')
    expr = expr.replace('[gt]','>')
    expr = expr.replace('[ge]','>=')
    
    expr = expr.replace('(',' ( ')
    expr = expr.replace(')',' ) ')
    expr = expr.replace(', ',' , ')
    expr = expr.replace('?',' ?')
    expr = expr.replace('.',' . ')
    tokens = expr.split(' ')

    print(tokens)

    segments = []
    prev_left_bracket = False
    prev_left_par = False
    cur_seg = ''

    for t in tokens:
        
        if t=='[':
            prev_left_bracket=True
            if cur_seg:
                segments.append(cur_seg)
        elif t==']':
            prev_left_bracket=False
            cur_seg = cur_seg.strip()
            
            # find in linear origin map
            processed = False
            # if linear_origin_map:
            #     for k,v in linear_origin_map.items():
            #         if k.lower() == f'[ {cur_seg} ]'.lower():
            #             cur_seg = v
            #             processed = True
            #             break

            if not processed:
                # find in label entity map
                if cur_seg.lower() in entity_label_map: # entity
                    cur_seg = entity_label_map[cur_seg.lower()]
                    processed = True
                elif cur_seg.lower() in type_label_map: # type
                    cur_seg = type_label_map[cur_seg.lower()]
                    processed = True
                else: # relation or unlinked entity
                    if ' , ' in cur_seg: 
                        if cur_seg.lower() in rel_label_map: # relation
                            cur_seg = rel_label_map[cur_seg.lower()]
                        elif cur_seg.lower() in train_entity_map: # entity in trainset
                            cur_seg = train_entity_map[cur_seg.lower()]
                        else: 
                            # try to link entity
                            cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(cur_seg,top_k=1)
                            if cand_entities:
                                cur_seg = list(cand_entities.keys())[0] # take the first entity
                            else: # view as relation
                                cur_seg = cur_seg.replace(' , ',',')
                                cur_seg = cur_seg.replace(',','.')
                                cur_seg = cur_seg.replace(' ', '_')
                            
                        cur_seg = "ns:"+cur_seg
                        processed = True
                    else:
                        # unlinked entity    
                        if cur_seg.lower() in train_entity_map:
                            cur_seg = train_entity_map[cur_seg.lower()]
                        else:
                            # keep it, time or integer
                            cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(cur_seg,top_k=1)
                            if cand_entities:
                                cur_seg = list(cand_entities.keys())[0]

                            cur_seg = cur_seg
            segments.append(cur_seg)
            cur_seg = ''
        else:
            if prev_left_bracket:
                # in a bracket
                cur_seg = cur_seg + ' '+t
            else:
                if t == '#':
                    segments.append('^^')
                else:
                    segments.append(t)
                
    # print(segments)
    expr = " ".join(segments)
    expr = expr.replace(" ( ","(")
    expr = expr.replace(" ) ",")")
    print(expr)
    return expr


def execute_normed_sparql_from_label_maps(normed_sparql, 
                                        entity_label_map,
                                        type_label_map,
                                        rel_label_map,
                                        train_entity_map,
                                        surface_index
                                        ):
    # print(normed_expr)
    try:
        denorm_sparql = denormalize_sparql_new(normed_sparql, 
                                        entity_label_map, 
                                        type_label_map,
                                        rel_label_map,
                                        train_entity_map,
                                        surface_index
                                        )
    except:
        return 'null', []
    
    query_expr = denorm_sparql.replace('( ','(').replace(' )', ')')
    if query_expr != 'null':
        try:
            # print('parse:', query_expr)
            # sparql_query = lisp_to_sparql(query_expr)
            # print('sparql:', sparql_query)
            # denotation = execute_query(sparql_query)
            denotation = execute_query_with_odbc(query_expr)
        except:
            denotation = []
 
    return query_expr, denotation


train_gen_dataset = {x['ID']:x for x in load_json(f'data/CWQ_train_expr.json')}
test_gen_dataset = {x['ID']:x for x in load_json(f'data/CWQ_test_expr.json')}
dev_gen_dataset = {x['ID']:x for x in load_json(f'data/CWQ_dev_expr.json')}


def aggressive_top_k_sparql_eval_new(split, predict_file):
    """Run top k predictions, using linear origin map"""
    predictions = load_json(predict_file)

    # print(os.path.dirname(predict_file))
    dirname = os.path.dirname(predict_file)
    
    augment_data_file = os.path.join(dirname, 'predict_data_all.json')

    augment_data = load_json(augment_data_file)

    if split=='dev':
        gen_dataset = dev_gen_dataset
    elif split=='train':
        gen_dataset = train_gen_dataset
    else:
        gen_dataset = test_gen_dataset

    use_goldEnt = "goldEnt" in predict_file # whether to use gold Entity for denormalization
    use_goldRel = "goldRel" in predict_file


    gold_label_maps = load_json(f"data/label_maps/CWQ_{split}_label_maps.json")    
    train_entity_map = load_json(f"data/label_maps/CWQ_train_entity_label_map.json")

    if not use_goldEnt:
        candidate_entity_map = load_json(f"data/linking_results/merged_CWQ_{split}_linking_results.json")
        train_type_map = load_json(f"data/label_maps/CWQ_train_type_label_map.json")
        
    if not use_goldRel:
        train_relation_map = load_json(f"data/label_maps/CWQ_train_relation_label_map.json")
    
    # load FACC1 Index
    surface_index = None
    # surface_index = surface_index_memory.EntitySurfaceIndexMemory(
    #     "entity_linker/data/entity_list_file_freebase_complete_all_mention", "entity_linker/data/surface_map_file_freebase_complete_all_mention",
    #     "entity_linker/data/freebase_complete_all_mention")

    
    ex_cnt = 0
    top_hit = 0
    lines = []
    failed_preds = []
    denormalize_failed = []

    gen_executable_cnt = 0
    final_executable_cnt = 0
    processed = 0
    for (qid,pred) in tqdm(predictions.items(), total = len(predictions)):
        
        denormed_pred = []

        # In RnG, they use disambiguated entities
        # entity_label_map = get_entity_mapping_from_el_results(split,qid,entity_dataset)
        if not use_goldEnt:
            entity_label_map = {e:l['label'] for e,l in candidate_entity_map[qid].items()}
            type_label_map = train_type_map
        else:
            # goldEnt label map
            entity_label_map = gold_label_maps[qid]['entity_label_map']
            type_label_map = gold_label_maps[qid]['type_label_map']
        
        if not use_goldRel:
            rel_label_map = train_relation_map # not use gold Relation, use relations from train
        else:
            # goldRel label map
            rel_label_map = gold_label_maps[qid]['rel_label_map']

        
        # exchange label and mid
        train_entity_map = {l.lower():e for e,l in train_entity_map.items()}
        entity_label_map = {l.lower():e for e,l in entity_label_map.items()}
        rel_label_map = {l.lower():r for r,l in rel_label_map.items()}
        type_label_map = {l.lower():t for t,l in type_label_map.items()}


        gen_feat = gen_dataset[qid]
        # linear_origin_map = augment_data[qid]['linear_origin_map']
        # normed_sexpr_exmatch = augment_data[qid]['ex_match']
        # gt_sexpr = augment_data[qid]['gt_s_expr']
        gt_normed_sparql = augment_data[qid]['normed_s_expr']

        
        found_executable = False

        # find the first executable lf
        for rank, p in enumerate(pred):
            lf, answers = execute_normed_sparql_from_label_maps(p, entity_label_map, 
                                                type_label_map, rel_label_map, train_entity_map,
                                                surface_index)
            
            denormed_pred.append(lf)


            if answers:
                lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer':answers}))
                found_executable = True
                if rank==0:
                    top_hit +=1
                break

        
        if found_executable:
            # found executable query from generated model
            gen_executable_cnt +=1
        else:
            # none executable query found
            failed_preds.append({'qid':qid, 
                                'gt_normed_sparql': gt_normed_sparql,
                                'pred': pred, 
                                'denormed_pred':denormed_pred})
        
            
        if found_executable:
            final_executable_cnt+=1
        
        processed+=1
        if processed%100==0:
            print(f'Processed:{processed}, gen_executable_cnt:{gen_executable_cnt}')

        
    print('STR Match', ex_cnt/ len(predictions))
    print('TOP 1 Executable', top_hit/ len(predictions))
    print('Gen Executable', gen_executable_cnt/ len(predictions))
    print('Final Executable', final_executable_cnt/ len(predictions))

    
    # write transferred logic forms
    # dump_json(lines,os.path.join(dirname,'gen_sexpr_results.json'))
    # with open(f'results/gen/CWQ_{split}/gen_results.txt', 'w') as f:
    result_file = os.path.join(dirname,'gen_sexpr_results.txt')
    with open(result_file, 'w') as f:
            f.writelines([x+'\n' for x in lines])

    # write failed predictions
    dump_json(failed_preds,os.path.join(dirname,'gen_failed_results.json'),indent=4)
    dump_json(denormalize_failed,os.path.join(dirname,'denormlize_failed_results.json'),indent=4)
    dump_json({
        'STR Match': ex_cnt/ len(predictions),
        'TOP 1 Executable': top_hit/ len(predictions),
        'Gen Executable': gen_executable_cnt/ len(predictions),
        'Final Executable': final_executable_cnt/ len(predictions)
    }, os.path.join(dirname,'statistics.json'),indent=4)

    # evaluate
    args.pred_file = result_file
    cwq_evaluate_valid_results(args)

    
def aggressive_top_k_eval_for_qid_new(split, predict_file, qid):
    """Run top k predictions one by one to get the executable logical form specially for qid"""
    predictions = load_json(predict_file)

    # print(os.path.dirname(predict_file))
    dirname = os.path.dirname(predict_file)
    
    augment_data_file = os.path.join(dirname, 'predict_data_all.json')

    augment_data = load_json(augment_data_file)

    if split=='dev':
        gen_dataset = dev_gen_dataset
    elif split=='train':
        gen_dataset = train_gen_dataset
    else:
        gen_dataset = test_gen_dataset

    use_goldEnt = "goldEnt" in predict_file # whether to use gold Entity for denormalization
    use_goldRel = "goldRel" in predict_file


    gold_label_maps = load_json(f"data/label_maps/CWQ_{split}_label_maps.json")    
    train_entity_map = load_json(f"data/label_maps/CWQ_train_entity_label_map.json")

    if not use_goldEnt:
        candidate_entity_map = load_json(f"data/linking_results/merged_CWQ_{split}_linking_results.json")
        train_type_map = load_json(f"data/label_maps/CWQ_train_type_label_map.json")
        
    if not use_goldRel:
        train_relation_map = load_json(f"data/label_maps/CWQ_train_relation_label_map.json")
    
    # load FACC1 Index
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "entity_linker/data/entity_list_file_freebase_complete_all_mention", "entity_linker/data/surface_map_file_freebase_complete_all_mention",
        "entity_linker/data/freebase_complete_all_mention")


    denormed_pred = []

    # In RnG, they use disambiguated entities
    # entity_label_map = get_entity_mapping_from_el_results(split,qid,entity_dataset)
    if not use_goldEnt:
        entity_label_map = {e:l['label'] for e,l in candidate_entity_map[qid].items()}
        type_label_map = train_type_map
    else:
        # goldEnt label map
        entity_label_map = gold_label_maps[qid]['entity_label_map']
        type_label_map = gold_label_maps[qid]['type_label_map']
        
    if not use_goldRel:
        rel_label_map = train_relation_map
    else:
        # goldRel label map
        rel_label_map = gold_label_maps[qid]['rel_label_map']

        
    # exchange label and mid
    train_entity_map = {l.lower():e for e,l in train_entity_map.items()}
    entity_label_map = {l.lower():e for e,l in entity_label_map.items()}
    rel_label_map = {l.lower():r for r,l in rel_label_map.items()}
    type_label_map = {l.lower():t for t,l in type_label_map.items()}


    gen_feat = gen_dataset[qid]
    # linear_origin_map = augment_data[qid]['linear_origin_map']
    normed_sexpr_exmatch = augment_data[qid]['ex_match']
    gt_sexpr = augment_data[qid]['gt_s_expr']
    gt_normed_sexpr = augment_data[qid]['normed_s_expr']

        
    found_executable = False

    # find the first executable lf
    pred = predictions[qid]
    lines = []
    ex_cnt = 0
    top_hit = 0
    for rank, p in enumerate(pred):
        lf, answers = execute_normed_sparql_from_label_maps(
                                            p, 
                                            entity_label_map, 
                                            type_label_map, 
                                            rel_label_map, 
                                            train_entity_map,
                                            surface_index)
        
        denormed_pred.append(lf)

        if rank == 0 and lf.lower() ==gen_feat['SExpr'].lower():
            ex_cnt +=1
        
        if rank == 0 and lf ==gen_feat['SExpr']:
            ex_cnt +=1
        
        if answers:
            lines.append(json.dumps({'qid': qid, 'logical_form': lf, 'answer':answers}))
            found_executable = True
            if rank==0:
                top_hit +=1
            break 
        
        if found_executable:
            # found executable query from generated model
            print('Sussessfully executed')
        
    print(f'ex_cnt:{ex_cnt}, top_hit:{top_hit}, predict result:{lines}')

    

if __name__=='__main__':
    """go down the top-k list to get the first executable locial form"""
    args = _parse_args()

    args.server_ip = '0.0.0.0'
    args.server_port = 12346

    # args.split = "test"
    # args.pred_file = "results/gen/CWQ_test_nlq_newqdt_candEnt_candRel_top1/top_k_predictions.json"
    
    # args.qid = "WebQTrn-60_68f0d0ad309d64a4af858a5ef4fb5713"
    # args.qid = "WebQTrn-3100_143c89d70679c3e5257c93d8e2bc4c67"
    # args.qid = "WebQTest-743_0a8cdba29cf260283b7c890b3609c0b9"
    # args.qid = "WebQTest-61_7bd6b37d01a372aa5af2a8d5ef53d827"

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    

    if args.qid:
        aggressive_top_k_eval_for_qid_new(args.split,args.pred_file,args.qid)
    else:
        # aggressive_top_k_eval(args.split, args.pred_file)
        aggressive_top_k_sparql_eval_new(args.split, args.pred_file)
        
        

        
    