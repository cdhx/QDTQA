#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   get_final_entity_linking.py
@Time    :   2022/02/25 14:16:14
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   Merge the entity linking results from multiple linkers
'''

# here put the import lib

from typing import DefaultDict, OrderedDict
from enumerate_candidates import arrange_disamb_results_in_lagacy_format
from components.utils import dump_json, load_json
from components.grail_utils import extract_mentioned_entities_from_sparql, extract_mentioned_relations_from_sparql
from executor.sparql_executor import execute_query_with_odbc, get_label, get_label_with_odbc, get_types, get_types_with_odbc
import os
from tqdm import tqdm


def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r


def load_entity_relation_type_label_from_dataset(split):
    train_databank =load_json(f"data/origin/ComplexWebQuestions_{split}.json")

    global_ent_label_map = {}
    global_rel_label_map = {}
    global_type_label_map = {}

    dataset_merged_label_map = {}

    for data in tqdm(train_databank, total=len(train_databank), desc=f"Processing {split}"):
        # print(data)
        qid = data['ID']
        sparql = data['sparql']

        ent_label_map = {}
        rel_label_map = {}
        type_label_map = {}

        # extract entity labels
        gt_entities = extract_mentioned_entities_from_sparql(sparql=sparql)
        for entity in gt_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)
            ent_label_map[entity] = entity_label
            global_ent_label_map[entity] = entity_label

            if is_type:
                type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

        # extract relation labels
        gt_relations = extract_mentioned_relations_from_sparql(sparql)
        for rel in gt_relations:
            linear_rel = _textualize_relation(rel)
            rel_label_map[rel] = linear_rel
            global_rel_label_map[rel] = linear_rel
        
        dataset_merged_label_map[qid] = {
            'entity_label_map':ent_label_map,
            'rel_label_map':rel_label_map,
            'type_label_map':type_label_map
        }

    dir_name = "data/label_maps"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    dump_json(dataset_merged_label_map,f'{dir_name}/CWQ_{split}_label_maps.json',indent=4)    

    dump_json(global_ent_label_map, f'{dir_name}/CWQ_{split}_entity_label_map.json',indent=4)
    dump_json(global_rel_label_map, f'{dir_name}/CWQ_{split}_relation_label_map.json',indent=4)
    dump_json(global_type_label_map, f'{dir_name}/CWQ_{split}_type_label_map.json',indent=4)

    print("done")


def merge_entity_linking_results(split):
    # print(f'Processing {split}')

    # get entity linking by facc1 and ranking
    # pred_file = f"results/disamb/CWQ_{split}/predictions.json"
    # facc1_el_results = arrange_disamb_results_in_lagacy_format(split, pred_file)
    # dump_json(facc1_el_results,f"data/linking_results/{split}_facc1_el_results.json", indent=4)

    elq_el_results = load_json(f"data/linking_results/CWQ_{split}_entities_elq.json")
    facc1_el_results = load_json(f"data/linking_results/CWQ_{split}_facc1_el_results.json")

    merged_el_results = {}

    for qid in tqdm(facc1_el_results, total=len(facc1_el_results), desc=f"Processing {split}"):
        facc1_pred = facc1_el_results[qid]['entities']
        elq_pred = elq_el_results[qid]

        ent_map = {}
        label_mid_map = {}
        for ent in facc1_pred:
            label = get_label_with_odbc(ent)
            ent_map[ent]={
                "label": label,
                "mention": facc1_pred[ent]["mention"],
                "perfect_match": label.lower()==facc1_pred[ent]["mention"].lower()
            }
            label_mid_map[label] = ent
        
        for ent in elq_pred:
            if ent["id"] not in ent_map:
                mid = ent['id']
                label = get_label_with_odbc(mid)

                if label in label_mid_map: # same label, different mid
                    ent_map.pop(label_mid_map[label]) # pop facc1 result, retain elq result

                if label:
                    ent_map[ent["id"]]= {
                        "label": label,
                        "mention": ent["mention"],
                        "perfect_match": label.lower()==ent['mention'].lower()
                    }
                        
        merged_el_results[qid] = ent_map

    dump_json(merged_el_results, f"data/linking_results/merged_CWQ_{split}_linking_results.json", indent=4)


def evaluate_linking_res(split, linker="facc1"):
    gold_et_bank = load_json(f"data/label_maps/CWQ_{split}_label_maps.json")
    if linker.lower()=="facc1":
        facc1_el_results = load_json(f"data/linking_results/CWQ_{split}_facc1_el_results.json")
    elif linker.lower()=="elq":
        elq_el_results = load_json(f"data/linking_results/CWQ_{split}_entities_elq.json")
    elif linker.lower()=="merged":
        merged_el_results = load_json(f"data/linking_results/merged_CWQ_{split}_linking_results.json")
    else:
        facc1_el_results = load_json(f"data/linking_results/CWQ_{split}_facc1_el_results.json")
        elq_el_results = load_json(f"data/linking_results/CWQ_{split}_entities_elq.json")

    avg_p = 0
    avg_r = 0
    avg_f = 0
    total = len(gold_et_bank)
    for qid in gold_et_bank:
        gold_et_map = gold_et_bank[qid]["entity_label_map"]
        gold_type_map = gold_et_bank[qid]["type_label_map"]
        gold_ets = gold_et_map.keys() - gold_type_map.keys()
        if linker=="facc1":
            pred_ets = facc1_el_results[qid]['entities'].keys()
        elif linker.lower()=="elq":
            pred_ets = set([x["id"] for x in elq_el_results[qid]])
        elif linker.lower()=="merged":
            pred_ets = merged_el_results[qid].keys()
        else:
            pred_ets = facc1_el_results[qid]['entities'].keys()
            pred_ets = pred_ets | set([x["id"] for x in elq_el_results[qid]])
        if len(pred_ets)==0:
            local_p,local_r,local_f = (1.0,1.0,1.0) if len(gold_ets)==0 else (0.0,0.0,0.0)
        elif len(gold_ets)==0:
            local_p,local_r,local_f = (1.0,1.0,1.0) if len(pred_ets)==0 else (0.0,0.0,0.0)
        else:
            local_p = len(pred_ets&gold_ets) /len(pred_ets)
            local_r = len(pred_ets&gold_ets) /len(gold_ets)
            local_f = 2*local_p*local_r/(local_p+local_r) if local_p+local_r >0 else 0
        avg_p+=local_p
        avg_r+=local_r
        avg_f+=local_f
        
    print(f"{linker.upper()}: AVG P:{avg_p/total}, AVG R:{avg_r/total}, AVG F:{avg_f/total}")


def get_all_type_class():
    get_all_type_sparql = """
    SELECT distinct ?resource WHERE{
    ?resource  rdf:type rdfs:Class .
    }
    """
    all_types = execute_query_with_odbc(get_all_type_sparql)
    print(f'TOTAL:{len(all_types)}')

    type_label_class = {}
    
    for fb_typ in tqdm(all_types,total=len(all_types),desc="Processing"):
        try:
            get_label_sparql = ("""
            SELECT ?label WHERE{
                """
                f'<{fb_typ}> rdfs:label ?label'
                """
                FILTER (lang(?label) = 'en')
            }
            """)
            # label = get_label_with_odbc(fb_typ)
            label = list(execute_query_with_odbc(get_label_sparql))[0]
            type_label_class[fb_typ]=label
        except Exception:
            continue
    
    

    # print(f"Get Label:{type_label_class}")

    # label_type_class = {l:t for t,l in type_label_class.items()}
    label_type_class = {}
    for t,l in type_label_class.items():
        if l not in label_type_class:
            label_type_class[l]=t
        else:
            if "ns/m." in t and not "ns/m." in label_type_class[l]:
                label_type_class[l]=t
            else:
                continue


    dump_json(type_label_class, "data/fb_type_label_class.json", indent=4)
    dump_json(label_type_class, "data/fb_label_type_class.json", indent=4)

    print("Done")


def get_candidate_entity_linking_with_logits_cwq(split):
    print(f'Preparing candidate entity linking results with logits for CWQ_{split}')
    logits_bank = load_json(f'results/disamb/CWQ_{split}/predict_logits.json')
    candidate_bank = load_json(f'data/CWQ_{split}_entities.json')
    res = OrderedDict()
    
    for qid,data in tqdm(candidate_bank.items(),total=len(candidate_bank),desc=f'Processing {split}'):
        problem_num = len(data)
        if problem_num ==0:
            res[qid]=[]
        entity_list = []

        problem_id = -1
        for problem in data:
            problem_id+=1
            logits = logits_bank.get(qid+'#'+str(problem_id),[1.0])
            for idx,cand_ent in enumerate(problem):
                logit = logits[idx]
                cand_ent['label'] = get_label_with_odbc(cand_ent['id'])
                cand_ent['logit'] = logit
                entity_list.append(cand_ent)

        entity_list.sort(key=lambda x:x['logit'],reverse=True)

        res[qid] = entity_list
    
    dump_json(res,f'data/cand_entities/CWQ_{split}_cand_entities_facc1.json',indent=4,ensure_ascii=False)


def get_candidate_entity_linking_with_logits_webqsp(split):
    print(f'Preparing candidate entity linking results with logits for WEBQSP_{split}')
    logits_bank = load_json(f'results/disamb/WEBQSP_{split}/predict_logits.json')
    candidate_bank = load_json(f'data/WEBQSP_{split}_entities.json')
    res = OrderedDict()
    
    for qid,data in tqdm(candidate_bank.items(),total=len(candidate_bank),desc=f'Processing {split}'):
        problem_num = len(data)
        if problem_num ==0:
            res[qid]=[]
        entity_list = []

        problem_id = -1
        for problem in data:
            problem_id+=1
            logits = logits_bank.get(qid+'#'+str(problem_id),[1.0]*len(problem))
            for idx,cand_ent in enumerate(problem):
                logit = logits[idx]
                cand_ent['label'] = get_label_with_odbc(cand_ent['id'])
                cand_ent['logit'] = logit
                entity_list.append(cand_ent)

        entity_list.sort(key=lambda x:x['logit'],reverse=True)

        res[qid] = entity_list
    
    dump_json(res,f'data/cand_entities/WEBQSP_{split}_cand_entities_facc1.json',indent=4,ensure_ascii=False)


def check_candidate_entities_recall(split):
    gold_label_maps = load_json(f'data/label_maps/CWQ_{split}_label_maps.json')
    candidate_entity_bank = load_json(f'data/cand_entities/CWQ_{split}_cand_entities.json')
    overall_recall = 0
    for qid,data in candidate_entity_bank.items():
        all_cand_entities = set([x['id'] for x in data])
        gold_entities = set(gold_label_maps[qid]['entity_label_map'].keys())
        gold_types = set(gold_label_maps[qid]['type_label_map'].keys())
        gold_entities = gold_entities - gold_types
        if len(gold_entities)==0:
            recall=1
        else:
            recall = len(all_cand_entities & gold_entities) / len(gold_entities)

        overall_recall+=recall
    
    print(f'Recall on {split}:',overall_recall/len(candidate_entity_bank))


def get_el_result_from_facc1_webqsp(split):
    print(f'Getting el_results of webqsp from facc1 for {split}')
    disamb_predictions_file = f'results/disamb/WEBQSP_{split}/predictions.json'
    disamb_predictions = load_json(disamb_predictions_file)

    cand_entities_file = f'data/WEBQSP_{split}_entities.json'
    cand_entities = load_json(cand_entities_file)

    el_result_map = DefaultDict(list)

    for qid,problem in cand_entities.items():
        for problem_idx,cand_ent_list in enumerate(problem):
            if f'{qid}#{problem_idx}' not in disamb_predictions:
                # no need to disambugation
                if len(cand_ent_list)>0:
                    el_result_map[qid].append(cand_ent_list[0])
            else:
                ent_idx = disamb_predictions[f'{qid}#{problem_idx}']
                el_result_map[qid].append(cand_ent_list[ent_idx])
                        
    dump_json(el_result_map,f'data/linking_results/WebQSP_{split}_facc1_el_results.json',indent=4)
    

def get_el_result_from_elq_webqsp(split):
    print(f'Getting el_results of webqsp from elq for {split}')
    elq_cand_ent_file = f'data/cand_entities/WEBQSP_test_cand_entities_elq.json'
    elq_cand_entities = load_json(elq_cand_ent_file)

    el_result_map = DefaultDict(list)
    for qid,cand_entities in elq_cand_entities.items():
        mention_entity_map = {}
        for ent in cand_entities:
            if ent['mention'] not in mention_entity_map:
                mention_entity_map[ent['mention']] = ent
        
        el_result_map[qid].extend([ent for mention,ent in mention_entity_map.items()])

    dump_json(el_result_map,f'data/linking_results/WebQSP_{split}_elq_results.json',indent=4)



def merge_el_result_webqsp(split):
    print(f'Merging el results of webqsp for {split}')
    facc1_el_results = load_json(f'data/linking_results/WebQSP_{split}_facc1_el_results.json')
    elq_el_results = load_json(f'data/linking_results/WebQSP_{split}_elq_results.json')

    origin_dataset = load_json(f'data/WebQSP/origin/WebQSP.{split}.json')


    merged_el_results = DefaultDict(dict)

    for data in origin_dataset['Questions']:
        qid = data['QuestionId']
        facc1_ent_list = facc1_el_results.get(qid,[])
        elq_ent_list = elq_el_results.get(qid,[])

        # elq entity
        for ent in elq_ent_list:
            merged_el_results[qid][ent['id']] = {
                'label': get_label_with_odbc(ent['id']),
                'mention':ent['mention'],
                'perfect_match':ent['perfect_match']
            }
        
        # facc1 entity
        for ent in facc1_ent_list:
            if ent['id'] not in merged_el_results[qid]:
                merged_el_results[qid][ent['id']] = {
                'label': get_label_with_odbc(ent['id']),
                'mention':ent['mention'],
                'perfect_match':ent['perfect_match']
            }
        
        if qid not in merged_el_results:
            merged_el_results[qid]=[]
        
    dump_json(merged_el_results, f'data/linking_results/merged_WebQSP_{split}_linking_results.json')


if __name__=='__main__':
        
    # test()
    
    # 处理数据集
    # split_list = ['test','dev','train']
    # for split in split_list:
    #     load_entity_relation_type_label_from_dataset(split)

    # 从Freebase获取所有type
    # get_all_type_class()

    
    # 处理实体链接结果
    # merge_entity_linking_results(split='test')
    # merge_entity_linking_results(split='dev')
    # merge_entity_linking_results(split='train')

    # 按链接器评估实体链接结果
    # evaluate_linking_res('test',linker="merged")

    # 返回实体链接经打分后的结果
    # get_candidate_entity_linking_with_logits_cwq('dev')
    # get_candidate_entity_linking_with_logits_cwq('test')
    # get_candidate_entity_linking_with_logits_cwq('train')

    # get_candidate_entity_linking_with_logits_webqsp('test')
    # get_candidate_entity_linking_with_logits_webqsp('train')

    # 检测候选实体链接召回率, 
    # check_candidate_entities_recall('test') # recall 0.670
    # check_candidate_entities_recall('dev') # recall 0.669
    # check_candidate_entities_recall('train') # recall 0.662

    # 获取FACC1在WebQSP上的实体链接结果
    get_el_result_from_facc1_webqsp('test')
    get_el_result_from_facc1_webqsp('train')

    # 获取ELQ在WebQSP上的实体链接结果
    get_el_result_from_elq_webqsp('test')
    get_el_result_from_elq_webqsp('train')

    # 合并FACC1和ELQ在WebQSP上的实体链接结果
    merge_el_result_webqsp('test')
    merge_el_result_webqsp('train')
    pass

