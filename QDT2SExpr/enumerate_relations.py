from typing import Set
import os
from inputDataset.disamb_dataset import _MODULE_DEFAULT
from inputDataset.gen_dataset import _tokenize_relation
from components.utils import dump_json, load_json
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize


from enumerate_candidates import USE_MASTER_CONFIG, arrange_disamb_results_in_lagacy_format, generate_candidate_file
from executor.cached_enumeration import (
    CacheBackend, 
    OntologyInfo,
    _grail_valid_intermediate_type_for_joining, 
    grail_rm_redundancy_adjancent_relations,
    grail_rm_redundancy_two_hop_paths,
    legal_relation,
    relations_info,
    resolve_cvt_sub_classes
)
from executor.sparql_executor import get_2hop_relations_with_odbc, get_adjacent_relations, get_2hop_relations, get_adjacent_relations_with_odbc
from components.grail_utils import extract_mentioned_entities, extract_mentioned_entities_from_sparql
from functools import partial
from tqdm import tqdm
import argparse


def cwq_enum_one_hop_one_entity_candidates(entity: str, use_master=True):
    if (CacheBackend.cache is not None):
        in_relations_e, out_relations_e = CacheBackend.cache.query_relations(entity)
    else:  # online executing the sparql query
        in_relations_e, out_relations_e = get_adjacent_relations_with_odbc(entity)
    
    in_relations_e, out_relations_e = grail_rm_redundancy_adjancent_relations(in_relations_e, out_relations_e, use_master=use_master)

    dataset = 'CWQ'
    
    # sub_domains = []
    relations = []
    
    if len(in_relations_e) > 0:
        for r in in_relations_e:
            if not legal_relation(r, dataset):
                continue

            relations.append(r)
            # domain_r = relations_info[r][0]
            # sub_domains.extend(resolve_cvt_sub_classes(domain_r,dataset))
        
    if len(out_relations_e) > 0:
        for r in out_relations_e:
            if not legal_relation(r, dataset):
                continue
            relations.append(r)
            # range_r = relations_info[r][1]
            # sub_domains.extend(resolve_cvt_sub_classes(range_r, dataset))
        
    return relations

def cwq_enum_two_hop_entity_candidates(entity:str, use_master=True):
    if (CacheBackend.cache is not None):
        paths = CacheBackend.cache.query_two_hop_paths(entity)
    else:
        paths = get_2hop_relations_with_odbc(entity)[2]
    
    paths = grail_rm_redundancy_two_hop_paths(paths, use_master)
    
    two_hop_candidates = []

    dataset='CWQ'
    for path in paths:
        if path[0][-2:] == "#R":
            if not legal_relation(path[0][:-2], dataset):
                continue
            relation0 = '(R ' + path[0][:-2] + ')'
            # relation0 = path[0][:-2]
            intermidiate_type = relations_info[path[0][:-2]][1]
        else:
            if not legal_relation(path[0], dataset):
                continue
            relation0 = path[0]
            intermidiate_type = relations_info[path[0]][0]

        if not _grail_valid_intermediate_type_for_joining(intermidiate_type):
            continue
        
        if path[1][-2:] == "#R":
            if not legal_relation(path[1][:-2], dataset):
                continue
            # typ = relations_info[path[1][:-2]][1]
            relation1 = '(R ' + path[1][:-2] + ')'
            # relation0 = path[1][:-2]
        else:
            if not legal_relation(path[1], dataset):
                continue
            # typ = relations_info[path[1]][0]
            relation1 = path[1]
        
        # sub_typs = [typ]

        two_hop_candidates.extend([relation0,relation1])

    return two_hop_candidates


def enumerate_relation_paths_from_entities(entities, use_master=True):
    relations = []
    if len(entities)>0:
        for entity in entities:
            
            # one hop relations
            relations.extend(cwq_enum_one_hop_one_entity_candidates(entity, use_master=use_master))
            
            # two hop relations
            relations.extend(cwq_enum_two_hop_entity_candidates(entity, use_master=use_master))
    return relations


def process_single_item(item, el_results,use_master=True):
    item['ID'] = str(item['ID'])
    # print(f"ID: {item['ID']}")
    
    if el_results is None:
        entities = []
        entity_map = {}
        entities_in_gt = {}
        if 'sparql' in item:
            sparql = item['sparql']
            entities_in_gt = set(extract_mentioned_entities_from_sparql(sparql))
        
        if 'SExpr' in item and entities_in_gt is None:
            SExpr = item['SExpr']
            entities_in_gt = set(extract_mentioned_entities(SExpr))
        
        entities = list(entities_in_gt)
    else:
        entity_map = el_results[item['ID']]['entities']
        entities = sorted(set(entity_map.keys()), key=lambda k: entity_map[k]["start"])
        
    
    candidate_relations = enumerate_relation_paths_from_entities(entities,use_master=use_master)

    question = item['question']
    question = question.lower()
    # question_tokens = word_tokenize(query.lower())

    proc_query_tokens = word_tokenize(question.lower())

    if _MODULE_DEFAULT.RELATION_FREQ is None:
        _MODULE_DEFAULT.RELATION_FREQ = load_json(_MODULE_DEFAULT.RELATION_FREQ_FILE)

    def key_func(r):
        """get the ranking key of relation r"""
        r_tokens = _tokenize_relation(r)
        overlapping_val = len(set(proc_query_tokens) & set(r_tokens))
        return(
            _MODULE_DEFAULT.RELATION_FREQ.get(r,1),
            -overlapping_val
        )
    
    candidate_relations = sorted(candidate_relations, key=lambda x: key_func(x))
    candidate_relations = candidate_relations[:20] # retain 10 relations

    return {'ID':item['ID'], 'Candidate relations': candidate_relations }



def generate_candidate_relations(dataset_file, el_results):
    file_contents = load_json(dataset_file)

    process_func = partial(process_single_item, el_results=el_results, use_master=USE_MASTER_CONFIG)
    candidates_info = []
    for i,item in tqdm(enumerate(file_contents), total=len(file_contents)):
        # print(f'example: {i}')
        candidates_info.append(process_func(item))
    
    return candidates_info


def enumerate_relations_from_entity(split, pred_file)->Set[str]:
    if split == 'train':
        el_results = None
    else:
        el_results = arrange_disamb_results_in_lagacy_format(split, pred_file)

    dataset_file = os.path.join('data', f'CWQ_{split}_expr.json')

    CacheBackend.init_cache_backend('CWQ')
    OntologyInfo.init_ontology_info('CWQ')

    candidate_relations_info = generate_candidate_relations(dataset_file, el_results)

    CacheBackend.exit_cache_backend()

    dump_json(candidate_relations_info, f'data/CWQ_{split}_candidate_relations.json')
    print('Candidate Relations are solved to '+ f'data/CWQ_{split}_candidate_relations.json')


def build_relation_classification_datset(split):
    label_maps = load_json(f"data/label_maps/CWQ_{split}_label_maps.json")
    gold_ent_maps = {x:y['entity_label_map'] for x,y in label_maps.items()}
    gold_type_maps = {x:y['type_label_map'] for x,y in label_maps.items()}
    gold_rel_maps = {x:y['rel_label_map'] for x,y in label_maps.items()}

    data_set = load_json(f"data/origin/ComplexWebQuestions_{split}.json")
    data_bank = {x["ID"]:x for x in data_set}



    pred_el_results = load_json(f"data/linking_results/merged_CWQ_{split}_linking_results.json")

    CacheBackend.init_cache_backend('CWQ')
    OntologyInfo.init_ontology_info('CWQ')

    rel_classify_data = {}
    for qid in tqdm(gold_ent_maps,total=len(gold_ent_maps),desc=f'Processing {split}'):
        
        gold_rel_set = gold_rel_maps[qid].keys()
        # modify bad relation
        gold_rel_set = set([rel.split('\t')[0] if '\t' in rel else rel for rel in gold_rel_set])
        gold_entities = gold_ent_maps[qid].keys()
        gold_types = gold_type_maps[qid].keys()
        
        gold_entities = gold_entities-gold_types
        
        pred_entities = pred_el_results[qid].keys()

        # entities = gold_entities | pred_entities

        gold_ent_relations = set(enumerate_relation_paths_from_entities(gold_entities,use_master=True))

        if split=='test' or split=='dev':
            cand_ent_relations = set(enumerate_relation_paths_from_entities(pred_entities,use_master=True))
            relations = gold_ent_relations | cand_ent_relations
        else:
            # for train, use gold entity for enumeration
            cand_ent_relations = None
            relations = gold_ent_relations

        

        # remove "(R xxx.xxx.xxx)"
        relations = [rel.split("(R ")[-1].replace(")","") if "(R " in rel else rel for rel in relations]
        
        if split=='test' or split=='dev':
            cand_ent_relations = [rel.split("(R ")[-1].replace(")","") if "(R " in rel else rel for rel in cand_ent_relations]
        else:
            cand_ent_relations = relations

        # print(relations)
        # positive_rels = set()
        negative_rels = set()
        for rel in relations:
            if rel not in gold_rel_set:
                negative_rels.add(rel)
            # else:
            #     positive_rels.add(rel)

        rel_classify_data[qid] = {
            "question": data_bank[qid]["question"],
            "positive_rels": list(set(gold_rel_set)),
            "negative_rels": list(set(negative_rels)),
            "cand_rels": list(set(cand_ent_relations))
        }

        if (len(rel_classify_data))%10000==0:
            print(f'Processed {len(rel_classify_data)} examples, saving.')
            dump_json(rel_classify_data,f"data/rel_match/CWQ_{split}_relation_data.json",indent=4)
            CacheBackend.cache.save()

    CacheBackend.exit_cache_backend()

    dump_json(rel_classify_data,f"data/rel_match/CWQ_{split}_relation_data.json",indent=4)
        


    # print(rel_classify_data[qid])
    
    



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, help='split to operate on')
    parser.add_argument('--pred_file', default=None, help='prediction file')
    parser.add_argument('--server_ip', default=None, help='server ip for debugging')
    parser.add_argument('--server_port', default=None, help='server port for debugging')
    args = parser.parse_args()
    if args.split != 'train':
        if args.pred_file is None:
            raise RuntimeError('A prediction file is required for evaluation and prediction (when split is not Train)')

    print('split', args.split, 'prediction', args.pred_file)
    return args

if __name__ == '__main__':

    # split = 'train'
    for split in ['test','dev','train']:
        build_relation_classification_datset(split)

    """
    args = _parse_args() 

    # args.server_ip = '0.0.0.0'
    # args.server_port = '12345'
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    
    enumerate_relations_from_entity(split=args.split, pred_file=args.pred_file)
    """
