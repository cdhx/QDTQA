"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import multiprocessing
import os
from typing import OrderedDict
import torch
from os.path import join
import sys
import argparse
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial

from components.utils import *
from executor.cached_enumeration import (
    CacheBackend,
    OntologyInfo,
    grail_enum_one_hop_one_entity_candidates,
    grail_enum_two_hop_one_entity_candidates,
    generate_all_logical_forms_for_literal,
    grail_enum_two_entity_candidates,
    grail_canonicalize_expr,
)
# get entity ambiguation prediction
from entity_linker.CWQ_Value_Extractor import CWQ_Value_Extractor
from inputDataset.disamb_dataset import (
    read_disamb_instances_from_entity_candidates,
)
from components.expr_parser import extract_entities, extract_relations

from grail_evaluate import process_ontology, SemanticMatcher
from nltk.tokenize import word_tokenize


MP_POOL_SIZE = 5
# True: use master property, False: use reserve property, None: do nothing
USE_MASTER_CONFIG = None

def _process_query(query):
    tokens = word_tokenize(query)
    proc_query = ' '.join(tokens).replace('``', '"').replace("''", '"')
    return proc_query

cnt = 0
def arrange_disamb_results_in_lagacy_format(split_id, entity_predictions_file):
    """process entity disambugation results in legacy format , return el_results"""
    dataset_id = 'CWQ'
    example_cache = join('feature_cache', f'{dataset_id}_{split_id}_disamb_example.bin')
    entities_file = f'data/CWQ_{split_id}_entities.json'
    if os.path.exists(example_cache):
        instances = torch.load(example_cache)
    else:
        dataset_file = join('data', f'CWQ_{split_id}_expr.json')
        instances = read_disamb_instances_from_entity_candidates(dataset_file, entities_file)
        torch.save(instances, example_cache)

    # build result index
    indexed_pred = load_json(entity_predictions_file) # predictions made by disambugation model
    # for (feat, pred) in zip(valid_features, predicted_indexes):

    el_results = OrderedDict()
    for inst in instances:
        inst_result = {}
        normed_query = _process_query(inst.query)
        inst_result['question'] = normed_query
        pred_entities = OrderedDict()
        for problem in inst.disamb_problems:
            if len(problem.candidates) == 0:
                continue
            if len(problem.candidates) == 1 or problem.pid not in indexed_pred: # no prediction
                pred_idx = 0 # use the first candidate by default
            else:
                # print('using predicted entity linking')
                pred_idx = indexed_pred[problem.pid]

            entity = problem.candidates[pred_idx]
            start_pos = normed_query.find(problem.mention) # -1 or strat
            pred_entities[entity.id] = {
                "mention": problem.mention,
                "label": entity.label,
                "friendly_name": entity.facc_label,
                "start": start_pos,
            }
        inst_result['entities'] = pred_entities
        el_results[inst.qid] = inst_result
        # pred_entities.append(problem.candidates[pred_idx].id)

    return el_results

def enumerate_candidates_from_entities_and_literals(entities, literals, use_master):
    """Enumerate candidate queries from entities and literals"""
    logical_forms = []
    # print(entities)
    # print(literals)
    if len(entities) > 0:
        for entity in entities:
            # enumerate one-hop-one-entity candidate queries
            logical_forms.extend(grail_enum_one_hop_one_entity_candidates(entity, use_master=use_master))
            # enumerate two-hop-one-entity candidate queries
            lfs_2 = grail_enum_two_hop_one_entity_candidates(entity, use_master=use_master)
            logical_forms.extend(lfs_2)
    if len(entities) == 2:
        # enumerate two-entity candidates
        logical_forms.extend(grail_enum_two_entity_candidates(entities[0], entities[1], use_master=use_master))
    
    # enumerate candidate queries from literal
    for literal in literals:
        logical_forms.extend(
            generate_all_logical_forms_for_literal(literal))

    return logical_forms

def process_single_item(item, el_results, extractor, use_master=True):
    """enumerate candidates for single item"""
    
    item['ID'] = str(item['ID'])
    print(f"ID: {item['ID']}")
    
    # no el results provided, using gt
    if el_results is None:
        # TODO el_results is None, use ground truth entity
        entities = []
        entity_map = {}
        for node in item['graph_query']['nodes']:
            if node['node_type'] == 'entity':
                if node['id'] not in entities:
                    entities.append(node['id'])
                    entity_map[node['id']] = ' '.join(
                        node['friendly_name'].replace(";", ' ').split()[:5])
        literals = []
        for node in item['graph_query']['nodes']:
            if node['node_type'] == 'literal' and node['function'] not in ['argmin', 'argmax']:
                if node['id'] not in literals:
                    literals.append(node['id'])
        if len(entities) > 1:
            normed_query = _process_query(item['question'])
            entities = sorted(entities, key=lambda k: normed_query.find(entity_map[k].lower()))
    # using el results, for testing    
    else:
        # find entity linking
        entity_map = el_results[item['ID']]['entities']
        entities = sorted(set(entity_map.keys()), key=lambda k: entity_map[k]["start"])

        # print("linked entities:", entities)
        # literals = set()
        
        # detect value mentions
        mentions = extractor.detect_mentions(item['question'])
        mentions = [extractor.process_literal(m) for m in mentions]
        literals = []
        for m in mentions:
            if m not in literals:
                literals.append(m)
    
    if 'SExpr' in item:
        # contains gold s_expression
        canonical_expr = grail_canonicalize_expr(item['SExpr'], use_master=use_master)
        # make logical forms
        logical_forms = enumerate_candidates_from_entities_and_literals(entities, literals, use_master=use_master)
        # if len(logical_forms) == 0:
        #     continue
        return {'ID': item['ID'], 'canonical_expr': canonical_expr, 'SExpr': item['SExpr'], 'candidates': logical_forms}
    else: # no canonical expression
        canonical_expr = 'null'
        logical_forms = enumerate_candidates_from_entities_and_literals(entities, literals, use_master=use_master)
        return {'ID': item['ID'], 'canonical_expr': canonical_expr, 'SExpr': 'null', 'candidates': logical_forms}


# generate candidates
def generate_candidate_file(dataset_file, el_results, is_parallel=False):
    """generate candidates from entity linking results
    Args:
        dataset_file: origin dataset file
        el_results: entity linking results
        is_parallel: whether to execute parallelly
    """
    extractor = CWQ_Value_Extractor()

    # el_fn = "graphq_el.json" if _gq1 else "grailqa_el.json"
    file_contents = load_json(dataset_file)

    process_func = partial(process_single_item, el_results=el_results, extractor=extractor, use_master=USE_MASTER_CONFIG)
    if is_parallel:
        CacheBackend.multiprocessing_preload()
        candidates_info = []
        with Pool(MP_POOL_SIZE) as p:
            candidates_info = p.map(process_func, file_contents, chunksize=100)
            # candidates_info = p.map(process_func, file_contents)
    else:
        candidates_info = []
        # file_contents = file_contents[:2] # for debug
        for i, item in enumerate(tqdm(file_contents)):
            print(f'example: {i}')
            candidates_info.append(process_func(item))
    candidates_info = [x for x in candidates_info if len(x['candidates'])]
    candidate_numbers = [len(x['candidates']) for x in candidates_info]
    print('AVG candidates', sum(candidate_numbers) / len(candidate_numbers), 'MAX', max(candidate_numbers))
    is_str_covered = [x['SExpr'] in x['candidates'] for x in candidates_info]
    print('Str coverage of orig expr', sum(is_str_covered) / len(is_str_covered), len(is_str_covered))
    is_str_covered = [x['canonical_expr'] in x['candidates'] for x in candidates_info]
    print('Str coverage of canonical expr', sum(is_str_covered) / len(is_str_covered), len(is_str_covered))
    is_str_same = [x['canonical_expr'] == x['SExpr'] for x in candidates_info]
    print('Canonical expr same with Orig ', sum(is_str_same) / len(is_str_covered), len(is_str_covered))
    return candidates_info

def pick_closest_target_expr(gt_expr, alter_exprs):
    gt_relations = set(extract_relations(gt_expr))
    
    sort_keys = []
    for expr in alter_exprs:
        e_relations = set(extract_relations(expr))
        r_dist = -len(gt_relations & e_relations) * 1.0 / len(gt_relations | e_relations)
        len_dist = -abs(len(expr) - len(gt_expr))
        # first relation overlapping then length difference
        sort_keys.append((r_dist, len_dist))
    print(sort_keys)
    selected_idx = min(list(range(len(alter_exprs))), key=lambda x: sort_keys[x])
    return alter_exprs[selected_idx]

def augment_edit_distance(candidates_info):
    """augment logic forms with """
    reverse_properties, relation_dr, relations, upper_types, types = process_ontology('ontology/fb_roles', 'ontology/fb_types', 'ontology/reverse_properties')
    matcher = SemanticMatcher(reverse_properties, relation_dr, relations, upper_types, types)
    hit_chance = 0
    ex_chance = 0
    count = 0
    augmented_lists = []
    for i, instance in enumerate(candidates_info):
        candidates = instance['candidates']
        gt = instance['canonical_expr']
        print(f'example: {i}, candidate num: {len(candidates)}')
        aux_candidates = []
        for c in candidates:
            if gt == 'null':
                ex = False
            else:
                ex = matcher.same_logical_form(gt, c)
            # tokens = []
            aux_candidates.append({'logical_form': c, 'ex': ex,})
        is_covered = any([x['ex'] for x in aux_candidates])
        hit_chance += is_covered
        is_exact = any([x['logical_form'] == gt for x in aux_candidates])
        ex_chance += is_exact

        if is_covered and not is_exact:
            # use relation overlapping to select the set with the
            alter_targets = [x['logical_form'] for x in aux_candidates if x['ex']]
            if len(alter_targets) == 1:
                target_expr = alter_targets[0]
            else:
               
                # exit()
                selected = pick_closest_target_expr(gt, alter_targets)
                target_expr = selected
        else:
            target_expr = gt

        instance['candidates'] = aux_candidates
        instance['target_expr'] = target_expr
        count += 1
        augmented_lists.append(instance)
    print('Coverage', hit_chance, count, hit_chance / count)
    print('Exact', ex_chance, count, ex_chance / count)
    return augmented_lists

# run evaluation
def sanity_check_proced_question():
        # el_fn = "graphq_el.json" if _gq1 else "grailqa_el.json"
    legacy_results = load_json("entity_linking/grailqa_el.json")
    new_results = load_json('tmp/tmp_el_results.json')
    count = 0
    for qid, leg_res in legacy_results.items():
        if qid not in new_results:
            continue
        new_res = new_results[str(qid)]
        if leg_res['question'] != new_res['question']:
            print('------------------------')
            print(new_res['question'])
            print(leg_res['question'])
            count += 1
    print(count)

def enumerate_candidates_for_ranking(split, pred_file):
    if split == 'train':
        el_results = None
    else:
        el_results = arrange_disamb_results_in_lagacy_format(split, pred_file)
        # temporalily save in case sth wrong
        # dump_json(el_results, f'tmp/tmp_el_results_{split}.json', indent=2)

    dataset_file = join('data', f'CWQ_{split}_expr.json')

    CacheBackend.init_cache_backend('CWQ')
    OntologyInfo.init_ontology_info('CWQ')
    candidates_info = generate_candidate_file(dataset_file, el_results)
    CacheBackend.exit_cache_backend()

    dump_json(candidates_info, f'misc/CWQ_{split}_candidates_intermidiate.json')
    # candidates_info = load_json(f'misc/grail_{split}_candidates_intermidiate.json')
    augmented_lists = augment_edit_distance(candidates_info)
    with open(f'outputs/CWQ_{split}_candidates-ranking.jsonline', 'w') as f:
        for info in augmented_lists:
            f.write(json.dumps(info) + '\n')

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
    # assert len(sys.argv) >= 2
    args = _parse_args()  
    
    args.server_ip = '0.0.0.0'
    args.server_port = 12345
    
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    enumerate_candidates_for_ranking(args.split, args.pred_file)
