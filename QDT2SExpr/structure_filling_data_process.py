from collections import defaultdict
import re 
import json
import os

from executor.sparql_executor import get_label_with_odbc
from components.xwu_expr_parser import parse_s_expr

from transformers import AutoTokenizer

"""
structure generation 数据构造: 
 - 把 merged_all_data 目录下的数据文件复制到 structure_filling 目录下
 - 调用 add_masked_sexpr_to_dataset()
"""


extra_relations_prefix = ['topic_server.schemastaging_corresponding_entities_type', 'topic_server.webref_cluster_members_type', 'topic_server.population_number']
literal_mask = '[LIT]'
entity_mask = '[ENT]'
relation_mask = '[REL]'

padding_structures = [
    "( ARGMAX ( JOIN ( R [REL] ) [ENT] ) [REL] )", 
    '( AND ( JOIN [REL] [ENT] ) ( JOIN ( R [REL] ) [ENT] ) )',
    '( JOIN ( R [REL] ) ( JOIN ( R [REL] ) [REL] ) )',
    '( JOIN ( R [REL] ) [REL] )',
    '( JOIN ( R [REL] ) ( AND ( JOIN [REL] [ENT] ) ( JOIN ( R [REL] ) [REL] ) ) )'
]

def post_process(structure):
    structure = structure.replace('<pad>', '')
    structure = structure.replace('</s>', '')
    structure = structure.replace('[REL]', ' [REL] ')
    structure = structure.replace('[ENT]', ' [ENT] ')
    structure = structure.replace('[LIT]', ' [LIT] ')
    toks = structure.split()
    return " ".join(toks)


def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def mask_relations(sexpr):
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split()
    for idx in range(len(toks)):
        t = toks[idx]
        if t in extra_relations_prefix:
            toks[idx] = relation_mask
        elif re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t) or re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t):
            toks[idx] = relation_mask
    return " ".join(toks)


def mask_entities(sexpr):
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split()
    for idx in range(len(toks)):
        t = toks[idx]
        if t.startswith('m.') or t.startswith('g.'):
            toks[idx] = entity_mask
    return " ".join(toks)


def mask_literals(sexpr):
    """ 使用一个很简单的方法, 不是 operator, 并且不是 [REL] [ENT] 的就换成 [LIT]"""
    operators_list = ['(', ')', 'AND', 'COUNT', 'R', 'JOIN', 'ARGMAX', 'ARGMIN', 'lt', 'gt', 'le', 'ge', '[ENT]', '[REL]', 'TC']
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split()
    for idx in range(len(toks)):
        t = toks[idx]
        if (
            t not in operators_list 
            and not (t.startswith('m.') or t.startswith('g.')) 
            and not t in extra_relations_prefix 
            and not re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t) or re.match(r"[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t)
        ):
            toks[idx] = literal_mask

    return " ".join(toks)


def mask_special_literals(sexpr):
    """
    特殊的 literal 形如 \"as Robby Ray Stewart\"@en
    其特点在于内部有空格，如果直接分词会影响结果
    """
    pattern = "\".*?\"(@en)?"
    sexpr = re.sub(pattern, literal_mask, sexpr)
    return sexpr


def get_all_masked(sexpr):
    sexpr = mask_special_literals(sexpr)
    sexpr = mask_literals(mask_entities(mask_relations(sexpr)))
    return sexpr

def get_relation_literal_masked(sexpr):
    sexpr = mask_special_literals(sexpr)
    sexpr = mask_literals(mask_relations(sexpr))
    return sexpr

def get_entity_literal_masked(sexpr):
    sexpr = mask_special_literals(sexpr)
    sexpr = mask_literals(mask_entities(sexpr))
    return sexpr

def get_literal_masked(sexpr):
    sexpr = mask_literals(mask_special_literals(sexpr))
    return sexpr


def _vanilla_linearization_method(expr, entity_label_map={}, relation_label_map={}, linear_origin_map={}):
    """
    textualize a logical form, replace mids with labels

    Returns:
        (str): normalized s_expr
    """
    expr = expr.replace("(", " ( ") # add space for parantheses
    expr = expr.replace(")", " ) ")
    toks = expr.split() # split by space
    toks = [x for x in toks if len(x)]

    norm_toks = []
    for t in toks:

        # original token
        origin_t = t

        if t.startswith("m.") or t.startswith("g."): # replace entity with its name
            if t in entity_label_map:
                t = entity_label_map[t]
            else:
                # name = get_label(t)
                name = get_label_with_odbc(t)
                if name is not None:
                    entity_label_map[t] = name
                    t = name
            t = '[ '+t+' ]'
        elif "XMLSchema" in t: # remove xml type
            format_pos = t.find("^^")
            t = t[:format_pos]
        elif t == "ge": # replace ge/gt/le/lt
            t = "GREATER EQUAL"
        elif t == "gt":
            t = "GREATER THAN"
        elif t == "le":
            t = "LESS EQUAL"
        elif t == "lt":
            t = "LESS THAN"
        else:
            # TODO 对于没有xml类型的float型数字，如"1.8"，会错误拆解
            t = t.replace("_", " ") # replace "_" with " "
            t = t.replace(".", " , ") # replace "." with " , "
            
            if "." in origin_t: # relation
                t = "[ "+t+" ]"
                relation_label_map[origin_t]=t
        
        norm_toks.append(t)
        linear_origin_map[t] = origin_t # for reverse transduction
        
    return " ".join(norm_toks)


def add_masked_sexpr_to_dataset(split):
    prev_dataset_path = 'data/CWQ/structure_filling/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_{}_all_data.json'.format(split)
    dataset = load_json(prev_dataset_path)
    new_data = []
    for data in dataset:
        sexpr = data["sexpr"]
        if sexpr == "null":
            data["normed_all_masked_sexpr"] = "null"
            data["normed_relation_literal_masked_sexpr"] = "null"
            data["normed_entity_literal_masked_sexpr"] = "null"
            data["normed_literal_masked_sexpr"] = "null"
        else:
            data["normed_all_masked_sexpr"] = _vanilla_linearization_method(get_all_masked(sexpr))
            data["normed_relation_literal_masked_sexpr"] = _vanilla_linearization_method(get_relation_literal_masked(sexpr))
            data["normed_entity_literal_masked_sexpr"] = _vanilla_linearization_method(get_entity_literal_masked(sexpr))
            data["normed_literal_masked_sexpr"] = _vanilla_linearization_method(get_literal_masked(sexpr))
        
        new_data.append(data)
    
    dirname = os.path.dirname(prev_dataset_path)
    dump_json(new_data, os.path.join(dirname, 'CWQ_{}_all_data_sexpr_masked.json'.format(split)))


def post_process_and_add_to_dataset(split, filter_null=False):
    """每个单词之间一个空格"""
    # dataset_path  = 'data/WebQSP/structure_filling/richRelation_2hopValidation_richEntity_CrossEntropyLoss_top100_1parse_ep6/FACC1_elq_entities/WebQSP_{}_all_data_sexpr_masked.json'.format(split)
    dataset_path = 'data/CWQ/structure_filling/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_{}_all_data_sexpr_masked.json'.format(split)
    dataset = load_json(dataset_path)
    dirname = os.path.dirname(dataset_path)
    structure_predictions = load_json('exps/structure_generation/CWQ_not_lower_beam_10_10epoch_new/10epoch_{}/beam_10_top_k_predictions_syntax_checked.json'.format(split))
    if filter_null:
        dataset = [item for item in dataset if item["sexpr"] != "null"]
    assert len(dataset) == len(structure_predictions), print(len(dataset), len(structure_predictions))
    new_dataset = []

    for (data, pred) in zip(dataset, structure_predictions):
        predictions_top5 = pred["predictions"][:5]
        if len(predictions_top5) < 5:
            # 候选结构都应该有 5 个，否则跑模型时会有问题
            predictions_top5 += padding_structures[:5-len(predictions_top5)]
            print(len(predictions_top5))

        def post_process(structure):
            structure = structure.replace('<pad>', '')
            structure = structure.replace('</s>', '')
            structure = structure.replace('[REL]', ' [REL] ')
            structure = structure.replace('[ENT]', ' [ENT] ')
            structure = structure.replace('[LIT]', ' [LIT] ')
            toks = structure.split()
            return " ".join(toks)
        
        predictions_top5 = [post_process(item) for item in predictions_top5]
        data["candidate_structures_list"] = predictions_top5
        
        new_dataset.append(data)

    dump_json(new_dataset, os.path.join(dirname, 'CWQ_{}_all_data_sexpr_masked_candidate_structures.json'.format(split)))



def post_process_and_remove_syntax_error(split):
    """每个单词之间一个空格"""
    prediction_path = 'exps/structure_generation/CWQ_not_lower_beam_10_10epoch_new/10epoch_{}/beam_10_top_k_predictions.json'.format(split)
    structure_predictions = load_json(prediction_path)
    dirname = os.path.dirname(prediction_path)
    new_pred = []

    for pred in structure_predictions:
        predictions = pred["predictions"]
        
        def filter_syntax_error(masked_sexpr):
            return parse_s_expr(masked_sexpr) is not None
        
        predictions = [post_process(item) for item in predictions]
        predictions = list(filter(filter_syntax_error, predictions))
        pred["predictions"] = predictions
        pred["gen_label"] = post_process(pred["gen_label"])
        new_pred.append(pred)

    dump_json(new_pred, os.path.join(dirname, 'beam_10_top_k_predictions_syntax_checked.json'))

def extract_sexpr_and_normed(split):
    prev_dataset_path = 'data/CWQ/structure_filling/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_{}_all_data_sexpr_masked.json'.format(split)
    dataset = load_json(prev_dataset_path)

    new_data = [
        {
            'sexpr': item["sexpr"],
            'normed_sexpr': item["normed_sexpr"],
            'normed_all_masked_sexpr': item["normed_all_masked_sexpr"]
        } for item in dataset
    ]

    dump_json(new_data, os.path.join(os.path.dirname(prev_dataset_path), 'CWQ_{}_sexprs.json'.format(split)))


def get_generation_target_maxlen(split):
    dataset = load_json('data/WebQSP/final/merged/WebQSP_{}.json'.format(split))
    tokenizer = AutoTokenizer.from_pretrained('/home2/xxhu/QDT2SExpr/CWQ/hfcache/t5-base')
    tokenizer.add_special_tokens(
            {"additional_special_tokens":["[DES]","[INQ]", "[des]","[inq]"]}
    )
    tokenizer.add_tokens(["[ENT]", "[REL]", "[LIT]", "[ent]", "[rel]", "[lit]"])

    length_dict = defaultdict(int)
    for data in dataset:
        length = len(tokenizer.tokenize(data["normed_sexpr"]))
        length_dict[length] += 1
    
    print(split)
    print(length_dict)


def calculate_top_prediction_sketch_accuracy():
    predictions = load_json('exps/structure_filling/WebQSP_structure_filling_structure_generation_concat_add_prefix_tgtlen_110_15epoch/beam_50_top_k_predictions.json_gen_sexpr_results.json')
    acc = 0.0
    consistency = 0.0
    for pred in predictions:
        if get_all_masked(pred["logical_form"]) == get_all_masked(pred["gt_sexpr"]):
            acc += 1.0
        if get_all_masked(pred["logical_form"]) == post_process(pred["pred"]["structure_predictions"][0]):
            consistency += 1.0
    print('accuracy: {}, consistency: {}'.format(acc/len(predictions), consistency/len(predictions)))



if __name__=='__main__':
    # for split in ['dev', 'test', 'train']:
    #     print(split)
    #     # add_masked_sexpr_to_dataset(split)
    #     extract_sexpr_and_normed(split)
    # for split in ['train', 'dev', 'test']:
    #     post_process_and_remove_syntax_error(split) 
    # post_process_and_add_to_dataset('train', filter_null=True)
    # post_process_and_add_to_dataset('dev', filter_null=True)
    # post_process_and_add_to_dataset('test', filter_null=False)

    # post_process_and_add_to_dataset('train', filter_null=True)
    # post_process_and_add_to_dataset('test')

    # for split in ['test', 'train']:
    #     get_generation_target_maxlen(split)

    dic = {25: 199, 2: 42, 90: 5, 30: 79, 29: 84, 26: 158, 73: 38, 97: 10, 36: 9, 74: 33, 28: 120, 51: 23, 32: 30, 31: 37, 52: 28, 59: 5, 72: 15, 91: 7, 78: 17, 83: 4, 27: 155, 47: 47, 24: 16, 49: 34, 54: 18, 66: 8, 71: 7, 121: 2, 75: 23, 50: 39, 76: 13, 58: 11, 35: 5, 68: 17, 80: 10, 77: 20, 69: 8, 86: 1, 92: 9, 67: 13, 60: 6, 55: 10, 98: 13, 53: 13, 61: 4, 48: 25, 70: 9, 34: 12, 46: 25, 84: 4, 88: 3, 45: 22, 82: 6, 81: 6, 95: 7, 96: 4, 63: 3, 65: 6, 33: 6, 62: 4, 37: 2, 79: 7, 87: 2, 89: 1, 40: 3, 103: 1, 93: 5, 99: 2, 104: 1, 102: 3, 94: 2, 57: 1, 64: 3, 56: 4, 110: 1, 43: 2, 42: 1, 114: 1, 105: 1, 39: 1, 108: 1, 111: 1, 116: 1, 44: 2, 85: 1, 126: 1, 100: 1}
    total = 0
    for i in range(0, 200):
        total += dic[i] if i in dic else 0
    print(total)

    # calculate_top_prediction_sketch_accuracy()



