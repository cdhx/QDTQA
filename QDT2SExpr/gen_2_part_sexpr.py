import json
import random

def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w', encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def clean_files(split='test'):
    qdt_file = load_json(
        f'/home3/stcheng/demo/QDT2SExpr/CWQ/data/qdt/cwq_qa_decomp_seq2seq_all_all_2inq_seq_prefix_decipher_{split}.json')
    for lines in qdt_file:
        for k in {'properties', 'filter_pred', 'topics', 'answers', 'qdt', 'qdt_no_decipher', 'subq1_no_decipher',
                  'subq2_no_decipher', 'pred_no_decipher'}:
            if k in lines.keys():
                del lines[k]
    dump_json(qdt_file, f'/home3/stcheng/demo/QDT2SExpr/CWQ/data/qdt/cwq_{split}_qdt_2_part_addSexp.json')


def extract_sub_questions(split='test'):
    qdt_file = load_json(f'/home3/stcheng/demo/QDT2SExpr/CWQ/data/qdt/cwq_{split}_qdt_2_part_addSexp.json')
    for index, item in enumerate(qdt_file):
        subq1 = ""
        subq2 = ""
        if item['compositionality_type'] == 'composition':
            for q1 in item['subq1']:
                subq1 = subq1 + q1
            subq1 = subq1.replace("[ENT]", "Curry")

            for q2 in item['subq2']:
                subq2 = subq2 + q2

        elif item['compositionality_type'] == 'conjunction':
            if len(item['subq2']) > 0:
                print(item)
            for q in item['subq1']:
                if q == item['subq1'][0]:
                    subq1 = subq1 + q
                else:
                    subq2 = subq2 + q
        else:
            print(item)
        qdt_file[index]['subq_1'] = subq1
        qdt_file[index]['subq_2'] = subq2

    dump_json(qdt_file, f'/home3/stcheng/demo/QDT2SExpr/CWQ/data/qdt/cwq_{split}_qdt_2_part_addSexp.json')


def concat_subq(split='test'):
    subq1_file = load_json(f'/home3/stcheng/demo/QDT2SExpr/CWQ/results/gen/WebQSP_{split}_nlq_CWQ_{split}_qdt_subq1_0720_candEnt/top_k_predictions.json')
    subq2_file = load_json(f'/home3/stcheng/demo/QDT2SExpr/CWQ/results/gen/WebQSP_{split}_nlq_CWQ_{split}_qdt_subq2_0720_candEnt/top_k_predictions.json')
    new_2_part = {}
    for index, item in enumerate(subq1_file):
        # rand1 = random.randint(0, 9)
        rand1 = 0
        sub1 = subq1_file[item][rand1].replace("curry", '[ENTITY]')
        sub2 = subq2_file[item][rand1]
        rand2 = random.randint(0, 9)
        rand2 = 1
        sub3 = subq1_file[item][rand2].replace("curry", '[ENTITY]')
        sub4 = subq2_file[item][rand2]
        new_ques = " [subq1] " + sub1 + " "+ sub3 + " [subq2] " + sub2 + " " + sub4
        # new_ques = " [subq1] " + sub1  + " [subq2] " + sub2
        new_2_part[item] = new_ques

    dump_json(new_2_part, f'/home3/stcheng/demo/QDT2SExpr/CWQ/data/qdt/CWQ_{split}_new_qdt_decipher_2_sexps_0720.json')


if __name__ == '__main__':
    clean_files('test')
    clean_files('dev')
    clean_files('train')
