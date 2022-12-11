import json
import os
ignored_tokens = ['<pad>', '</s>']

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

def remove_ignored_tokens(sexpr):
    for tok in ignored_tokens:
        sexpr = sexpr.replace(tok, ' '+tok+' ')
    new_sexpr = []
    for tok in sexpr.split():
        if tok not in ignored_tokens:
            new_sexpr.append(tok)
    return " ".join(new_sexpr)


def calculate_prediction_metrics_ignore_special_tokens(split):
    dirname = 'exps/structure_generation/CWQ_not_lower_beam_10_10epoch_new/10epoch_{}'.format(split)
    predictions = load_json(os.path.join(dirname, 'beam_10_top_k_predictions.json'))
    real_len = 0
    hit1 = 0.0
    hit5 = 0.0
    hit10 = 0.0
    idx = 0
    missed_ids = []

    for pred in predictions:
        golden = pred["gen_label"]
        preds = pred["predictions"]
        
        processed_golden = remove_ignored_tokens(golden)
        processed_preds = [remove_ignored_tokens(item) for item in preds]

        if processed_golden != "null":
            real_len += 1
        if len(processed_preds) == 0:
            continue
        if processed_golden == processed_preds[0]:
            hit1 += 1.0
        if processed_golden in processed_preds[:5]:
            hit5 += 1.0
        if processed_golden in processed_preds[:10]:
            hit10 += 1.0
        if processed_golden != "null" and processed_golden not in processed_preds[:10]:
            missed_ids.append(idx)
        idx += 1
    
    res = 'Hit@1: {}, Hit@5: {}, Hit@10: {}\n'.format(hit1/len(predictions), hit5/len(predictions), hit10/len(predictions))
    res += 'Real: Hit@1: {}, Hit@5: {}, Hit@10: {}\n'.format(hit1/real_len, hit5/real_len, hit10/real_len)
    # res += 'missed: {}\n'.format(missed_ids)

    print(res)
    print('missed: {}'.format(missed_ids))

    dump_json(res, os.path.join(dirname, 'eval_results.txt'))



if __name__=='__main__':
    for split in ['train', 'dev', 'test']:
        calculate_prediction_metrics_ignore_special_tokens(split)
