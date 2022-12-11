from cProfile import label
from pandas import cut
import torch
import os
from inputDataset.gen_dataset import ListDataset
from inputDataset.disamb_dataset import _tokenize_relation, _normalize_relation


from components.utils import load_json
from tqdm import tqdm
from nltk.tokenize import word_tokenize

class CWQRelationClassificationExample:
    def __init__(self, qid, question, cand_rels, labels):
        self.qid = qid
        self.question = question
        self.cand_rels = cand_rels
        self.labels = labels
        # self.positive_rels = positive_rels
        # self.negative_rels = negative_rels
        # self.cand_rels = cand_rels


class CWQRelationClassificationFeature:
    def __init__(self, pid, input_ids, token_type_ids, labels):
        self.pid = pid
        self.candidate_input_ids = input_ids
        self.candidate_token_type_ids = token_type_ids
        self.labels = labels      

def proc_instance(ex, rel_map, evaluate=False, cutoff=100):
    qid = ex['ID']
    question = ex['question']

    positive_rels = rel_map['positive_rels']
    negative_rels = rel_map['negative_rels']
    cand_rels = rel_map['cand_rels']

    if evaluate:
        cand_rels = cand_rels[:cutoff]
    else: # do training, force add positive rels
        cand_rels = positive_rels+cand_rels[:cutoff-len(positive_rels)]

    if len(cand_rels)==0:
            # read from negative_rels
            cand_rels = positive_rels+negative_rels[:cutoff-len(positive_rels)]
    
    labels = []
    for rel in cand_rels[:cutoff]:
        if rel in positive_rels:
            labels.append(True)
        else:
            labels.append(False)

    return CWQRelationClassificationExample(qid,question,cand_rels,labels)



def read_rel_classify_examples_from_rel_candidates(dataset_file, candidate_file, evaluate=False):
    dataset = load_json(dataset_file)
    candidate_rel_maps = load_json(candidate_file)

    examples = []

    for data in tqdm(dataset, total = len(dataset), desc='Reading'):
        qid = data['ID']
        rel_map = candidate_rel_maps[qid]
        examples.append(proc_instance(data,rel_map,evaluate))
    
    return examples



def extract_rel_classify_features_from_examples(args, tokenizer, examples, do_predict=False):
    features = []
    for ex in tqdm(examples,total=len(examples),desc='Indexing'):
        qid = ex.qid
        question = ex.question
        candidate_rels = ex.cand_rels
        candidate_rels = [_normalize_relation(r) for r in candidate_rels]
        labels = [1 if v else 0 for v in ex.labels]

        candidate_input_ids = []
        candidate_token_type_ids = []
        for rel in candidate_rels:
            c_encoded = tokenizer(question,rel,truncation=True,max_length=args.max_seq_length, return_token_type_ids=True)
            candidate_input_ids.append(c_encoded['input_ids'])
            candidate_token_type_ids.append(c_encoded['token_type_ids'])
        
        rel_ex_feature = CWQRelationClassificationFeature(qid,candidate_input_ids,candidate_token_type_ids, labels)
        features.append(rel_ex_feature)        
    return features
        


def load_and_cache_rel_classify_examples(args, tokenizer, evaluate=False, output_examples=False):
    logger = args.logger

    if args.local_rank not in [-1,0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    split_file = args.predict_file if evaluate else args.train_file # predict file for evaluate
    dataset_id = os.path.basename(split_file).split('_')[0]
    split_id = os.path.basename(split_file).split('_')[1]

    # split_file = '_'.(join(os.path.basename(split_file).split('_')[:2])
    cached_features_file = os.path.join('feature_cache',"rel_classify_{}_{}_{}_{}".format(dataset_id, split_id,args.model_type,args.max_seq_length))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # cache exists, load it
        logger.info("Loading features from cached file %s", cached_features_file)
        data = torch.load(cached_features_file)
        examples = data['examples']
        features = data['features']
    else:
        # cache not exists, create it
        logger.info("Creating features from dataset file at %s", input_dir)
        candidate_file = args.predict_file if evaluate else args.train_file

        example_cache = os.path.join('feature_cache', f'{dataset_id}_{split_id}_rel_classify_example.bin')
        if os.path.exists(example_cache) and not args.overwrite_cache:
            examples = torch.load(example_cache)
        else:
            orig_split = split_id
            dataset_file = os.path.join('data', f'CWQ_{orig_split}_expr.json')
            examples = read_rel_classify_examples_from_rel_candidates(dataset_file, candidate_file, evaluate=evaluate)
            torch.save(examples, example_cache)

        features = extract_rel_classify_features_from_examples(args, tokenizer, examples, do_predict=args.do_predict)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({'examples': examples, 'features': features}, cached_features_file)
        
    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    if output_examples:
        return ListDataset(features), examples
    else:
        return ListDataset(features)





