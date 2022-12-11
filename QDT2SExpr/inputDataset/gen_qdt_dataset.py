

from pyexpat import features
from transformers import BartTokenizer
from components.utils import load_json
from inputDataset.gen_dataset import ListDataset
import os
import torch
from typing import List
from tqdm import tqdm


class QDTGenerationExample:
    """
    Generation Example from a raw query to a qdt
    """

    def __init__(self, qid, question, qdt) -> None:
        self.qid = qid
        self.question = question  # raw question
        self.qdt = qdt

    def __str__(self):
        return "{}\n\t->{}\n".format(self.question, self.qdt)

    def __repr__(self):
        return self.__str__()


class QDTGenerationFeature:
    """
    Feature for qdt generation
    """

    def __init__(self, ex, src_inputs_ids, tgt_input_ids):
        self.ex = ex
        self.src_input_ids = src_inputs_ids
        self.tgt_input_ids = tgt_input_ids


def cwq_read_gen_qdt_examples_from_json(
    split_file,
    is_eval=False
    ) -> List[QDTGenerationExample]:
    """Read cwq dataset file to qdt generation examples"""

    data_bank = load_json(split_file)
    examples = []
    for data in tqdm(data_bank, desc="Reading", total=len(data_bank)):
        qid = data['ID']
        question = data['question']
        qdt = data['linear_qdt']
        ex = QDTGenerationExample(qid=qid, question=question, qdt=qdt)

        examples.append(ex)

    return examples


def extract_gen_qdt_features_from_examples(args, tokenizer, examples) -> List[QDTGenerationFeature]:
    """Extract QDT Generation Features from examples with Huggingface Tokenizer"""
    features = []
    # whether to add prefix space
    add_prefix_space = isinstance(tokenizer, BartTokenizer)

    # indexing the examples to generate features
    for ex in tqdm(examples, desc='Indexing', total=len(examples)):
        question = ex.question
        qdt = ex.qdt
        # do lower case
        if args.do_lower_case:
            question = question.lower()
            qdt = qdt.lower()

        src_text = question
        dst_text = qdt

        if add_prefix_space:
            batch_encoding = tokenizer.prepare_seq2seq_batch(
                [src_text],
                [dst_text],
                max_length=args.max_source_length,
                max_target_length=args.max_target_length,
                return_tensors='pt',
                add_prefix_space=add_prefix_space
            ).data
        else:
            batch_encoding = tokenizer.prepare_seq2seq_batch(
                [src_text],
                [dst_text],
                max_length=args.max_source_length,
                max_target_length=args.max_target_length,
                return_tensors='pt'
            ).data
            input_ids, labels = batch_encoding['input_ids'][0], batch_encoding['labels'][0]
            feat = QDTGenerationFeature(ex, input_ids, labels)

        features.append(feat)
    return features


def cwq_load_and_cache_qdt_examples(args, tokenizer, evaluate=False) -> ListDataset:
    """Load and cache generation examples of CWQ, return a ListDataset"""
    # load CWQ generate qdt examples
    logger = args.logger

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    # split_id = args.split
    split_file = (
        args.predict_file if evaluate else args.train_file
    )  # if evaluate, use predict file
    dataset_id = os.path.basename(split_file).split("_")[
                                  0]  # CWQ, Grail, WebQSP
    split_id = os.path.basename(split_file).split("_")[1]  # dev, test, train

    # make feature cache dir
    if not os.path.exists("feature_cache"):
        os.mkdir("feature_cache")

    cachefile_name = "qdtgen_{}_{}".format(
        dataset_id, split_id, args.model_type)

    cached_features_file = os.path.join(
        "feature_cache", cachefile_name
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:  # feature cache not exists
        logger.info("Creating features from dataset file at %s", input_dir)

        examples = cwq_read_gen_qdt_examples_from_json(
            split_file, is_eval=evaluate)
                                                                                                
        features=extract_gen_qdt_features_from_examples(
            args, tokenizer, examples)

        logger.info("Saving features into cached file %s",
                    cached_features_file)
        torch.save(features, cached_features_file)

        return ListDataset(features)
