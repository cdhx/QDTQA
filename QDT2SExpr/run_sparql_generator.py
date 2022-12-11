#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_sparql_generator.py
@Time    :   2022/01/06 15:18:10
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   Generate SPARQL directly from input questions
'''

# here put the import lib

from curses import meta
from email.policy import default
from importlib_metadata import metadata
import torch
from tqdm import tqdm
import timeit
import logging
import os
import sys
import numpy as np
import math
from functools import partial
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, OrderedDict, Tuple
from transformers import (
    BartTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    EvalPrediction
)
from components.generation_trainer import GenerationTrainer
from components.utils import dump_json
from torch.utils.data import DataLoader, SequentialSampler

from inputDataset.gen_sparql_dataset import ListDataset, cwq_load_and_cache_gen_sparql_examples, generation_collate_fn

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing:
        sortish_sampler:
        predict_with_generate
    """
    warmup_ratio: Optional[float] = field(default=0.0, metadata={"help": "The warmup ratio"})
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_type: str = field(
        metadata={"help": "type of the model, t5 or bart"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset: str = field(default=None,metadata={"help": "dataset id"})
    train_file: str = field(default=None,metadata={"help": "path to training file"})
    predict_file: str = field(default=None,metadata={"help": "path to predict file"})

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='hfcache', metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_source_length: Optional[int] = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    # top_k_candidates: Optional[int] = field(default=5, metadata={"help": "# top k candidates used for generation."})
    do_lower_case: bool = field(default=False)
    overwrite_cache: bool = field(default=False)
    use_qdt: bool = field(
        default=False, metadata={"help": "Whether to use QDT to generate S_expression"}
    )

    qdt_only: bool = field(
        default=False, metadata={"help": "Whether to use QDT only"}
    )

    add_entity: bool = field(
        default=False, metadata={"help": "Whether to add entity in the input_seq"}
    )

    gold_entity: bool = field(
        default=False, metadata={"help": "Whether to use gold entity in the input_seq"}
    )

    add_relation: bool = field(
        default=False, metadata={"help": "Whether to add relation in the input_seq"}
    )

    gold_relation: bool = field(
        default=False, metadata={"help": "Whether to use relation in the input_seq"}
    )

    gold_relation_num: int = field(
        default=0, metadata={"help": "How many gold relations are added, 0 means all"}
    )

    cand_relation_num: int = field(
        default=5, metadata={"help":"How many candidate relations are retained, 5 by default"}
    )

    new_qdt: bool = field(
        default=False, metadata={"help": "Whether to use new qdt generated by T5"}
    )
    
    # overwrite_output_dir: bool = field(default=False)
    # local_rank: Optional[int] = field(default=-1,metadata={"help": "local_rank for distributed training on gpus."})

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def run_prediction(args, model_args,dataset, model, tokenizer, output_prediction=False):
    """Use generation model to predict logical forms"""
    
    # check output_dir existence
    if not os.path.exists(args.output_dir) and args.local_rank in [-1,0]:
        os.makedirs(args.output_dir)
    
    # For evaluating, we ensure batch size is only one
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.per_device_eval_batch_size, 
                                collate_fn=partial(generation_collate_fn, tokenizer=tokenizer)
                                )
    
    max_length = (
        model.config.max_generate_length 
        if hasattr(model.config, "max_generate_length") 
        else model.config.max_position_embeddings
    )

    num_beams = model.config.num_beams
    pad_token_id = model.config.pad_token_id

    # only allow using one gpu here
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.per_device_eval_batch_size)
    logger.info("  Beam size = %d", model_args.eval_beams)

    start_time = timeit.default_timer()
    model = model.to(args.device)

    all_predictions = []
    all_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", total=math.ceil(len(dataset)/args.per_device_eval_batch_size)):
        model.eval()
        batch = dict([(k, v.to(args.device)) for k,v in batch.items()])

        labels = batch.pop('labels')
        [all_labels.append(l.cpu().numpy()) for l in labels]

        with torch.no_grad():
            generated_tokens = model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=max_length
            )
            generated_tokens = torch.reshape(generated_tokens, (labels.size(0), num_beams, -1))
            [all_predictions.append(p.cpu().numpy()) for p in generated_tokens]

    assert len(all_predictions) == len(all_labels)

    ex_cnt = 0
    total = 0
    pred_outputs = OrderedDict()
    entity_label_map = OrderedDict()
    predict_data_map = OrderedDict()
    for feat, pred in tqdm(zip(dataset, all_predictions), total=len(all_predictions),desc="Decoding"):
        ex = feat.ex
        decoded_pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        # the first prediction exactly matches
        ex_cnt +=(decoded_pred[0].replace(' ','') == ex.gt.normed_expr.replace(' ', '').lower())
        # ex_cnt += (pred[0].equal(tokenizer.encode(ex.gt.normed_expr)))

        if ex.gt.s_expr!='' and ex.gt.s_expr.lower()!='null':
            total+=1
        
        entity_label_map[ex.qid] = ex.entity_label_map
        pred_outputs[ex.qid] = decoded_pred

        predict_data={}
        predict_data['predictions'] = decoded_pred
        predict_data['gt_s_expr'] = ex.gt.s_expr
        predict_data['normed_s_expr'] = ex.gt.normed_expr
        predict_data['question'] = ex.query
        predict_data['qdt'] = ex.qdt
        predict_data['entity_label_map'] = ex.entity_label_map
        predict_data['candidate_entity_map'] = ex.candidate_entity_map
        predict_data['candidate_relation_list'] = ex.candidate_relation_list
        predict_data['gold_relation_list'] = ex.gold_relation_list
        predict_data['linear_origin_map'] = ex.linear_origin_map
        # predict_data['src_input_ids'] = feat.src_input_ids.tolist()
        # predict_data['tgt_input_ids'] = feat.tgt_input_ids.tolist()
        predict_data['ex_match'] = (decoded_pred[0] == ex.gt.normed_expr.replace(' ,', ',').lower())

        predict_data_map[ex.qid] = predict_data


    if output_prediction:
        dump_json(pred_outputs, os.path.join(args.output_dir, 'top_k_predictions.json'),indent=4)
        dump_json(entity_label_map, os.path.join(args.output_dir, 'entity_label.json'),indent=4)
        dump_json(predict_data_map, os.path.join(args.output_dir, 'predict_data_all.json'),indent=4)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # only count questions with predictions
    return  {
            'total': len(all_predictions),
            'with_target_total:': total, 
            'ex': ex_cnt / len(all_predictions), 
            'ex_real': ex_cnt/total
            }


def main():
    parser = HfArgumentParser((ModelArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args.local_rank = training_args.local_rank

    # check whether the model already exists
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)

    # set seed
    set_seed(training_args.seed)


    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    if model_args.model_type.lower()=='bart':
        tokenizer = BartTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )


    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    tokenizer.add_special_tokens({"additional_special_tokens":["{","}"]})
    if model_args.use_qdt:
        # try new tag, in the form like: NLQ+QDT+[ENT]xx[REL]xx,xx[REL]xx,xx
        tokenizer.add_special_tokens(
            {"additional_special_tokens":["[DES]","[INQ]","[ENT]","[REL]","[des]","[inq]","[ent]","[rel]","{","}"]}
        )
    model.resize_token_embeddings(len(tokenizer))

    # set num_beams for evaluation
    if model_args.eval_beams is not None:
        model.config.num_beams = model_args.eval_beams
    
    assert model.config.num_beams >= 1, f"got eval_beams={model.config.num_beams}. Need an integer >= 1"
    model_args.logger = logger

    # set max length for generation
    model.config.max_generate_length = model_args.max_target_length

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def non_pad_len(tokens: np.ndarray) -> int:
            """return non padding token length"""
            return np.count_nonzero(tokens != tokenizer.pad_token_id)

        def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
            """decode predictions and labels"""
            pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
            pred_str = lmap(str.strip, pred_str)
            label_str = lmap(str.strip, label_str)
            return pred_str, label_str

        # with decoding
        def _exact_match_metrics(pred: EvalPrediction) -> Dict:
            # firstly decode
            pred_str, label_str = decode_pred(pred)
            ex = sum([a == b for (a,b) in zip(pred_str, label_str)])/len(pred_str)
            result = {'ex': ex}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result

        # without decoding
        def exact_match_metrics(pred: EvalPrediction) -> Dict:
            # print(pred)
            # pred_str, label_str = decode_pred(pred)
            ex = np.sum(np.all(pred.label_ids == pred.predictions, axis=1)) / pred.label_ids.shape[0]
            # for a, b in zip(pred.label_ids, pred.predictions):
            #     print(a)
            #     print(b)
            # exit()
            result = {'ex': ex, 'num_total': pred.label_ids.shape[0]}
            gen_len = np.mean(lmap(non_pad_len, pred.predictions))
            result.update({"gen_len": gen_len})
            return result
        
        # compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
        compute_metrics_fn = exact_match_metrics
        return compute_metrics_fn
        
    # Get datasets
    if training_args.do_train:
        #train_dataset = ListDataset([])
        train_dataset = ListDataset(cwq_load_and_cache_gen_sparql_examples(model_args, tokenizer, evaluate=False))
    else:
        train_dataset = ListDataset([])

    if training_args.do_eval:
        # eval_dataset = ListDataset(load_and_cache_examples(model_args, tokenizer, evaluate=True))
        eval_dataset = ListDataset(cwq_load_and_cache_gen_sparql_examples(model_args, tokenizer, evaluate=True))
        # pass
    else:
        eval_dataset = ListDataset([])

    print('train_dataset:{}'.format(len(train_dataset)))
    print('eval_dataset:{}'.format(len(eval_dataset)))

    # Training
    if training_args.do_train:
        # Initialize our Trainer
        trainer = GenerationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=partial(generation_collate_fn, tokenizer=tokenizer),
            # prediction_loss_only = True,
            compute_metrics=build_compute_metrics_fn()
        )

        trainer.train(model_path=None)
        # trainer.train(
        #     model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        # )
        
        trainer.save_model()

        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

        
    # evaluating
    eval_results = {}
    if training_args.do_eval:
        logging.info("*** Test ***")

        result = run_prediction(training_args, model_args, eval_dataset, model, tokenizer, output_prediction=True)

        logger.info("***** Test results *****")

        for key, value in result.items():
            logger.info("  %s = %s", key, value)
        
        eval_results.update(result)

        dump_json(eval_results,os.path.join(training_args.output_dir, "gen_eval_result.json") )

    return eval_results


if __name__ == '__main__':
    
    # do_debug = False
    do_debug = True
    if do_debug:
        import ptvsd
        server_ip = "0.0.0.0"
        server_port = 12345
        print('Waiting for debugger attach...')
        ptvsd.enable_attach(address=(server_ip,server_port),redirect_output=True)
        ptvsd.wait_for_attach()

    main()
