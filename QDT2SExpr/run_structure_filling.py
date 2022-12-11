import argparse
from collections import defaultdict
import os
import copy
from inputDataset.get_structure_filling_dataset import StructureFillingDataset, StructureFillingExample
from models.T5_structure_multitask import T5_MultiTask_Concat_Relation_Concat_Entity, T5_SExpr_Generation_Structure_Generation, T5_SExpr_Generation_Structure_Generation_Concat, T5_Structure_Filling
from components.utils import dump_json, load_json

from functools import partial
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np


def load_data(split):
    # read data
    data_file_name = f'data/CWQ/structure_filling/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_{split}_all_data_sexpr_masked_candidate_structures.json'
    # WebQSP
    # data_file_name = f'data/WebQSP/structure_filling/richRelation_2hopValidation_richEntity_CrossEntropyLoss_top100_1parse_ep6/FACC1_elq_entities/WebQSP_{split}_all_data_sexpr_masked_candidate_structures.json'
    print('Loading data from:',data_file_name)
    data_dict = load_json(data_file_name)
    return data_dict


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_debug',default='False',help='whether to do training')
    parser.add_argument('--do_train',default=False,action='store_true',help='whether to do training')
    parser.add_argument('--do_eval',default=False,action='store_true',help='whether to do eval when training')
    parser.add_argument('--do_predict',default=False,action='store_true',help='whether to do training')
    parser.add_argument('--predict_split',default='test',help='which dataset to perform prediction')
    parser.add_argument('--pretrained_model_path', default="t5-base", help='model name like "t5-base" or a local directory with t5 model in it')
    parser.add_argument('--model_save_dir', default="exps/gen_multitask/model_saved", help='model path for saving and loading model')
    parser.add_argument('--max_src_len',default=256, type=int, help='maximum source length')
    parser.add_argument('--max_tgt_len',default=196, type=int, help='maximum target length')
    parser.add_argument('--train_batch_size', default=4, type=int, help='batch_size for training')
    parser.add_argument('--eval_batch_size', default=4, type=int, help='batch_size for evaluation')
    parser.add_argument('--test_batch_size',default=8, type=int, help='batch_size for testing')
    parser.add_argument('--lr',default=2e-5,type=float,help='learning_rate')
    parser.add_argument('--weight_decay',default=1e-3,type=float,help='weight_decay')
    parser.add_argument('--epochs',default=2,type=int,help='epochs')
    parser.add_argument('--iters_to_accumulate',default=1,type=int,help='the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size')
    parser.add_argument('--print_every',default=100,type=int,help='every steps to print training information')
    parser.add_argument('--save_every_epochs',default=5,type=int,help='save the model every n eopchs')
    parser.add_argument('--warmup_ratio',default=0.1,type=float,help='the ratio of warm up steps')
    parser.add_argument('--output_dir',default='exps/gen_multitask',help='where to save model')
    parser.add_argument('--overwrite_output_dir',default=False,action='store_true',help='whether to overwrite the output dir')
    parser.add_argument('--eval_beams',default=5,type=int, help="beam size for generating")
    parser.add_argument('--do_lower',default=False,action='store_true',help='whether to do lower for both inputs and outputs')
    parser.add_argument('--normalize_relations', default=False, action='store_true')
    parser.add_argument('--relation_sample_size',default=10,type=int)
    parser.add_argument('--entity_sample_size',default=10,type=int)
    parser.add_argument('--structure_sample_size',default=5,type=int)
    parser.add_argument('--model',type=str, default='T5_Structure_Filling', help='T5_Structure_Filling | T5_MultiTask_Concat_Relation_Concat_Entity | T5_SExpr_Generation_Structure_Generation')
    parser.add_argument('--add_top1_structure',default=False, action='store_true')
    parser.add_argument('--add_prefix', default=False, action='store_true', help='True to add prefix for classification task')
    parser.add_argument('--structure_gen_beam_size', default=1, type=int)
    parser.add_argument('--max_structure_tgt_len', default=50, type=int)
    args = parser.parse_args()
    return args


def generate_entity_label_map_by_classification_res_sorted(predictions, dirname, dataset):
    """
    根据多任务中实体分类的结果，生成 candidate_entity_label_map
    对于多个实体有相同 label 的情况，按照分类的 logits 来排序
    """
    predicted_entities = defaultdict(dict)

    assert len(predictions) == len(dataset), print(len(predictions), len(dataset))

    for (pred, data) in zip(predictions, dataset):
        qid = data["ID"]
        pred_clf_logits = pred["pred_entity_clf_labels"]
        pred_clf_indexes = [idx for (idx, value) in enumerate(pred_clf_logits) if float(value) > 0.5]
        for idx in pred_clf_indexes:
            cand_entity = data["cand_entity_list"][idx]
            logits = float(pred_clf_logits[idx])
            if cand_entity['label'].lower() in predicted_entities[qid]:
                # 相同 label 的实体，按照 logits 排序；如果 logits 一样，按照 idx (也就是ELQ 和 FACC1 的排名)排序
                prev_logit = predicted_entities[qid][cand_entity['label'].lower()]['pred_logits']
                if logits > prev_logit:
                    predicted_entities[qid][cand_entity['label'].lower()] = {
                        'id': cand_entity['id'],
                        'pred_logits': logits
                    }
            else:
                predicted_entities[qid][cand_entity['label'].lower()] = {
                    'id': cand_entity['id'],
                    'pred_logits': logits
                }
            # predicted_entities[qid][cand_entity["id"]] = {"label": cand_entity["label"]}
    
    dump_json(predicted_entities, os.path.join(dirname, 'CWQ_test_predicted_entity_linking_results_sorted.json'))


def _collate_fn(data,tokenizer):
    all_tgt_input_ids = []
    all_relation_clf_input_ids = []
    all_structure_clf_input_ids = []
    all_relation_clf_labels = []
    all_entity_clf_labels = []
    all_structure_clf_labels = []
    candidate_relations = []
    candidate_entities = []
    candidate_structures = []
    input_srcs = []
    all_tgt_structure_input_ids = []
    for data_tuple in data:
        all_tgt_input_ids.append(data_tuple[0])
        all_relation_clf_input_ids.extend(data_tuple[1])
        all_structure_clf_input_ids.extend(data_tuple[2])
        all_relation_clf_labels.extend(data_tuple[3])
        all_entity_clf_labels.extend(data_tuple[4])
        all_structure_clf_labels.extend(data_tuple[5])
        candidate_relations.extend(data_tuple[6])
        candidate_entities.extend(data_tuple[7])
        candidate_structures.extend(data_tuple[8])
        input_srcs.extend(data_tuple[9])
        all_tgt_structure_input_ids.append(data_tuple[10])
    
    tgt_encoded = tokenizer.pad({'input_ids': all_tgt_input_ids},return_tensors='pt')
    tgt_structure_encoded = tokenizer.pad({'input_ids': all_tgt_structure_input_ids},return_tensors='pt')
    relation_clf_encoded = tokenizer.pad({'input_ids': all_relation_clf_input_ids},return_tensors='pt')
    structure_clf_encoded = tokenizer.pad({'input_ids': all_structure_clf_input_ids},return_tensors='pt')
    relation_clf_labels = torch.tensor(all_relation_clf_labels)
    entity_clf_labels = torch.tensor(all_entity_clf_labels)
    structure_clf_labels = torch.tensor(all_structure_clf_labels)

    return (
        tgt_encoded,
        relation_clf_encoded,
        structure_clf_encoded,
        relation_clf_labels,
        entity_clf_labels,
        structure_clf_labels,
        candidate_relations,
        candidate_entities,
        candidate_structures,
        input_srcs,
        tgt_structure_encoded
    )


def prepare_dataloader(args,split,tokenizer,batch_size):
    assert split in ['train','test','dev','train_sample','dev_sample','test_sample']

    data = load_data(split)
    print(f'Origin {split} dataset len: {len(data)}')
    assert type(data)==list
    if 'train' in split or 'dev' in split:
        # for train and dev, filter the examples without sexpr
        examples = []
        for x in data:
            if x['sexpr'].lower()!="null":
                examples.append(StructureFillingExample(x))                
    else:
        examples = [StructureFillingExample(x) for x in data]
    print(f'Real {split} dataset len: {len(examples)}')

    # examples = examples[:100]
    dataset = StructureFillingDataset(examples, 
                            tokenizer=tokenizer,
                            do_lower=args.do_lower,
                            normalize_relations=args.normalize_relations,
                            max_src_len=args.max_src_len,
                            max_tgt_len=args.max_tgt_len,
                            add_top1_structure=args.add_top1_structure,
                            add_prefix=args.add_prefix
                            )

    # print(train_dataset.__getitem__(0))
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=partial(_collate_fn,tokenizer=tokenizer),
                            shuffle=False
                            )
    return dataloader


def save_model(model_save_dir,model_to_save,tokenizer,epoch,is_final_epoch=False):
    if is_final_epoch:
        output_model_file = os.path.join(model_save_dir,'pytorch_model.bin')    
    else:
        output_model_file = os.path.join(model_save_dir,f'pytorch_model_epoch_{epoch}.bin')
    
    output_config_file = os.path.join(model_save_dir,'config_file.json')
    # output_vocab_file = os.path.join(model_save_dir,'vocab_file.bin')
    output_tokenizer_dir = os.path.join(model_save_dir,'custom_tokenizer')
    
    torch.save(model_to_save.state_dict(),output_model_file)
    model_to_save.t5.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_tokenizer_dir)
    # tokenizer.save_vocabulary(output_vocab_file)
    if is_final_epoch:
        print("The final model has been saved at {}".format(output_model_file))
    else:
        print("The model of eopch {} has been saved at {}".format(epoch,output_model_file))


def train_model(args,model,tokenizer,device,train_dataloader, epochs, dev_dataloader=None,model_save_dir=None):
    # train
    print('Start training...')
    # set parameters
    lr = args.lr # learning rate
    iters_to_accumulate = args.iters_to_accumulate  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
    print_every = args.print_every
    # set weight_decay for different parameters
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    opti = AdamW(
                optimizer_grouped_parameters, 
                lr=lr, 
                )
    # opti = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    warmup_ratio = args.warmup_ratio # The number of steps for the warmup phase.
    # num_training_steps = epochs * len(train_dataloader)  # The total number of training steps
    t_total = (len(train_dataloader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation
    lr_scheduler = get_linear_schedule_with_warmup(
                                        optimizer=opti,
                                        num_warmup_steps=t_total * warmup_ratio,
                                        num_training_steps=t_total
                                        )
    # scaler = GradScaler()
    best_loss = np.Inf
    best_epoch = 1
    num_iterations = len(train_dataloader)

    # dir to save model
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # train step
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for it,data in enumerate(tqdm(train_dataloader,desc=f'Epoch {epoch+1}')):
            tgt_encoded = data[0]
            relation_clf_encoded = data[1]
            structure_clf_encoded = data[2]
            relation_clf_labels = data[3]
            entity_clf_labels = data[4]
            structure_clf_labels = data[5]
            candidate_relations = data[6]
            candidate_entities = data[7]
            candidate_structures = data[8]
            input_srcs = data[9]
            tgt_structure_encoded = data[10]

            if isinstance(model, T5_Structure_Filling):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    input_ids_structure_clf=structure_clf_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    structure_clf_labels=structure_clf_labels.to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    structure_clf_attention_mask=structure_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_candidate_entities=candidate_entities,
                    textual_candidate_structures=candidate_structures,
                    textual_nlq=input_srcs,
                    normalize_relations=args.normalize_relations
                )
            elif isinstance(model, T5_MultiTask_Concat_Relation_Concat_Entity):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_srcs,
                    normalize_relations=args.normalize_relations,
                    textual_candidate_entities=candidate_entities,
                )
            elif isinstance(model, T5_SExpr_Generation_Structure_Generation):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    structure_gen_labels=tgt_structure_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_candidate_entities=candidate_entities,
                    textual_nlq=input_srcs,
                    normalize_relations=args.normalize_relations,
                )
            elif isinstance(model, T5_SExpr_Generation_Structure_Generation_Concat):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    structure_gen_labels=tgt_structure_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_candidate_entities=candidate_entities,
                    textual_nlq=input_srcs,
                    normalize_relations=args.normalize_relations,
                )

            loss = loss / iters_to_accumulate

            if (it+1)%iters_to_accumulate == 0:
                loss.backward()
                opti.step()
                lr_scheduler.step()
                opti.zero_grad()
            
            running_loss += loss.item()

            if (it + 1) % print_every == 0:
                print(flush=True)
                print("Iteration {}/{} of epoch {} (Total:{}) complete. Loss : {} "
                    .format(it+1, num_iterations, epoch+1, epochs, running_loss / print_every)
                    ,flush=True)
                running_loss = 0.0
        
        if args.do_eval:
            dev_loss = evaluate_loss(model,device,dev_dataloader,args)
            print()
            print("Epoch {} complete! Validation Loss : {}".format(epoch+1, dev_loss))

            if dev_loss < best_loss:
                print('Best validation loss improved from {} to {}'.format(best_loss, dev_loss))
                print()
                model_copy = copy.deepcopy(model) # save a copy of the model
                best_loss = dev_loss
                best_epoch = epoch+1
            # save the best model
            model_to_save = model_copy
        else:
            print()
            print("Epoch {} complete!".format(epoch+1))
            model_to_save = model
        
        # save intermediate models after every n epochs
        if (epoch+1)%args.save_every_epochs==0:
            save_model(model_save_dir,model_to_save,tokenizer,(epoch+1),is_final_epoch=False)
    
    # empty cache
    torch.cuda.empty_cache()
    # save final model
    # save_model(model_save_dir,model_to_save,tokenizer,epochs,is_final_epoch=True)

    return model_to_save


def evaluate_loss(model,device,dataloader,args):
    model.eval()
    mean_loss = 0
    count = 0
    with torch.no_grad():
        for it, data in enumerate(tqdm(dataloader,desc='Evaluating')):
            tgt_encoded = data[0]
            relation_clf_encoded = data[1]
            structure_clf_encoded = data[2]
            relation_clf_labels = data[3]
            entity_clf_labels = data[4]
            structure_clf_labels = data[5]
            candidate_relations = data[6]
            candidate_entities = data[7]
            candidate_structures = data[8]
            input_srcs = data[9]
            tgt_structure_encoded = data[10]

            if isinstance(model, T5_Structure_Filling):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    input_ids_structure_clf=structure_clf_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    structure_clf_labels=structure_clf_labels.to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    structure_clf_attention_mask=structure_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_candidate_entities=candidate_entities,
                    textual_candidate_structures=candidate_structures,
                    textual_nlq=input_srcs,
                    normalize_relations=args.normalize_relations
                )
            elif isinstance(model, T5_MultiTask_Concat_Relation_Concat_Entity):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_input_src_gen=input_srcs,
                    normalize_relations=args.normalize_relations,
                    textual_candidate_entities=candidate_entities,
                )
            elif isinstance(model, T5_SExpr_Generation_Structure_Generation):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    structure_gen_labels=tgt_structure_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_candidate_entities=candidate_entities,
                    textual_nlq=input_srcs,
                    normalize_relations=args.normalize_relations,
                )
            elif isinstance(model, T5_SExpr_Generation_Structure_Generation_Concat):
                loss = model(
                    input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                    relation_clf_labels=relation_clf_labels.to(device),
                    entity_clf_labels=entity_clf_labels.to(device),
                    structure_gen_labels=tgt_structure_encoded['input_ids'].to(device),
                    gen_labels=tgt_encoded['input_ids'].to(device),
                    relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                    textual_candidate_relations=candidate_relations,
                    textual_candidate_entities=candidate_entities,
                    textual_nlq=input_srcs,
                    normalize_relations=args.normalize_relations,
                )

            mean_loss += loss.item()
            count+=1
    
    return mean_loss/count


def run_prediction(args,model,device,dataloader,tokenizer,output_dir,output_predictions=True):
    print()
    print(f'Start predicting {args.predict_split}, beam_size:{args.eval_beams}, batch_size:{args.test_batch_size}')
    model.eval()

    all_gen_predictions = []
    all_gen_labels = []
    all_relation_clf_predictions = []
    all_relation_clf_labels = []
    all_entity_clf_predictions = []
    all_entity_clf_labels = []
    all_structure_clf_predictions = []
    all_structure_clf_labels = []
    all_structure_gen_predictions = []
    all_structure_gen_labels = []
    for it,data in enumerate(tqdm(dataloader,desc='Predicting')):
        tgt_encoded = data[0]
        relation_clf_encoded = data[1]
        structure_clf_encoded = data[2]
        relation_clf_labels = data[3]
        entity_clf_labels = data[4]
        structure_clf_labels = data[5]
        candidate_relations = data[6]
        candidate_entities = data[7]
        candidate_structures = data[8]
        input_srcs = data[9]
        tgt_structure_encoded = data[10]

        structure_clf_outputs = None
        structure_gen_outputs = None
        
        if isinstance(model, T5_Structure_Filling):
            gen_outputs, relation_clf_outputs, entity_clf_outputs, structure_clf_outputs = model.inference(
                input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                input_ids_structure_clf=structure_clf_encoded['input_ids'].to(device),
                relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                structure_clf_attention_mask=structure_clf_encoded['attention_mask'].to(device),
                num_beams=args.eval_beams,
                textual_candidate_relations=candidate_relations,
                textual_candidate_entities=candidate_entities,
                textual_candidate_structures=candidate_structures,
                textual_nlq=input_srcs,
                normalize_relations=args.normalize_relations
            )
        elif isinstance(model, T5_MultiTask_Concat_Relation_Concat_Entity):
            gen_outputs, relation_clf_outputs, entity_clf_outputs = model.inference(
                input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                num_beams=args.eval_beams,
                max_length=args.max_tgt_len,
                textual_candidate_relations=candidate_relations,
                textual_input_src_gen=input_srcs,
                normalize_relations=args.normalize_relations,
                textual_candidate_entities=candidate_entities
            )
        elif isinstance(model, T5_SExpr_Generation_Structure_Generation):
            gen_outputs, relation_clf_outputs, entity_clf_outputs = model.inference(
                input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                num_beams=args.eval_beams,
                textual_candidate_relations=candidate_relations,
                textual_candidate_entities=candidate_entities,
                textual_nlq=input_srcs,
                normalize_relations=args.normalize_relations,
            )
        elif isinstance(model, T5_SExpr_Generation_Structure_Generation_Concat):
            gen_outputs, relation_clf_outputs, entity_clf_outputs, structure_gen_outputs = model.inference(
                input_ids_relation_clf=relation_clf_encoded['input_ids'].to(device),
                relation_clf_attention_mask=relation_clf_encoded['attention_mask'].to(device),
                num_beams=args.eval_beams,
                textual_candidate_relations=candidate_relations,
                textual_candidate_entities=candidate_entities,
                textual_nlq=input_srcs,
                normalize_relations=args.normalize_relations,
            )

        gen_outputs = [p.cpu().numpy() for p in gen_outputs]
        gen_labels = tgt_encoded['input_ids'].numpy()
        all_gen_predictions.extend(gen_outputs)
        all_gen_labels.extend(gen_labels)

        relation_clf_outputs = torch.sigmoid(relation_clf_outputs).detach().cpu().reshape(-1,args.relation_sample_size)
        relation_clf_labels = relation_clf_labels.cpu().reshape(-1,args.relation_sample_size)
        all_relation_clf_predictions.extend([p.numpy() for p in relation_clf_outputs])
        all_relation_clf_labels.extend([l.numpy() for l in relation_clf_labels])

        entity_clf_outputs = torch.sigmoid(entity_clf_outputs).detach().cpu().reshape(-1,args.entity_sample_size)
        entity_clf_labels = entity_clf_labels.cpu().reshape(-1,args.entity_sample_size)
        all_entity_clf_predictions.extend([p.numpy() for p in entity_clf_outputs])
        all_entity_clf_labels.extend([l.numpy() for l in entity_clf_labels])

        if structure_clf_outputs is not None:
            structure_clf_outputs = torch.sigmoid(structure_clf_outputs).detach().cpu().reshape(-1,args.structure_sample_size)
            structure_clf_labels = structure_clf_labels.cpu().reshape(-1, args.structure_sample_size)
            all_structure_clf_predictions.extend([p.numpy() for p in structure_clf_outputs])
            all_structure_clf_labels.extend([l.numpy() for l in structure_clf_labels])
        
        if structure_gen_outputs is not None:
            structure_gen_outputs = [p.cpu().numpy() for p in structure_gen_outputs]
            structure_gen_labels = tgt_structure_encoded['input_ids'].numpy()
            all_structure_gen_predictions.extend(structure_gen_outputs)
            all_structure_gen_labels.extend(structure_gen_labels)

    ex_cnt = 0
    contains_ex_cnt = 0
    output_list = []
    real_total = 0
    for i,pred in enumerate(all_gen_predictions):
        predictions = tokenizer.batch_decode(pred, skip_special_tokens=True)
        gen_label = tokenizer.decode(all_gen_labels[i], skip_special_tokens=True)
        if len(all_structure_clf_predictions) > 0:
            output_list.append({
                'predictions': predictions,
                'gen_label': gen_label,
                'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
                'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],
                'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
                'gold_entity_clf_labels':[float(l) for l in list(all_entity_clf_labels[i])],
                'pred_structure_clf_labels':[float(p) for p in list(all_structure_clf_predictions[i])],
                'gold_structure_clf_labels':[float(l) for l in list(all_structure_clf_labels[i])],
            })
        elif len(all_structure_gen_predictions) > 0:
            structure_predictions = tokenizer.batch_decode(all_structure_gen_predictions[i],skip_special_tokens=True)
            structure_label = tokenizer.decode(all_structure_gen_labels[i], skip_special_tokens=True)
            output_list.append({
                'predictions': predictions,
                'gen_label': gen_label,
                'structure_predictions': structure_predictions,
                'structure_label': structure_label,
                'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
                'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],
                'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
                'gold_entity_clf_labels':[float(l) for l in list(all_entity_clf_labels[i])],
            })
        else:
            output_list.append({
                'predictions': predictions,
                'gen_label': gen_label,
                'pred_relation_clf_labels':[float(p) for p in list(all_relation_clf_predictions[i])],
                'gold_relation_clf_labels':[float(l) for l in list(all_relation_clf_labels[i])],
                'pred_entity_clf_labels':[float(p) for p in list(all_entity_clf_predictions[i])],
                'gold_entity_clf_labels':[float(l) for l in list(all_entity_clf_labels[i])],
            })

        if predictions[0].lower()==gen_label.lower():
            ex_cnt+=1
        if any([x.lower()==gen_label.lower() for x in predictions]):
            contains_ex_cnt+=1
        if gen_label.lower()!='null':
            real_total+=1
    
    print(f"""total:{len(output_list)}, 
                    ex_cnt:{ex_cnt}, 
                    ex_rate:{ex_cnt/len(output_list)}, 
                    real_ex_rate:{ex_cnt/real_total}, 
                    contains_ex_cnt:{contains_ex_cnt}, 
                    contains_ex_rate:{contains_ex_cnt/len(output_list)}
                    real_contains_ex_rate:{contains_ex_cnt/real_total}
                    """)
    
    if output_predictions:
        file_path = os.path.join(output_dir,f'beam_{args.eval_beams}_top_k_predictions.json')
        gen_statistics_file_path = os.path.join(output_dir,f'beam_{args.eval_beams}_gen_statistics.json')
        gen_statistics = {
            'total':len(output_list),
            'exmatch_num': ex_cnt,
            'exmatch_rate': ex_cnt/len(output_list),
            'real_exmatch_rate':ex_cnt/real_total, 
            'contains_ex_num':contains_ex_cnt,
            'contains_ex_rate':contains_ex_cnt/len(output_list),
            'real_contains_ex_rate':contains_ex_cnt/real_total
        }
        dump_json(output_list, file_path, indent=4)
        dump_json(gen_statistics, gen_statistics_file_path,indent=4)
        dataset = load_json('data/CWQ/structure_filling/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_test_all_data_sexpr_masked_candidate_structures.json')
        # dataset = load_json('data/WebQSP/structure_filling/richRelation_2hopValidation_richEntity_CrossEntropyLoss_top100_1parse_ep6/FACC1_elq_entities/WebQSP_test_all_data_sexpr_masked_candidate_structures.json')
        generate_entity_label_map_by_classification_res_sorted(output_list, output_dir, dataset)


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__=='__main__':
    args = _parse_args()
    print(args)

    # do_debug = False
    do_debug = args.do_debug
    
    if do_debug=='True':
        import ptvsd
        server_ip = "0.0.0.0"
        server_port = 12345
        print('Waiting for debugger attach...',flush=True)
        ptvsd.enable_attach(address=(server_ip,server_port))
        ptvsd.wait_for_attach()
    
    set_seed(42) # default seed
    # set parameters
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs
    lr = args.lr
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    tokenizer.add_special_tokens(
            {"additional_special_tokens":["[DES]","[INQ]", "[des]","[inq]"]}
    )
    tokenizer.add_tokens(["[ENT]", "[REL]", "[LIT]", "[ent]", "[rel]", "[lit]"])

    if args.do_train:
        # load data
        train_dataloader = prepare_dataloader(args,'train',tokenizer, batch_size=train_batch_size)
        if args.do_eval:
            dev_dataloader = prepare_dataloader(args,'dev',tokenizer, batch_size=eval_batch_size)
        else: 
            dev_dataloader = None

        if args.model == 'T5_Structure_Filling':
            print('T5_Structure_Filling')
            model = T5_Structure_Filling(
                pretrained_model_path=args.pretrained_model_path, 
                is_test=False,
                device=device,
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                tokenizer=tokenizer,
                entity_sample_size=args.entity_sample_size,
                relation_sample_size=args.relation_sample_size,
                structure_sample_size=args.structure_sample_size,
                do_lower=args.do_lower,
            )
        elif args.model == 'T5_MultiTask_Concat_Relation_Concat_Entity':
            print('T5_MultiTask_Concat_Relation_Concat_Entity')
            model = T5_MultiTask_Concat_Relation_Concat_Entity(
                pretrained_model_path=args.pretrained_model_path, 
                is_test=False,
                device=device,
                max_src_len=args.max_src_len,
                tokenizer=tokenizer,
                entity_sample_size=args.entity_sample_size,
                relation_sample_size=args.relation_sample_size,
                do_lower=args.do_lower,
                add_prefix=args.add_prefix
            )
        elif args.model == 'T5_SExpr_Generation_Structure_Generation':
            print('T5_SExpr_Generation_Structure_Generation')
            model = T5_SExpr_Generation_Structure_Generation(
                pretrained_model_path=args.pretrained_model_path, 
                is_test=False,
                device=device,
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                tokenizer=tokenizer,
                entity_sample_size=args.entity_sample_size,
                relation_sample_size=args.relation_sample_size,
                do_lower=args.do_lower,
                add_prefix=args.add_prefix
            )
        elif args.model == 'T5_SExpr_Generation_Structure_Generation_Concat':
            print('T5_SExpr_Generation_Structure_Generation_Concat')
            model = T5_SExpr_Generation_Structure_Generation_Concat(
                pretrained_model_path=args.pretrained_model_path, 
                is_test=False,
                device=device,
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                tokenizer=tokenizer,
                entity_sample_size=args.entity_sample_size,
                relation_sample_size=args.relation_sample_size,
                do_lower=args.do_lower,
                add_prefix=args.add_prefix,
                structure_gen_beam_size=args.structure_gen_beam_size,
                max_structure_tgt_len=args.max_structure_tgt_len
            )
        model.t5.resize_token_embeddings(len(tokenizer))
        # model = model.to(device)
        model.to(device)
        # define model path to
        output_dir = args.output_dir
        model_save_dir = args.model_save_dir
        model = train_model(args,model,tokenizer,device,train_dataloader,epochs,dev_dataloader,model_save_dir=model_save_dir)
    
    if args.do_predict:
        # test load model
        if args.do_train:
            print()
            print('Use trained model to do prediction')
            model = model.to(device)
        else:
            print()
            print("Loading the weights of the model...")
            if args.model == 'T5_Structure_Filling':
                print('T5_Structure_Filling')
                model = T5_Structure_Filling(
                    pretrained_model_path=args.pretrained_model_path, 
                    is_test=False,
                    device=device,
                    max_src_len=args.max_src_len,
                    max_tgt_len=args.max_tgt_len,
                    tokenizer=tokenizer,
                    entity_sample_size=args.entity_sample_size,
                    relation_sample_size=args.relation_sample_size,
                    structure_sample_size=args.structure_sample_size,
                    do_lower=args.do_lower,
                )
            elif args.model == 'T5_MultiTask_Concat_Relation_Concat_Entity':
                print('T5_MultiTask_Concat_Relation_Concat_Entity')
                model = T5_MultiTask_Concat_Relation_Concat_Entity(
                    pretrained_model_path=args.pretrained_model_path, 
                    is_test=False,
                    device=device,
                    max_src_len=args.max_src_len,
                    tokenizer=tokenizer,
                    entity_sample_size=args.entity_sample_size,
                    relation_sample_size=args.relation_sample_size,
                    do_lower=args.do_lower,
                    add_prefix=args.add_prefix
                )
            elif args.model == 'T5_SExpr_Generation_Structure_Generation':
                print('T5_SExpr_Generation_Structure_Generation')
                model = T5_SExpr_Generation_Structure_Generation(
                    pretrained_model_path=args.pretrained_model_path, 
                    is_test=False,
                    device=device,
                    max_src_len=args.max_src_len,
                    max_tgt_len=args.max_tgt_len,
                    tokenizer=tokenizer,
                    entity_sample_size=args.entity_sample_size,
                    relation_sample_size=args.relation_sample_size,
                    do_lower=args.do_lower,
                    add_prefix=args.add_prefix
                )
            elif args.model == 'T5_SExpr_Generation_Structure_Generation_Concat':
                print('T5_SExpr_Generation_Structure_Generation_Concat')
                model = T5_SExpr_Generation_Structure_Generation_Concat(
                    pretrained_model_path=args.pretrained_model_path, 
                    is_test=False,
                    device=device,
                    max_src_len=args.max_src_len,
                    max_tgt_len=args.max_tgt_len,
                    tokenizer=tokenizer,
                    entity_sample_size=args.entity_sample_size,
                    relation_sample_size=args.relation_sample_size,
                    do_lower=args.do_lower,
                    add_prefix=args.add_prefix,
                    structure_gen_beam_size=args.structure_gen_beam_size,
                    max_structure_tgt_len=args.max_structure_tgt_len
                )
            model.t5.resize_token_embeddings(len(tokenizer))
            state_dict = torch.load(os.path.join(args.model_save_dir,'pytorch_model_epoch_15.bin'))
            model.load_state_dict(state_dict)
            # tokenizer = T5Tokenizer(os.path.join(args.model_save_dir,'vocab_file.bin'))
            model.to(device)
            print('Model loaded')
        
        test_dataloader = prepare_dataloader(args, args.predict_split,tokenizer=tokenizer,batch_size=test_batch_size)
        # print('Predicting Num:', len(test_dataloader)*test_batch_size)
        run_prediction(args,model,device,test_dataloader,tokenizer,output_dir=args.output_dir,output_predictions=True)
        print('Prediction Finished')