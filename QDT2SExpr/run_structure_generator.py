from inputDataset.get_structure_filling_dataset import StructureGenDataset, StructureGenerationExample
from models.T5_structure_multitask import T5_Structure_Generation
from components.utils import dump_json, load_json
import argparse
import os
from functools import partial
import copy
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup


def load_data(split):
    # read data
    # data_file_name = f'data/CWQ/structure_filling/2hopValidation_maskMention_richRelation_CrossEntropyLoss_top100_candidate_entities_merged_FACC1_elq/CWQ_{split}_all_data_sexpr_masked.json'
    # WebQSP
    data_file_name = f'data/WebQSP/structure_filling/richRelation_2hopValidation_richEntity_CrossEntropyLoss_top100_1parse_ep6/FACC1_elq_entities/WebQSP_{split}_all_data_sexpr_masked.json'
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
    parser.add_argument('--max_src_len',default=128, type=int, help='maximum source length')
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
    args = parser.parse_args()
    return args


def _collate_fn(data,tokenizer):
    """For mini-batch dynamic padding"""
    all_src_input_ids = []
    all_tgt_input_ids = []
    # print(len(data))
    for data_tuple in data:
        # print(data_tuple)
        all_src_input_ids.append(data_tuple[0])
        all_tgt_input_ids.append(data_tuple[1])
    
    src_encoded = tokenizer.pad({'input_ids': all_src_input_ids},return_tensors='pt')
    tgt_encoded = tokenizer.pad({'input_ids': all_tgt_input_ids},return_tensors='pt')

    return (
        src_encoded,
        tgt_encoded
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
                examples.append(StructureGenerationExample(x))                
    else:
        examples = [StructureGenerationExample(x) for x in data]
    print(f'Real {split} dataset len: {len(examples)}')

    # examples = examples[:100]
    dataset = StructureGenDataset(examples, 
                            tokenizer=tokenizer,
                            do_lower=args.do_lower,
                            max_src_len=args.max_src_len,
                            max_tgt_len=args.max_tgt_len
                            )

    # print(train_dataset.__getitem__(0))
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            collate_fn=partial(_collate_fn,tokenizer=tokenizer),
                            shuffle=False
                            )
    return dataloader


def save_model(model_save_dir,model_to_save,epoch,tokenizer,is_final_epoch=False):
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


def train_model(args,model,tokenizer,device,train_dataloader,dev_dataloader=None,model_save_dir=None):
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
    t_total = (len(train_dataloader) // iters_to_accumulate) * args.epochs  # Necessary to take into account Gradient accumulation
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
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for it,data in enumerate(tqdm(train_dataloader,desc=f'Epoch {epoch+1}')):
            src_encoded = data[0]
            tgt_encoded = data[1]
            # print(data)
            loss = model(
                input_ids_gen=src_encoded['input_ids'].to(device),
                gen_attention_mask=src_encoded['attention_mask'].to(device),
                gen_labels=tgt_encoded['input_ids'].to(device)
            )

            loss = loss / iters_to_accumulate

            if (it+1)%iters_to_accumulate == 0:
                loss.backward()
                opti.step()
                lr_scheduler.step()
                opti.zero_grad()
            
            running_loss += loss.item()

            if (it + 1) % print_every == 0: # Print training loss inforamtion
                print(flush=True)
                print("Iteration {}/{} of epoch {} (Total:{}) complete. Loss : {} "
                    .format(it+1, num_iterations, epoch+1, args.epochs, running_loss / print_every)
                    ,flush=True)

            running_loss = 0.0

        if args.do_eval:
            # after training on one epoch, check dev_loss
            dev_loss = evaluate_loss(model,device,dev_dataloader)
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
            save_model(model_save_dir,model_to_save,(epoch+1), tokenizer, is_final_epoch=False)

        
    # empty cache
    torch.cuda.empty_cache()
    # save final model
    # save_model(model_save_dir,model_to_save,args.epochs,tokenizer,is_final_epoch=True)
    
    return model_to_save


def evaluate_loss(model,device,dataloader):
    model.eval()
    mean_loss = 0
    count = 0
    with torch.no_grad():
        for it, data in enumerate(tqdm(dataloader,desc='Evaluating')):
            src_encoded = data[0]
            tgt_encoded = data[1]

            loss = model(
                input_ids_gen=src_encoded['input_ids'].to(device),
                gen_attention_mask=src_encoded['attention_mask'].to(device),
                gen_labels=tgt_encoded['input_ids'].to(device)
            )
            
            mean_loss += loss.item()
            count+=1
    
    # torch.cuda.empty_cache()
    
    return mean_loss/count


def run_prediction(args,model,device,dataloader,tokenizer,output_dir,output_predictions=True):
    print()
    print(f'Start predicting {args.predict_split}, beam_size:{args.eval_beams}, batch_size:{args.test_batch_size}')
    
    model.eval()
    all_gen_predictions = []
    all_gen_labels = []
    
    for it,data in enumerate(tqdm(dataloader,desc='Predicting')):
            src_encoded = data[0]
            tgt_encoded = data[1]
            
            gen_outputs = model.inference(
                input_ids_gen=src_encoded['input_ids'].to(device),
                gen_attention_mask=src_encoded['attention_mask'].to(device),
                num_beams=args.eval_beams,
                max_length=args.max_tgt_len,
            )

            gen_outputs = [p.cpu().numpy() for p in gen_outputs]
            gen_labels = tgt_encoded['input_ids'].numpy()
            all_gen_predictions.extend(gen_outputs)
            all_gen_labels.extend(gen_labels)

    ex_cnt = 0
    contains_ex_cnt = 0
    output_list = []
    real_total = 0
    for i,pred in enumerate(all_gen_predictions):
        predictions = tokenizer.batch_decode(pred, skip_special_tokens=True)
        gen_label = tokenizer.decode(all_gen_labels[i], skip_special_tokens=True)
        output_list.append({
            'predictions':predictions,
            'gen_label':gen_label,
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

    # set seed, for reproduce
    set_seed(42) # default seed
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
    # tokenizer.add_special_tokens(
    #         {"additional_special_tokens":["[DES]","[INQ]","[ENT]","[REL]","[LIT]", "[des]","[inq]","[ent]","[rel]","[lit]"]}
    # )

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
    
        model = T5_Structure_Generation(
            args.pretrained_model_path,
            is_test=False,
        )
        model.t5.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        output_dir = args.output_dir
        model_save_dir = args.model_save_dir

        model = train_model(args,model,tokenizer,device,train_dataloader,dev_dataloader,model_save_dir=model_save_dir)
    
    if args.do_predict:
        if args.do_train:
            print()
            print('Use trained model to do prediction')
            model = model.to(device)
        else:
            print()
            print("Loading the weights of the model...")
            model = T5_Structure_Generation(
                args.pretrained_model_path,
                is_test=False,
            )
            model.t5.resize_token_embeddings(len(tokenizer))
            state_dict = torch.load(os.path.join(args.model_save_dir,'pytorch_model_epoch_10.bin'))
            model.load_state_dict(state_dict)
            # tokenizer = T5Tokenizer(os.path.join(args.model_save_dir,'vocab_file.bin'))
            model.to(device)
            print('Model loaded')

        test_dataloader = prepare_dataloader(args, args.predict_split,tokenizer=tokenizer,batch_size=test_batch_size)
        # print('Predicting Num:', len(test_dataloader)*test_batch_size)
        run_prediction(args,model,device,test_dataloader,tokenizer,output_dir=args.output_dir,output_predictions=True)

        print('Prediction Finished')
