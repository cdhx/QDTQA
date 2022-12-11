
from torch.utils import data
from sys import path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
from datasets import load_dataset
import torch
import os
import json
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForTokenClassification
from datasets import load_metric
import pandas as pd
from sys import path

path.append('..')
# from utils.SPARQL_utils import FBExecutor
from utils.utils import *
import math
import stanza
import logging
import numpy as np
from utils.similarity_util import *
from modeling.clue_decipher_mc import *


class seq2seq_decomp_model:
    def __init__(self, args):
        self.args = args
        self.GPU_NUM = args.gpu_num
        self.max_length = self.args.max_length
        self.seq_eval = self.args.seq_eval
        self.save_dir = '../../model/' + self.args.file_name + '.pkl'
        self.batch_size = self.args.batch_size
        self.inference_batch_size = self.args.inference_batch_size
        self.epoch=args.epoch

        logging.info('Model will saved in ' + self.save_dir)
        self.loded_model = False

        self.tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.tokenizer.add_tokens(['[DES]', '[INQL]', '[INQR]'])
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = torch.device("cuda:" + str(self.GPU_NUM)) if torch.cuda.is_available() else 'cpu'
        logging.info('seq2seq decomp model device ' + str(self.device))

        self.clue_decipher_mc_config = {'max_length': 64,
                                        'gpu_num': self.GPU_NUM,
                                        'batch_size': 64,
                                        'inference_batch_size': 256,
                                        'sample_rate': 1.0,
                                        'file_name': 'clue_decipher_mc'}
        self.clue_decipher_mc_args = Dict2Obj(self.clue_decipher_mc_config)
        self.clue_decipher_model = clue_decipher_mc_model(self.clue_decipher_mc_args)
        self.clue_decipher_model.load_model()

    def set_param(self):
        self.bleu_metric = load_metric("bleu")
        self.get_dataset()

    def get_dataset(self):
        logging.info('Generate training data...')

        def deal_data(part):

            train=readjson('../../data/qdtrees/qdt_'+part+'.json')

            train_json = [
                {'ID': train[index]['ID'],
                 'question': clean_question(train[index]['question']),
                 'decomposition': clean_question(train[index]['decomposition']),
                 'decomposition_2_part': clean_question(train[index]['decomposition_2_part']),
                 'source':train[index]['source']
                 } for index in range(len(train))]


            return train_json

        self.train_json, self.dev_json, self.test_json = deal_data('train'), deal_data(
            'dev'), deal_data( 'test')

        logging.info('Data generated ! ')

    def do_eval(self, eval_json):

        self.model.eval()

        question_list = [row['question'] for row in eval_json]
        prefix_list = ["Decomp" for row in eval_json]
        inputbatch = [prefix_list[index] + ": " + question_list[index] for index in range(len(question_list))]
        if self.seq_eval:
            labelbatch = [row['decomposition_2_part'] for row in eval_json]
        else:
            labelbatch = [row['decomposition'] for row in eval_json]

        modi_eval_pred, eval_pred = self.batch_predict(question_list, prefix_list)

        eval_example_num = len(inputbatch)
        eval_acc = sum([labelbatch[index] == eval_pred[index] for index in
                         range(len(inputbatch))])
        eval_modi_acc = sum([labelbatch[index] == modi_eval_pred[index] for index in
                                  range(len(inputbatch))])

        eval_modi_acc = eval_modi_acc / eval_example_num
        eval_acc = eval_acc / eval_example_num

        eval_bleu=cal_bleu(eval_pred,labelbatch)
        eval_rouge = cal_rouge(eval_pred, labelbatch)
        eval_modi_bleu=cal_bleu(modi_eval_pred,labelbatch)
        eval_modi_rouge = cal_rouge(modi_eval_pred, labelbatch)

        return {'modi_acc': eval_modi_acc, 'modi_bleu':eval_modi_bleu,'modi_rouge':eval_modi_rouge,
                    'acc': eval_acc,'bleu':eval_bleu,'rouge':eval_rouge}


    def train_model(self, num_epoch=1,direct_train=False):
        self.set_param()
        self.num_train_epochs = num_epoch
        if not direct_train:
            self.load_model()
        logging.info('Evaluate exist model on dev and test set...')
        if not self.loded_model:
            self.dev_best_matrix = self.test_best_matrix = {'modi_acc': 0, 'acc': 0}
        else:
            self.test_best_matrix = self.do_eval(self.test_json)
            self.dev_best_matrix = self.do_eval(self.dev_json)
        logging.info('dev  last best:' + str(self.dev_best_matrix))
        logging.info('test last best:' + str(self.test_best_matrix))
        logging.info('Start training')
        optimizer = Adafactor(
            self.model.parameters(),
            lr=1e-4,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

        for epoch in range(self.num_train_epochs):
            loss_sum = 0.0
            example_num = 0
            self.model.train()

            num_of_batches = math.ceil(len(self.train_json) / self.batch_size)
            for i in tqdm(range(num_of_batches)):
                new_json = self.train_json[i * self.batch_size:i * self.batch_size + self.batch_size]
                question_list = [row['question'] for row in new_json]
                prefix_list = ["Decomp" for row in new_json]
                inputbatch = [prefix_list[index] + ": " + question_list[index] for index in range(len(question_list))]

                if self.seq_eval:
                    labelbatch = [row['decomposition_2_part'] for row in new_json]
                else:
                    labelbatch = [row['decomposition'] for row in new_json]

                inputbatch = \
                    self.tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=self.max_length,
                                                     truncation=True,
                                                     return_tensors='pt')[
                        "input_ids"]
                labelbatch = self.tokenizer.batch_encode_plus(labelbatch, padding=True, max_length=self.max_length,
                                                              return_tensors="pt")[
                    "input_ids"]
                inputbatch = inputbatch.to(self.device)
                labelbatch = labelbatch.to(self.device)
                example_num += len(inputbatch)

                optimizer.zero_grad()
                out = self.model(input_ids=inputbatch, labels=labelbatch)  # output
                loss = out.loss
                loss.backward()
                optimizer.step()
                loss_sum += loss.cpu().data.numpy()

                # calculate active accuracy
                example_num += labelbatch.shape[0]
                # calculate after mask

            self.loded_model = True
            loss_sum = loss_sum / example_num
            train_matrix = self.do_eval(self.train_json)

            dev_matrix = self.do_eval(self.dev_json)
            test_matrix = self.do_eval(self.test_json)
            if self.seq_eval:# if sequence-based evaluation
                logging.info(
                    "epoch % d,train Macc:%.5f, dev Macc:%.5f,acc:%.5f, test Macc:%.5f,acc:%.5f" % (
                        epoch, train_matrix['modi_acc'],
                        dev_matrix['modi_acc'], dev_matrix['acc'],
                        test_matrix['modi_acc'],test_matrix['acc']))
            else:# if tree-based evaluation
                logging.info(
                    "epoch % d,train Macc:%.5f, dev Macc:%.5f,acc:%.5f, test Macc:%.5f,acc:%.5f" % (
                        epoch, train_matrix['modi_acc'],
                        dev_matrix['modi_acc'], dev_matrix['acc'],
                        test_matrix['modi_acc'],test_matrix['acc']))

            if dev_matrix['modi_acc'] > self.dev_best_matrix['modi_acc']:
                torch.save(self.model.state_dict(), self.save_dir)
                logging.info('Best model on dev set saved! acc increase:' + str(
                    (dev_matrix['modi_acc'] - self.dev_best_matrix['modi_acc'])))
                self.dev_best_matrix = dev_matrix


    def predict(self, question, prefix):
        if not self.loded_model:
            self.load_model()
        self.model.eval()
        input_ids = self.tokenizer.encode(prefix+": "+question, return_tensors="pt")  # Batch size 1
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            pred = self.model.generate(input_ids, max_length=64)
            pred = self.tokenizer.decode(pred[0], skip_special_tokens=True).strip()
            pred = pred.replace('[DES] ', '[DES]').replace('[INQL] ', '[INQL]').replace('[INQR] ', '[INQR]').replace(
                '[INQ] ',  '[INQ]').strip()

            pred = pred.replace('[DES]', ' [DES] ').replace('[INQL]', ' [INQL] ').replace('[INQR]',  ' [INQR] ').replace(
                '[INQ]', ' [INQ] ').strip()
            modi_text_result = self.clue_decipher_model.e2e_predict(pred, question)

            text_result = ' '.join(pred.split()).strip()
        return modi_text_result, text_result

    def batch_predict(self, question_list, prefix_list):
        # The new version of batch_prediction integrates the batch part into main, which can pass arbitrary large arrays
        if not self.loded_model:
            self.load_model()
        self.model.eval()
        pred=[]
        modi_pred=[]
        num_of_batches = math.ceil(len(question_list) / self.inference_batch_size)
        for i in tqdm(range(num_of_batches)):
            question_list_batch = question_list[i * self.inference_batch_size:i * self.inference_batch_size + self.inference_batch_size]
            prefix_list_batch = prefix_list[i * self.inference_batch_size:i * self.inference_batch_size + self.inference_batch_size]

            inputbatch = [prefix_list_batch[index] + ": " + question_list_batch[index] for index in range(len(question_list_batch))]
            inputbatch = self.tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=self.max_length,
                                                          truncation=True, return_tensors='pt')
            inputbatch = inputbatch.to(self.device)
            with torch.no_grad():
                pred_batch = self.model.generate(input_ids=inputbatch['input_ids'], max_length=64)
                pred_batch = self.tokenizer.batch_decode(pred_batch, skip_special_tokens=True)
                pred_batch = [
                    x.replace('[DES] ', '[DES]').replace('[INQL] ', '[INQL]').replace('[INQR] ', '[INQR]').replace('[INQ] ',
                                                                                                                   '[INQ]').strip()
                    for x in pred_batch]
                pred_batch = [
                    x.replace('[DES]', ' [DES] ').replace('[INQL]', ' [INQL] ').replace('[INQR]', ' [INQR] ').replace(
                        '[INQ]', ' [INQ] ').strip() for x in pred_batch]

                modi_pred_batch = self.clue_decipher_model.e2e_batch_predict(pred_batch, question_list_batch)
                modi_pred_batch = [' '.join(x.split()) for x in modi_pred_batch]

                pred_batch = [' '.join(x.split()) for x in pred_batch]
                modi_pred_batch = [' '.join(x.split()) for x in modi_pred_batch]
            pred += pred_batch
            modi_pred += modi_pred_batch

        return modi_pred, pred

    def load_model(self):
        if os.path.exists(self.save_dir):
            self.model.load_state_dict(torch.load(self.save_dir, map_location=self.device))
            self.model.to(self.device)
            logging.info('Successfully load exist model! ' + self.save_dir)
            self.loded_model = True
        else:
            logging.info('No model exist in ' + self.save_dir + ', training new model!')
            self.model.to(self.device)
            self.train_model(self.epoch,direct_train=True) 
