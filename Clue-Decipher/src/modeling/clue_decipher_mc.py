
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
from transformers import AutoTokenizer, BertForMultipleChoice, AutoConfig, AutoModel, AutoModelForTokenClassification
from datasets import load_metric
import pandas as pd
from sys import path

path.append('..')
from utils.SPARQL_utils import FBExecutor
from utils.utils import *
import math
import stanza
import logging
import numpy as np
import random


class clue_decipher_mc_model:
    def __init__(self, args):
        self.args=args
        self.save_dir = '../../model/' + args.file_name + '.pkl'

        logging.info('Model will saved in ' + self.save_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.tokenizer.add_tokens(['[des]', '[inql]', '[inqr]'])

        self.model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.GPU_NUM = args.gpu_num
        self.device = torch.device("cuda:" + str(self.GPU_NUM)) if torch.cuda.is_available() else 'cpu'
        self.loded_model = False

        logging.info('multi-choice model device ' + str(self.device))
        self.max_length = args.max_length


    def set_param(self):

        self.training_data = self.args.training_data
        self.eval_data = self.args.eval_data
        self.task = self.args.task
        self.token2label = {'N': 0, 'Y': 1}

        self.batch_size = self.args.batch_size
        self.inference_batch_size = self.args.inference_batch_size
        self.sample_rate = self.args.sample_rate
        self.get_dataset()

    def corrupt_question(self, question):
        question_token_list = question.split(' ')
        question_token_list = [x for x in question_token_list if x != '']
        question_token_list_without_sp_tok = [x for x in question_token_list if
                                              x not in ['[DES]', '[INQL]', '[INQR]', '[SEP]', '[INQ]']]
        corrupted_question_token_list = []
        for index, question_token in enumerate(question_token_list):
            if question_token not in ['[DES]', '[INQL]', '[INQR]', '[DES]', '[INQ]']:
                if question_token[-1] in ['s'] and len(question_token) > 3:  # Plural processing, the 's' of too short word  can not be removed
                    if random.random() < 0.2:
                        corrupted_question_token_list.append(question_token[:-1])
                    else:
                        corrupted_question_token_list.append(question_token)
                elif question_token[-1] in [',']:  # Handling of commas
                    if random.random() < 0.2:
                        corrupted_question_token_list.append(question_token[:-1])
                    else:
                        corrupted_question_token_list.append(question_token)
                elif question_token[-2:] == "'s":  # Handling of ’s
                    if random.random() > 0.2:
                        corrupted_question_token_list.append(question_token[:-2])
                        corrupted_question_token_list.append("'s")
                    else:
                        corrupted_question_token_list.append(question_token)
                else:  # 其他词
                    if random.random() > 0.1:  # no change
                        corrupted_question_token_list.append(question_token)
                    elif random.random() > 0.003:  # Replace with the result of encoding and decoding back
                        self.tokenizer.convert_tokens_to_string(
                            self.tokenizer.convert_ids_to_tokens(
                                self.tokenizer(question_token)['input_ids'][:-1]))
                    elif random.random() > 0.002:  # Change a word at random
                        corrupted_question_token_list.append(question_token_list_without_sp_tok[random.randint(0, len(
                            question_token_list_without_sp_tok) - 1)])
                    elif random.random() > 0.001:  # Add a word at random
                        corrupted_question_token_list.append(question_token)
                        corrupted_question_token_list.append(question_token_list_without_sp_tok[random.randint(0, len(
                            question_token_list_without_sp_tok) - 1)])
                    else:  # Drop this word
                        pass
            else:
                corrupted_question_token_list.append(question_token)
        return ' '.join(corrupted_question_token_list)

    def corrupt_data(self, linear_qdt):
        special_token_num = linear_qdt.count('[INQR]') + linear_qdt.count(
            '[INQL]') + linear_qdt.count('[DES]') + linear_qdt.count('[INQ]')

        qdt_token_list = linear_qdt.split(' ')
        question_token_list = [x for x in qdt_token_list if x not in ['[INQR]', '[INQL]', '[INQ]', '[DES]', '[SEP]']]

        clue_list = []  # original question with only one special symbol
        decipher_list = []
        label_list = []
        for example_index in range(special_token_num):  # How many special tokens have you processed in that linear_qdt
            this_clue_token_list = []  # current input
            this_special_token = ''  # the special token in the current example
            special_token_count = 0  # How many special symbols have gone through
            special_token_position = 0  # Record the token position of this special in the sentence to record the approximate position
            # The decipher and label for each example are a list
            label_example = []
            decipher_example = []
            for index, qdt_token in enumerate(qdt_token_list):
                if qdt_token not in ['[DES]', '[INQL]', '[INQR]', '[DES]', '[INQ]']:
                    this_clue_token_list.append(qdt_token)
                else:
                    if special_token_count == example_index:  # If this special symbol is the one that should be added currently
                        this_clue_token_list.append(qdt_token)
                        this_special_token = qdt_token
                        special_token_position = index - special_token_count
                    special_token_count += 1
            # original clue(branch)
            this_clue = ' '.join(this_clue_token_list)
            # corrupt clue
            corrupt_clue = self.corrupt_question(this_clue)

            # Find an approximate insertion point

            # Constructs examples of decoding results to judge

            if special_token_position > len(question_token_list) - 2:
                around_index = [x for x in range(len(question_token_list) - 4, len(question_token_list) + 1)]
            elif special_token_position < 2:
                around_index = [x for x in range(5)]
            else:
                around_index = [x for x in range(special_token_position - 2, special_token_position + 3)]
            for index in around_index:
                this_decipher_token_list = question_token_list[:]
                this_decipher_token_list.insert(index, this_special_token)
                this_decipher = ' '.join(this_decipher_token_list)
                decipher_example.append(this_decipher)
                if this_decipher == this_clue:
                    label_example.append(self.token2label['Y'])
                else:
                    label_example.append(self.token2label['N'])
            concat_data = list(zip(decipher_example, label_example))
            random.Random(random.randint(0, 100)).shuffle(concat_data)
            decipher_example, label_example = zip(*concat_data)
            decipher_example, label_example = list(decipher_example), list(label_example)

            clue_list.append([corrupt_clue] * len(decipher_example))

            if sum(label_example) != 1:
                print('emmM?')
            label = label_example.index(1)


            decipher_list.append(decipher_example)
            label_list.append(label)

        return clue_list, decipher_list, label_list

    def get_dataset(self):
        # read annotate file，train test split
        logging.info('Generate training data...')

        def deal_data(part):
            train = readjson('../../data/qdtrees/qdt_' + part + '.json')

            if self.task == 'conj':
                train = [x for x in train if
                         'inner_questions' not in sum([list(y.keys()) for y in x['decomposition']['root_question']],
                                                      [])]
            elif self.task == 'comp':
                train = [x for x in train if
                         'inner_questions' in sum([list(y.keys()) for y in x['decomposition']['root_question']],
                                                  [])]
            train_clue = []
            train_decipher = []
            train_label = []
            for exemple in train:
                linear_qdt=exemple['decomposition']
                clue_list, decipher_list, label_list = self.corrupt_data(linear_qdt)
                train_clue = train_clue + clue_list
                train_decipher = train_decipher + decipher_list
                train_label = train_label + label_list


            train_json = [{
                'clue': train_clue[index], 'decipher': train_decipher[index], 'label': train_label[index]} for
                index in
                range(len(train_decipher))]
            return train_json

        self.train_json, self.dev_json, self.test_json = deal_data('train'), deal_data(
       'dev'), deal_data('test')

        logging.info('Data generated ! ')

    def do_eval(self, eval_json):
        eval_acc = 0
        eval_example_num = 0
        self.model.eval()
        num_of_batches = math.ceil(len(eval_json) / self.inference_batch_size)
        for i in tqdm(range(num_of_batches)):
            new_json = eval_json[
                       i * self.inference_batch_size:i * self.inference_batch_size + self.inference_batch_size]
            input_ids, attention_mask, token_type_ids = [], [], []
            for row in new_json:
                input_prompt = row['clue']
                input_choice = row['decipher']

                text = self.tokenizer(input_prompt, text_pair=input_choice, padding='max_length', truncation=True,
                                      max_length=self.max_length,
                                      return_tensors='pt')

                input_ids.append(text['input_ids'].tolist())
                attention_mask.append(text['attention_mask'].tolist())
                token_type_ids.append(text['token_type_ids'].tolist())
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            token_type_ids = torch.tensor(token_type_ids)
            label = torch.tensor([x['label'] for x in new_json])

            input_ids, attention_mask, token_type_ids, label = input_ids.to(self.device), attention_mask.to(
                self.device), token_type_ids.to(self.device), label.to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 labels=label).logits  # output
            eval_example_num += label.shape[0]
            eval_acc += (out.argmax(1) == label).sum().item()
        eval_acc = eval_acc / eval_example_num
        return {'acc': eval_acc}

    def train_model(self, num_epoch=1,direct_train=False):
        self.set_param()
        self.num_train_epochs = num_epoch
        if not direct_train:
            self.load_model()
        logging.info('Evaluate exist model on dev and test set...')
        if not self.loded_model:  # If not loaded into the existing model, no evaluation, directly set to zero, save time
            self.dev_best_matrix = self.test_best_matrix = {'acc': 0}
        else:
            self.dev_best_matrix = self.do_eval(self.dev_json)
            self.test_best_matrix = self.do_eval(self.test_json)
        logging.info('dev  last best:' + str(self.dev_best_matrix))
        logging.info('test last best:' + str(self.test_best_matrix))
        logging.info('Start training')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=1e-4)  # AdamW
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

        for epoch in range(self.num_train_epochs):
            loss_sum = 0.0
            example_num = 0
            self.model.train()

            num_of_batches = math.ceil(len(self.train_json) / self.batch_size)
            for i in tqdm(range(num_of_batches)):

                new_json = self.train_json[i * self.batch_size:i * self.batch_size + self.batch_size]

                input_ids, attention_mask, token_type_ids = [], [], []
                for row in new_json:
                    input_prompt = row['clue']
                    input_choice = row['decipher']

                    text = self.tokenizer(input_prompt, text_pair=input_choice, padding='max_length', truncation=True,
                                          max_length=self.max_length,
                                          return_tensors='pt')

                    input_ids.append(text['input_ids'].tolist())
                    attention_mask.append(text['attention_mask'].tolist())
                    token_type_ids.append(text['token_type_ids'].tolist())
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.tensor(attention_mask)
                token_type_ids = torch.tensor(token_type_ids)
                label = torch.tensor([x['label'] for x in new_json])

                input_ids, attention_mask, token_type_ids, label = input_ids.to(self.device), attention_mask.to(
                    self.device), token_type_ids.to(self.device), label.to(self.device)

                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 labels=label)  # output
                loss = out.loss
                loss.backward()
                optimizer.step()
                loss_sum += loss.cpu().data.numpy()

                # calculate active accuracy
                example_num += label.shape[0]
                # calculate after mask
            loss_sum = loss_sum / example_num
            train_matrix = self.do_eval(self.train_json)
            dev_matrix = self.do_eval(self.dev_json)
            test_matrix = self.do_eval(self.test_json)
            logging.info(
                "epoch % d,train loss:%f,acc:%f,dev acc:%f,test acc:%f" % (
                    epoch, loss_sum, train_matrix['acc'], dev_matrix['acc'], test_matrix['acc']))

            if dev_matrix['acc'] > self.dev_best_matrix['acc']:
                torch.save(self.model.state_dict(), self.save_dir)
                logging.info('Best model on dev set saved! acc increase:' + str(
                    (dev_matrix['acc'] - self.dev_best_matrix['acc'])))
                self.dev_best_matrix = dev_matrix
                self.loded_model = True  # The model has been loaded

    def predict(self, clue, original_question):
        # Given a prediction result with only one special symbol (clue) and the original question, return the original question with the special symbol inserted
        if not self.loded_model:
            self.load_model()
        self.model.eval()
        # What's the special token in clue
        this_special_token = [x for x in clue.split() if x in ['[SEP]', '[DES]', '[INQL]', '[INQR]', '[INQ]']][0]
        # What is the approximate location of the special token
        special_token_position = [index for index, x in enumerate(clue.split()) if x == this_special_token][0]

        # Original problem construction option
        question_token_list = original_question.split()
        # Construct candidate insertion points
        if special_token_position > len(question_token_list) - 2:
            around_index = [x for x in range(len(question_token_list) - 4, len(question_token_list) + 1)]
        elif special_token_position < 2:
            around_index = [x for x in range(5)]
        else:
            around_index = [x for x in range(special_token_position - 2, special_token_position + 3)]
        # Construct candidate options
        decipher_example = []
        for index in around_index:
            this_decipher_token_list = question_token_list[:]
            this_decipher_token_list.insert(index, this_special_token)
            this_decipher = ' '.join(this_decipher_token_list)
            decipher_example.append(this_decipher)

        # predict
        encoding = self.tokenizer([clue for x in range(len(decipher_example))], decipher_example,
                                  return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**{k: v.unsqueeze(0) for k, v in encoding.items()}).logits
        answer_choice = outputs.argmax(1)
        text_result = decipher_example[answer_choice[0]]

        return text_result
    def clues_to_clue_list(self,linear_qdt):
        #linear qdt convert clue list
        special_token_num = linear_qdt.count('[INQR]') + linear_qdt.count(
            '[INQL]') + linear_qdt.count('[DES]') + linear_qdt.count('[INQ]') + linear_qdt.count('[SEP]')
        if special_token_num==0:
            return []
        qdt_token_list = linear_qdt.split(' ')
        clue_list = []
        for example_index in range(special_token_num):  # How many special tokens have you processed in that linear_qdt
            this_clue_token_list = []  # current input
            special_token_count = 0
            for qdt_token in qdt_token_list:
                if qdt_token not in ['[SEP]', '[DES]', '[INQL]', '[INQR]', '[INQ]']:
                    this_clue_token_list.append(qdt_token)
                else:
                    if special_token_count == example_index:
                        this_clue_token_list.append(qdt_token)
                    special_token_count += 1
            this_clue = ' '.join(this_clue_token_list)
            clue_list.append(this_clue)
        return clue_list

    def e2e_predict(self, linear_qdt, original_question):
        # Given the complete prediction result (clue) and the original question, return the original question with all the special symbols inserted
        # Break the complete prediction into several clues
        special_token_num = linear_qdt.count('[INQR]') + linear_qdt.count(
            '[INQL]') + linear_qdt.count('[DES]') + linear_qdt.count('[INQ]') +linear_qdt.count('[SEP]')
        if special_token_num==0:
            return original_question
        qdt_token_list = linear_qdt.split(' ')
        clue_list=[]
        for example_index in range(special_token_num):
            this_clue_token_list = []  # current input
            special_token_count = 0
            for qdt_token in qdt_token_list:
                if qdt_token not in ['[SEP]', '[DES]', '[INQL]', '[INQR]', '[INQ]']:
                    this_clue_token_list.append(qdt_token)
                else:
                    if special_token_count == example_index:
                        this_clue_token_list.append(qdt_token)
                    special_token_count += 1
            this_clue = ' '.join(this_clue_token_list)
            clue_list.append(this_clue)
        decipher_list=self.batch_predict(clue_list,[original_question for x in range(len(clue_list))])

        # Merge decoding result
        text_result=self.clue_list2full_qdt(decipher_list)

        return text_result
    def e2e_batch_predict(self,linear_qdt_list,original_question_list):
        clue_list=[]
        flat_original_question_list=[]# Copy the flattened original question list according to the number of special symbols
        clue_num_for_every_question=[]
        for index,linear_qdt in enumerate(linear_qdt_list):
            special_token_num = linear_qdt.count('[INQR]') + linear_qdt.count(
                '[INQL]') + linear_qdt.count('[DES]') + linear_qdt.count('[INQ]') +linear_qdt.count('[SEP]')
            clue_num_for_every_question.append(special_token_num)
            if special_token_num==0:
                continue

            qdt_token_list = linear_qdt.split(' ')
            clue_list_for_a_question=[]
            for example_index in range(special_token_num):
                this_clue_token_list = []  # current input
                special_token_count = 0
                for qdt_token in qdt_token_list:
                    if qdt_token not in ['[SEP]', '[DES]', '[INQL]', '[INQR]', '[INQ]']:
                        this_clue_token_list.append(qdt_token)
                    else:
                        if special_token_count == example_index:
                            this_clue_token_list.append(qdt_token)
                        special_token_count += 1
                this_clue = ' '.join(this_clue_token_list)
                clue_list_for_a_question.append(this_clue)
            clue_list+=clue_list_for_a_question
            flat_original_question_list+=[original_question_list[index] for x in range(len(clue_list_for_a_question))]

        decipher_list=self.batch_predict(clue_list,flat_original_question_list)

        final_prediction_list=[]
        clue_have_visit=0
        for index,clue_num in enumerate(clue_num_for_every_question):
            if clue_num==0:
                final_prediction=original_question_list[index]
            else:
                final_prediction=self.clue_list2full_qdt(decipher_list[clue_have_visit:clue_have_visit+clue_num])
            final_prediction_list.append(final_prediction)
            clue_have_visit+=clue_num
        return final_prediction_list
    def batch_predict(self, clue_list, flat_original_question_list):
        #batch predict, given a list of clues and original questions, returns the result after insertion of the split point
        if not self.loded_model:
            self.load_model()
        self.model.eval()

        input_ids, attention_mask, token_type_ids,decipher_list= [], [], [],[]
        for clue_index,clue in enumerate(clue_list):
            original_question=flat_original_question_list[clue_index]
            try:
                this_special_token = [x for x in clue.split() if x in ['[SEP]', '[DES]', '[INQL]', '[INQR]', '[INQ]']][0]
            except:
                pass
            special_token_position = [index for index, x in enumerate(clue.split()) if x == this_special_token][0]
            question_token_list = original_question.split()

            if special_token_position > len(question_token_list) - 2:
                around_index = [x for x in range(len(question_token_list) - 4, len(question_token_list) + 1)]
            elif special_token_position < 2:
                around_index = [x for x in range(5)]
            else:
                around_index = [x for x in range(special_token_position - 2, special_token_position + 3)]

            decipher_example = []
            for pos_index in around_index:
                this_decipher_token_list = question_token_list[:]
                this_decipher_token_list.insert(pos_index, this_special_token)
                this_decipher = ' '.join(this_decipher_token_list)
                decipher_example.append(this_decipher)
            decipher_list.append(decipher_example)
            text = self.tokenizer([clue for x in range(len(decipher_example))], text_pair=decipher_example, padding='max_length', truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')
            input_ids.append(text['input_ids'].tolist())
            attention_mask.append(text['attention_mask'].tolist())
            token_type_ids.append(text['token_type_ids'].tolist())
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        input_ids, attention_mask, token_type_ids = input_ids.to(self.device), attention_mask.to(
            self.device), token_type_ids.to(self.device)
        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits  # output
        except:
            pass

        answer_choice_list=out.argmax(1)
        text_result_list=[x[answer_choice_list[index]] for index, x in enumerate(decipher_list)]

        return text_result_list


    def load_model(self):
        if os.path.exists(self.save_dir):
            logging.info('Loadding model from ' + self.save_dir+' ...')
            self.model.load_state_dict(torch.load(self.save_dir, map_location=self.device))
            self.model.to(self.device)
            logging.info('Successfully load exist model! ' + self.save_dir)
            self.loded_model = True
        else:
            logging.info('No model exist in ' + self.save_dir + ', training new model!')
            self.model.to(self.device)
            self.train_model(20,direct_train=True)

    def clue_list2full_qdt(self, clue_list):
        # Converts the clue list into a complete qdt with all the special symbols
        # becomes one if there are two identical special symbols next to each other
        if len(clue_list)==1:
            return clue_list[0]
        clue_list_token_list = [clue.split() for clue in clue_list]
        for ind,_ in enumerate(clue_list_token_list):
            clue_list_token_list[ind]+=['<padding>' for x in range(len(clue_list))]

        shift_list = [0 for x in range(len(clue_list))]  #
        assert len(list(set([len(x) for x in clue_list_token_list]))) == 1, "Every clue do not have same length！"
        linear_qdt_token_list = []
        token_index = 0
        while True:

            this_position_token_list = [clue_list_token_list[clue_index][token_index + shift_list[clue_index]] for
                                        clue_index in range(len(clue_list_token_list))]

            # If there is a special symbol (in the current position, there are at least two different words, namely the special symbol and the words in the original question)
            if len(list(set([x for x in this_position_token_list if x!='<padding>']))) > 1:
                # Instead of adding the words of the original sentence, a special symbol is added, so token_index does not need to be added by one
                for clue_index, token_in_different_clue in enumerate(
                        this_position_token_list):  # Each token in a different clue in the current position
                    if token_in_different_clue in ['[DES]', '[INQ]', '[INQR]', '[INQL]', '[SEP]']:
                        shift_list[clue_index] += 1  # Make an offset for the corresponding clue
                        linear_qdt_token_list.append(token_in_different_clue)
                        # print(token_index,token_in_different_clue)
            # If there are no different words, it's either a common word in the original question, token_inde+1 or it's the last word in the last clue, and you need to decide whether it's a special symbol, whether token_index should be +1 or not
            elif len(list(set([x for x in this_position_token_list if x!='<padding>']))) == 1:
                linear_qdt_token_list.append([x for x in this_position_token_list if x!='<padding>'][0])
                token_index += 1  # Add one only if you add the token in the original question
                # print(token_index, this_position_token_list[0])
            elif len(list(set([x for x in this_position_token_list if x!='<padding>'])))  == 0:
                break
            # print(shift_list)
            # print(linear_qdt_token_list)
        linear_qdt = ' '.join(linear_qdt_token_list)
        return linear_qdt































