from functools import reduce
import json
import os

import torch
from transformers import BertTokenizer
from torchtext.legacy import data

des_token = "[DES]"
inq_token = "[INQ]"

def add_qdt_sequence_to_json(input_json):
    """
    往原数据里头加入序列化后的 qdt 表示
    :param input_json: {"ID", "question", "qdt"...}
    :return: input_json: {"ID", "question", "qdt", "serialization"}
    """
    idx = 0
    for item in input_json:
        if len(item["qdt"]) > 1:
            print(">1: {}".format(item["qdt"]))
        # normalize white space
        item["serialization"] = " ".join(qdt_serialization(item["qdt"]).split())
        item["idx"] = idx
        idx += 1
        print(item["qdt"])
        print(item["serialization"])
        print()
    return list(map(lambda item: {
        "ID": item["ID"],
        "question": item["question"],
        "qdt": item["qdt"],
        "serialization": item["serialization"],
        "idx": item["idx"]
    }, input_json))

def qdt_serialization(qdt_tree):
    """
    具体规则:
    在每个 description 开头加上 [DES] token
    在每个 Inner Question 两侧加上 [INQ] token

    递归函数:
    - 出口: 是一个 item, 且没有 inner_questions 属性: [DES] + description
    - 是一个 item, 但是有 inner question: replace [INQ} with ([INQ] recur(children) [INQ])
    - 有多个 item: recur(child1) + recur(child2) ...

    格式: 所有空格加在后面
    """
    print(qdt_tree)
    if not isinstance(qdt_tree, list):
        if 'inner_questions' not in qdt_tree:
            return '[DES] ' + qdt_tree["description"]
        else:
            return '[DES] ' + qdt_tree["description"].replace(
                "[INQ]",
                '[INQ] ' + qdt_serialization(qdt_tree['inner_questions']['INQ']) + ' [INQ]'
            )
    else:
        return reduce(lambda x, y: x + ' ' + qdt_serialization(y), qdt_tree, '')

def preprocess_qdt(input_path, output_path):
    with open(input_path, 'r') as f:
        input_json = json.load(f)
    output_json = add_qdt_sequence_to_json(input_json)
    with open(output_path, 'w') as f:
        # json.dump(output_json, f, indent=4)
        for item in output_json:
            f.write(json.dumps(item) + "\n")


def get_maxLen(jsonl_path):
    max_len = 0
    with open(jsonl_path, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        res = json.loads(json_str)
        # + 2 because in BERT, add [CLS] in beginning and [SEP] at the end
        max_len = max(len(res["serialization"].split(' ')) + 2, max_len)
    return max_len

class SerializationDataset:
    def __init__(self, no_cuda=False, bert_model_name='bert-base-uncased', fix_length=40, batch_size=16, gpu_num="4",
                 data_dir="/home2/xxhu/QDT2sExpression/data/CWQ",
                 train_file="cwq_train_qdt_preprocessed.jsonl",
                 dev_file="cwq_dev_qdt_preprocessed.jsonl",
                 test_file="cwq_test_qdt_preprocessed.jsonl",
                 data_format="json",
                 verbose=False
                 ):
        """
        :param no_cuda:
        :param bert_model_name:
        :param fix_length: suggested 40 for CWQ.
        :param batch_size:
        :param gpu_num:
        :param data_dir: the folder of qdt data
        :param train_file: file name of training data. PS: Please convert json file to jsonl format
        :param dev_file:
        :param test_file:
        :param data_format: format of training data. SUpporting 'json', 'csv', 'tsv'...
        :param verbose: print debugging information
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # 添加我们自定的 token
        special_tokens_dict = {'additional_special_tokens': [des_token, inq_token]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # 将各种默认 token 与 BERT 对齐
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id
        unk_token_id = tokenizer.unk_token_id
        # print(tokenizer.get_vocab())

        self.SERIALIZATION = data.Field(
            batch_first=True, # BERT requirement
            tokenize=tokenizer.tokenize, # BERT standard tokenize
            use_vocab=False, # Using tokenizer's vocab
            lower=True,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=cls_token_id,
            eos_token=sep_token_id,
            pad_token=pad_token_id,
            unk_token=unk_token_id,
            fix_length=fix_length # padding all sentences to fix_length
        )
        self.serialization_vocab = tokenizer.get_vocab()
        # IDX is not optimization targets
        self.IDX = data.LabelField(use_vocab=False, is_target=False, sequential=False)
        # ignore field "decomposition"
        fields = {'serialization': ('serialization', self.SERIALIZATION), 'idx': ('idx', self.IDX)}

        self.train_data, self.validation_data, self.test_data = data.TabularDataset.splits(
            path=data_dir,
            train=train_file,
            validation=dev_file,
            test=test_file,
            format=data_format,
            fields=fields
        )

        if verbose:
            for i in range(0, len(self.test_data)):
                print(vars(self.test_data[i]))

        self.train_iter = data.BucketIterator(self.train_data, batch_size=batch_size, device=device)
        self.validation_iter = data.BucketIterator(self.validation_data, batch_size=batch_size, device=device)
        self.test_iter = data.BucketIterator(self.test_data, batch_size=batch_size, device=device)

    def get_iterators(self):
        return self.train_iter, self.validation_iter, self.test_iter

    def get_serialization_vocab(self):
        return self.serialization_vocab

if __name__ == '__main__':
    # To serailize QDT
    # preprocess_qdt(
    #     '/home3/xwu/workspace/QDT2SExpr/CWQ/data/cwq_test_qdt.json',
    #     '/home3/xwu/workspace/QDT2SExpr/CWQ/data/cwq_test_qdt_preprocessed.jsonl'
    # )

    # calculate max length of field "serialization"
    # train_max_len = get_maxLen('/home3/xwu/workspace/QDT2SExpr/CWQ/data/cwq_train_qdt_preprocessed.jsonl') # 39
    # dev_max_len = get_maxLen('/home3/xwu/workspace/QDT2SExpr/CWQ/data/cwq_dev_qdt_preprocessed.jsonl') # 34
    # test_max_len = get_maxLen('/home3/xwu/workspace/QDT2SExpr/CWQ/data/cwq_test_qdt_preprocessed.jsonl') # 33
    # print(train_max_len, dev_max_len, test_max_len)


    # To get data iterators and use it
    serializationDataset = SerializationDataset(data_dir="/home3/xwu/workspace/QDT2SExpr/CWQ/data/")
    train_iter, validation_iter, test_iter = serializationDataset.get_iterators()
    vocab = serializationDataset.get_serialization_vocab()
    print(vocab)
    print(train_iter.batch_size)

    for batch in test_iter:
        print("{}: {} ".format(batch.idx[0], batch.serialization[0]))
