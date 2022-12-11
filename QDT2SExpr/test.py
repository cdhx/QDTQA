# WebQTrn-1358_f354a5347b642e1d24124b034f0f1812
from transformers import T5ForConditionalGeneration
import torch
from executor.logic_form_util import lisp_to_sparql
from components.utils import load_json
import numpy as np
import re
import sys
import os

print(os.getcwd())

"""
data = load_json('data/origin/ComplexWebQuestions_train.json',mode='rb')
# print(data)
for ex in data:
    if ex['ID']=='WebQTrn-1358_f354a5347b642e1d24124b034f0f1812':
        question = ex['question']
        print(question)
        print(type(question))
        # s = eval(question.__repr__())
        # print(s)
        # question = re.sub(r'\\u.{4}','', s.__repr__())
        # print(question)
        
        
        break
"""

def test_rel_coverage():
    train_rel_map = load_json("data/label_maps/CWQ_train_relation_label_map.json")
    test_rel_map = load_json("data/label_maps/CWQ_test_relation_label_map.json")

    print(set(test_rel_map.keys()).issubset(train_rel_map.keys()))
    print(test_rel_map.keys()-train_rel_map.keys())



def test_model_to_gpu():
    model = T5ForConditionalGeneration.from_pretrained('hfcache/t5-base')
    modle = model.to('cuda:0')
    print('Done')

def test_tensor_to_gpu():
    criterion = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0,10.0])).float().cuda())
    print(criterion.weight)

if __name__=='__main__':
    
    # from inputDataset import gen_dataset
    # expr = '(JOIN (R location.religion_percentage.religion) (JOIN (R location.statistical_region.religions) (JOIN location.country.administrative_divisions m.04pwfx)))'
    # normed_expr = gen_dataset._vanilla_linearization_method(expr,{},{},{})
    # print(normed_expr)

    # test_rel_coverage()

    # test_model_to_gpu()
    # weight_data = np.random.random([100]).astype("float64")
    # print(weight_data.shape)

    # test_tensor_to_gpu()

    lc_databank = load_json('data/lc/lc_test.json')
    success_num = 0
    failed_num = 0
    total = 0
    for data in lc_databank:
        s_expr = data['SExpr']
        if s_expr!="":
            total+=1
            try:
                sparql = lisp_to_sparql(s_expr)
                success_num+=1
            except:
                print(s_expr)
                failed_num+=1
    print(f'Total:{total}, Success:{success_num}, Failed:{failed_num}')

    
    # split = 'test'
    # qid = 'WebQTest-61_7bd6b37d01a372aa5af2a8d5ef53d827'
    # normed_expr = "( and ( join [ religion, religion, sacred sites ] [ jersulam ] ) ( join ( r [ location, religion percentage, religion ] ) ( join ( r [ location, statistical region, religions ] ) [ indonesia ] ) ) )"
    


