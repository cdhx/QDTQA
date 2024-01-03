# QDTQA

Code for AAAI 2023 research track paper "Question Decomposition Tree for Answering Complex Questions over Knowledge
Bases"

[Download Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26519/26291)

## Requirements

### Knowledge Base
The latest offical data dump of Freebase can be downloaded [here](https://developers.google.com/freebase).

### Python enviroment
- Python >= 3.6
- Pytorch = 1.8.0
- CUDA >= 11.0 (for GPU)
- tensorflow = 2.6.2

To install, run ```pip install -r requirements.txt``` in the derectory```Clue-Decipher``` and ```QDT2Expr```. In case of CUDA problems, consult the official PyTorch [Get Started guide](https://pytorch.org/get-started/locally/).

### Data and cache

 
The cache of QDTQA can be found in [here](https://drive.google.com/file/d/1GEfsE9jiYhWOaiztJJWX8V4rCKC971Yz/view?usp=sharing)

Please put them under the ```QDT2Expr``` directory for running QA model.



## Run Clue-Decipher

Clue-Decipher is the proposed two-stage decomposition model.

The source code is in QDTQA/Clude-Decipher, hyperparameter is set in ```config``` in the ```run/run_decomp.py```

```    
config = {
        'gpu_num': find_free_gpu(0, 20000),
        "max_length": 48,  
        "batch_size": 64,  
        'seed': 42, # random seed
        "epoch": 100,
        'other': '',  
        'action': 'train',  
        'seq_eval': False, # sequence-based or tree-based
    }
```

### Train
The trained model will be saved in ```model/your_exp_name.pkl```

Run ```run/run_decomp.py``` with  ```action=train``` in ```config```

### Predict

Predict decomposition file will be saved in ```data/aaai_out``` 

Run ```run/run_decomp.py``` with  ```action=cwq_qa``` in ```config```



## Run QDTQA


QDTQA is the proposed decomposition-based KBQA model, with the decomposition of question obtained by Clue-Decipher as part of the input.

The source code is in QDTQA/QDT2Expr.  

Run within ```QDT2SExpr``` directoty

### Train 
Model will be saved in ```QDT2SExpr/exps/gen_CWQ_nlq_newqdt_candEnt```

Please specify the path of your decomposition result obtained from Clue-Decipher in ```inputDataset/gen_dataset.py```,

```
cwq_load_and_cache_gen_examples()        

if args.new_qdt:
    qdt_file = your_decomposition_result_path        
```
Then run this scipt
```
CUDA_VISIBLE_DEVICES=0 sh scripts/run_gen_nlq_newqdt_candEnt.sh train nlq_newqdt_candEnt
```

### Predict
Predict result will be saved in  ```QDT2SExpr/results/gen/CWQ_test_nlq_newqdt_candEnt```

```
CUDA_VISIBLE_DEVICES=0 sh scripts/run_gen_nlq_newqdt_candEnt.sh predict nlq_newqdt_candEnt test
```


### Evaluate
Evaluate result will be saved in ```QDT2SExpr/results/gen/CWQ_test_nlq_newqdt_candEnt/final_eval_results.txt```

```
python eval_topk_prediction.py --split test --pred_file results/gen/CWQ_test_nlq_newqdt_candEnt/top_k_predictions.json
```

# Run EDGQA with Clue-Decipher

The code for EDGQA can be found in [EDGQA](https://github.com/HXX97/EDGQA/).


A modified version of EDGQA that can be run with EDG format decomposition files is available in ```EDGQA_Exp/EDGQA```.

Scripts and resources for converting QDT format to EDG format are available in ```EDGQA_Exp/QDT2EDG_convert```. 

Shell scripts for running tests are in ```EDGQA_Exp/run```.


# Cite
```bibtex
@article{Huang_Cheng_Shu_Bao_Qu_2023,
  title={Question Decomposition Tree for Answering Complex Questions over Knowledge Bases},
  volume={37},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/26519},
  DOI={10.1609/aaai.v37i11.26519},
  number={11},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Huang, Xiang and Cheng, Sitao and Shu, Yiheng and Bao, Yuheng and Qu, Yuzhong},
  year={2023},
  month={Jun.},
  pages={12924-12932}
}
```
