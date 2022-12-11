from os import sep
import pandas as pd


def process_data(set_type:str):
    
    #input_filename = 'data/lcquad_queryrank_'+set_type+"_new.tsv"
    #output_filename = 'lcquad_queryrank_block_sparql_'+set_type+'.tsv'
    
    input_filename = set_type+'.tsv'
    output_filename = set_type+'.tsv'

    with open(input_filename,'r') as f:
        test_df = pd.read_csv(f,sep='\t')

    # question	blockString	subQuery	score

    block_string = test_df['blockString']
    sub_query = test_df['subQuery']
    score = test_df['score']

    score = [1 if s>=0.1 else 0 for s in score]

    id1 = list(range(len(score)))
    id2 = id1

    dict_to_write = {'score':score,'id1':id1,'id2':id2,"blockString":block_string,"subQuery":sub_query}

    df = pd.DataFrame(dict_to_write)

    df.to_csv(output_filename,sep='\t',index=False)


process_data('test')
process_data('train')