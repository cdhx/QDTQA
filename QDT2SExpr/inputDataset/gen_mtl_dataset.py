from torch.utils.data import Dataset
from inputDataset.gen_dataset import _vanilla_linearization_method
import torch


class MTLGenerationExample:
    """
    Multi Task Generation Example
    """
    def __init__(self, dict_data) -> None:
        """ Initialize from dict data"""
        self.ID = dict_data['ID']
        self.question = dict_data['question']
        self.comp_type = dict_data['comp_type']
        self.sprql = dict_data['sparql']
        self.sexpr = dict_data['sexpr']
        self.normed_sexpr = dict_data['normed_sexpr']
        self.gold_entity_map = dict_data['gold_entity_map']
        self.gold_relation_map = dict_data['gold_relation_map']
        self.gold_type_map = dict_data['gold_type_map']
        self.cand_relation_list = dict_data['cand_relation_list']
        self.answer = dict_data['answer']
        self.cand_entity_list = dict_data['cand_entity_list']


    def __str__(self) -> str:
        return f'{self.question}\n\t->{self.normed_sexpr}'

    def __repr__(self) -> str:
        return self.__str__()


class MTLGenDataset(Dataset):
    """Dataset for MTLGeneration"""

    def __init__(
        self, 
        examples, 
        tokenizer, 
        do_lower=True,
        normalize_relations=False,
        max_src_len=128, 
        max_tgt_len=196,
        add_prefix=False
    ):
        # super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.do_lower = do_lower
        self.normalize_relations = normalize_relations
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.add_prefix = add_prefix
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        
        ID = example.ID
        question = example.question
        normed_sexpr = example.normed_sexpr

        candidate_relations = [x[0] for x in example.cand_relation_list]
        gold_relation_set = set(example.gold_relation_map.keys())
        
        relation_labels = [(rel in gold_relation_set) for rel in candidate_relations]
        relation_clf_pairs_labels = torch.LongTensor(relation_labels)

        # entity id 可以唯一表示实体
        gold_entities_ids_set = set([item.lower() for item in example.gold_entity_map.keys()])

        entity_labels = [(ent['id'] in gold_entities_ids_set) for ent in example.cand_entity_list]
        entity_clf_pairs_labels = torch.LongTensor(entity_labels)

        input_src = question # the question it self

        if self.do_lower:
            input_src = input_src.lower()
            normed_sexpr = normed_sexpr.lower()
        
        gen_src = input_src
        if self.add_prefix:
            gen_src = 'Translate to S-Expression: ' + input_src
        if self.do_lower:
            gen_src = gen_src.lower()
        tokenized_src = self.tokenizer(
                        gen_src,
                        max_length=self.max_src_len,
                        truncation=True,
                        return_tensors='pt',
                        #padding='max_length',
                        ).data['input_ids'].squeeze(0)
        
        with self.tokenizer.as_target_tokenizer():
            tokenized_tgt = self.tokenizer(
                normed_sexpr,
                max_length=self.max_tgt_len,
                truncation=True,
                return_tensors='pt',
                #padding='max_length',
            ).data['input_ids'].squeeze(0)
        
        tokenized_relation_clf_pairs = []
        
        for cand_rel in candidate_relations:
            if self.normalize_relations:
                cand_rel = _textualize_relation(cand_rel)
            
            rel_src = input_src
            if self.add_prefix:
                rel_src = 'Relation Classification: ' + rel_src
            
            if self.do_lower:
                rel_src = rel_src.lower()
                cand_rel = cand_rel.lower()

            tokenized_relation_pair = self.tokenizer(
                rel_src,
                cand_rel,
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt',
                # padding='max_length',
            ).data['input_ids'].squeeze(0)
            
            tokenized_relation_clf_pairs.append(tokenized_relation_pair)
        
        # tokenized_clf_pairs = self.tokenizer.pad({'input_ids':tokenized_clf_pairs}, return_tensors='pt')

        tokenized_entity_clf_pairs = []

        for cand_ent in example.cand_entity_list:
            label = cand_ent['label']
            in_relations = cand_ent['in_relations'] if 'in_relations' in cand_ent else []
            out_relations = cand_ent['out_relations'] if 'out_relations' in cand_ent else []
            ent_info = label
            # TODO, there is no `in_relations` and `out_relations` in the data file now
            for rel in in_relations:
                if self.normalize_relations:
                    ent_info += ("|" + _textualize_relation(rel))
                else:
                    ent_info += ("|" + rel)
            for rel in out_relations:
                if self.normalize_relations:
                    ent_info += ("|" + _textualize_relation(rel))
                else:
                    ent_info += ("|" + rel)
            if self.do_lower:
                ent_info = ent_info.lower()
            
            ent_src = input_src
            if self.add_prefix:
                ent_src = 'Entity Classification: ' + input_src
            
            tokenized_entity_pair = self.tokenizer(
                ent_src,
                ent_info, 
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt'
            ).data['input_ids'].squeeze(0)

            tokenized_entity_clf_pairs.append(tokenized_entity_pair)

        return (
            tokenized_src, 
            tokenized_tgt, 
            tokenized_relation_clf_pairs, 
            relation_clf_pairs_labels,
            # ID,
            [input_src],
            candidate_relations,
            tokenized_entity_clf_pairs,
            entity_clf_pairs_labels,
            example.cand_entity_list # rich information of entities, including one_hop_relations
        )



def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r       
        
