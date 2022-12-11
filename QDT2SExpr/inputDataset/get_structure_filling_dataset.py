from torch.utils.data import Dataset
import torch


def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r 

class StructureGenerationExample:
    """
    Strucure Generation Example
    From S-Expression to (all) masked S-Expression
    """
    def __init__(self, dict_data):
        self.ID = dict_data['ID']
        self.question = dict_data['question']
        self.normed_sexpr = dict_data['normed_sexpr']
        self.normed_all_masked_sexpr = dict_data["normed_all_masked_sexpr"]
    
    def __str__(self):
        return f'{self.sexpr}\n\t->{self.normed_all_masked_sexpr}'

    def __repr__(self):
        return self.__str__()

class StructureGenDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        do_lower=True,
        max_src_len=128, 
        max_tgt_len=128, 
    ):
        super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.do_lower = do_lower
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]
        
        qid = example.ID
        question = example.question
        normed_sexpr = example.normed_sexpr
        normed_all_masked_sexpr = example.normed_all_masked_sexpr

        if self.do_lower:
            question = question.lower()
            normed_all_masked_sexpr = normed_all_masked_sexpr.lower()
        
        tokenized_src = self.tokenizer(
            question,
            max_length=self.max_src_len,
            truncation=True,
            return_tensors='pt',
        ).data['input_ids'].squeeze(0)

        with self.tokenizer.as_target_tokenizer():
            tokenized_tgt = self.tokenizer(
                normed_all_masked_sexpr,
                max_length=self.max_tgt_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
        
        return (
            tokenized_src,
            tokenized_tgt
        )


class StructureFillingExample:
    """
    Strucure Filling Example
    """
    def __init__(self, dict_data):
        self.ID = dict_data['ID']
        self.question = dict_data['question']
        self.answer = dict_data['answer']
        self.sparql = dict_data['sparql']
        self.sexpr = dict_data['sexpr']
        self.normed_sexpr = dict_data['normed_sexpr']
        self.gold_relation_map = dict_data['gold_relation_map']
        self.cand_relation_map = dict_data['cand_relation_map']
        self.gold_entity_map = dict_data['gold_entity_map']
        self.cand_entity_list = dict_data['cand_entity_list']
        self.gold_type_map = dict_data['gold_type_map']
        self.gold_structure = dict_data['normed_all_masked_sexpr']
        self.candidate_structures_list = dict_data['candidate_structures_list']
    
    def __str__(self):
        return f'{self.question}\n\t->{self.normed_sexpr}'

    def __repr__(self):
        return self.__str__()


class StructureFillingDataset(Dataset):
    """
    Dataset for structure-based slot filling
    """

    def __init__(
        self,
        examples,
        tokenizer,
        add_top1_structure=False,
        do_lower=True,
        normalize_relations=True,
        max_src_len=256,
        max_tgt_len=192,
        add_prefix=False
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.add_top1_structure = add_top1_structure
        self.do_lower = do_lower
        self.normalize_relations = normalize_relations
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.add_prefix = add_prefix # 对比实验，分类任务加不加 prefix
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        example = self.examples[index]
        qid = example.ID
        question = example.question
        normed_sexpr = example.normed_sexpr

        candidate_relations = [x[0] for x in example.cand_relation_map]
        gold_relations_set = set(example.gold_relation_map.keys())
        relation_labels = [(rel in gold_relations_set) for rel in candidate_relations]
        relation_clf_labels = torch.LongTensor(relation_labels)

        # candidate_entities = [item['label'] for item in example.cand_entity_list]
        # entity id 可以唯一表示实体
        gold_entities_ids_set = set([item for item in example.gold_entity_map.keys()])
        entity_labels = [(ent['id'] in gold_entities_ids_set) for ent in example.cand_entity_list]
        entity_clf_labels = torch.LongTensor(entity_labels)

        candidate_structures = example.candidate_structures_list
        gold_structure = example.gold_structure
        structure_labels = [(item == gold_structure) for item in candidate_structures]
        structure_clf_labels = torch.LongTensor(structure_labels)

        input_src = question
        if self.add_top1_structure:
            input_src = question + " " + candidate_structures[0]
        if self.do_lower:
            normed_sexpr = normed_sexpr.lower()
            input_src = input_src.lower()

        with self.tokenizer.as_target_tokenizer():
            tokenized_tgt = self.tokenizer(
                normed_sexpr,
                max_length=self.max_tgt_len,
                truncation=True,
                return_tensors='pt',
                #padding='max_length',
            ).data['input_ids'].squeeze(0)
        
        with self.tokenizer.as_target_tokenizer():
            tokenized_structure_gen_tgt = self.tokenizer(
                gold_structure,
                max_length=self.max_tgt_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
        
        tokenized_relation_clf = []
        for cand_rel in candidate_relations:
            if self.normalize_relations:
                cand_rel = _textualize_relation(cand_rel)
            if self.add_prefix:
                rel_src = 'Relation Classification: ' + input_src
            else:
                rel_src = input_src
            
            if self.do_lower:
                cand_rel = cand_rel.lower()
                rel_src = rel_src.lower()

            tokenized_relation = self.tokenizer(
                rel_src,
                cand_rel,
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt',
                # padding='max_length',
            ).data['input_ids'].squeeze(0)

            tokenized_relation_clf.append(tokenized_relation)
        
        tokenized_structure_clf = []

        for cand_structure in candidate_structures:
            if self.add_prefix:
                structure_src = 'Structure Classification: ' + input_src
            else:
                structure_src = input_src
            if self.do_lower:
                cand_structure = cand_structure.lower()
                structure_src = structure_src.lower()
            
            tokenized_structure = self.tokenizer(
                structure_src,
                cand_structure,
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            tokenized_structure_clf.append(tokenized_structure)
        
        return (
            tokenized_tgt,
            tokenized_relation_clf,
            tokenized_structure_clf,
            relation_clf_labels,
            entity_clf_labels,
            structure_clf_labels,
            candidate_relations,
            example.cand_entity_list,
            candidate_structures,
            [input_src],
            tokenized_structure_gen_tgt
        )