import numpy as np
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer


def list_reshape(prev_list, dim0, dim1):
    assert len(prev_list) == dim0 * dim1
    new_list = [prev_list[i*dim1: (i+1)*dim1] for i in range(dim0)]
    return new_list


def filter_candidates_by_classification_results(candidates, classification_results, cross_entropy_loss=False, extract_label=False):
    """
    candidates: list of [dim0, dim1]
    classification_results: tensor of [dim0, dim1]
    filter condition: logits > 0 if BCELoss, logits = 1.0 if CrossEntropyLoss
    """
    assert len(candidates) == classification_results.shape[0]
    assert len(candidates[0]) == classification_results.shape[1], print(len(candidates[0]), classification_results.shape[1])

    if cross_entropy_loss:
        indices = np.argwhere(classification_results.detach().cpu().numpy() == 1.0).tolist()
    else:
        indices = np.argwhere(classification_results.detach().cpu().numpy() > 0.0).tolist()

    filtered_candidates = []
    for i in range(len(candidates)):
        row = []
        for j in range(len(candidates[0])):
            if [i,j] in indices:
                if extract_label:
                    row.append(candidates[i][j]['label'])
                else:
                    row.append(candidates[i][j])
        filtered_candidates.append(row)
    
    return filtered_candidates


def find_best_candidate_by_classification_results(candidates, classification_results):
    """
    candidates: list of [dim0, dim1]
    classification_results: tensor of [dim0, dim1]
    与上一个函数的区别：对于每一行，取分类得分最高的那一项，在 candidates 里头的对应
    """
    assert len(candidates) == classification_results.shape[0]
    assert len(candidates[0]) == classification_results.shape[1], print(len(candidates[0]), classification_results.shape[1])

    indices = torch.argmax(classification_results, dim=1).tolist()

    filtered_candidates = []
    for (row, col) in enumerate(indices):
        filtered_candidates.append(candidates[row][col])
    
    return filtered_candidates


def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r

class T5_Structure_Generation(nn.Module):
    """
    From Question to (all) masked S-Expression
    """
    def __init__(self, pretrained_model_path, is_test=False):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    
    def forward(
        self,
        input_ids_gen=None,
        gen_labels=None,
        gen_attention_mask=None,
    ):
        gen_outputs = self.t5(
            input_ids=input_ids_gen,
            attention_mask=gen_attention_mask,
            labels=gen_labels,
        )

        gen_loss = gen_outputs['loss']

        return gen_loss
    
    def inference(
        self,
        input_ids_gen=None,
        gen_attention_mask=None,
        num_beams=5,
        max_length=128,
    ):
        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=input_ids_gen,
                attention_mask=gen_attention_mask,
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=max_length
            )

            # [batch_size, num_beams, -1]
            gen_outputs = torch.reshape(gen_outputs,(input_ids_gen.size(0),num_beams,-1))

        return gen_outputs


class T5_Structure_Filling(nn.Module):
    """
    Slot Filling on S-Expr structure to get final S-Expr
    """
    def __init__(
        self, 
        pretrained_model_path, 
        is_test=False,
        device='cuda',
        max_src_len=256, 
        max_tgt_len=196,
        tokenizer=None,
        entity_sample_size=10,
        relation_sample_size=10,
        structure_sample_size=5,
        do_lower=False,
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.entity_sample_size = entity_sample_size
        self.relation_sample_size = relation_sample_size
        self.structure_sample_size = structure_sample_size
        self.do_lower = do_lower
        self.REL_TOKEN = ' [REL] '
        self.ENT_TOKEN = ' [ENT] '
        self.LITERAL_TOKEN = ' [LIT] '
        self.SEPERATOR = ' | '

        if 't5-large' in pretrained_model_path.lower():
            hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            hidden_size = 512
        else:
            hidden_size = 768
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        input_ids_relation_clf=None,
        input_ids_structure_clf=None,
        relation_clf_labels=None,
        entity_clf_labels=None,
        structure_clf_labels=None,
        gen_labels=None,
        relation_clf_attention_mask=None,
        structure_clf_attention_mask=None,
        textual_candidate_relations=None,
        textual_candidate_entities=None,
        textual_candidate_structures=None,
        textual_nlq=None,
        normalize_relations=True
    ):
        # data preprocess
        textual_candidate_relations = list_reshape(
            textual_candidate_relations, 
            int(len(textual_candidate_relations)/self.relation_sample_size), 
            self.relation_sample_size
        )
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities)/self.entity_sample_size),
            self.entity_sample_size
        )
        textual_candidate_structures = list_reshape(
            textual_candidate_structures,
            int(len(textual_candidate_structures)/self.structure_sample_size),
            self.structure_sample_size
        )

        # Task 1: Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        if relation_clf_labels is not None:
            relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
            relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
            relation_clf_loss = self.criterion(relation_clf_predict_logits.float(), relation_clf_labels.unsqueeze(1).float())
            relation_clf_predict_logits = torch.reshape(relation_clf_predict_logits, (gen_labels.size(0), self.relation_sample_size, -1))
            relation_clf_predict_logits = relation_clf_predict_logits.squeeze(2)
        
            filtered_candidate_relations = filter_candidates_by_classification_results(
                textual_candidate_relations,
                relation_clf_predict_logits
            )
        else:
            relation_clf_loss = None
            filtered_candidate_relations = None
        
        # Task 2: Entity Classification
        input_ids_entity_concatenated = []
        for (nlq, cand_ents, cand_relations) in zip(textual_nlq, textual_candidate_entities, filtered_candidate_relations):
            if self.do_lower:
                nlq = nlq.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    nlq,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        if entity_clf_labels is not None:
            entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
            entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
            entity_clf_loss = self.criterion(entity_clf_predict_logits.float(), entity_clf_labels.unsqueeze(1).float())
            entity_clf_predict_logits = torch.reshape(entity_clf_predict_logits, (gen_labels.size(0), self.entity_sample_size, -1))
            entity_clf_predict_logits = entity_clf_predict_logits.squeeze(2)

            filtered_candidate_entities = filter_candidates_by_classification_results(
                textual_candidate_entities,
                entity_clf_predict_logits,
                extract_label=True
            )
        else:
            entity_clf_loss = None
            filtered_candidate_entities = None
        
        # Task 3: Structure Classification
        # TODO: concat entity, relation, literal informations
        structure_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_structure_clf, attention_mask=structure_clf_attention_mask)
        if structure_clf_labels is not None:
            structure_sentence_embedding = torch.mean(structure_clf_encoder_outputs.last_hidden_state,dim=1)
            structure_clf_predict_logits = self.cls_layer(self.dropout(structure_sentence_embedding))
            structure_clf_loss = self.criterion(structure_clf_predict_logits.float(), structure_clf_labels.unsqueeze(1).float())
            structure_clf_predict_logits = torch.reshape(structure_clf_predict_logits, (gen_labels.size(0), self.structure_sample_size, -1))
            structure_clf_predict_logits = structure_clf_predict_logits.squeeze(2)
            filtered_candidate_structures = find_best_candidate_by_classification_results(
                textual_candidate_structures,
                structure_clf_predict_logits
            )
        else:
            structure_clf_loss = None
            filtered_candidate_structures = None
        
        # Task 4: Slot Filling to generate final S-Expression
        # Format: Structure | [ENT] ... [ENT] | [REl] ....[REL] | [LIT] 2005-01-10 | NLQ
        # TODO: add extracted Literal mentions
        input_ids_gen_concatenated = []
        for (nlq, cand_ents, cand_rels, cand_structure) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations, filtered_candidate_structures):
            input_src = cand_structure + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel
            input_src += self.SEPERATOR
            input_src += nlq

            if self.do_lower:
                input_src = input_src.lower()
            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )
        gen_loss = gen_outputs['loss']

        if entity_clf_loss is not None and relation_clf_loss is not None and structure_clf_loss is not None:
            total_loss = gen_loss + entity_clf_loss + relation_clf_loss + structure_clf_loss
        else:
            total_loss = gen_loss
        
        return total_loss
    
    def inference(
        self,
        input_ids_relation_clf=None,
        input_ids_structure_clf=None,
        relation_clf_attention_mask=None,
        structure_clf_attention_mask=None,
        num_beams=50,
        textual_candidate_relations=None,
        textual_candidate_entities=None,
        textual_candidate_structures=None,
        textual_nlq=None,
        normalize_relations=True
    ):
        # data preprocess
        textual_candidate_relations = list_reshape(
            textual_candidate_relations, 
            int(len(textual_candidate_relations)/self.relation_sample_size), 
            self.relation_sample_size
        )
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities)/self.entity_sample_size),
            self.entity_sample_size
        )
        textual_candidate_structures = list_reshape(
            textual_candidate_structures,
            int(len(textual_candidate_structures)/self.structure_sample_size),
            self.structure_sample_size
        )

        # Task 1. Get prediction from Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
        relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
        relation_clf_outputs = torch.reshape(relation_clf_predict_logits, (-1, self.relation_sample_size)) # [batch_size, sample_size]

        filtered_candidate_relations = filter_candidates_by_classification_results(
            textual_candidate_relations,
            relation_clf_outputs
        )

        # Task 2. Get prediction from Entity Classification
        input_ids_entity_concatenated = []
        for (nlq, cand_ents, cand_relations) in zip(textual_nlq, textual_candidate_entities, filtered_candidate_relations):
            if self.do_lower:
                nlq = nlq.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    nlq,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
        entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
        entity_clf_outputs = torch.reshape(entity_clf_predict_logits, (-1, self.entity_sample_size)) # [batch_size, sample_size]

        filtered_candidate_entities = filter_candidates_by_classification_results(
            textual_candidate_entities,
            entity_clf_outputs,
            extract_label=True
        )

        # Task 3: Structure Classification
        structure_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_structure_clf, attention_mask=structure_clf_attention_mask)
        structure_sentence_embedding = torch.mean(structure_clf_encoder_outputs.last_hidden_state,dim=1)
        structure_clf_predict_logits = self.cls_layer(self.dropout(structure_sentence_embedding))
        structure_clf_outputs = torch.reshape(structure_clf_predict_logits, (-1, self.structure_sample_size))
        
        filtered_candidate_structures = find_best_candidate_by_classification_results(
            textual_candidate_structures,
            structure_clf_outputs
        )

        # Task 4: Slot Filling to generate final S-Expression
        input_ids_gen_concatenated = []
        for (nlq, cand_ents, cand_rels, cand_structure) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations, filtered_candidate_structures):
            input_src = cand_structure + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel
            input_src += self.SEPERATOR
            input_src += nlq

            if self.do_lower:
                input_src = input_src.lower()
            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )

            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))
        
        return gen_outputs, relation_clf_outputs, entity_clf_outputs, structure_clf_outputs


class T5_MultiTask_Concat_Relation_Concat_Entity(nn.Module):
    """ 
    基本上是从 T5_Multitask 复制过来的；可以用于直接拼接 top1 结构的方法
    训练任务：关系分类 + 实体分类 + 生成
    主要区别在于先做关系分类，后做实体分类
    - 先关系分类
    - 预测为正的关系和 candidate_entity 的 one hop relation 取交集，交集部分拼接到实体上，用于消岐
    - 关系分类和实体分类的结果，拼到最后的生成模型输入上
    """
    def __init__(
        self, 
        pretrained_model_path, 
        device='cuda', 
        max_src_len=128, 
        tokenizer=None, 
        is_test=False, 
        relation_sample_size=10, 
        entity_sample_size=10, 
        do_lower=False,
        add_prefix=False
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(pretrained_model_path)
        self.REL_TOKEN = ' [REL] '
        self.ENT_TOKEN = ' [ENT] '
        self.relation_sample_size = relation_sample_size
        self.entity_sample_size = entity_sample_size
        self.do_lower = do_lower
        self.add_prefix = add_prefix

        if 't5-large' in pretrained_model_path.lower():
            hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            hidden_size = 512
        else:
            hidden_size = 768

        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(hidden_size, 1) # binary classification
        self.criterion = nn.BCEWithLogitsLoss()
            
    
    def forward(self,
        input_ids_relation_clf=None,
        gen_labels=None,
        relation_clf_labels=None,
        entity_clf_labels=None,
        relation_clf_attention_mask=None,
        textual_candidate_relations=None,
        textual_input_src_gen=None,
        normalize_relations=False,
        textual_candidate_entities=None # entity 的各类信息，包括 oneHopRelations 等
        ):
        # clf_encoder_outputs: [batch_size, max_len, hidden_size]
        textual_candidate_relations = list_reshape(textual_candidate_relations, int(len(textual_candidate_relations)/self.relation_sample_size), self.relation_sample_size)
        textual_candidate_entities = list_reshape(textual_candidate_entities, int(len(textual_candidate_entities) / self.entity_sample_size), self.entity_sample_size) # list of [batch_size, sample_size]

        # Task 1: Relation classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        if relation_clf_labels is not None:
            # sentence_embedding: [batch_size*sample_size, embedding_dim]
            relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)

            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
            
            # classification loss
            relation_clf_loss = self.criterion(relation_clf_predict_logits.float(), relation_clf_labels.unsqueeze(1).float())
            
            relation_clf_predict_logits = torch.reshape(relation_clf_predict_logits, (len(textual_candidate_relations), self.relation_sample_size))
            
            filtered_candidate_relations = filter_candidates_by_classification_results(
                textual_candidate_relations,
                relation_clf_predict_logits,
            )
        else:
            relation_clf_loss = None
            filtered_candidate_relations = None     

        # Task2: Entity classification
        # 对于每个实体，拼接其一跳关系 和 filtered_candidate_relations 之间的交集
        input_ids_entity_concatenated = []
        for (input_src, cand_ents, cand_relations) in zip(textual_input_src_gen, textual_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + input_src
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
    
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        if entity_clf_labels is not None:
            # sentence_embedding: [batch_size*sample_size, embedding_dim]
            entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)

            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
            
            # classification loss
            entity_clf_loss = self.criterion(entity_clf_predict_logits.float(), entity_clf_labels.unsqueeze(1).float())
            
            entity_clf_predict_logits = torch.reshape(entity_clf_predict_logits, (len(textual_candidate_entities), self.entity_sample_size))
            
            filtered_candidate_entities = filter_candidates_by_classification_results(
                textual_candidate_entities,
                entity_clf_predict_logits,
                extract_label=True
            )
        else:
            entity_clf_loss = None
            filtered_candidate_entities = None 

        
        input_ids_gen_concatenated = []
        for (input_src, cand_ents, cand_rels) in zip(textual_input_src_gen, filtered_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Translate to S-Expression: ' + input_src
            for rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(rel)
                else:
                    input_src += self.REL_TOKEN + rel
            for ent in cand_ents:
                input_src += self.ENT_TOKEN + ent
            if self.do_lower:
                input_src = input_src.lower()

            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
        
        # TODO, maybe we can change the input according to the classification results
        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )

        gen_loss = gen_outputs['loss']
        if entity_clf_loss is not None and relation_clf_loss is not None:
            total_loss = gen_loss + entity_clf_loss + relation_clf_loss
        else:
            total_loss = gen_loss

        return total_loss
    
    # TODO:
    def inference(self,
        input_ids_relation_clf=None,
        relation_clf_attention_mask=None,
        num_beams=5,
        max_length=196,
        textual_candidate_relations=None,
        textual_input_src_gen=None,
        normalize_relations=False,
        textual_candidate_entities=None
        ):
        textual_candidate_relations = list_reshape(textual_candidate_relations, int(len(textual_candidate_relations)/self.relation_sample_size), self.relation_sample_size)
        textual_candidate_entities = list_reshape(textual_candidate_entities, int(len(textual_candidate_entities) / self.entity_sample_size), self.entity_sample_size) # list of [batch_size, sample_size]

        # 1. Get prediction from Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
        # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
        relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
        relation_clf_outputs = torch.reshape(relation_clf_predict_logits, (len(textual_candidate_relations), self.relation_sample_size)) # [batch_size, sample_size]
        
        filtered_candidate_relations = filter_candidates_by_classification_results(
            textual_candidate_relations,
            relation_clf_outputs,
        )
       
        # 2. Get prediction from entity classification
        # 对于每个实体，拼接其一跳关系 和 filtered_candidate_relations 之间的交集
        input_ids_entity_concatenated = []
        for (input_src, cand_ents, cand_relations) in zip(textual_input_src_gen, textual_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + input_src
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
        # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
        entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
        entity_clf_outputs = torch.reshape(entity_clf_predict_logits,(len(textual_candidate_entities), self.entity_sample_size)) # [batch_size, sample_size]
        

        filtered_candidate_entities = filter_candidates_by_classification_results(
            textual_candidate_entities,
            entity_clf_outputs,
            extract_label=True
        )

        input_ids_gen_concatenated = []
        for (input_src, cand_ents, cand_rels) in zip(textual_input_src_gen, filtered_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Translate to S-Expression: ' + input_src
            for rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(rel)
                else:
                    input_src += self.REL_TOKEN + rel
            for ent in cand_ents:
                input_src += self.ENT_TOKEN + ent
            if self.do_lower:
                input_src = input_src.lower()

            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
        
        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=max_length
            )
            # src_encoded['input_ids']: [batch_size, padding_length]
            # [batch_size, num_beams, -1]
            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))
            # gen_outputs = [p.cpu().numpy() for p in gen_outputs]

        return gen_outputs, relation_clf_outputs, entity_clf_outputs


class T5_SExpr_Generation_Structure_Generation(nn.Module):
    """
    Structure Generation and SExpr Generation multitask
    """
    def __init__(
        self, 
        pretrained_model_path, 
        is_test=False,
        device='cuda',
        max_src_len=256, 
        max_tgt_len=196,
        tokenizer=None,
        entity_sample_size=10,
        relation_sample_size=10,
        do_lower=False,
        add_prefix=False,
    ):
        super().__init__()
        self._is_test = is_test
        # All tasks use the same T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(pretrained_model_path)
        self.entity_sample_size = entity_sample_size
        self.relation_sample_size = relation_sample_size
        self.do_lower = do_lower
        self.add_prefix = add_prefix # 对比实验，分类任务加不加 prefix
        self.REL_TOKEN = ' [REL] '
        self.ENT_TOKEN = ' [ENT] '
        self.LITERAL_TOKEN = ' [LIT] '
        self.SEPERATOR = ' | '

        if 't5-large' in pretrained_model_path.lower():
            self.hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            self.hidden_size = 512
        else:
            self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(self.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        input_ids_relation_clf=None,
        relation_clf_labels=None,
        entity_clf_labels=None,
        structure_gen_labels=None, 
        gen_labels=None,
        relation_clf_attention_mask=None,
        textual_candidate_relations=None,
        textual_candidate_entities=None,
        textual_nlq=None,
        normalize_relations=True
    ):
        # data preprocess
        textual_candidate_relations = list_reshape(
            textual_candidate_relations, 
            int(len(textual_candidate_relations)/self.relation_sample_size), 
            self.relation_sample_size
        )
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities)/self.entity_sample_size),
            self.entity_sample_size
        )

        # Task 1: Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        if relation_clf_labels is not None:
            relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
            relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
            relation_clf_loss = self.criterion(relation_clf_predict_logits.float(), relation_clf_labels.unsqueeze(1).float())
            relation_clf_predict_logits = torch.reshape(relation_clf_predict_logits, (gen_labels.size(0), self.relation_sample_size))
        
            filtered_candidate_relations = filter_candidates_by_classification_results(
                textual_candidate_relations,
                relation_clf_predict_logits
            )
        else:
            relation_clf_loss = None
            filtered_candidate_relations = None
        
        # Task 2: Entity Classification
        input_ids_entity_concatenated = []
        for (nlq, cand_ents, cand_relations) in zip(textual_nlq, textual_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + nlq # prefix
            else:
                input_src = nlq
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        if entity_clf_labels is not None:
            entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
            entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
            entity_clf_loss = self.criterion(entity_clf_predict_logits.float(), entity_clf_labels.unsqueeze(1).float())
            entity_clf_predict_logits = torch.reshape(entity_clf_predict_logits, (gen_labels.size(0), self.entity_sample_size))

            filtered_candidate_entities = filter_candidates_by_classification_results(
                textual_candidate_entities,
                entity_clf_predict_logits,
                extract_label=True
            )
        else:
            entity_clf_loss = None
            filtered_candidate_entities = None
        
        # Task 3: Structure Generation
        input_ids_structure_gen_concatenated = []
        for (nlq, cand_ents, cand_rels) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations):
            input_src = 'Translate to Structure: ' # prefix
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel
            if self.do_lower:
                input_src = input_src.lower()
            
            tokenized_structure_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_structure_gen_concatenated.append(tokenized_structure_src)
        
        structure_src_encoded = self.tokenizer.pad({'input_ids': input_ids_structure_gen_concatenated},return_tensors='pt')
        structure_gen_outputs = self.t5(
            input_ids=structure_src_encoded['input_ids'].to(self.device),
            attention_mask=structure_src_encoded['attention_mask'].to(self.device),
            labels=structure_gen_labels,
        )
        structure_loss = structure_gen_outputs['loss']
        
        # Task 4: Slot Filling to generate final S-Expression
        # Format: Structure | [ENT] ... [ENT] | [REl] ....[REL] | [LIT] 2005-01-10 | NLQ
        # TODO: add extracted Literal mentions
        input_ids_gen_concatenated = []
        for (nlq, cand_ents, cand_rels) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations):
            input_src = 'Translate to S-Expression: ' # prefix
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel
            if self.do_lower:
                input_src = input_src.lower()
            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )
        gen_loss = gen_outputs['loss']

        if entity_clf_loss is not None and relation_clf_loss is not None and structure_loss is not None:
            total_loss = gen_loss + entity_clf_loss + relation_clf_loss + structure_loss
        else:
            total_loss = gen_loss
        
        return total_loss
    
    def inference(
        self,
        input_ids_relation_clf=None,
        relation_clf_attention_mask=None,
        num_beams=50,
        textual_candidate_relations=None,
        textual_candidate_entities=None,
        textual_nlq=None,
        normalize_relations=True
    ):
        # data preprocess
        textual_candidate_relations = list_reshape(
            textual_candidate_relations, 
            int(len(textual_candidate_relations)/self.relation_sample_size), 
            self.relation_sample_size
        )
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities)/self.entity_sample_size),
            self.entity_sample_size
        )

        # Task 1. Get prediction from Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
        relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
        relation_clf_outputs = torch.reshape(relation_clf_predict_logits, (-1, self.relation_sample_size)) # [batch_size, sample_size]

        filtered_candidate_relations = filter_candidates_by_classification_results(
            textual_candidate_relations,
            relation_clf_outputs
        )

        # Task 2. Get prediction from Entity Classification
        input_ids_entity_concatenated = []
        for (nlq, cand_ents, cand_relations) in zip(textual_nlq, textual_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + nlq # prefix
            else:
                input_src = nlq
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
        entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
        entity_clf_outputs = torch.reshape(entity_clf_predict_logits, (-1, self.entity_sample_size)) # [batch_size, sample_size]

        filtered_candidate_entities = filter_candidates_by_classification_results(
            textual_candidate_entities,
            entity_clf_outputs,
            extract_label=True
        )

        # Task 3: Structure Generation
        # No Need to do this in inference, because the output of structure is not what we need

        # Task 4: Slot Filling to generate final S-Expression
        input_ids_gen_concatenated = []
        for (nlq, cand_ents, cand_rels) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations):
            input_src = 'Translate to S-Expression: ' # prefix
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel

            if self.do_lower:
                input_src = input_src.lower()
            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )

            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))
        
        return gen_outputs, relation_clf_outputs, entity_clf_outputs


class T5_SExpr_Generation_Structure_Generation_Concat(nn.Module):
    """
    Structure Generation and SExpr Generation multitask
    + 结构生成的时候，拼接分类得到的关系和实体的数量
    + 结构生成的结果拼接到最后的生成模型，调用 generate 方法，beam_size=1
    + max_tgt_len: 110
    """
    def __init__(
        self, 
        pretrained_model_path, 
        is_test=False,
        device='cuda',
        max_src_len=256, 
        max_tgt_len=110,
        tokenizer=None,
        entity_sample_size=10,
        relation_sample_size=10,
        do_lower=False,
        add_prefix=False,
        structure_gen_beam_size=1,
        max_structure_tgt_len=50,
    ):
        super().__init__()
        self._is_test = is_test
        # All tasks use the same T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(pretrained_model_path)
        self.entity_sample_size = entity_sample_size
        self.relation_sample_size = relation_sample_size
        self.do_lower = do_lower
        self.add_prefix = add_prefix # 对比实验，分类任务加不加 prefix
        self.REL_TOKEN = ' [REL] '
        self.ENT_TOKEN = ' [ENT] '
        self.LITERAL_TOKEN = ' [LIT] '
        self.SEPERATOR = ' | '
        self.structure_gen_beam_size = structure_gen_beam_size
        self.max_structure_tgt_len = max_structure_tgt_len

        if 't5-large' in pretrained_model_path.lower():
            self.hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            self.hidden_size = 512
        else:
            self.hidden_size = 768
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(self.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        input_ids_relation_clf=None,
        relation_clf_labels=None,
        entity_clf_labels=None,
        structure_gen_labels=None, 
        gen_labels=None,
        relation_clf_attention_mask=None,
        textual_candidate_relations=None,
        textual_candidate_entities=None,
        textual_nlq=None,
        normalize_relations=True
    ):
        # data preprocess
        textual_candidate_relations = list_reshape(
            textual_candidate_relations, 
            int(len(textual_candidate_relations)/self.relation_sample_size), 
            self.relation_sample_size
        )
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities)/self.entity_sample_size),
            self.entity_sample_size
        )

        # Task 1: Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        if relation_clf_labels is not None:
            relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
            relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
            relation_clf_loss = self.criterion(relation_clf_predict_logits.float(), relation_clf_labels.unsqueeze(1).float())
            relation_clf_predict_logits = torch.reshape(relation_clf_predict_logits, (gen_labels.size(0), self.relation_sample_size))
        
            filtered_candidate_relations = filter_candidates_by_classification_results(
                textual_candidate_relations,
                relation_clf_predict_logits
            )
        else:
            relation_clf_loss = None
            filtered_candidate_relations = None
        
        # Task 2: Entity Classification
        input_ids_entity_concatenated = []
        for (nlq, cand_ents, cand_relations) in zip(textual_nlq, textual_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + nlq # prefix
            else:
                input_src = nlq
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        if entity_clf_labels is not None:
            entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
            entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
            entity_clf_loss = self.criterion(entity_clf_predict_logits.float(), entity_clf_labels.unsqueeze(1).float())
            entity_clf_predict_logits = torch.reshape(entity_clf_predict_logits, (gen_labels.size(0), self.entity_sample_size))

            filtered_candidate_entities = filter_candidates_by_classification_results(
                textual_candidate_entities,
                entity_clf_predict_logits,
                extract_label=True
            )
        else:
            entity_clf_loss = None
            filtered_candidate_entities = None
        
        # Task 3: Structure Generation
        input_ids_structure_gen_concatenated = []
        for (nlq, cand_ents, cand_rels) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations):
            input_src = 'Translate to Structure: ' # prefix
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN # 只加 [ENT], 不加具体的实体
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                input_src += self.REL_TOKEN
            input_src += self.SEPERATOR
            if self.do_lower:
                input_src = input_src.lower()
            tokenized_structure_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_structure_gen_concatenated.append(tokenized_structure_src)
        
        structure_src_encoded = self.tokenizer.pad({'input_ids': input_ids_structure_gen_concatenated},return_tensors='pt')
        structure_gen_outputs = self.t5(
            input_ids=structure_src_encoded['input_ids'].to(self.device),
            attention_mask=structure_src_encoded['attention_mask'].to(self.device),
            labels=structure_gen_labels,
        )
        structure_loss = structure_gen_outputs['loss']
        with torch.no_grad():
            structure_gen_prediction = self.t5.generate(
                input_ids=structure_src_encoded['input_ids'].to(self.device),
                attention_mask=structure_src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=self.structure_gen_beam_size,
                num_return_sequences=self.structure_gen_beam_size,
                max_length=self.max_structure_tgt_len
            )
            structure_gen_prediction = torch.reshape(structure_gen_prediction,(structure_src_encoded['input_ids'].shape[0],self.structure_gen_beam_size,-1))
            structure_gen_prediction = [p.cpu().numpy() for p in structure_gen_prediction]
            structure_gen_prediction = [self.tokenizer.batch_decode(pred, skip_special_tokens=True) for pred in structure_gen_prediction]
        
        # Task 4: Slot Filling to generate final S-Expression
        # Format: Structure | [ENT] ... [ENT] | [REl] ....[REL] | [LIT] 2005-01-10 | NLQ
        input_ids_gen_concatenated = []
        for (nlq, cand_ents, cand_rels, structure_pred) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations, structure_gen_prediction):
            input_src = 'Translate to S-Expression: ' # prefix
            input_src += structure_pred[0] + self.SEPERATOR # structure beam size should be 1
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel
            if self.do_lower:
                input_src = input_src.lower()
            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )
        gen_loss = gen_outputs['loss']

        if entity_clf_loss is not None and relation_clf_loss is not None and structure_loss is not None:
            total_loss = gen_loss + entity_clf_loss + relation_clf_loss + structure_loss
        else:
            total_loss = gen_loss
        
        return total_loss
    
    def inference(
        self,
        input_ids_relation_clf=None,
        relation_clf_attention_mask=None,
        num_beams=50,
        textual_candidate_relations=None,
        textual_candidate_entities=None,
        textual_nlq=None,
        normalize_relations=True
    ):
        # data preprocess
        textual_candidate_relations = list_reshape(
            textual_candidate_relations, 
            int(len(textual_candidate_relations)/self.relation_sample_size), 
            self.relation_sample_size
        )
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities)/self.entity_sample_size),
            self.entity_sample_size
        )

        # Task 1. Get prediction from Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
        relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
        relation_clf_outputs = torch.reshape(relation_clf_predict_logits, (-1, self.relation_sample_size)) # [batch_size, sample_size]

        filtered_candidate_relations = filter_candidates_by_classification_results(
            textual_candidate_relations,
            relation_clf_outputs
        )

        # Task 2. Get prediction from Entity Classification
        input_ids_entity_concatenated = []
        for (nlq, cand_ents, cand_relations) in zip(textual_nlq, textual_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + nlq # prefix
            else:
                input_src = nlq
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
        entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
        entity_clf_outputs = torch.reshape(entity_clf_predict_logits, (-1, self.entity_sample_size)) # [batch_size, sample_size]

        filtered_candidate_entities = filter_candidates_by_classification_results(
            textual_candidate_entities,
            entity_clf_outputs,
            extract_label=True
        )

        # Task 3: Structure Generation
        input_ids_structure_gen_concatenated = []
        for (nlq, cand_ents, cand_rels) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations):
            input_src = 'Translate to Structure: ' # prefix
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN # 只加 [ENT], 不加具体的实体
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                input_src += self.REL_TOKEN
            input_src += self.SEPERATOR
            if self.do_lower:
                input_src = input_src.lower()
            
            tokenized_structure_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_structure_gen_concatenated.append(tokenized_structure_src)
        
        structure_src_encoded = self.tokenizer.pad({'input_ids': input_ids_structure_gen_concatenated},return_tensors='pt')
        with torch.no_grad():
            structure_gen_prediction = self.t5.generate(
                input_ids=structure_src_encoded['input_ids'].to(self.device),
                attention_mask=structure_src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=self.structure_gen_beam_size,
                num_return_sequences=self.structure_gen_beam_size,
                max_length=self.max_structure_tgt_len
            )

            structure_gen_prediction = torch.reshape(structure_gen_prediction,(structure_src_encoded['input_ids'].shape[0],self.structure_gen_beam_size,-1))
            structure_gen_outputs = structure_gen_prediction
            structure_gen_prediction = [p.cpu().numpy() for p in structure_gen_prediction]
            structure_gen_prediction = [self.tokenizer.batch_decode(pred, skip_special_tokens=True) for pred in structure_gen_prediction]
        # Task 4: Slot Filling to generate final S-Expression
        input_ids_gen_concatenated = []
        for (nlq, cand_ents, cand_rels, structure_pred) in zip(textual_nlq, filtered_candidate_entities, filtered_candidate_relations, structure_gen_prediction):
            input_src = 'Translate to S-Expression: ' # prefix
            input_src += structure_pred[0] + self.SEPERATOR
            input_src += nlq + self.SEPERATOR
            for cand_ent in cand_ents:
                input_src += self.ENT_TOKEN + cand_ent
            input_src += self.SEPERATOR
            for cand_rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(cand_rel)
                else:
                    input_src += self.REL_TOKEN + cand_rel

            if self.do_lower:
                input_src = input_src.lower()
            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )

            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))
        
        return gen_outputs, relation_clf_outputs, entity_clf_outputs, structure_gen_outputs