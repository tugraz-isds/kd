import os
import json

import torch
from transformers import BertTokenizer
from transformers import pipeline

from bertconve.helper import log

logger = log.get_logger(__name__)


class BertFeatureExtractor:
    def __init__(self, nlp):
        self.nlp = nlp
        

    def get_head_embeddings_from_triple(self, triple_sents, heads):
        # triple_sents: ['baseball is a sport.', 'yo-yo is a toy.', 'dog capable of bark.']
        # heads: ['baseball', 'yo-yo', 'dog']
        inputs = self.nlp.tokenizer(triple_sents, return_tensors='pt', padding=True, truncation=True)['input_ids']
        head_id = self.nlp.tokenizer(heads, return_tensors='pt', padding=True, add_special_tokens=False, truncation=True)['input_ids']

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            head_id = head_id.cuda()

        last_hidden = torch.tensor(self.nlp(triple_sents, truncation=True)).to(inputs.device)
        # last_hidden.shape

        head_id_idx = self._pad_head_id_to_triple_size(head_id, inputs.shape[1])
        emb = last_hidden*head_id_idx.unsqueeze_(dim=2)
        return emb.sum(dim=1)/head_id_idx.sum(dim=1)



    @staticmethod
    def _pad_head_id_to_triple_size(head_id, sent_len):
        # torch.tensor(s).shape
        is_head = head_id != 0
        head_len = is_head.shape[1]
        m = torch.nn.ConstantPad2d((1, sent_len - head_len - 1, 0, 0), False)
        return m(is_head)



