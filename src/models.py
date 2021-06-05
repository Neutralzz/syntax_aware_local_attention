from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.modeling_bert import *

class TokenClsModel(BertPreTrainedModel):
    def __init__(self, config):
        super(TokenClsModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.set_reg_weights()

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def set_reg_weights(self):
        for scale_module in self.bert.encoder.scale_layer:
            scale_module.reg.weight.data.zero_()
            scale_module.reg.bias.data.fill_(5.0)

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        dep_dist_matrix = batch['dep_dist_matrix']
        labels = batch['labels']
        loss_mask = batch['loss_mask'] if 'loss_mask' in batch else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            dep_dist_matrix=dep_dist_matrix)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)

class SentClsModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SentClsModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        self.set_reg_weights()

    @staticmethod
    def build_batch(batch, tokenizer):
        return batch

    def set_reg_weights(self):
        for scale_module in self.bert.encoder.scale_layer:
            scale_module.reg.weight.data.zero_()
            scale_module.reg.bias.data.fill_(5.0)

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        dep_dist_matrix = batch['dep_dist_matrix']
        labels = batch['labels']

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            dep_dist_matrix=dep_dist_matrix)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)

