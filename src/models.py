#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: models.py
Description: 
Author: Barry Chow
Date: 2020/12/20 9:13 PM
Version: 0.1
"""
from transformers import RobertaTokenizer,RobertaModel
from transformers import BertModel,BertTokenizer
import torch.nn as nn
import os

class BertBaseLinear(nn.Module):
    '''Simple BERT with three Linear output'''
    def __init__(self):
        super(BertBaseLinear, self).__init__()
        self.model_path = os.path.join(os.getcwd(),MODELS[self.__class__.__name__]['path'])

        self.bert_base = BertModel.from_pretrained(self.model_path)

        self.ocemo_linear = nn.Linear(768,7)
        self.ocnli_linear = nn.Linear(768,3)
        self.tnews_linear = nn.Linear(768,15)

    def forward(self, task_type, *inputs):
        pooled_outputs = self.bert_base(*inputs)
        if task_type=='OCEMOTION':
            return self.ocemo_linear(pooled_outputs[1])
        elif task_type=='OCNLI':
            return self.ocnli_linear(pooled_outputs[1])
        elif task_type=='TNEWS':
            return self.tnews_linear(pooled_outputs[1])
        else:
            raise ValueError("Unknown task type for "+task_type)

class RobertaLargeLinear(nn.Module):
    '''roberta-large'''
    def __init__(self):
        super(RobertaLargeLinear, self).__init__()
        self.model_path = os.path.join(os.getcwd(),MODELS[self.__class__.__name__]['path'])

        self.roberta_large = RobertaModel.from_pretrained(self.model_path)

        self.ocemo_linear = nn.Linear(1024, 7)
        self.ocnli_linear = nn.Linear(1024, 3)
        self.tnews_linear = nn.Linear(1024, 15)

    def forward(self, task_type, *inputs):
        pooled_outputs = self.roberta_large(*inputs)
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(pooled_outputs[1])
        elif task_type == 'OCNLI':
            return self.ocnli_linear(pooled_outputs[1])
        elif task_type == 'TNEWS':
            return self.tnews_linear(pooled_outputs[1])
        else:
            raise ValueError("Unknown task type for " + task_type)


class RobertaBaseLinear(nn.Module):
    '''roberta-base'''
    def __init__(self):
        super(RobertaBaseLinear,self).__init__()
        self.model_path = os.path.join(os.getcwd(), MODELS[self.__class__.__name__]['path'])

        self.roberta_base = RobertaModel.from_pretrained(self.model_path)

        self.ocemo_linear = nn.Linear(768, 7)
        self.ocnli_linear = nn.Linear(768, 3)
        self.tnews_linear = nn.Linear(768, 15)

    def forward(self,task_type,*inputs):
        pooled_outputs = self.roberta_base(*inputs)
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(pooled_outputs[1])
        elif task_type == 'OCNLI':
            return self.ocnli_linear(pooled_outputs[1])
        elif task_type == 'TNEWS':
            return self.tnews_linear(pooled_outputs[1])
        else:
            raise ValueError("Unknown task type for " + task_type)


class ChineseRobertaLinear(nn.Module):
    '''hfl/chinese-roberta-wwm-ext'''
    def __init__(self):
        super(ChineseRobertaLinear,self).__init__()
        self.model_path = os.path.join(os.getcwd(), MODELS[self.__class__.__name__]['path'])
        #chinese-roberta-wwm-ext使用BertModel BertTokenizer
        self.chinese_roberta = BertModel.from_pretrained(self.model_path)
        self.ocemo_linear = nn.Linear(768, 7)
        self.ocnli_linear = nn.Linear(768, 3)
        self.tnews_linear = nn.Linear(768, 15)

    def forward(self,task_type,*inputs):
        pooled_outputs = self.chinese_roberta(*inputs)
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(pooled_outputs[1])
        elif task_type == 'OCNLI':
            return self.ocnli_linear(pooled_outputs[1])
        elif task_type == 'TNEWS':
            return self.tnews_linear(pooled_outputs[1])
        else:
            raise ValueError("Unknown task type for " + task_type)

#所有模型
MODELS =  {
    'BertBaseLinear':{
        'class':BertBaseLinear,
        'tokenizer':BertTokenizer,
        'path':'pretrain_model/bert-base-chinese/',
    },
    'RobertaLargeLinear':{
        'class':RobertaLargeLinear,
        'tokenizer':RobertaTokenizer,
        'path':'pretrain_model/roberta-large/'
    },
    'RobertaBaseLinear':{
        'class':RobertaBaseLinear,
        'tokenizer':RobertaTokenizer,
        'path':'pretrain_model/roberta-base/'
    },
    'ChineseRobertaLinear':{
        'class':ChineseRobertaLinear,
        'tokenizer':BertTokenizer,
        'path':'pretrain_model/chinese-roberta-wwm-ext'
    }
}

