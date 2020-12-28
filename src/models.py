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
import torch.nn.functional as F
import torch
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
        #pooled_outputs = self.bert_base(*inputs)
        #class embedding outputs
        cls_embs = self.bert_base(*inputs)[0][:,0,:].squeeze(1)
        if task_type=='OCEMOTION':
            return self.ocemo_linear(cls_embs)
        elif task_type=='OCNLI':
            return self.ocnli_linear(cls_embs)
        elif task_type=='TNEWS':
            return self.tnews_linear(cls_embs)
        else:
            raise ValueError("Unknown task type for "+task_type)

class RobertaLargeLinear(nn.Module):
    '''roberta-large with linear output'''
    def __init__(self):
        super(RobertaLargeLinear, self).__init__()
        self.model_path = os.path.join(os.getcwd(),MODELS[self.__class__.__name__]['path'])
        self.roberta_large = RobertaModel.from_pretrained(self.model_path)
        self.ocemo_linear = nn.Linear(1024, 7)
        self.ocnli_linear = nn.Linear(1024, 3)
        self.tnews_linear = nn.Linear(1024, 15)

    def forward(self, task_type, *inputs):
        cls_embs = self.roberta_large(*inputs)[0][:,0,:].squeeze(1)
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(cls_embs)
        elif task_type == 'OCNLI':
            return self.ocnli_linear(cls_embs)
        elif task_type == 'TNEWS':
            return self.tnews_linear(cls_embs)
        else:
            raise ValueError("Unknown task type for " + task_type)

class RobertaBaseLinear(nn.Module):
    '''roberta-base with linear output'''
    def __init__(self):
        super(RobertaBaseLinear,self).__init__()
        self.model_path = os.path.join(os.getcwd(), MODELS[self.__class__.__name__]['path'])
        self.roberta_base = RobertaModel.from_pretrained(self.model_path)
        self.ocemo_linear = nn.Linear(768, 7)
        self.ocnli_linear = nn.Linear(768, 3)
        self.tnews_linear = nn.Linear(768, 15)

    def forward(self,task_type,*inputs):
        cls_embs = self.roberta_base(*inputs)[0][:,0,:].squeeze(1)
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(cls_embs)
        elif task_type == 'OCNLI':
            return self.ocnli_linear(cls_embs)
        elif task_type == 'TNEWS':
            return self.tnews_linear(cls_embs)
        else:
            raise ValueError("Unknown task type for " + task_type)

class ChineseRobertaLinear(nn.Module):
    '''hfl/chinese-roberta-wwm-ext with linear output'''
    def __init__(self):
        super(ChineseRobertaLinear,self).__init__()
        self.model_path = os.path.join(os.getcwd(), MODELS[self.__class__.__name__]['path'])
        #chinese-roberta-wwm-ext使用BertModel BertTokenizer
        self.chinese_roberta = BertModel.from_pretrained(self.model_path)
        self.ocemo_linear = nn.Linear(768, 7)
        self.ocnli_linear = nn.Linear(768, 3)
        self.tnews_linear = nn.Linear(768, 15)

    def forward(self,task_type,*inputs):
        cls_embs = self.chinese_roberta(*inputs)[0][:,0,:].squeeze(1)
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(cls_embs)
        elif task_type == 'OCNLI':
            return self.ocnli_linear(cls_embs)
        elif task_type == 'TNEWS':
            return self.tnews_linear(cls_embs)
        else:
            raise ValueError("Unknown task type for " + task_type)

class BertBaseAttention(nn.Module):
    '''
        bert-base-chinese with self-attention output
        The implementated self-attention mechamism is same as
       'A Structured Self-attentive Sentence Embedding' in ICLR2017
       without penalization item.
    '''
    def __init__(self):
        super(BertBaseAttention,self).__init__()
        self.model_path = os.path.join(os.getcwd(), MODELS[self.__class__.__name__]['path'])
        self.bert_base = BertModel.from_pretrained(self.model_path)

        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        da = 350
        r = 30
        self.W_s1 = nn.Linear(768, da)
        self.W_s2 = nn.Linear(da, r)
        self.fc = nn.Linear(r*768, 2000)

        self.ocemo_linear = nn.Linear(2000, 7)
        self.ocnli_linear = nn.Linear(2000, 3)
        self.tnews_linear = nn.Linear(2000, 15)

    def forward(self,task_type,*inputs):
        outputs = self.bert_base(*inputs)[0]
        hidden_matrix = self.self_att(outputs)
        fc_output = self.fc(hidden_matrix.view(-1,hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        if task_type == 'OCEMOTION':
            return self.ocemo_linear(fc_output)
        elif task_type == 'OCNLI':
            return self.ocnli_linear(fc_output)
        elif task_type == 'TNEWS':
            return self.tnews_linear(fc_output)
        else:
            raise ValueError("Unknown task type for " + task_type)

    def self_att(self,embs):
        ''''''
        attention_matrix = self.W_s2(torch.tanh(self.W_s1(embs)))
        attention_matrix = F.softmax(attention_matrix.permute(0, 2, 1), dim=2)
        hidden_matrix = torch.bmm(attention_matrix,embs)
        return hidden_matrix

#所有模型
MODELS =  {
    'BertBaseLinear':{
        'class':BertBaseLinear,
        'tokenizer':BertTokenizer,
        'path':'pretrain_model/bert-base-chinese/',
    },
    'BertBaseAttention': {
        'class': BertBaseAttention,
        'tokenizer': BertTokenizer,
        'path': 'pretrain_model/bert-base-chinese/',
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

