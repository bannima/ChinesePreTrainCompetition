#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: data_generator.py
Description: 生成数据loader并预处理
Author: Barry Chow
Date: 2020/12/22 10:41 AM
Version: 0.1
"""

from torch.utils.data import DataLoader,TensorDataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import RandomSampler,SequentialSampler
from tqdm import tqdm
import csv

#将数据转换为bert的输入
def convert_to_bert_dataset(data, labels, tokenizer, max_length):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for _,row in tqdm(data.iterrows(),total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(
            row['document'],max_length=max_length,pad_to_max_length=True,\
            return_attention_mask=True,return_tensors='pt',truncation=True)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        #对于bert输入有token_type_ids,其他模型没有
        try:
            token_type_ids.append(encoded_dict['token_type_ids'])
        except:
            pass

    #convert lists to tensor
    input_ids = torch.cat(input_ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)

    if len(token_type_ids)!=0:
        token_type_ids = torch.cat(token_type_ids,dim=0)

    #对于测试集，没有labels
    if labels is None:
        #对于非Bert类输入，没有token_type_ids
        if len(token_type_ids)==0:
            return TensorDataset(input_ids,attention_masks)
        return TensorDataset(input_ids,attention_masks,token_type_ids)
    #训练集和测试集，有labels
    else:
        labels = torch.tensor(labels.values)
        if len(token_type_ids)==0:
            return TensorDataset(input_ids,attention_masks,labels)
        return TensorDataset(input_ids,attention_masks,token_type_ids,labels)

class Corpus():

    def __init__(self,tokenizer,batch_size,seed_val):

        self.batch_size = batch_size
        self.seed_val = seed_val
        #保持验证集和测试集大致数量相等,约为1500
        #self.test_scale = test_scale
        self.test_scale = {
            'OCEMOTION':0.04202,#1500/35693=0.04203
            'OCNLI':0.028093, #1500/53386=0.0281
            'TNEWS':0.02367, #1500/63359=0.02367
        }

        # 加载模型分词器具
        self.tokenizer = tokenizer
        print("### Dataset processing... ###")

        # 加载数据集
        # OCEMO
        print(" processing OCEMOTION ...")

        OCEMO_train = pd.read_csv('../dataset/OCEMOTION_train1128.csv', sep='\t',
                                  quoting=csv.QUOTE_NONE,names=['id', 'document', 'class'])
        OCEMO_test = pd.read_csv('../dataset/OCEMOTION_a.csv',sep='\t',quoting=csv.QUOTE_NONE,names=['id','document'])

        #在去除掉验证集1500之后，计算训练集的长度
        ocemo_train_len = int(len(OCEMO_train)*(1-self.test_scale['OCEMOTION']))

        # OCNLI
        print(" processing OCNLI ... ")
        OCNLI_train = pd.read_csv('../dataset/OCNLI_train1128.csv', sep='\t',
                                  quoting=csv.QUOTE_NONE,names=['id', 'sentence1', 'sentence2', 'class'])
        OCNLI_train['document'] = OCNLI_train['sentence1'].str.cat(OCNLI_train['sentence2'], sep='[SEP]')
        OCNLI_test = pd.read_csv('../dataset/OCNLI_a.csv',quoting=csv.QUOTE_NONE,sep='\t',names=['id','sentence1','sentence2'])
        OCNLI_test['document'] = OCNLI_test['sentence1'].str.cat(OCNLI_test['sentence2'], sep='[SEP]')
        ocnli_train_len = int(len(OCNLI_train)*(1-self.test_scale['OCNLI']))

        # TNEWS
        print(" processing TNEWS ... ")
        TNEWS_train = pd.read_csv('../dataset/TNEWS_train1128.csv', sep='\t',
                                  quoting=csv.QUOTE_NONE,names=['id', 'document', 'class'])
        TNEWS_test = pd.read_csv('../dataset/TNEWS_a.csv',quoting=csv.QUOTE_NONE,sep='\t',names=['id','document'])
        tnews_train_len = int(len(TNEWS_train)*(1-self.test_scale['TNEWS']))

        #根据三个数据集的长度，来计算每个batch中各个数据集的mini batch size,确保三个数据集训练的次数相同
        total_len = sum([ocemo_train_len,ocnli_train_len,tnews_train_len])
        ocemo_batch_size = round(float(ocemo_train_len/total_len)*self.batch_size)
        ocnli_batch_size = round(float(ocnli_train_len/total_len)*self.batch_size)
        tnews_batch_size = round(float(tnews_train_len/total_len)*self.batch_size)
        print("## dataset size:  OCEMOTION is {}, OCNLI is {}, TNEWS is {} ##".format(len(OCEMO_train),len(OCNLI_train),len(TNEWS_train)))
        print("## min batch size:  OCEMOTION is {}, OCNLI is {}, TNEWS is {} ##".format(ocemo_batch_size,ocnli_batch_size,tnews_batch_size))

        #预处理数据集
        #计算max length, 计算耗时，直接保存结果512
        self.max_length=512
        print(" max token length of FineTune model : ", self.max_length)

        '''
        ocemo_maxlen = self.calc_max_length(OCEMO_train)
        ocnli_maxlen = self.calc_max_length(OCNLI_train)
        tnews_maxlen = self.calc_max_length(TNEWS_train)

        max_len = max((ocemo_maxlen,ocnli_maxlen,tnews_maxlen))
        # 将输入统一为相同的最大长度, BERT最多接受长度为512的输入
        self.max_length = min(512,pow(2, int(np.log2(max_len) + 1)))
        '''

        # OCEMO
        print(' Load OCEMOTION dataset ... ')
        self.ocemo_train_loader,self.ocemo_valid_loader,self.ocemo_test_loader,\
            self.ocemo_idx2label = self.loader_generator(self.test_scale['OCEMOTION'], OCEMO_train[['document', 'class']], \
                                                         OCEMO_test[['document']], ocemo_batch_size)

        #OCNLI
        print(' Load OCNLI dataset ... ')
        self.ocnli_train_loader,self.ocnli_valid_loader,self.ocnli_test_loader,\
            self.ocnli_idx2label = self.loader_generator(self.test_scale['OCNLI'], OCNLI_train[['document', 'class']], \
                                                         OCNLI_test[['document']], ocnli_batch_size)

        #TNEWS
        print(' Load TNEWS dataset ... ')
        self.tnews_train_loader,self.tnews_valid_loader,self.tnews_test_loader,\
            self.tnews_idx2label = self.loader_generator(self.test_scale['TNEWS'], TNEWS_train[['document', 'class']], \
                                                         TNEWS_test[['document']], tnews_batch_size)

        print('## All Train and Test Data loaded ! ##')

    def loader_generator(self, test_size, train_data, test_data, mini_batch_size):
        '''生成训练、验证、测试集合的dataloader，注意训练集batch size根据三个任务不同长度计算而来'''
        labels = set(train_data['class'])
        # 计算label2idx, id2label
        label2idx = {}
        idx2label = {}
        idx = 0
        for label in labels:
            label2idx[label] = idx
            idx += 1
        for label, idx in label2idx.items():
            idx2label[idx] = label

        #将原始类别class(如 sadness,happiness)转为label(0,1,2)
        transfer_labels = train_data.loc[:,'class'].map(label2idx)
        train_data.insert(0,'labels',transfer_labels)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            train_data[['document']], train_data['labels'], test_size=test_size, random_state=self.seed_val
        )

        train = convert_to_bert_dataset(X_train, y_train, self.tokenizer, self.max_length)
        validation = convert_to_bert_dataset(X_val, y_val, self.tokenizer, self.max_length)
        test = convert_to_bert_dataset(test_data[['document']], None, self.tokenizer, self.max_length)

        # 生成训练 验证 dataloader
        train_dataloader = DataLoader(
            train,
            sampler=RandomSampler(train),
            batch_size=mini_batch_size
        )

        valid_dataloader = DataLoader(
            validation,
            sampler=SequentialSampler(validation),
            batch_size=self.batch_size
        )

        test_dataloader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=False
        )
        return train_dataloader,valid_dataloader,test_dataloader,idx2label


    def calc_max_length(self,data):
        # 计算输入最大长度和labels set,
        labels = set()
        max_len = 0
        for _, row in data.iterrows():
            max_len = max(max_len, len(self.tokenizer(row['document'])['input_ids']))
            labels.add(row['class'])
        return max_len,labels

    def get_train_dataloader(self):
        return self.ocemo_train_loader,self.ocnli_train_loader,self.tnews_train_loader

    def get_valid_dataloader(self):
        return self.ocemo_valid_loader,self.ocnli_valid_loader,self.tnews_valid_loader

    def get_test_dataloader(self):
        return self.ocemo_test_loader,self.ocnli_test_loader,self.tnews_test_loader

    def get_idx2label(self):
        '''输出测试结果时使用'''
        return (self.ocemo_idx2label,self.ocnli_idx2label,self.tnews_idx2label)
