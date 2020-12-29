#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: loss_strategy.py
Description: dynamic weight loss strategy
Author: Barry Chow
Date: 2020/12/28 6:53 PM
Version: 0.1
"""
import numpy as np
from abc import abstractmethod,ABCMeta
from sklearn.metrics import accuracy_score
import torch.nn as nn

class LossStrategy(metaclass=ABCMeta):
    #loss optimizer abstract class 
    def __init__(self):
        super(LossStrategy, self).__init__()

    @abstractmethod
    def __call__(self, **kwargs):
        #calculate the loss
        pass

class Average(LossStrategy):
    ''' average mean loss, average '''
    def __init__(self,criterion=nn.CrossEntropyLoss()):
        super(Average,self).__init__()
        self.criterion = criterion

    def __call__(self, **batch_outputs_labels):
        total_loss = 0
        for task,(outputs,labels) in batch_outputs_labels.items():
            loss = self.criterion(outputs,labels)
            #preds = np.argmax(outputs,dim=1).flatten().tolist()
            total_loss+=loss
        return total_loss

class Weighted(LossStrategy):
    ''' average mean loss, average '''
    def __init__(self,criterion=nn.CrossEntropyLoss(),weights={'OCEMOTION':0.36442725,'OCNLI':0.28560992,'TNEWS':0.34996283}):
        super(Weighted,self).__init__()
        self.criterion = criterion
        self.weights = weights

    def __call__(self, **batch_outputs_labels):
        total_loss = 0
        for task_type,(outputs,labels) in batch_outputs_labels.items():
            loss = self.criterion(outputs,labels)
            #preds = np.argmax(outputs,dim=1).flatten().tolist()
            total_loss+=loss*self.weights[task_type]
        return total_loss

class Dtp(LossStrategy):
    '''
    Dynamic Task Prioritization, dtp, based
    on "Dynamic Task Prioritization for Multitask Learning"
    In this implementation, we take accuracy score as kpi.
    '''
    def __init__(self,criterion=nn.CrossEntropyLoss(),kpi=accuracy_score):
        super(Dtp,self).__init__()
        self.criterion = criterion
        self.kpi = kpi

    def __call__(self, **batch_outputs_labels):
        pass

LOSS_STRATEGY = {
    'Average':Average,
    'Weighted':Weighted,
    'Dtp':Dtp,
}