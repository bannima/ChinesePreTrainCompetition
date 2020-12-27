#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: multitask_finetune.py
Description: 
Author: Barry Chow
Date: 2020/12/9 6:13 PM
Version: 0.1
"""

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm
import random
import datetime
import time
from data_generator import Corpus
import json
import os
from itertools import zip_longest
from models import MODELS
from stats_visualizer import stats_visualize

TASK_TYPES = ['OCEMOTION','OCNLI','TNEWS']

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

class MultiTaskFineTune():

    def __init__(self, model_name='BertBaseLinear', retrain_model_path=None):
        super(MultiTaskFineTune).__init__()

        #是否重新开始训练，若从已训练好的模型继续训练，则输入模型路径，否则为None
        self.retrain_model_path = retrain_model_path
        self._set_device()
        self._set_random_seed()
        HYPERS['MODEL_NAME'] = model_name

        #model_path = os.path.join(os.getcwd(),'pretrain_model/bert-base-chinese/')

        # 加载model
        # self.model = BERTLinearModel(model_path)
        self.model,self.tokenizer = self._load_model_tokenizer(model_name)

        self.model_path = os.path.join(os.getcwd(), MODELS[model_name]['path'])

        #demo experiment
        '''self.roberta = RobertaModel.from_pretrained(self.model_path)
        text = "Replace me by any text you'd like."
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.roberta(**encoded_input)'''

        corpus = Corpus(self.tokenizer,HYPERS['BATCH_SIZE'],self.seed_val)

        self.ocemo_train_loader, self.ocnli_train_loader, self.tnews_train_loader = corpus.get_train_dataloader()

        print("## train times with batch size 64 is {}".format(len(self.ocemo_train_loader)))

        self.valid_loaders = corpus.get_valid_dataloader()

        print("## valid times with batch size 64 is {}".format(len(self.valid_loaders[0])))

        self.test_loaders = corpus.get_test_dataloader()

        self.all_idx2label = corpus.get_idx2label()

        # move to GPU
        if torch.cuda.is_available():
            self.model.cuda()
            if self.n_gpu>1:
                self.model = torch.nn.DataParallel(self.model)

        # optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = HYPERS['LEARNING_RATE'],
            eps = 1e-8
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

    def _load_model_tokenizer(self,model_name):
        model_path = os.path.join(os.getcwd(), MODELS[model_name]['path'])

        assert model_name in MODELS
        if self.retrain_model_path is None:

            print("### 从 model path加载预训练模型, 重新训练 ###")
            model = MODELS[model_name]['class']()
        else:
            print("### 加载已预训练好模型，继续训练 ###")
            model = self.load_model(self.retrain_model_path)

        print('## Model {} loaded. ##'.format(HYPERS['MODEL_NAME']))

        tokenizer = MODELS[model_name]['tokenizer'].from_pretrained(model_path)

        print("## Tokenizer {} loaded. ##".format(tokenizer.__class__.__name__))

        return model,tokenizer

    def _set_random_seed(self):
        # Set the seed value all over the place to make this reproducible.
        self.seed_val = 2020
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)

    def _set_device(self):
        #设置运行环境 GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
            logging.info(msg='\n Using GPU, {} device available'.format(self.n_gpu))
        else:
            logging.info(msg='\n Using CPU ... ')
            self.device = torch.device("cpu")

    def run_epoch(self):
        '''分批次训练模型，每批次均进行train-validation-test的过程，并报告训练过程统计数据'''
        # list to store a number of quantities such as
        # training and validation loss, validation accuracy, and timings.
        training_stats = []

        # Total number of training steps is [number of batches] x [number of epochs].
        total_steps = len(self.ocnli_train_loader) * HYPERS['EPOCHS']

        # create the learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        total_t0 = time.time()

        for epoch in tqdm(range(1,HYPERS['EPOCHS']+1),desc="Training All {} Epochs".format(HYPERS['EPOCHS']),unit='epoch'):
            # Measure how long the training epoch takes.
            #print("Train for epoch: %d ".format(epoch))
            t0 = time.time()

            # Put the model into training model
            self.model.train()

            epoch_train_loss = self._train(epoch)
            avg_train_loss = epoch_train_loss/len(self.ocemo_train_loader)
            print("## train loss for epoch {} is {} ".format(epoch,epoch_train_loss))

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            t0 = time.time()

            #eval mode
            self.model.eval()

            epoch_eval_loss, eval_metrics = self._eval(epoch)
            avg_eval_loss = epoch_eval_loss/len(self.valid_loaders[0])

            print("## Validation loss for epoch {} is {} ".format(epoch,epoch_eval_loss))
            mean_f1_score = np.mean(eval_metrics)
            print("## Validation macro f1 score for epoch {} is {} ".format(epoch,mean_f1_score))

            for task_type_,task_f1_ in zip(TASK_TYPES,eval_metrics):
                print("## Validation f1 for {} is {} ".format(task_type_, task_f1_))

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            self.test(epoch)

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'Epoch':epoch,
                    'Train Loss':epoch_train_loss,
                    'Avg Train Loss':avg_train_loss,
                    'Train Time': training_time,
                    'Valid Loss':epoch_eval_loss,
                    'Avg Valid Loss':avg_eval_loss,
                    'Valid Time':validation_time,
                    'Ocemotion F1': eval_metrics[0],
                    'Ocnli F1':eval_metrics[1],
                    'Tnews F1':eval_metrics[2],
                    'Valid F1': np.mean(eval_metrics),
                    'Test Time':time.strftime("%Y%m%d_%H%M",time.localtime()),
                    'Test Results Dir':'Model{}_BATCH{}_Epoch{}_LR{}/'.format(HYPERS['MODEL_NAME'],HYPERS['BATCH_SIZE'],epoch,HYPERS['LEARNING_RATE'])
                }
            )
            self._save_stats(training_stats)

        print("\n### Training complete! ###")
        print("### Total training took {} ###".format(str(format_time(time.time()-total_t0))))
        #可视化结果
        try:
            stats_visualize(training_stats,HYPERS)
        except:
            pass

    def _train(self, epoch):
        total_train_loss = 0
        for batches in zip_longest(tqdm(self.ocemo_train_loader,desc="Epoch {}".format(epoch),unit='batch'),self.ocnli_train_loader,self.tnews_train_loader):
            for batch,task_type in zip(batches,TASK_TYPES):
                #zip longest对于不同长度的迭代器，会默认补None
                if batch is None: continue

                """input_ids = batch[0].to(self.device)
                attention_masks = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                labels = batch[3].to(self.device)"""
                inputs = [data.to(self.device) for data in batch[:-1]]

                labels = batch[-1].to(self.device)

                # clear any previously calculated gradients before performing a backward pass.
                self.model.zero_grad()

                #outputs = self.model(task_type,input_ids,token_type_ids,attention_masks)
                outputs = self.model(task_type,*inputs)

                loss = self.criterion(outputs,labels)

                # perform a backward pass to calculate the gradients
                loss.backward()

                total_train_loss  += loss.item()

                # normalization of the gradients to 1.0 to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # update parameters and take a step using the computed gradient.
                self.optimizer.step()

            # update the learning rate
            self.scheduler.step()

        return total_train_loss

    def _eval(self, epoch, metric=f1_score):
        '''逐个任务分别预测'''
        #task_metrics = {}
        epoch_eval_loss = 0
        eval_metrics = []
        for task_type,valid_loader in zip(TASK_TYPES,self.valid_loaders):
            predicted_labels = []
            target_labels = []
            for batch in valid_loader:
                # unpack batch from dataloader
                '''input_ids = batch[0].to(self.device)
                attention_masks = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                labels = batch[3].to(self.device)'''
                inputs = [data.to(self.device) for data in batch[:-1]]
                labels = batch[-1].to(self.device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    #forward pass, calculate logit predictions
                    outputs = self.model(task_type,*inputs)
                    loss = self.criterion(outputs, labels)
                    epoch_eval_loss += loss.item()

                    # move logits and labels to CPU
                    logits = outputs.detach().cpu().numpy()
                    label_ids = labels.to("cpu").numpy()

                    y_pred = np.argmax(logits, axis=1).flatten()

                    predicted_labels.extend(y_pred.tolist())
                    target_labels.extend(label_ids)

            #计算某一任务的metric
            task_f1 = metric(predicted_labels,target_labels,average='macro')
            eval_metrics.append(task_f1)
        return epoch_eval_loss,eval_metrics

    #预测结果
    def test(self,epoch=0):
        #逐个任务的分别预测，与train模式一批次分三次循环训练不同
        for task_type,idx2label,test_loader in zip(TASK_TYPES,self.all_idx2label,self.test_loaders):
            print('Predict task '+str(task_type)+ ' ... ')
            predictions = []
            for batch in tqdm(test_loader, desc='Testing', unit='batch'):
                # unpack batch from dataloader
                '''input_ids = batch[0].to(self.device)
                attention_masks = batch[1].to(self.device)
                token_type_ids = batch[2].to(self.device)
                '''
                #inputs = batch.to(self.device)
                inputs = [data.to(self.device) for data in batch]

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    #forward pass, calculate logit predictions
                    outputs = self.model(task_type,*inputs)
                    # move logits and labels to CPU
                    logits = outputs.detach().cpu().numpy()
                    y_pred = np.argmax(logits, axis=1).flatten()
                    predictions.extend(y_pred.tolist())

            self._save_predictions(epoch, task_type, predictions, idx2label)

    def _save_predictions(self, epoch, task_type, predicts, idx2label):
        filepath = './finetuned_results/' +'Model{}_BATCH{}_Epoch{}_LR{}/'.format(HYPERS['MODEL_NAME'],HYPERS['BATCH_SIZE'],epoch,HYPERS['LEARNING_RATE'])
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        filename = task_type.lower() + '_predict.json'
        with open(filepath + filename, 'w') as fw:
            idx = 0
            for predict in predicts:
                predict = {'id':str(idx),'label':str(idx2label[predict])}
                json.dump(predict,fw)
                fw.write('\n')
                idx += 1

    def _save_stats(self,stats):
        '''保存训练过程数据'''
        df = pd.DataFrame(stats)
        filename = 'Stats_{}_BATCH{}_Epoch{}_LR{}.csv'.format(HYPERS['MODEL_NAME'],HYPERS['BATCH_SIZE'],\
                HYPERS['EPOCHS'],HYPERS['LEARNING_RATE'])
        df.to_csv('./finetuned_results/'+filename, sep=',', encoding='utf-8', index=False)

    def save_model(self):
        #保存训练好的模型
        model_name = 'Model{}_BATCH{}_Epoch{}_LR{}_TIME{}.pkl'.format(HYPERS['MODEL_NAME'],HYPERS['BATCH_SIZE'],\
                HYPERS['EPOCHS'],HYPERS['LEARNING_RATE'],time.strftime("%Y%m%d_%H%M",time.localtime()))
        torch.save(self.model, './finetuned_model/'+model_name)
        print("Finetuned model saved! ...")

    def load_model(self, model_name):
        #加载训练好的模型
        return torch.load('./finetuned_model/'+model_name)


if __name__ =='__main__':
    # global hpyer parameters
    HYPERS = {
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 5e-5,
        'EPOCHS': 5,
    }
    #加载数据和模型
    #app = MultiTaskFineTune(model_name='BertBaseLinear')
    #app = MultiTaskFineTune(model_name='RobertaLargeLinear')
    #app = MultiTaskFineTune(model_name='RobertaBaseLinear')
    app = MultiTaskFineTune(model_name='ChineseRobertaLinear')
    #按照EPOCH数进行训练
    app.run_epoch()
    #app.test()
    #保存训练好的模型
    app.save_model()





