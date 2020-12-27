#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
FileName: stats_visualizer.py
Description: 
Author: Barry Chow
Date: 2020/12/26 9:36 PM
Version: 0.1
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import os

global_colors = ['darkgreen',"crimson",'dodgerblue','darkturquoise','dimgray','darkviolet','darkorange']

def stats_visualize(stats, hypers):

    plt.cla()
    x_axis = stats['Epoch']
    # visualize twin y axis
    fig = plt.figure()
    plt.xlabel('Epochs', fontsize='large')

    ax1 = fig.add_subplot(111)
    #f1 值在 (0，1)之间
    ax1.set_ylim(0.4,0.9)

    ax1.plot(x_axis, stats['Ocemotion F1'], color=global_colors[0], marker='.', lw=1.5)
    ax1.plot(x_axis, stats['Ocnli F1'], color=global_colors[1], marker='.', lw=1.5)
    ax1.plot(x_axis, stats['Tnews F1'], color=global_colors[2], marker='.', lw=1.5)
    ax1.plot(x_axis, stats['Valid F1'], color=global_colors[4], marker='.', lw=1.5)
    ax1.legend(['Ocemotion F1', 'Ocnli F1','Tnews F1','Valid F1'], loc='upper left', fontsize=10)
    ax1.set_ylabel('Three tasks F1 Score', fontsize='large')

    ax2 = ax1.twinx()
    #ax2.plot(x_axis, stats['Train Loss'].map(lambda val:val/2280), color=global_colors[6], marker='*', lw=1.5, linestyle='--')
    #ax2.plot(x_axis, stats['Valid Loss'].map(lambda val:val/24), color=global_colors[5], marker='*', lw=1.5, linestyle='--')
    ax2.plot(x_axis, stats['Avg Train Loss'], color=global_colors[6], marker='*', lw=1.5, linestyle='--')
    ax2.plot(x_axis, stats['Avg Valid Loss'], color=global_colors[5], marker='*', lw=1.5, linestyle='--')

    ax2.legend(['Avg Train Loss','Avg Valid Loss'], loc='upper right', fontsize=10)
    ax2.set_ylabel('Avg Train & Valid Loss', fontsize='large')

    plt.grid(ls='--')
    fig_name = 'Stats_{}_BATCH{}_Epoch{}_LR{}'.format(hypers['MODEL_NAME'], hypers['BATCH_SIZE'], hypers['EPOCHS'], hypers['LEARNING_RATE'])
    plt.title(fig_name,color='black',fontsize='large')

    #plt.show()
    plt.savefig('./finetuned_results/'+fig_name+'.jpg')

if __name__ == '__main__':
    hypers = {
        'BATCH_SIZE': 64,
        'LEARNING_RATE': 5e-5,
        'EPOCHS': 10,
        'MODEL_NAME': 'BertBaseLinear'
    }
    file_name = 'Stats_{}_BATCH{}_Epoch{}_LR{}.csv'.format(hypers['MODEL_NAME'], hypers['BATCH_SIZE'], hypers['EPOCHS'], hypers['LEARNING_RATE'])
    file_path = os.path.join(os.getcwd(), 'finetuned_results/' + file_name)
    stats = pd.read_csv(file_path, sep=',')

    stats_visualize(stats,hypers)
