## [NLP中文预训练模型泛化能力挑战赛](https://tianchi.aliyun.com/competition/entrance/531841/introduction)
(since 2020/11/20)

### Requirements

- pytorch > 1.0.0
- transformers
- sklearn
- numpy
- tqdm
- pandas

### Model Archietcture

<img src="./model_architecture.jpg" width="80%" height="50%">


### Run the following command to train:

`python3 multitask_finetune.py`  or

`nohup python3 -u multitask_finetune.py > output.log 2>&1 &`

###  Models

- bert-base-chinese + Linear : **BertBaseLinear**

- roberta-large + Linear : **RobertaLargeLinear**

- roberta-base + Linear : **RobertaBaseLinear**

- chinese-roberta-wwm-ext + Linear : **ChineseRobertaLinear**

- bert-base-chinese + Self Attention : **BertBaseAttention**


### Experiment Results

| Model | Batch Size | Learning Rate | Finetuned Model Output | Epochs | Valid F1 | Test F1 | Date | Statistics |
| :----:| :----: | :----: | :----: |:----: | :----: | :----: |  :----: |  :----: |
| BertBaseLinear | 64 | 5e-5 | pooled output | 7 | 0.5708366690203711 | 0.5834| 2020/12/26 09:30|Stats_BertBaseLinear_BATCH64_Epoch10_LR5e-05.csv |
| BertBaseLinear | 64 | 5e-5 | pooled output |10 | 0.5669902108325462 | 0.5788| 2020/12/26 11:15 |Stats_BertBaseLinear_BATCH64_Epoch10_LR5e-05.csv |
| ChineseRoberteLinear | 64 | 5e-5 | pooled output | 4 | 0.609208530047737 | 0.6041| 2020/12/27 8:30 |Stats_ChineseRobertaLinear_BATCH64_Epoch5_LR5e-05.csv |
| BertBaseAttention | 32 | 5e-5 | cls embedding |3 | 0.5805297492177061 | 0.5774 | 2020/12/28 11:10 |Stats_BertBaseAttention_BATCH32_Epoch5_LR5e-05.csv |



### References

- [关于中文预训练模型泛化能力挑战赛的调研](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.12.25a02494RQLgEY&postId=145917)

- [深度学习中的注意力机制](https://blog.csdn.net/tg229dvt5i93mxaq5a6u/article/details/78422216)