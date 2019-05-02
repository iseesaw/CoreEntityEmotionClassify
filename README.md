### Core Entity Emotion Classify
[2019搜狐校园算法大赛](https://www.biendata.com/competition/sohu2019/)

### 竞赛任务
给定若干文章，目标是判断文章的核心实体以及对核心实体的情感态度。每篇文章识别最多三个核心实体，并分别判断文章对上述核心实体的情感倾向（积极、中立、消极三种）。

### 模型说明
#### 命名实体识别
- BERT + BiLSTM + CRF
    + [SEQ]Sentence[SEQ] 
- 预处理
    + 将训练集文章拆分成句子级样本作为模型输入

#### 情感分类
- BERT
    + [CLS]Entity[SEQ]Article[SEQ]